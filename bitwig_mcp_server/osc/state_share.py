"""
Share Bitwig's OSC state between several server instances

Bitwig (DrivenByMoss) sends its feedback to a single UDP port, so only one
process can receive it. Claude Desktop, however, starts one server process for
chats and another one for Cowork and Code sessions.

The process that owns the UDP port runs a small read-only HTTP endpoint on
127.0.0.1 (`StateShareServer`). Every other process mirrors the state from
there (`RemoteOSCState`) and keeps sending its commands straight to Bitwig.
If the owner goes away, a mirror takes over the UDP port and the endpoint.
"""

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .server import BitwigOSCServer

logger = logging.getLogger(__name__)

STATE_PATH = "/state"


class StateShareServer:
    """Read-only HTTP endpoint that serves the state of a BitwigOSCServer

    GET /state?since=N returns {"version": V, "messages": {...}} with every
    OSC message received after version N.
    """

    def __init__(self, osc_server: BitwigOSCServer, host: str, port: int):
        self.osc_server = osc_server
        self.host = host
        self.port = port
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start serving. Raises OSError if the port is taken."""
        osc_server = self.osc_server

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                url = urlparse(self.path)
                if url.path != STATE_PATH:
                    self.send_error(404)
                    return
                try:
                    since = int(parse_qs(url.query).get("since", ["0"])[0])
                except ValueError:
                    self.send_error(400, "since must be an integer")
                    return

                version, messages = osc_server.changes_since(since)
                body = json.dumps(
                    {"version": version, "messages": messages}, default=str
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                # Keep the MCP log free of one line per poll
                pass

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"Sharing Bitwig state on http://{self.host}:{self.port}")

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


class RemoteOSCState:
    """Mirror of another instance's OSC state, with BitwigOSCServer's interface

    Polls the owner's StateShareServer and merges the changes into
    `received_messages`. If the owner stops answering, it tries to take over
    the UDP receive port and the share endpoint itself.
    """

    def __init__(
        self,
        ip: str,
        port: int,
        state_port: int,
        poll_interval: float = 0.1,
        takeover_after: int = 10,
        request_timeout: float = 1.0,
    ):
        """
        Args:
            ip: Address of Bitwig and of the share endpoint
            port: Bitwig's OSC feedback port (taken over if the owner is gone)
            state_port: Port of the owner's share endpoint
            poll_interval: Seconds between polls
            takeover_after: Failed polls in a row before taking over
            request_timeout: Timeout of one poll request in seconds
        """
        self.ip = ip
        self.port = port
        self.state_port = state_port
        self.poll_interval = poll_interval
        self.takeover_after = takeover_after
        self.request_timeout = request_timeout

        self.received_messages: dict[str, Any] = {}
        self.version = 0
        self.running = False
        self.owner: Optional[BitwigOSCServer] = None
        self.share: Optional[StateShareServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://{self.ip}:{self.state_port}{STATE_PATH}"

    def _fetch(self) -> None:
        with urllib.request.urlopen(
            f"{self.url}?since={self.version}", timeout=self.request_timeout
        ) as response:
            data = json.load(response)
        with self._lock:
            self.received_messages.update(data["messages"])
            self.version = int(data["version"])

    def start(self) -> None:
        """Load the owner's state and start mirroring it

        Raises:
            OSError: If the owner's share endpoint is not reachable
        """
        if self.running:
            return
        try:
            self._fetch()
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            raise OSError(f"Bitwig state endpoint {self.url} not reachable: {e}")

        self.running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"Mirroring Bitwig state from {self.url}")

    def _poll_loop(self) -> None:
        failures = 0
        while self.running and self.owner is None:
            time.sleep(self.poll_interval)
            try:
                self._fetch()
                failures = 0
            except Exception as e:  # owner busy, restarting or gone
                failures += 1
                if failures >= self.takeover_after and self._take_over():
                    return
                logger.debug(f"Polling {self.url} failed ({failures}): {e}")

    def _take_over(self) -> bool:
        """Become the owner of the UDP port and the share endpoint"""
        server = BitwigOSCServer(self.ip, self.port)
        # Keep the same dict, and give the mirrored state a version so that
        # new mirrors receive it as well
        server.received_messages = self.received_messages
        server.version = 1
        server.versions = {address: 1 for address in self.received_messages}
        try:
            server.start()
        except OSError:
            return False  # someone else owns the port (e.g. another mirror)

        self.owner = server
        share = StateShareServer(server, self.ip, self.state_port)
        try:
            share.start()
            self.share = share
        except OSError as e:
            logger.warning(f"Took over port {self.port}, but not sharing: {e}")
        logger.info(f"Took over Bitwig's OSC feedback on port {self.port}")
        return True

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.share:
            self.share.stop()
        if self.owner:
            self.owner.stop()

    def get_message(self, address: str) -> Optional[Any]:
        return self.received_messages.get(address)

    def wait_for_message(self, address: str, timeout: float = 3.0) -> Optional[Any]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if address in self.received_messages:
                return self.received_messages[address]
            time.sleep(0.05)
        return None

    def clear_messages(self) -> None:
        with self._lock:
            self.received_messages.clear()
