"""Tests for sharing Bitwig's OSC state between server instances.

These tests use real sockets on free localhost ports, but never Bitwig.
"""

import socket
import time

import pytest
from pythonosc.udp_client import SimpleUDPClient

from bitwig_mcp_server.osc.server import BitwigOSCServer
from bitwig_mcp_server.osc.state_share import RemoteOSCState, StateShareServer


def free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(condition, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def ports():
    return free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_STREAM)


@pytest.fixture
def owner(ports):
    udp_port, state_port = ports
    server = BitwigOSCServer("127.0.0.1", udp_port)
    server.start()
    share = StateShareServer(server, "127.0.0.1", state_port)
    share.start()
    yield server, share
    share.stop()
    server.stop()


def test_changes_since_returns_only_newer_messages():
    server = BitwigOSCServer("127.0.0.1", 0)
    server._default_handler("/tempo/raw", 120)
    version, changed = server.changes_since(0)
    assert changed == {"/tempo/raw": 120}

    server._default_handler("/play", 1)
    server._default_handler("/tempo/raw", 121)
    new_version, changed = server.changes_since(version)
    assert changed == {"/play": 1, "/tempo/raw": 121}
    assert new_version == version + 2
    assert server.changes_since(new_version)[1] == {}


def test_mirror_loads_and_follows_owner_state(owner, ports):
    server, _ = owner
    udp_port, state_port = ports
    server._default_handler("/device/name", "Deep Pad")

    mirror = RemoteOSCState("127.0.0.1", udp_port, state_port, poll_interval=0.02)
    mirror.start()
    try:
        assert mirror.get_message("/device/name") == "Deep Pad"

        # Feedback from Bitwig arriving at the owner's UDP port
        SimpleUDPClient("127.0.0.1", udp_port).send_message("/tempo/raw", 106)
        assert mirror.wait_for_message("/tempo/raw", timeout=3.0) == 106

        # A repeated value is delivered again (needed for send_and_wait)
        mirror.received_messages.pop("/tempo/raw")
        server._default_handler("/tempo/raw", 106)
        assert mirror.wait_for_message("/tempo/raw", timeout=3.0) == 106
    finally:
        mirror.stop()


def test_mirror_start_fails_without_owner(ports):
    udp_port, state_port = ports
    mirror = RemoteOSCState("127.0.0.1", udp_port, state_port, request_timeout=0.2)
    with pytest.raises(OSError, match="not reachable"):
        mirror.start()
    assert not mirror.running


def test_mirror_takes_over_when_owner_stops(owner, ports):
    server, share = owner
    udp_port, state_port = ports
    server._default_handler("/device/name", "Deep Pad")

    mirror = RemoteOSCState(
        "127.0.0.1",
        udp_port,
        state_port,
        poll_interval=0.02,
        takeover_after=3,
        request_timeout=0.2,
    )
    mirror.start()
    try:
        share.stop()
        server.stop()

        assert wait_until(lambda: mirror.owner is not None)

        # The mirror now receives Bitwig's feedback itself ...
        SimpleUDPClient("127.0.0.1", udp_port).send_message("/play", 1)
        assert mirror.wait_for_message("/play", timeout=3.0) == 1
        assert mirror.get_message("/device/name") == "Deep Pad"

        # ... and shares the complete state with new instances
        newcomer = RemoteOSCState("127.0.0.1", udp_port, state_port)
        newcomer.start()
        try:
            assert newcomer.get_message("/device/name") == "Deep Pad"
            assert newcomer.get_message("/play") == 1
        finally:
            newcomer.stop()
    finally:
        mirror.stop()
