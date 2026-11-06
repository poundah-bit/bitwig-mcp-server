"""
Bitwig OSC Controller

High-level controller that combines client and server functionality
"""

import errno
import logging
import socket
import time
from typing import Any, Dict, List, Optional

from .client import BitwigOSCClient
from .error_handler import ErrorHandler
from .exceptions import (
    BitwigNotRespondingError,
    ConnectionError,
    ResourceNotFoundError,
    TimeoutError,
)
from .server import BitwigOSCServer
from .state_share import RemoteOSCState, StateShareServer

logger = logging.getLogger(__name__)

SEND_ONLY_MESSAGE = (
    "Running in send-only mode: another Bitwig MCP server instance already "
    "receives Bitwig's feedback on port {port} and does not share its state "
    "(it may run an older version; restart Claude Desktop to update it). "
    "Commands are sent to Bitwig, but its current state cannot be read from "
    "this instance."
)


def _is_port_in_use(error: OSError) -> bool:
    """True if binding failed because another socket already uses the port"""
    return (
        error.errno == errno.EADDRINUSE
        or getattr(error, "winerror", None) == 10048  # WSAEADDRINUSE
    )


class BitwigOSCController:
    """Controller for bidirectional OSC communication with Bitwig Studio"""

    def __init__(
        self,
        ip: str = "127.0.0.1",
        send_port: int = 8000,
        receive_port: int = 9000,
        connection_timeout: float = 10.0,  # Increased timeout for Bitwig operations
        reconnect_attempts: int = 5,  # More reconnection attempts
        state_port: Optional[int] = None,
    ):
        """Initialize the controller

        Args:
            ip: IP address of Bitwig instance
            send_port: Port to send messages to
            receive_port: Port to receive messages on
            connection_timeout: Timeout for initial connection in seconds
            reconnect_attempts: Number of reconnection attempts
            state_port: Local port on which the instance that owns
                receive_port shares Bitwig's state (default: receive_port + 1)
        """
        self.ip = ip
        self.send_port = send_port
        self.receive_port = receive_port
        self.state_port = state_port if state_port is not None else receive_port + 1
        self.connection_timeout = connection_timeout
        self.reconnect_attempts = reconnect_attempts

        # Create client and server. The server is replaced by a RemoteOSCState
        # mirror if another instance already owns receive_port.
        self.client = BitwigOSCClient(ip, send_port)
        self.server: Any = BitwigOSCServer(ip, receive_port)
        self.state_share: Optional[StateShareServer] = None
        self.mirroring = False

        # Create error handler
        self.error_handler = ErrorHandler()

        # Controller state
        self.ready = False
        self.connected = False
        # True when another instance owns the receive port and does not share
        # its state: commands are still sent, but no feedback is received.
        self.send_only = False
        self.last_refresh = 0.0
        self.connection_attempts = 0

    def start(self) -> None:
        """Start the controller

        Raises:
            ConnectionError: If unable to connect to Bitwig
        """
        try:
            # Start the OSC server, or mirror the instance that owns the port
            try:
                self.server.start()
                self._start_state_share()
            except OSError as e:
                if not _is_port_in_use(e):
                    raise
                mirror = RemoteOSCState(self.ip, self.receive_port, self.state_port)
                try:
                    mirror.start()
                except OSError as mirror_error:
                    self.send_only = True
                    self.ready = True
                    self.connected = True
                    logger.warning(
                        f"{SEND_ONLY_MESSAGE.format(port=self.receive_port)} "
                        f"({mirror_error})"
                    )
                    return
                self.server = mirror
                self.mirroring = True
                logger.info(
                    f"Port {self.receive_port} is owned by another instance; "
                    f"mirroring its state from port {self.state_port}"
                )

            # Wait briefly for server to start
            time.sleep(0.1)

            # Try to connect to Bitwig with timeout
            self._connect_with_timeout()

            # Mark as ready
            self.ready = True
            self.connected = True
            self.connection_attempts = 0
            self.error_handler.record_success()

            logger.info(
                f"OSC controller connected to Bitwig at {self.ip}:{self.send_port}"
            )

        except Exception as e:
            self.ready = False
            self.connected = False

            # Clean up
            self._stop_state_share()
            try:
                self.server.stop()
            except Exception:
                pass

            error = ConnectionError(details=str(e))
            self.error_handler.record_error("start", error)
            raise error

    def _start_state_share(self) -> None:
        """Share the received state with other instances (best effort)"""
        share = StateShareServer(self.server, self.ip, self.state_port)
        try:
            share.start()
        except OSError as e:
            logger.warning(
                f"Cannot share Bitwig state on port {self.state_port}: {e}. "
                "Other instances will run in send-only mode."
            )
            return
        self.state_share = share

    def _stop_state_share(self) -> None:
        if self.state_share:
            self.state_share.stop()
            self.state_share = None

    def _connect_with_timeout(self) -> None:
        """Attempt to connect to Bitwig with timeout

        Raises:
            ConnectionError: If unable to connect to Bitwig
            TimeoutError: If connection times out
        """
        start_time = time.time()
        self.connection_attempts += 1

        # Send ping to check if Bitwig is responding
        try:
            # Request initial state from Bitwig
            self.client.refresh()

            # Wait for a response with timeout
            while time.time() - start_time < self.connection_timeout:
                # Check if we've received any messages
                if self.server.received_messages:
                    return

                # Wait a bit before checking again
                time.sleep(0.1)

            # If we get here, we timed out
            if self.connection_attempts < self.reconnect_attempts:
                logger.warning(
                    f"Connection attempt {self.connection_attempts} timed out, retrying..."
                )
                return self._connect_with_timeout()

            raise TimeoutError("connect", self.connection_timeout)

        except socket.error as e:
            raise ConnectionError(details=f"Socket error: {e}")
        except Exception as e:
            raise ConnectionError(details=str(e))

    def stop(self) -> None:
        """Stop the controller"""
        self.ready = False
        self.connected = False
        self._stop_state_share()
        self.server.stop()
        logger.info("OSC controller stopped")

    def __enter__(self) -> "BitwigOSCController":
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        self.stop()

    def ping(self, timeout: float = 2.0) -> bool:
        """Check if Bitwig is responding

        Args:
            timeout: Timeout in seconds

        Returns:
            True if Bitwig is responding, False otherwise
        """
        try:
            # Clear all messages
            self.server.clear_messages()

            # Send a query that should always return a value
            self.client.send("/refresh", 1)

            # Wait for any response
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.server.received_messages:
                    self.error_handler.record_success()
                    return True
                time.sleep(0.1)

            # If we get here, we timed out
            error = BitwigNotRespondingError()
            self.error_handler.record_error("ping", error)
            return False

        except Exception as e:
            logger.error(f"Error pinging Bitwig: {e}")
            return False

    def refresh(self, timeout: float = 2.0) -> bool:
        """Refresh state from Bitwig

        Args:
            timeout: Timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        if self.send_only:
            return False

        try:
            # Rate limit refreshes
            now = time.time()
            if (
                now - self.last_refresh < 0.2
            ):  # Don't refresh more than 5 times per second
                return True

            self.last_refresh = now

            # Send refresh command
            self.client.refresh()

            # Wait for any response
            start_time = time.time()
            received_count = len(self.server.received_messages)

            # Wait for new messages
            while time.time() - start_time < timeout:
                # If we've received new messages, success
                if len(self.server.received_messages) > received_count:
                    logger.debug(
                        f"Received {len(self.server.received_messages) - received_count} new messages during refresh"
                    )
                    self.error_handler.record_success()
                    return True
                time.sleep(0.1)

            # If we get here, we timed out waiting for new messages
            logger.debug(
                f"No new messages received after {timeout}s. Testing connection..."
            )

            # Explicitly test connection with a ping
            if self.ping(timeout=1.0):
                # Connection is good, Bitwig just didn't send new messages
                # This can happen when Bitwig has already sent all the state info
                # and doesn't have anything new to report
                logger.info(
                    "No new messages from Bitwig, but connection is good. Proceeding with cached state."
                )
                self.error_handler.record_success()
                return True

            # Connection test failed - we have a real problem
            logger.warning("Connection test failed after refresh timeout")
            if self.error_handler.connection_status["consecutive_timeouts"] > 3:
                error = BitwigNotRespondingError()
                self.error_handler.record_error("refresh", error)
                self.connected = False

            return False

        except Exception as e:
            logger.error(f"Error refreshing Bitwig state: {e}")
            return False

    # Synchronized command methods with response waiting
    def send_and_wait(
        self,
        address: str,
        value: Any,
        response_address: Optional[str] = None,
        timeout: float = 2.0,
    ) -> Any:
        """Send command and wait for response

        Args:
            address: The OSC address to send to
            value: The value to send
            response_address: The address to wait for (defaults to same as sent)
            timeout: Timeout in seconds

        Returns:
            The response value

        Raises:
            TimeoutError: If the operation times out
            BitwigNotRespondingError: If Bitwig is not responding
        """
        if self.send_only:
            raise ConnectionError(
                details=SEND_ONLY_MESSAGE.format(port=self.receive_port)
            )

        # Ensure connection is healthy
        if not self.ready or not self.connected:
            if not self._attempt_reconnect():
                raise BitwigNotRespondingError(address)

        if response_address is None:
            response_address = address

        # Define the operation to retry
        def _send_and_wait_operation():
            # Clear any previous messages with this address
            self.server.received_messages.pop(response_address, None)

            # Send the command
            self.client.send(address, value)

            # Wait for response
            result = self.server.wait_for_message(response_address, timeout)
            if result is None:
                raise TimeoutError(f"send_and_wait({address})", timeout)
            return result

        # Retry the operation with timeout handling
        try:
            return self.error_handler.retry_with_timeout(
                _send_and_wait_operation,
                f"send_and_wait({address})",
                max_retries=2,
                retry_delay=0.2,
                timeout=timeout,
            )
        except Exception:
            # If still failing after retries, check connection
            if not self.ping():
                self.connected = False
                raise BitwigNotRespondingError(address)
            raise

    def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to Bitwig

        Returns:
            True if successful, False otherwise
        """
        if self.send_only:
            return False

        if self.mirroring:
            # The mirror reconnects or takes over the port by itself
            return bool(self.server.running)

        if self.connection_attempts >= self.reconnect_attempts:
            logger.error("Maximum reconnection attempts reached")
            return False

        logger.info("Attempting to reconnect to Bitwig...")

        try:
            # Stop everything
            self._stop_state_share()
            try:
                self.server.stop()
            except Exception:
                pass

            time.sleep(1.0)

            # Restart
            self.server = BitwigOSCServer(self.ip, self.receive_port)
            self.server.start()
            self._start_state_share()

            # Attempt connection
            self._connect_with_timeout()

            self.ready = True
            self.connected = True
            self.error_handler.record_success()

            logger.info("Successfully reconnected to Bitwig")
            return True

        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")
            return False

    # High-level control methods
    def get_track_info(self, track_index: int) -> Dict[str, Any]:
        """Get information about a track

        Args:
            track_index: Track index (1-based)

        Returns:
            Dict containing track information

        Raises:
            InvalidParameterError: If track_index is invalid
            ResourceNotFoundError: If track not found
            BitwigNotRespondingError: If Bitwig is not responding
        """
        # Validate track index
        track_index = self.error_handler.validate_track_index(track_index)

        # Refresh to get latest state
        if not self.refresh():
            raise BitwigNotRespondingError(f"/track/{track_index}")

        # Check if track exists
        name = self.server.get_message(f"/track/{track_index}/name")
        if not name:
            raise ResourceNotFoundError("Track", str(track_index))

        prefix = f"/track/{track_index}/"
        track_info = {"name": name, "index": track_index}

        # Extract all track properties from received messages
        for address, value in self.server.received_messages.items():
            if address.startswith(prefix) and not address.endswith("/name"):
                # Extract property name from address
                prop = address[len(prefix) :]
                track_info[prop] = value

        return track_info

    def set_device_parameter_verified(
        self,
        param_index: int,
        value: float,
        timeout: float = 1.0,
        automation_touch: bool = True,
    ) -> Dict[str, Any]:
        """Set a device parameter and check that Bitwig applied it

        Bitwig treats the OSC value like a hardware knob with soft takeover:
        a jump far away from the current value is ignored until the "knob"
        has once been at the parameter's current value. So the current value
        is sent first ("pick up"), then the target value.

        Bitwig reports a changed value back on /device/param/{i}/value. If
        nothing comes back, the value was not applied.

        Args:
            param_index: Parameter index (1-8)
            value: Parameter value (0-127)
            timeout: Seconds to wait for Bitwig's feedback
            automation_touch: Wrap the change in /device/param/{i}/touched 1
                and 0, like touching a hardware knob (for automation
                recording in touch mode)

        Returns:
            Dictionary with the parameter name, the sent value, the value
            before and after, the displayed value, "verified" (whether
            Bitwig's state could be checked) and "applied"
        """
        value_address = f"/device/param/{param_index}/value"
        target = int(round(min(max(value, 0), 127)))
        before = self.server.get_message(value_address)

        if automation_touch:
            self.client.touch_device_parameter(param_index, True)
        try:
            if isinstance(before, (int, float)) and int(before) != target:
                self.client.set_device_parameter(param_index, before)  # pick up
            self.client.set_device_parameter(param_index, value)
        finally:
            if automation_touch:
                self.client.touch_device_parameter(param_index, False)

        result: Dict[str, Any] = {
            "index": param_index,
            "name": self.server.get_message(f"/device/param/{param_index}/name"),
            "target": target,
            "before": before,
            "after": before,
            "value_str": None,
            "verified": False,
            "applied": None,
        }
        if self.send_only or before is None:
            return result  # no feedback available to compare with

        after = before
        if before != target:
            deadline = time.time() + timeout
            while time.time() < deadline:
                time.sleep(0.05)
                after = self.server.get_message(value_address)
                if after != before:
                    break
        else:
            # Bitwig reports nothing if the value does not change
            time.sleep(0.2)

        result.update(
            after=after,
            value_str=self.server.get_message(f"/device/param/{param_index}/valueStr"),
            verified=True,
            applied=after == target or after != before,
        )
        return result

    def get_parameter_page_info(self) -> Dict[str, Any]:
        """Get the selected parameter page and the names of its 8 parameters

        DrivenByMoss reports pages in windows of 8 slots, so the page number
        is the slot within the currently shown window, not an absolute index.

        Returns:
            Dictionary with device name, selected slot and name, the page
            slots of the current window and the 8 parameters

        Raises:
            ConnectionError: In send-only mode (no feedback from Bitwig)
            ResourceNotFoundError: If no device is selected
        """
        if self.send_only:
            raise ConnectionError(
                details=SEND_ONLY_MESSAGE.format(port=self.receive_port)
            )

        messages = self.server.received_messages
        if "/device/page/selected/name" not in messages:
            self.refresh()

        if not self.server.get_message("/device/exists"):
            raise ResourceNotFoundError("Device", "No device selected")

        pages = []
        selected_slot = None
        for i in range(1, 9):
            if not self.server.get_message(f"/device/page/{i}/exists"):
                continue
            selected = bool(self.server.get_message(f"/device/page/{i}/selected"))
            if selected:
                selected_slot = i
            pages.append(
                {
                    "slot": i,
                    "name": self.server.get_message(f"/device/page/{i}/name"),
                    "selected": selected,
                }
            )

        params = []
        for i in range(1, 9):
            params.append(
                {
                    "index": i,
                    "exists": bool(
                        self.server.get_message(f"/device/param/{i}/exists")
                    ),
                    "name": self.server.get_message(f"/device/param/{i}/name"),
                    "value_str": self.server.get_message(f"/device/param/{i}/valueStr"),
                }
            )

        return {
            "device": self.server.get_message("/device/name"),
            "selected_slot": selected_slot,
            "selected_name": self.server.get_message("/device/page/selected/name"),
            "pages": pages,
            "params": params,
        }

    def get_device_params(self) -> List[Dict[str, Any]]:
        """Get information about device parameters

        Returns:
            List of parameter information dictionaries

        Raises:
            ResourceNotFoundError: If no device is selected
            BitwigNotRespondingError: If Bitwig is not responding
        """
        # Refresh to get latest state
        if not self.refresh():
            raise BitwigNotRespondingError("/device")

        # Check if a device is selected
        device_exists = self.server.get_message("/device/exists")
        if not device_exists:
            raise ResourceNotFoundError("Device", "No device selected")

        params = []

        # Find all parameters
        for i in range(1, 9):  # Assuming 8 parameters max
            prefix = f"/device/param/{i}/"
            param_exists = self.server.get_message(f"{prefix}exists")

            if not param_exists:
                continue

            param_info = {"index": i}

            # Extract parameter properties
            for address, value in self.server.received_messages.items():
                if address.startswith(prefix) and not address.endswith("/exists"):
                    # Extract property name from address
                    prop = address[len(prefix) :]
                    param_info[prop] = value

            # Only add if we have some properties
            if len(param_info) > 1:
                params.append(param_info)

        return params

    def get_status(self) -> Dict[str, Any]:
        """Get controller status information

        Returns:
            Dict with status information
        """
        connection_health = self.error_handler.check_connection_health()

        # If connection doesn't seem healthy but we think we're connected,
        # do a quick ping test
        if not connection_health and self.connected:
            self.connected = self.ping()

        return {
            "ready": self.ready,
            "connected": self.connected,
            "ip": self.ip,
            "send_port": self.send_port,
            "receive_port": self.receive_port,
            "connection_health": connection_health,
            "errors": self.error_handler.recent_errors,
            "diagnostics": self.error_handler.get_diagnostic_info(),
        }
