"""Tests for the BitwigOSCController class"""

import errno
import unittest
from unittest.mock import MagicMock, patch

from bitwig_mcp_server.osc.controller import BitwigOSCController
from bitwig_mcp_server.osc.exceptions import ConnectionError


class TestBitwigOSCController(unittest.TestCase):
    """Test cases for BitwigOSCController"""

    def setUp(self):
        """Set up test environment"""
        # Create controller with mocked client and server
        with (
            patch(
                "bitwig_mcp_server.osc.controller.BitwigOSCClient"
            ) as mock_client_class,
            patch(
                "bitwig_mcp_server.osc.controller.BitwigOSCServer"
            ) as mock_server_class,
        ):
            self.mock_client = MagicMock()
            self.mock_server = MagicMock()

            mock_client_class.return_value = self.mock_client
            mock_server_class.return_value = self.mock_server

            self.controller = BitwigOSCController()

        # Never open real sockets for state sharing in these tests
        share_patcher = patch("bitwig_mcp_server.osc.controller.StateShareServer")
        mirror_patcher = patch("bitwig_mcp_server.osc.controller.RemoteOSCState")
        self.mock_share_class = share_patcher.start()
        self.mock_mirror_class = mirror_patcher.start()
        self.addCleanup(share_patcher.stop)
        self.addCleanup(mirror_patcher.stop)

    def test_start_stop(self):
        """Test starting and stopping controller"""
        # Test start
        self.controller.start()
        self.mock_server.start.assert_called_once()
        self.mock_client.refresh.assert_called_once()
        self.assertTrue(self.controller.ready)

        # The port owner shares its state on receive_port + 1
        self.mock_share_class.assert_called_once_with(
            self.mock_server, "127.0.0.1", 9001
        )
        share = self.mock_share_class.return_value
        share.start.assert_called_once()
        self.assertFalse(self.controller.mirroring)

        # Test stop
        self.controller.stop()
        self.mock_server.stop.assert_called_once()
        share.stop.assert_called_once()

    def test_start_mirrors_owner_when_port_in_use(self):
        """A taken receive port makes the controller mirror the owner's state"""
        self.mock_server.start.side_effect = OSError(
            errno.EADDRINUSE, "Address already in use"
        )
        mirror = self.mock_mirror_class.return_value
        mirror.received_messages = {"/tempo/raw": 120}

        self.controller.start()

        self.mock_mirror_class.assert_called_once_with("127.0.0.1", 9000, 9001)
        mirror.start.assert_called_once()
        self.assertIs(self.controller.server, mirror)
        self.assertTrue(self.controller.mirroring)
        self.assertFalse(self.controller.send_only)
        self.assertTrue(self.controller.ready)
        # The usual connection check runs against the mirrored state
        self.mock_client.refresh.assert_called_once()
        self.mock_share_class.assert_not_called()

    def test_start_send_only_when_port_in_use(self):
        """Without a reachable owner the controller falls back to send-only"""
        port_in_use = OSError(errno.EADDRINUSE, "Address already in use")
        self.mock_server.start.side_effect = port_in_use
        self.mock_mirror_class.return_value.start.side_effect = OSError("not reachable")

        self.controller.start()

        self.assertTrue(self.controller.send_only)
        self.assertFalse(self.controller.mirroring)
        self.assertTrue(self.controller.ready)
        # No connection check: replies from Bitwig cannot be received
        self.mock_client.refresh.assert_not_called()

        # Commands are still sent
        self.controller.client.set_tempo(120)
        self.mock_client.set_tempo.assert_called_once_with(120)

        # Anything that needs a reply fails fast with a clear message
        self.assertFalse(self.controller.refresh())
        with self.assertRaises(ConnectionError) as ctx:
            self.controller.send_and_wait("/tempo/raw", 120)
        self.assertIn("send-only mode", str(ctx.exception))

    def test_start_other_os_error_still_fails(self):
        """Other socket errors are not hidden by send-only mode"""
        self.mock_server.start.side_effect = OSError(errno.EACCES, "Access denied")

        with self.assertRaises(ConnectionError):
            self.controller.start()

        self.assertFalse(self.controller.send_only)
        self.assertFalse(self.controller.ready)

    def test_context_manager(self):
        """Test context manager protocol"""
        with (
            patch.object(self.controller, "start") as mock_start,
            patch.object(self.controller, "stop") as mock_stop,
        ):
            with self.controller:
                mock_start.assert_called_once()

            mock_stop.assert_called_once()

    def test_send_and_wait(self):
        """Test sending command and waiting for response"""
        # Set up mock response and ready state
        self.controller.ready = True
        self.controller.connected = True
        self.mock_server.received_messages = {"/test/response": None}
        self.mock_server.wait_for_message.return_value = 42

        # Mock the retry_with_timeout method
        with patch.object(
            self.controller.error_handler, "retry_with_timeout"
        ) as mock_retry:
            mock_retry.return_value = 42

            # Test with default response address
            result = self.controller.send_and_wait("/test/command", 1)
            self.assertEqual(result, 42)

            # Verify retry was called
            mock_retry.assert_called()

            # Test with custom response address
            result = self.controller.send_and_wait("/test/command", 1, "/test/response")
            self.assertEqual(result, 42)

    def test_get_track_info(self):
        """Test retrieving track information"""
        # Set up mock track data
        self.mock_server.received_messages = {
            "/track/1/name": "Audio Track",
            "/track/1/volume": 64,
            "/track/1/pan": 64,
            "/track/1/mute": 0,
            "/track/1/solo": 0,
            "/track/2/name": "Other Track",  # Should be ignored
        }

        # Set up mock for refresh
        self.controller.refresh = MagicMock(return_value=True)

        # Set up get_message mock to return values from received_messages
        self.mock_server.get_message = (
            lambda addr: self.mock_server.received_messages.get(addr)
        )

        # Mock the validate_track_index method
        with patch.object(
            self.controller.error_handler, "validate_track_index"
        ) as mock_validate:
            mock_validate.return_value = 1

            # Get track info
            track_info = self.controller.get_track_info(1)

            # Verify validate and refresh called
            mock_validate.assert_called_with(1)
            self.controller.refresh.assert_called_once()

            # Verify track info extracted correctly
            expected = {
                "name": "Audio Track",
                "volume": 64,
                "pan": 64,
                "mute": 0,
                "solo": 0,
                "index": 1,
            }
            self.assertEqual(track_info, expected)

    def test_get_device_params(self):
        """Test retrieving device parameters"""
        # Set up mock parameter data
        self.mock_server.received_messages = {
            "/device/exists": 1,
            "/device/param/1/exists": 1,
            "/device/param/1/name": "Cutoff",
            "/device/param/1/value": 64,
            "/device/param/2/exists": 1,
            "/device/param/2/name": "Resonance",
            "/device/param/2/value": 32,
            "/device/param/3/exists": 0,
        }

        # Mock get_message to return values from received_messages
        self.mock_server.get_message = (
            lambda addr: self.mock_server.received_messages.get(addr)
        )

        # Set up mock for refresh
        self.controller.refresh = MagicMock(return_value=True)

        # Get device parameters
        params = self.controller.get_device_params()

        # Verify client refreshed
        self.controller.refresh.assert_called_once()

        # Verify parameters extracted correctly
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0]["name"], "Cutoff")
        self.assertEqual(params[0]["value"], 64)
        self.assertEqual(params[1]["name"], "Resonance")
        self.assertEqual(params[1]["value"], 32)


if __name__ == "__main__":
    unittest.main()
