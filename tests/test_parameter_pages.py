"""Tests for switching and reading parameter pages of the cursor device."""

from unittest.mock import MagicMock, patch

import pytest

from bitwig_mcp_server.mcp.tools import execute_tool, get_bitwig_tools
from bitwig_mcp_server.osc.client import BitwigOSCClient
from bitwig_mcp_server.osc.controller import BitwigOSCController
from bitwig_mcp_server.osc.exceptions import (
    ConnectionError,
    InvalidParameterError,
    ResourceNotFoundError,
)

PAGE_TOOLS = {
    "select_parameter_page",
    "next_parameter_page",
    "previous_parameter_page",
    "next_parameter_page_bank",
    "previous_parameter_page_bank",
    "get_parameter_page_info",
}

# State as DrivenByMoss reports it for a Polysynth with two preset pages
POLYSYNTH_STATE = {
    "/device/exists": 1,
    "/device/name": "Deep Pad",
    "/device/page/selected/name": "Page 2",
    "/device/page/1/exists": 1,
    "/device/page/1/name": "Perform",
    "/device/page/1/selected": 0,
    "/device/page/2/exists": 1,
    "/device/page/2/name": "Page 2",
    "/device/page/2/selected": 1,
    "/device/page/3/exists": 1,
    "/device/page/3/name": "OSC1",
    "/device/page/3/selected": 0,
    "/device/page/4/exists": 0,
    "/device/param/1/exists": 1,
    "/device/param/1/name": "Filt Freq",
    "/device/param/1/valueStr": "1.20 kHz",
    "/device/param/2/exists": 1,
    "/device/param/2/name": "Reso",
    "/device/param/2/valueStr": "20.0 %",
    "/device/param/3/exists": 0,
    "/device/param/3/name": "",
}


@pytest.fixture
def client():
    osc_client = BitwigOSCClient()
    osc_client.client = MagicMock()
    return osc_client


@pytest.fixture
def controller():
    with (
        patch("bitwig_mcp_server.osc.controller.BitwigOSCClient"),
        patch("bitwig_mcp_server.osc.controller.BitwigOSCServer"),
    ):
        ctrl = BitwigOSCController()
    ctrl.server.received_messages = dict(POLYSYNTH_STATE)
    ctrl.server.get_message.side_effect = ctrl.server.received_messages.get
    return ctrl


# --- OSC client: addresses from the DrivenByMoss manual -----------------------


def test_select_parameter_page_address(client):
    client.select_parameter_page(2)
    client.client.send_message.assert_called_once_with("/device/page/selected", 2)


@pytest.mark.parametrize("bad", [0, 9, "2", 2.0, True])
def test_select_parameter_page_rejects_invalid(client, bad):
    with pytest.raises(InvalidParameterError):
        client.select_parameter_page(bad)
    client.client.send_message.assert_not_called()


@pytest.mark.parametrize(
    "method, address",
    [
        ("next_parameter_page", "/device/param/+"),
        ("previous_parameter_page", "/device/param/-"),
        ("next_parameter_page_bank", "/device/param/bank/page/+"),
        ("previous_parameter_page_bank", "/device/param/bank/page/-"),
    ],
)
def test_page_navigation_addresses(client, method, address):
    getattr(client, method)()
    client.client.send_message.assert_called_once_with(address, None)


# --- Controller -----------------------------------------------------------------


def test_get_parameter_page_info(controller):
    info = controller.get_parameter_page_info()

    assert info["device"] == "Deep Pad"
    assert info["selected_slot"] == 2
    assert info["selected_name"] == "Page 2"
    assert info["pages"] == [
        {"slot": 1, "name": "Perform", "selected": False},
        {"slot": 2, "name": "Page 2", "selected": True},
        {"slot": 3, "name": "OSC1", "selected": False},
    ]
    assert len(info["params"]) == 8
    assert info["params"][0] == {
        "index": 1,
        "exists": True,
        "name": "Filt Freq",
        "value_str": "1.20 kHz",
    }
    assert info["params"][2]["exists"] is False
    controller.client.refresh.assert_not_called()


def test_get_parameter_page_info_refreshes_empty_cache(controller):
    controller.server.received_messages.pop("/device/page/selected/name")
    with patch.object(controller, "refresh") as refresh:
        controller.get_parameter_page_info()
    refresh.assert_called_once()


def test_get_parameter_page_info_without_device(controller):
    controller.server.received_messages["/device/exists"] = 0
    with pytest.raises(ResourceNotFoundError):
        controller.get_parameter_page_info()


def test_get_parameter_page_info_send_only(controller):
    controller.send_only = True
    with pytest.raises(ConnectionError, match="send-only mode"):
        controller.get_parameter_page_info()


# --- MCP tools ------------------------------------------------------------------


def test_page_tools_are_listed():
    tools = {tool.name: tool for tool in get_bitwig_tools()}
    assert PAGE_TOOLS <= tools.keys()
    assert tools["select_parameter_page"].inputSchema["required"] == ["page"]


@pytest.mark.asyncio
async def test_execute_select_parameter_page():
    controller = MagicMock()
    result = await execute_tool(controller, "select_parameter_page", {"page": 2})
    controller.client.select_parameter_page.assert_called_once_with(2)
    assert "slot 2" in result[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [{}, {"page": 0}, {"page": 9}, {"page": "2"}])
async def test_execute_select_parameter_page_invalid(args):
    controller = MagicMock()
    result = await execute_tool(controller, "select_parameter_page", args)
    assert result[0].text.startswith("Error:")
    controller.client.select_parameter_page.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        "next_parameter_page",
        "previous_parameter_page",
        "next_parameter_page_bank",
        "previous_parameter_page_bank",
    ],
)
async def test_execute_page_navigation(tool):
    controller = MagicMock()
    result = await execute_tool(controller, tool, {})
    getattr(controller.client, tool).assert_called_once_with()
    assert not result[0].text.startswith("Error:")


@pytest.mark.asyncio
async def test_execute_get_parameter_page_info(controller):
    with patch("bitwig_mcp_server.mcp.tools.asyncio.sleep"):
        result = await execute_tool(controller, "get_parameter_page_info", {})
    text = result[0].text

    assert "Device: Deep Pad" in text
    assert "Selected page: slot 2 - Page 2" in text
    assert " * 2: Page 2" in text
    assert "1: Filt Freq = 1.20 kHz" in text
    assert "3: (none)" in text


@pytest.mark.asyncio
async def test_execute_get_parameter_page_info_send_only(controller):
    controller.send_only = True
    with patch("bitwig_mcp_server.mcp.tools.asyncio.sleep"):
        result = await execute_tool(controller, "get_parameter_page_info", {})
    assert "send-only mode" in result[0].text
