"""Tests for set_device_parameter checking that Bitwig applied the value."""

from unittest.mock import MagicMock, call, patch

import pytest

from bitwig_mcp_server.mcp.tools import execute_tool
from bitwig_mcp_server.osc.controller import BitwigOSCController

FX_STATE = {
    "/device/param/7/name": "FX",
    "/device/param/7/value": 71,
    "/device/param/7/valueStr": "56.0 %",
}


@pytest.fixture
def controller():
    with (
        patch("bitwig_mcp_server.osc.controller.BitwigOSCClient"),
        patch("bitwig_mcp_server.osc.controller.BitwigOSCServer"),
    ):
        ctrl = BitwigOSCController()
    ctrl.server.received_messages = dict(FX_STATE)
    ctrl.server.get_message.side_effect = ctrl.server.received_messages.get
    return ctrl


def bitwig_applies(controller, value_str):
    """Make the mocked client behave like Bitwig accepting the value."""

    def apply(index, value):
        controller.server.received_messages[f"/device/param/{index}/value"] = int(
            round(value)
        )
        controller.server.received_messages[f"/device/param/{index}/valueStr"] = (
            value_str
        )

    controller.client.set_device_parameter.side_effect = apply


def test_applied_value_is_reported(controller):
    bitwig_applies(controller, "70.9 %")

    outcome = controller.set_device_parameter_verified(7, 90, timeout=0.5)

    # current value first (pick up), then the target
    assert controller.client.set_device_parameter.call_args_list == [
        call(7, 71),
        call(7, 90),
    ]
    assert outcome["verified"] is True
    assert outcome["applied"] is True
    assert (outcome["before"], outcome["after"]) == (71, 90)
    assert outcome["value_str"] == "70.9 %"


def test_ignored_value_is_detected(controller):
    # Bitwig sends nothing back when it ignores the value
    outcome = controller.set_device_parameter_verified(7, 90, timeout=0.2)

    assert outcome["verified"] is True
    assert outcome["applied"] is False
    assert outcome["after"] == 71


def test_same_value_counts_as_applied(controller):
    # Bitwig reports nothing when the value does not change
    outcome = controller.set_device_parameter_verified(7, 71, timeout=0.2)
    assert outcome["applied"] is True


def test_send_only_cannot_verify(controller):
    controller.send_only = True
    controller.server.received_messages.clear()
    outcome = controller.set_device_parameter_verified(7, 90, timeout=0.2)

    controller.client.set_device_parameter.assert_called_once_with(7, 90)
    assert outcome["verified"] is False
    assert outcome["applied"] is None


@pytest.mark.asyncio
async def test_tool_reports_applied_value(controller):
    bitwig_applies(controller, "70.9 %")
    result = await execute_tool(
        controller, "set_device_parameter", {"param_index": 7, "value": 90}
    )
    assert result[0].text == "Device parameter 7 (FX) set to 90 = 70.9 %"


@pytest.mark.asyncio
async def test_tool_warns_when_bitwig_ignores_value(controller):
    with patch.object(
        controller,
        "set_device_parameter_verified",
        wraps=lambda i, v, **kw: BitwigOSCController.set_device_parameter_verified(
            controller, i, v, timeout=0.2, **kw
        ),
    ):
        result = await execute_tool(
            controller, "set_device_parameter", {"param_index": 7, "value": 90}
        )
    text = result[0].text
    assert text.startswith("Warning: Bitwig did not apply the value 90")
    assert "still 71 (56.0 %)" in text
    assert "automated" in text


@pytest.mark.asyncio
async def test_tool_reports_unverified_in_send_only_mode(controller):
    controller.send_only = True
    result = await execute_tool(
        controller, "set_device_parameter", {"param_index": 7, "value": 90}
    )
    assert "not verified" in result[0].text


def sent_messages(controller):
    """Record touch and value messages in the order they are sent."""
    order = MagicMock()
    order.attach_mock(controller.client.touch_device_parameter, "touch")
    order.attach_mock(controller.client.set_device_parameter, "value")
    return order


def test_touch_wraps_pick_up_and_value(controller):
    order = sent_messages(controller)
    bitwig_applies(controller, "70.9 %")

    controller.set_device_parameter_verified(7, 90, timeout=0.5)

    assert order.mock_calls == [
        call.touch(7, True),
        call.value(7, 71),
        call.value(7, 90),
        call.touch(7, False),
    ]


def test_without_touch(controller):
    bitwig_applies(controller, "70.9 %")

    outcome = controller.set_device_parameter_verified(
        7, 90, timeout=0.5, automation_touch=False
    )

    controller.client.touch_device_parameter.assert_not_called()
    assert outcome["applied"] is True


def test_no_pick_up_when_value_unchanged(controller):
    controller.set_device_parameter_verified(7, 71, timeout=0.2)
    controller.client.set_device_parameter.assert_called_once_with(7, 71)


def test_touch_is_released_when_sending_fails(controller):
    controller.client.set_device_parameter.side_effect = ConnectionError("down")

    with pytest.raises(ConnectionError):
        controller.set_device_parameter_verified(7, 90, timeout=0.2)

    controller.client.touch_device_parameter.assert_called_with(7, False)


@pytest.mark.asyncio
async def test_tool_passes_automation_touch(controller):
    with patch.object(controller, "set_device_parameter_verified") as verified:
        verified.return_value = {
            "index": 7,
            "name": "FX",
            "target": 90,
            "after": 90,
            "value_str": "70.9 %",
            "verified": True,
            "applied": True,
        }
        await execute_tool(
            controller,
            "set_device_parameter",
            {"param_index": 7, "value": 90, "automation_touch": False},
        )
    verified.assert_called_once_with(7, 90, automation_touch=False)


@pytest.mark.asyncio
async def test_tool_rejects_invalid_automation_touch(controller):
    result = await execute_tool(
        controller,
        "set_device_parameter",
        {"param_index": 7, "value": 90, "automation_touch": "yes"},
    )
    assert "automation_touch" in result[0].text
    controller.client.set_device_parameter.assert_not_called()
