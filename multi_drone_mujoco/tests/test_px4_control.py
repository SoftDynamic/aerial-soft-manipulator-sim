import sys
import types

import pytest

from multi_drone_mujoco.backends.px4_control import (
    Px4FlightControlConfig,
    Px4MavlinkFlightControl,
)


class FakeMav:
    def __init__(self):
        self.setpoints = []
        self.commands = []

    def set_position_target_local_ned_send(self, *args):
        self.setpoints.append(args)

    def command_long_send(self, *args):
        self.commands.append(args)


class FakeConnection:
    def __init__(self):
        self.mav = FakeMav()
        self.messages = []
        self.closed = False

    def recv_match(self, blocking=False):
        return self.messages.pop(0) if self.messages else None

    def close(self):
        self.closed = True


def install_fake_pymavlink(monkeypatch, connection):
    dialect = types.SimpleNamespace(
        MAV_FRAME_LOCAL_NED=1,
        MAV_CMD_DO_SET_MODE=176,
        MAV_CMD_COMPONENT_ARM_DISARM=400,
        MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1,
        MAV_MODE_FLAG_SAFETY_ARMED=128,
        MAV_SEVERITY_WARNING=4,
    )
    mavutil = types.SimpleNamespace(
        mavlink=dialect,
        mavlink_connection=lambda *args, **kwargs: connection,
    )
    module = types.ModuleType("pymavlink")
    module.mavutil = mavutil
    monkeypatch.setitem(sys.modules, "pymavlink", module)


def test_position_setpoint_converts_enu_to_ned(monkeypatch):
    connection = FakeConnection()
    install_fake_pymavlink(monkeypatch, connection)
    control = Px4MavlinkFlightControl(Px4FlightControlConfig("127.0.0.1", 14550))
    control.open()
    control.send_position_setpoint_enu(1_234_567, (2.0, 3.0, 4.0))
    values = connection.mav.setpoints[0]
    assert values[0] == 1234
    assert values[5:8] == (3.0, 2.0, -4.0)
    control.request_offboard_mode()
    control.request_arm(True)
    control.request_arm(False)
    control.request_auto_land()
    assert [item[2] for item in connection.mav.commands] == [176, 400, 400, 176]
    assert connection.mav.commands[-1][5:7] == (4, 6)
    control.close()
    assert connection.closed


def test_flight_control_open_and_config_are_explicit(monkeypatch):
    with pytest.raises(ValueError, match="host"):
        Px4FlightControlConfig("", 14550)
    with pytest.raises(ValueError, match="port"):
        Px4FlightControlConfig("127.0.0.1", 0)
    connection = FakeConnection()
    install_fake_pymavlink(monkeypatch, connection)
    control = Px4MavlinkFlightControl(Px4FlightControlConfig("0.0.0.0", 14550))
    assert control.endpoint == "udpin:0.0.0.0:14550"
    control.open()
    with pytest.raises(RuntimeError, match="already open"):
        control.open()
