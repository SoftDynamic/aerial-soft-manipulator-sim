"""Gate P1 tests for generic controller and actuator contracts."""

from dataclasses import FrozenInstanceError

import pytest

from multi_drone_mujoco.backends import (
    CommandWatchdog,
    ControllerBackend,
    NormalizedMotorCommand,
    PwmMotorCommand,
    RotorSpeedCommand,
    SimulatorClock,
)


def test_motor_contract_shape_finite_range_and_immutability():
    command = PwmMotorCommand((900, 1000, 1500, 2000), True, 12, 3)
    assert command.pulse_width_us == (900.0, 1000.0, 1500.0, 2000.0)
    with pytest.raises(FrozenInstanceError):
        command.armed = False
    with pytest.raises(ValueError, match="exactly 4"):
        RotorSpeedCommand((1, 2, 3), 0, 0)
    with pytest.raises(ValueError, match="finite"):
        NormalizedMotorCommand((0, 0, float("nan"), 0), True, 0, 0)
    with pytest.raises(ValueError, match="non-negative"):
        RotorSpeedCommand((0, -1, 0, 0), 0, 0)


def test_simulator_clock_units_and_validation():
    assert SimulatorClock.from_seconds(0.0015, 4) == SimulatorClock(1500, 4)
    with pytest.raises(ValueError):
        SimulatorClock(-1, 0)


def test_watchdog_timestamp_reset_close_and_safe_timeout():
    watchdog = CommandWatchdog(timeout_us=100)
    command = RotorSpeedCommand((1, 2, 3, 4), timestamp_us=10, sequence=1)
    watchdog.accept(command)
    assert watchdog.output(110, 9) is command
    safe = watchdog.output(111, 10)
    assert safe.rpm == (0.0, 0.0, 0.0, 0.0)
    assert not safe.armed
    with pytest.raises(ValueError, match="monotonically"):
        watchdog.accept(command)
    watchdog.reset()
    assert not watchdog.output(0, 0).armed
    watchdog.reset()
    watchdog.close()
    watchdog.close()
    assert not watchdog.output(1000, 2).armed


def test_backend_protocol_is_structural_and_does_not_import_px4():
    class Backend:
        health = None

        def reset(self, clock):
            pass

        def step(self, sensor_frame):
            return RotorSpeedCommand((0, 0, 0, 0), 0, 0, armed=False)

        def close(self):
            pass

    assert isinstance(Backend(), ControllerBackend)
