"""Gate P6 fake-peer integration tests for the complete PX4 backend."""

import pytest

from multi_drone_mujoco.backends.actuation import (
    NormalizedToPwmConfig,
    PwmToRotorSpeedConfig,
    PwmToRotorSpeedModel,
    Px4NormalizedOutputToPwm,
)
from multi_drone_mujoco.backends.contracts import (
    ControllerBackend,
    ControllerState,
    NormalizedMotorCommand,
    SimulatorClock,
)
from multi_drone_mujoco.backends.lockstep import LockstepConfig
from multi_drone_mujoco.backends.px4_sitl import PX4SITLBackend, Px4BackendInput
from multi_drone_mujoco.backends.sensors import (
    IdealSensorAdapter,
    IdealSensorConfig,
    MujocoTruthFrame,
)
from multi_drone_mujoco.backends.transport import FakeTransport


def make_backend(packets=()):
    directions = (-1, 1, -1, 1)
    transport = FakeTransport(packets)
    sensor = IdealSensorAdapter(IdealSensorConfig(
        30, 120, 100, 9.81, (0.2, 0, 0.4), 1000, 10, 50, 100
    ))
    pwm = Px4NormalizedOutputToPwm(NormalizedToPwmConfig(
        900, 1000, 2000, 0, 1, (2, 0, 3, 1), directions, True
    ))
    esc = PwmToRotorSpeedModel(PwmToRotorSpeedConfig(
        900, 1000, 2000, 0, 1000, 21000, 0, directions
    ))
    backend = PX4SITLBackend(
        sensor, transport, pwm, esc, LockstepConfig(0.01, (0, 0, 0, 0))
    )
    return backend, transport


def truth():
    return MujocoTruthFrame(
        (0, 0, 1), (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0)
    )


def input_at(timestamp, sequence):
    return Px4BackendInput(truth(), SimulatorClock(timestamp, sequence), 0.001)


def test_fake_px4_sensor_to_normalized_pwm_rpm_and_motor_order():
    packet = NormalizedMotorCommand((0.0, 0.25, 0.5, 1.0), True, 1000, 9)
    backend, transport = make_backend([packet])
    assert isinstance(backend, ControllerBackend)
    backend.reset(SimulatorClock(0, 0))
    result = backend.step(input_at(1000, 1))
    # order (2,0,3,1): normalized (0.5, 0, 1, 0.25)
    assert result.rotor_speed.rpm == (11000, 1000, 21000, 6000)
    assert result.advance_physics
    assert result.health.state == ControllerState.READY
    assert transport.sent_sensor_frames[0].timestamp_us == 1000
    assert backend.last_normalized_command == packet
    assert backend.last_pwm_command.pulse_width_us == (1500, 1000, 2000, 1250)


def test_disarmed_px4_output_and_timeout_both_stop_motors_safely():
    disarmed = NormalizedMotorCommand((1, 1, 1, 1), False, 1000, 1)
    backend, _ = make_backend([disarmed])
    backend.reset(SimulatorClock(0, 0))
    assert backend.last_normalized_command is None
    assert backend.last_pwm_command is None
    result = backend.step(input_at(1000, 1))
    assert result.rotor_speed.rpm == (0, 0, 0, 0)
    assert result.advance_physics
    timeout = backend.step(input_at(2000, 2))
    assert timeout.rotor_speed.rpm == (0, 0, 0, 0)
    assert not timeout.advance_physics
    assert timeout.health.state == ControllerState.TIMED_OUT


def test_reset_reinitializes_sensor_motor_and_packet_state():
    backend, transport = make_backend([
        NormalizedMotorCommand((1, 1, 1, 1), True, 1000, 1)
    ])
    backend.reset(SimulatorClock(0, 0))
    assert backend.step(input_at(1000, 1)).rotor_speed.rpm == (21000,) * 4
    backend.reset(SimulatorClock(0, 0))
    transport.queue_actuator(NormalizedMotorCommand((0, 0, 0, 0), True, 1000, 1))
    assert backend.step(input_at(1000, 1)).rotor_speed.rpm == (1000,) * 4


def test_close_is_idempotent_and_does_not_manage_external_processes():
    backend, transport = make_backend()
    backend.reset(SimulatorClock(0, 0))
    backend.close()
    backend.close()
    assert backend.health.state == ControllerState.CLOSED
    with pytest.raises(RuntimeError, match="closed"):
        backend.step(input_at(1000, 1))


def test_paused_backend_does_not_consume_timestamp_before_single_step():
    backend, transport = make_backend([
        NormalizedMotorCommand((0.5,) * 4, True, 1000, 1)
    ])
    backend.reset(SimulatorClock(0, 0))
    backend.pause()
    paused = backend.step(input_at(1000, 1))
    assert not paused.advance_physics
    assert not transport.sent_sensor_frames
    backend.request_single_step()
    stepped = backend.step(input_at(1000, 1))
    assert stepped.advance_physics
