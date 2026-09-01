"""Gate P2 tests for the two explicit PX4 actuator conversion layers."""

import numpy as np
import pytest

from multi_drone_mujoco.backends.actuation import (
    NormalizedToPwmConfig,
    PwmToRotorSpeedConfig,
    PwmToRotorSpeedModel,
    Px4NormalizedOutputToPwm,
)
from multi_drone_mujoco.backends.contracts import (
    NormalizedMotorCommand,
    PwmMotorCommand,
)
from multi_drone_mujoco.envs.base_aviary import BaseAviary


DIRECTIONS = (-1, 1, -1, 1)


def pwm_model(order=(0, 1, 2, 3), clip=True):
    return Px4NormalizedOutputToPwm(
        NormalizedToPwmConfig(
            pwm_disarmed_us=900,
            pwm_min_us=1000,
            pwm_max_us=2000,
            normalized_min=0,
            normalized_max=1,
            motor_order=order,
            motor_directions=DIRECTIONS,
            clip_out_of_range=clip,
        )
    )


def rpm_model(time_constant=0.0, rpm_max=20000.0):
    return PwmToRotorSpeedModel(
        PwmToRotorSpeedConfig(
            pwm_disarmed_us=900,
            pwm_min_us=1000,
            pwm_max_us=2000,
            rpm_disarmed=0,
            rpm_idle=1000,
            rpm_max=rpm_max,
            motor_time_constant_s=time_constant,
            motor_directions=DIRECTIONS,
        )
    )


def normalized(values, armed=True):
    return NormalizedMotorCommand(values, armed, timestamp_us=10, sequence=2)


def test_normalized_zero_half_full_and_disarmed_pwm():
    model = pwm_model()
    result = model.convert(normalized((0.0, 0.5, 1.0, 0.25)))
    assert result.pulse_width_us == (1000, 1500, 2000, 1250)
    result = model.convert(normalized((1, 1, 1, 1), armed=False))
    assert result.pulse_width_us == (900, 900, 900, 900)
    assert not result.armed


def test_normalized_channel_permutation_saturation_and_rejection():
    result = pwm_model(order=(2, 0, 3, 1)).convert(normalized((-1, 0.25, 2, 0.75)))
    assert result.pulse_width_us == (2000, 1000, 1750, 1250)
    with pytest.raises(ValueError, match="outside"):
        pwm_model(clip=False).convert(normalized((0, 0, 1.01, 0)))
    with pytest.raises(ValueError, match="finite"):
        normalized((0, float("inf"), 0, 0))


def test_pwm_idle_midpoint_full_disarmed_and_direction_metadata():
    model = rpm_model()
    command = PwmMotorCommand((999, 1000, 1500, 2100), True, 4, 1)
    result = model.convert(command, dt_s=0.01)
    assert result.rpm == (1000, 1000, 10500, 20000)
    assert model.rotor_directions == DIRECTIONS
    stopped = model.convert(PwmMotorCommand((2000,) * 4, False, 5, 2), 0.01)
    assert stopped.rpm == (0, 0, 0, 0)
    assert not stopped.armed


def test_first_order_response_is_frequency_consistent_and_resettable():
    command = PwmMotorCommand((2000,) * 4, True, 0, 0)
    slow = rpm_model(time_constant=0.1)
    fast = rpm_model(time_constant=0.1)
    for _ in range(10):
        slow_result = slow.convert(command, 0.01)
    for _ in range(20):
        fast_result = fast.convert(command, 0.005)
    assert np.allclose(slow_result.rpm, fast_result.rpm, rtol=0, atol=1e-10)
    slow.reset()
    idle = slow.convert(PwmMotorCommand((900,) * 4, False, 1, 1), 0.01)
    assert idle.rpm == (0, 0, 0, 0)


def test_rotor_rpm_preserves_existing_squared_thrust_semantics():
    env = BaseAviary()
    try:
        model = rpm_model(rpm_max=env.MAX_RPM)
        rpm = np.asarray(
            model.convert(PwmMotorCommand((1500,) * 4, True, 0, 0), 0.01).rpm
        )
        assert np.array_equal(env.KF * rpm**2, np.asarray(rpm**2) * env.KF)
        direct = env._preprocessAction(rpm)
        assert np.array_equal(direct.reshape(4), rpm)
    finally:
        env.close()
