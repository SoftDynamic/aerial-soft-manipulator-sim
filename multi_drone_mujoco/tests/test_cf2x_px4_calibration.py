"""Derived rigid-CF2X PX4 geometry and actuator calibration tests."""

import math
from pathlib import Path

import numpy as np

from multi_drone_mujoco.backends.actuation import (
    PwmToRotorSpeedModel,
    Px4NormalizedOutputToPwm,
)
from multi_drone_mujoco.backends.cf2x_px4 import (
    CF2X_PX4_MOTOR_ORDER,
    CF2X_ROTOR_DIRECTIONS,
    cf2x_27g_calibration,
    cf2x_normalized_to_pwm_config,
    cf2x_pwm_to_rpm_config,
)
from multi_drone_mujoco.backends.contracts import NormalizedMotorCommand
from multi_drone_mujoco.backends.px4_environment import read_px4_parameter_file


def test_hover_pwm_and_rpm_are_derived_from_frozen_plant():
    calibration = cf2x_27g_calibration()
    assert math.isclose(calibration.mass_kg, 0.027)
    assert calibration.inertia_kg_m2 == (1.4e-5, 1.4e-5, 2.17e-5)
    assert math.isclose(calibration.hover_normalized, 2.0 / 3.0)
    assert math.isclose(calibration.px4_hover_thrust, 4.0 / 9.0)
    assert math.isclose(
        math.sqrt(calibration.px4_hover_thrust),
        calibration.hover_normalized,
    )
    assert math.isclose(calibration.hover_pwm_us, 1666.6666666666665)
    assert math.isclose(
        4 * calibration.kf_n_per_rpm2 * calibration.hover_rpm**2,
        calibration.mass_kg * 9.81,
    )
    assert math.isclose(
        calibration.max_thrust_n,
        2.25 * calibration.mass_kg * 9.81,
    )

    pwm = Px4NormalizedOutputToPwm(cf2x_normalized_to_pwm_config()).convert(
        NormalizedMotorCommand((2 / 3,) * 4, True, 1, 1)
    )
    rpm = PwmToRotorSpeedModel(cf2x_pwm_to_rpm_config()).convert(pwm, 0.004)
    assert np.allclose(rpm.rpm, calibration.hover_rpm, rtol=0, atol=1e-10)


def test_px4_channel_to_mujoco_rotor_mapping_and_moment_signs():
    assert CF2X_PX4_MOTOR_ORDER == (2, 1, 3, 0)
    assert CF2X_ROTOR_DIRECTIONS == (-1, 1, -1, 1)
    # MuJoCo rotor order: FL, RL, RR, FR in an FLU body frame.
    positions_flu = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    roll_sign = (1, 1, -1, -1)
    pitch_sign = (-1, 1, 1, -1)
    yaw_sign = (-1, 1, -1, 1)
    channel_to_rotor = tuple(CF2X_PX4_MOTOR_ORDER.index(i) for i in range(4))
    assert channel_to_rotor == (3, 1, 0, 2)
    assert tuple(roll_sign[i] for i in channel_to_rotor) == (-1, 1, 1, -1)
    assert tuple(pitch_sign[i] for i in channel_to_rotor) == (-1, 1, -1, 1)
    assert tuple(yaw_sign[i] for i in channel_to_rotor) == (1, 1, -1, -1)
    # FLU (x forward, y left) -> FRD (x forward, y right).
    positions_frd = tuple((positions_flu[i][0], -positions_flu[i][1]) for i in channel_to_rotor)
    assert positions_frd == ((1, 1), (-1, -1), (1, -1), (-1, 1))


def test_repository_parameter_files_freeze_real_probed_yaw_and_hover_model():
    root = Path(__file__).resolve().parents[2]
    rigid = read_px4_parameter_file(root / "configs/px4/cf2x_27g.params")
    passive = read_px4_parameter_file(
        root / "configs/px4/cf2x_scaled_stablesoft_passive.params"
    )
    for parameters in (rigid, passive):
        assert tuple(
            parameters[f"CA_ROTOR{index}_KM"] for index in range(4)
        ) == ("-0.025126582", "-0.025126582", "0.025126582", "0.025126582")
        assert parameters["THR_MDL_FAC"] == "1.0"
    assert rigid["MPC_THR_HOVER"] == "0.444444444"
    assert passive["MPC_THR_HOVER"] == "0.493827160"
