"""Quantitative final-state comparison with the ideal-tension baseline."""

from types import MappingProxyType

import numpy as np
import pytest

from multi_drone_mujoco.control.scaled_stablesoft_bending_control import (
    MAX_REFERENCE_L1_M,
    scaled_stablesoft_bending_controller_config,
)
from multi_drone_mujoco.control.tendon_modal_coordinates import (
    TendonModalCoordinates,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    TendonCommandRunConfig,
    run_tendon_command_characterization,
)
from multi_drone_mujoco.tests._scaled_stablesoft_c2_c2_closed_loop import (
    CompatibleClosedLoopConfig,
    run_compatible_closed_loop,
)


TENSIONS_N = (0.002, 0.005, 0.010)
KINDS = ("axis", "diagonal")
DT = 1.0 / 240.0


@pytest.fixture(scope="module")
def comparisons():
    result = {}
    for kind in KINDS:
        for tension in TENSIONS_N:
            command = (
                (tension, tension, 0.0, 0.0)
                if kind == "axis"
                else (tension, 0.0, 0.0, 0.0)
            )
            ideal = run_tendon_command_characterization(
                TendonCommandRunConfig(command, sim_freq=2400)
            )
            coordinates = TendonModalCoordinates.from_tendon_names(
                ideal.tendon_names
            )
            contraction = ideal.tendon_length_m[0] - ideal.tendon_length_m
            ideal_q = np.mean(
                np.asarray(
                    [coordinates.tendon_to_bending(value) for value in contraction]
                )[ideal.steady_mask],
                axis=0,
            )
            controller = None
            if float(np.sum(np.abs(ideal_q))) <= MAX_REFERENCE_L1_M:
                direction = ideal_q / np.linalg.norm(ideal_q)
                maximum_along_velocity = 0.002 / np.sum(np.abs(direction))
                pulse_ticks = max(
                    12,
                    int(
                        np.ceil(
                            np.linalg.norm(ideal_q)
                            / maximum_along_velocity
                            / DT
                        )
                    ),
                )
                pulse_s = pulse_ticks * DT
                acceleration = np.linalg.norm(ideal_q) / pulse_s ** 2
                controller = run_compatible_closed_loop(
                    CompatibleClosedLoopConfig(
                        2400,
                        scaled_stablesoft_bending_controller_config(),
                        tuple(direction),
                        acceleration,
                        pulse_s,
                        1.5,
                    )
                )
            result[kind, tension] = (ideal, ideal_q, controller)
    return MappingProxyType(result)


def test_ideal_tension_baselines_record_force_pose_power_energy_and_safety(comparisons):
    for ideal, ideal_q, _ in comparisons.values():
        assert np.all(np.isfinite(ideal.tendon_length_m))
        assert np.all(np.isfinite(ideal.tendon_velocity_m_s))
        assert np.all(np.isfinite(ideal.tip_world_m))
        assert np.all(np.isfinite(ideal.joint_rotation_vectors_rad))
        np.testing.assert_allclose(
            ideal.actuator_force_n, ideal.tension_command_n, atol=1e-15
        )
        np.testing.assert_allclose(
            ideal.generalized_actuator_power_w,
            ideal.actuator_space_power_w,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            ideal.actuator_space_power_w,
            ideal.tendon_coordinate_power_w,
            atol=1e-12,
        )
        assert np.all(np.isfinite(ideal.qfrc_actuator_nm))
        assert np.all(np.isfinite(ideal.position_energy_j))
        assert np.all(np.isfinite(ideal.kinetic_energy_j))
        assert np.all(np.isfinite(ideal.total_energy_j))
        assert ideal.safety.warning_events == ()
        assert ideal.safety.maximum_contacts == 0
        assert ideal.safety.minimum_tendon_range_margin_m >= -1e-9
        assert np.linalg.norm(ideal_q) > 0.0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tension", (0.002, 0.005))
def test_controller_reproduces_ideal_modal_and_tip_final_effect(
    comparisons, kind, tension
):
    ideal, ideal_q, controller = comparisons[kind, tension]
    assert controller is not None
    controller_q = np.mean(controller.q_actual_m[-48:], axis=0)
    controller_tip = (
        np.mean(controller.tip_position_m[-48:], axis=0)
        - controller.tip_position_m[0]
    )
    np.testing.assert_allclose(controller_q, ideal_q, atol=0.012e-6)
    np.testing.assert_allclose(
        controller_tip,
        ideal.steady_tip_displacement_m,
        atol=0.5e-6,
    )
    assert float(controller_q @ ideal_q) > 0.0
    assert np.max(controller.tension_command_n) <= 1.30 * tension
    ideal_work = float(
        np.trapezoid(ideal.actuator_space_power_w, ideal.time_s)
    )
    controller_work = float(controller.tendon_work_j[-1])
    assert 1.0 <= controller_work / ideal_work <= 1.7
    assert np.max(np.abs(controller.actuator_power_w)) <= (
        4.1 * np.max(np.abs(ideal.actuator_space_power_w))
    )


@pytest.mark.parametrize("kind", KINDS)
def test_ten_millinewton_baseline_is_outside_controller_reference_envelope(
    comparisons, kind
):
    ideal, ideal_q, controller = comparisons[kind, 0.010]
    assert controller is None
    assert np.sum(np.abs(ideal_q)) >= 380e-6
    assert MAX_REFERENCE_L1_M == 300e-6
    assert np.max(ideal.tension_command_n) == pytest.approx(0.010)
