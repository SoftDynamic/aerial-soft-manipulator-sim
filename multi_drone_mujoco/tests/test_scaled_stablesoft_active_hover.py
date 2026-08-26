"""C1-B3 gravity and constant-RPM active UAV-arm acceptance tests."""

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_active_hover import (
    CASE_CORNERS,
    CORNERS,
    EARLY_END_S,
    FORMAL_CASES,
    FORMAL_TENSION_N,
    FORCED_END_S,
    RELEASE_TICK,
    TOTAL_ARM_LENGTH_M,
    C1B3RunConfig,
    diagnostic_summary,
    early_angular_impulse_limit,
    early_energy_ledger_limit,
    early_linear_impulse_limit,
    energy_window_metrics,
    forced_energy_ledger_limit,
    forced_linear_impulse_limit,
    full_duration_energy_guard,
    impulse_metrics,
    convergence_metrics,
    response_metrics,
    run_c1_b3,
)


SIM_FREQUENCIES = (2400, 4800)
ACTIVE_CASES = ("single_xp_yp", "plus_x_pair")
COMMON_MODE_DISPLACEMENT_LIMIT_M = 1e-10
COMMON_MODE_ANGLE_LIMIT_RAD = 1e-10
COMMON_MODE_TENDON_EXCURSION_LIMIT_M = 1e-10
FORCED_ANGULAR_IMPULSE_LIMIT_NMS = 6.5e-8
FULL_LINEAR_IMPULSE_ABSOLUTE_GUARD_NS = 6e-2
FULL_ANGULAR_IMPULSE_GUARD_NMS = 6e-5


@pytest.fixture(scope="module")
def c1_b3_results():
    results = {}
    for sim_freq in SIM_FREQUENCIES:
        baseline = run_c1_b3(C1B3RunConfig(sim_freq, "baseline"))
        results[(sim_freq, "baseline")] = baseline
        for case in FORMAL_CASES[1:]:
            results[(sim_freq, case)] = run_c1_b3(
                C1B3RunConfig(sim_freq, case), baseline=baseline
            )
    return MappingProxyType(results)


def _diagnostics(trace):
    return diagnostic_summary(trace)


def _matrix(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _response_peaks(values, start_tick, end_tick):
    values = np.asarray(values)
    return np.asarray(
        [
            values[index]
            for index in range(start_tick + 1, end_tick)
            if values[index] >= values[index - 1]
            and values[index] > values[index + 1]
        ]
    )


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b3_gravity_hover_model_removes_only_floor(c1_b3_results, sim_freq):
    trace = c1_b3_results[(sim_freq, "baseline")]
    proof = trace.topology

    assert proof.production_counts[4] == proof.hover_counts[4] + 1
    assert proof.production_counts[:4] == proof.hover_counts[:4]
    assert proof.production_counts[5:] == proof.hover_counts[5:]
    assert proof.production_floor_count == 1
    assert proof.hover_floor_count == 0
    assert proof.production_gravity == proof.hover_gravity == (0.0, 0.0, -9.81)
    assert proof.named_topology_match
    assert proof.physical_arrays_match
    assert proof.actuator_arrays_match
    assert proof.timestep_match
    assert proof.integrator_match
    assert proof.system_mass_kg == pytest.approx(0.030, abs=1e-12)
    assert len(proof.tendon_names) == len(proof.actuator_names) == 4


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b3_hover_mass_rpm_and_initial_force_balance_include_arm(
    c1_b3_results, sim_freq
):
    trace = c1_b3_results[(sim_freq, "baseline")]
    proof = trace.initial_hover
    diagnostics = _diagnostics(trace)

    assert proof.environment_mass_kg == pytest.approx(0.030, abs=1e-12)
    assert proof.model_system_mass_kg == pytest.approx(0.030, abs=1e-12)
    assert proof.arm_mass_kg == pytest.approx(0.003, abs=1e-12)
    assert proof.hover_rpm == pytest.approx(15258.842632421787, rel=1e-12)
    assert proof.hover_rpm < proof.maximum_rpm
    np.testing.assert_allclose(
        proof.rotor_rpm, np.full(4, proof.hover_rpm), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        proof.rotor_force_world_n, [0.0, 0.0, 0.2943], rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(proof.rotor_torque_world_nm, 0.0, atol=1e-15)
    assert np.linalg.norm(proof.force_balance_world_n) <= 1e-14, diagnostics
    np.testing.assert_allclose(proof.initial_quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(
        proof.initial_root_to_tip_body_m,
        [0.0, 0.0, -TOTAL_ARM_LENGTH_M],
        rtol=0.0,
        atol=1e-12,
    )
    assert proof.initial_qvel_max <= 1e-15
    assert proof.initial_contacts == 0
    assert not any(proof.initial_warning_counts)


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b3_zero_tension_open_loop_hover_baseline_is_stationary(
    c1_b3_results, sim_freq
):
    trace = c1_b3_results[(sim_freq, "baseline")]
    control = trace.control
    diagnostics = _diagnostics(trace)

    assert trace.status == "completed", diagnostics
    np.testing.assert_array_equal(control["command"], 0.0)
    np.testing.assert_array_equal(control["actuator_force"], 0.0)
    assert np.max(
        np.linalg.norm(
            control["uav_position_world_m"] - control["uav_position_world_m"][0],
            axis=1,
        )
    ) <= 1e-8
    assert np.max(
        np.linalg.norm(control["uav_linear_velocity_world_m_s"], axis=1)
    ) <= 1e-8
    assert np.max(trace.relative["uav_quaternion_geodesic_rad"]) <= 1e-8
    assert np.max(
        np.linalg.norm(control["uav_angular_velocity_world_rad_s"], axis=1)
    ) <= 1e-8
    assert np.max(
        np.linalg.norm(
            control["root_to_tip_uav_body_m"]
            - control["root_to_tip_uav_body_m"][0],
            axis=1,
        )
    ) <= 1e-10


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", FORMAL_CASES)
def test_c1_b3_ramp_hold_release_aligns_exactly_to_control_grid(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    control = trace.control
    by_corner = {
        corner: next(
            index
            for index, binding in enumerate(trace.bindings)
            if binding.tendon_name.endswith(f"_{corner}_tendon")
        )
        for corner in CORNERS
    }
    expected = np.zeros_like(control["command"])
    active = set(CASE_CORNERS[case])
    for tick in range(RELEASE_TICK):
        scale = (
            1.0
            if tick >= 60
            else 0.5 * (1.0 - np.cos(np.pi * tick / 60.0))
        )
        for corner in active:
            expected[tick, by_corner[corner]] = FORMAL_TENSION_N * scale

    np.testing.assert_allclose(control["time"], control["tick"] / 240.0, atol=1e-12)
    np.testing.assert_allclose(control["command"], expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(control["actuator_force"], expected, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(control["command"][RELEASE_TICK:], 0.0)


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", ACTIVE_CASES)
def test_c1_b3_active_cases_deform_arm_relative_to_matched_baseline(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    metrics = response_metrics(trace)
    diagnostics = _diagnostics(trace)
    if case == "single_xp_yp":
        assert metrics["release_tip_m"] >= 5e-4, diagnostics
        assert metrics["release_bend_rad"] >= 1e-2, diagnostics
    else:
        assert metrics["release_tip_m"] >= 7e-4, diagnostics
        assert metrics["release_bend_rad"] >= 2e-2, diagnostics
    assert metrics["maximum_tendon_excursion_m"] >= 5e-5, diagnostics
    if case == "plus_x_pair":
        tendon_norm = np.linalg.norm(trace.relative["tendon_excursion_m"], axis=1)
        assert np.max(tendon_norm) >= 1e-4, diagnostics


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", ACTIVE_CASES)
def test_c1_b3_active_cases_disturb_uav_relative_to_matched_baseline(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    metrics = response_metrics(trace)
    diagnostics = _diagnostics(trace)
    if case == "single_xp_yp":
        assert metrics["release_position_m"] >= 1e-2, diagnostics
        assert metrics["release_attitude_rad"] >= 3e-2, diagnostics
    else:
        assert metrics["release_position_m"] >= 1.5e-2, diagnostics
        assert metrics["release_attitude_rad"] >= 5e-2, diagnostics
    assert metrics["final_position_m"] > metrics["release_position_m"]
    assert metrics["final_attitude_rad"] > metrics["release_attitude_rad"]


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", ACTIVE_CASES)
def test_c1_b3_root_to_tip_body_transform_and_so3_states_are_consistent(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    control = trace.control
    reconstructed = np.einsum(
        "nij,nj->ni",
        control["uav_rotation_matrix"],
        control["root_to_tip_uav_body_m"],
    )
    np.testing.assert_allclose(
        reconstructed, control["root_to_tip_world_m"], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        control["uav_angular_velocity_world_rad_s"],
        np.einsum(
            "nij,nj->ni",
            control["uav_rotation_matrix"],
            control["uav_angular_velocity_body_rad_s"],
        ),
        rtol=1e-10,
        atol=1e-12,
    )
    for matrix, quaternion in zip(
        trace.relative["uav_rotation_matrix"],
        trace.relative["uav_quaternion"],
    ):
        np.testing.assert_allclose(matrix, _matrix(quaternion), atol=1e-12)
    np.testing.assert_allclose(
        trace.relative["uav_quaternion_geodesic_rad"],
        np.linalg.norm(trace.relative["uav_rotation_vector_rad"], axis=1),
        atol=1e-12,
    )


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", FORMAL_CASES)
def test_c1_b3_nonzero_rotor_wrench_reconstructs_generalized_force(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    control, physics = trace.control, trace.physics
    diagnostics = _diagnostics(trace)

    np.testing.assert_allclose(
        control["rotor_xfrc_applied"][:, :3],
        control["rotor_force_world_n"],
        atol=0.0,
    )
    np.testing.assert_allclose(
        control["rotor_xfrc_applied"][:, 3:],
        control["rotor_torque_world_nm"],
        atol=0.0,
    )
    assert np.min(np.linalg.norm(control["rotor_force_world_n"], axis=1)) >= 0.2942
    assert np.min(physics["rotor_generalized_force_norm"]) >= 0.2942
    assert np.max(np.abs(control["rotor_torque_world_nm"])) <= 1e-15
    assert np.all(np.isfinite(control["rotor_generalized_force"])), diagnostics


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", FORMAL_CASES)
def test_c1_b3_rotor_generalized_and_cartesian_mechanical_power_match(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    physics = trace.physics
    scale = np.maximum(
        np.abs(physics["rotor_generalized_power_w"]),
        np.abs(physics["rotor_cartesian_power_w"]),
    )
    residual = np.abs(
        physics["rotor_generalized_power_w"]
        - physics["rotor_cartesian_power_w"]
    )
    assert np.all(residual <= 1e-12 + 1e-9 * scale), _diagnostics(trace)
    if case in ACTIVE_CASES:
        early_tick = int(EARLY_END_S * 240)
        assert control_value(trace, "rotor_work_j", early_tick) >= (
            5e-3 if case == "single_xp_yp" else 1e-2
        )
    else:
        assert np.max(np.abs(physics["rotor_work_j"])) <= 1e-12


def control_value(trace, name, tick):
    return float(trace.control[name][tick])


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", ACTIVE_CASES)
def test_c1_b3_external_force_impulse_matches_system_linear_momentum_change(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    metrics = impulse_metrics(trace)
    forced = metrics["forced"]
    early = metrics["early"]
    full = metrics["full"]
    diagnostics = _diagnostics(trace)

    assert forced["maximum_linear_residual_ns"] <= forced_linear_impulse_limit(
        forced["peak_linear_impulse_ns"]
    ), diagnostics
    assert early["maximum_linear_residual_ns"] <= early_linear_impulse_limit(
        early["peak_linear_impulse_ns"]
    ), diagnostics
    assert full["maximum_linear_residual_ns"] <= max(
        FULL_LINEAR_IMPULSE_ABSOLUTE_GUARD_NS,
        0.20 * full["peak_linear_impulse_ns"],
    ), diagnostics


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", ACTIVE_CASES)
def test_c1_b3_external_torque_impulse_matches_system_angular_momentum_change(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    metrics = impulse_metrics(trace)
    forced = metrics["forced"]
    early = metrics["early"]
    diagnostics = _diagnostics(trace)

    assert forced["maximum_angular_residual_nms"] <= (
        FORCED_ANGULAR_IMPULSE_LIMIT_NMS
    ), diagnostics
    assert early["maximum_angular_residual_nms"] <= early_angular_impulse_limit(
        early["peak_delta_h_nms"], early["peak_angular_impulse_nms"]
    ), diagnostics
    assert metrics["full"]["maximum_angular_residual_nms"] <= (
        FULL_ANGULAR_IMPULSE_GUARD_NMS
    ), diagnostics
    assert trace.safety.maximum_gravity_torque_residual_nm <= 1e-12


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", FORMAL_CASES)
def test_c1_b3_tendon_rotor_damping_mechanical_energy_ledger(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    metrics = energy_window_metrics(trace)
    forced, early, full = (
        metrics["forced"],
        metrics["early"],
        metrics["full"],
    )
    diagnostics = _diagnostics(trace)

    assert forced["maximum_abs_ledger_residual_j"] <= forced_energy_ledger_limit(
        forced["peak_abs_combined_work_j"]
    ), diagnostics
    assert early["maximum_abs_ledger_residual_j"] <= early_energy_ledger_limit(
        early["peak_abs_combined_work_j"]
    ), diagnostics
    assert full["maximum_abs_ledger_residual_j"] <= full_duration_energy_guard(
        full["peak_abs_combined_work_j"]
    ), diagnostics
    assert np.min(trace.physics["damping_power_w"]) >= -1e-15
    assert trace.safety.maximum_actuator_moment_residual <= 1e-12
    assert trace.safety.maximum_tendon_power_identity_residual_w <= 1e-12


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", ACTIVE_CASES)
def test_c1_b3_release_reduces_relative_arm_deformation_in_early_window(
    c1_b3_results, sim_freq, case
):
    trace = c1_b3_results[(sim_freq, case)]
    metrics = response_metrics(trace)
    diagnostics = _diagnostics(trace)
    assert metrics["early_tip_m"] <= 0.10 * metrics["release_tip_m"], diagnostics
    assert metrics["early_bend_rad"] <= 0.10 * metrics["release_bend_rad"], diagnostics

    tip = np.linalg.norm(trace.relative["root_to_tip_uav_body_m"], axis=1)
    peaks = _response_peaks(tip, RELEASE_TICK, int(EARLY_END_S * 240))
    assert len(peaks) >= 4, {"peaks_m": peaks.tolist(), **diagnostics}
    assert peaks[3] <= 0.40 * peaks[0], {"peaks_m": peaks.tolist(), **diagnostics}


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b3_common_mode_structural_zero_transverse_response(
    c1_b3_results, sim_freq
):
    trace = c1_b3_results[(sim_freq, "common_mode")]
    relative = trace.relative
    diagnostics = _diagnostics(trace)
    active_force = trace.control["actuator_force"][60:RELEASE_TICK]

    assert np.min(active_force) == pytest.approx(FORMAL_TENSION_N, abs=1e-15)
    assert np.max(np.linalg.norm(relative["uav_position_world_m"], axis=1)) <= (
        COMMON_MODE_DISPLACEMENT_LIMIT_M
    ), diagnostics
    assert np.max(relative["uav_quaternion_geodesic_rad"]) <= (
        COMMON_MODE_ANGLE_LIMIT_RAD
    ), diagnostics
    assert np.max(
        np.abs(relative["composite_rotation_vector_rad"][:, :2])
    ) <= COMMON_MODE_ANGLE_LIMIT_RAD, diagnostics
    assert np.max(
        np.abs(relative["composite_rotation_vector_rad"][:, 2])
    ) <= COMMON_MODE_ANGLE_LIMIT_RAD, diagnostics
    assert np.max(np.abs(relative["tendon_excursion_m"])) <= (
        COMMON_MODE_TENDON_EXCURSION_LIMIT_M
    ), diagnostics


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", FORMAL_CASES)
def test_c1_b3_physics_rate_safety_contract(c1_b3_results, sim_freq, case):
    trace = c1_b3_results[(sim_freq, case)]
    safety = trace.safety
    diagnostics = _diagnostics(trace)

    assert trace.status == "completed", diagnostics
    assert safety.first_failure is None, diagnostics
    assert safety.completed_steps == safety.requested_steps == 3 * sim_freq
    assert len(trace.physics["time"]) == 3 * sim_freq + 1
    assert len(trace.control["time"]) == 721
    assert safety.maximum_contacts == 0
    assert safety.minimum_tendon_margin_m >= -1e-9
    assert safety.maximum_abs_qvel_native < 1000.0
    assert safety.maximum_abs_qacc_native < 2e5
    assert safety.maximum_root_to_tip_length_m <= TOTAL_ARM_LENGTH_M + 1e-9
    assert safety.maximum_force_command_error_n <= 1e-15
    assert np.all(trace.physics["warnings"] == 0)
    assert np.all(trace.physics["contacts"] == 0)
    for values in (*trace.physics.values(), *trace.control.values(), *trace.relative.values()):
        assert not values.flags.writeable
        assert np.all(np.isfinite(values))


def test_c1_b3_shared_results_and_metadata_are_immutable(c1_b3_results):
    trace = c1_b3_results[(2400, "single_xp_yp")]
    with pytest.raises((ValueError, RuntimeError)):
        trace.control["time"][0] = 1.0
    with pytest.raises((TypeError, AttributeError)):
        trace.relative["new"] = np.zeros(1)
    with pytest.raises(FrozenInstanceError):
        trace.config.case = "baseline"
    with pytest.raises(FrozenInstanceError):
        trace.bindings[0].tendon_id = -1


@pytest.mark.parametrize("case", FORMAL_CASES)
def test_c1_b3_2400_converges_to_4800_on_common_control_grid(
    c1_b3_results, case
):
    candidate = c1_b3_results[(2400, case)]
    reference = c1_b3_results[(4800, case)]
    metrics = convergence_metrics(candidate, reference)
    diagnostics = {
        "case": case,
        "metrics": dict(metrics),
        "candidate": _diagnostics(candidate),
        "reference": _diagnostics(reference),
    }
    np.testing.assert_allclose(
        candidate.control["time"], reference.control["time"], atol=1e-12
    )
    assert metrics["passed"], diagnostics
