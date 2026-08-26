"""C1-B2 active free-base reaction and conservation acceptance tests."""

from types import MappingProxyType

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_active_coupled import (
    C1B2RunConfig,
    TOTAL_ARM_LENGTH_M,
    activity_metrics,
    conservation_metrics,
    convergence_metrics,
    diagnostic_summary,
    energy_metrics,
    run_c1_b2,
)


SIM_FREQUENCIES = (2400, 4800)


@pytest.fixture(scope="module")
def c1_b2_results():
    return MappingProxyType(
        {
            sim_freq: run_c1_b2(C1B2RunConfig(sim_freq=sim_freq))
            for sim_freq in SIM_FREQUENCIES
        }
    )


def _diagnostics(trace):
    return diagnostic_summary(trace)


def _quat_matrices(quaternions):
    matrices = []
    for quaternion in quaternions:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        matrices.append(matrix.reshape(3, 3))
    return np.asarray(matrices)


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_active_free_space_build_preserves_production_topology(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    proof = trace.topology

    assert proof.production_counts[4] == proof.free_counts[4] + 1
    assert proof.production_counts[:4] == proof.free_counts[:4]
    assert proof.production_counts[5:] == proof.free_counts[5:]
    assert proof.production_gravity == (0.0, 0.0, -9.81)
    assert proof.free_gravity == (0.0, 0.0, 0.0)
    assert proof.production_floor_count == 1
    assert proof.free_floor_count == 0
    assert proof.named_topology_match
    assert proof.mass_inertia_routing_match
    assert proof.timestep_match
    assert proof.integrator_match
    assert proof.free_joint_name == "drone0_joint"
    assert proof.uav_body_names == ("drone0",)
    assert len(proof.arm_body_names) == 9
    assert proof.system_mass_kg == pytest.approx(0.030, abs=1e-12)
    assert len(proof.tendon_names) == 4
    assert len(proof.actuator_names) == 4
    assert len(proof.routing_site_body_names) == 72
    assert set(proof.routing_site_body_names) == set(proof.arm_body_names)
    assert len(trace.bindings) == 4
    assert trace.bindings[trace.target_binding_index].tendon_name.endswith(
        "_xp_yp_tendon"
    )


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_initial_state_and_external_force_isolation(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    physics, control = trace.physics, trace.control
    diagnostics = _diagnostics(trace)

    assert trace.status == "completed", diagnostics
    np.testing.assert_allclose(control["qvel"][0], 0.0, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(control["command"][0], 0.0)
    np.testing.assert_array_equal(control["actuator_force"][0], 0.0)
    np.testing.assert_array_equal(control["rotor_rpm"], 0.0)
    np.testing.assert_array_equal(control["rotor_wrench"], 0.0)
    np.testing.assert_array_equal(control["rotor_qfrc"], 0.0)
    assert np.all(physics["warning_counts"] == 0)
    assert np.all(physics["contacts"] == 0)
    assert np.all(np.isfinite(control["qpos"]))
    assert np.all(np.isfinite(control["qvel"]))
    assert np.all(np.isfinite(control["qacc"]))


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_manifest_resolved_half_cosine_command_drives_arm_bending(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    control = trace.control
    target = trace.target_binding_index
    non_target = np.asarray([index for index in range(4) if index != target])
    ticks = control["tick"]
    expected_scale = np.where(
        ticks >= 60,
        1.0,
        0.5 * (1.0 - np.cos(np.pi * ticks / 60.0)),
    )
    expected = 0.002 * expected_scale
    metrics = activity_metrics(trace)
    diagnostics = _diagnostics(trace)

    np.testing.assert_allclose(
        control["command"][:, target], expected, rtol=0.0, atol=1e-15
    )
    np.testing.assert_array_equal(control["command"][:, non_target], 0.0)
    np.testing.assert_allclose(
        control["actuator_force"], control["command"], rtol=0.0, atol=1e-15
    )
    assert control["command"][0, target] == 0.0
    assert control["command"][60, target] == 0.002
    assert np.all(control["command"][60:, target] == 0.002)
    assert metrics["maximum_bend_rad"] >= 1e-2, diagnostics
    assert metrics["maximum_body_tip_excursion_m"] >= 5e-4, diagnostics
    assert metrics["maximum_tendon_excursion_m"] >= 5e-5, diagnostics


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_uav_exhibits_measurable_reaction_motion(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    metrics = activity_metrics(trace)
    diagnostics = _diagnostics(trace)

    assert metrics["maximum_uav_displacement_m"] >= 5e-6, diagnostics
    assert metrics["maximum_uav_orientation_rad"] >= 1e-3, diagnostics
    assert metrics["linear_component_activity_ns"] >= 1e-8, diagnostics
    assert metrics["angular_component_activity_nms"] >= 1e-10, diagnostics


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_root_to_tip_uav_body_transform_is_directionally_correct(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    control = trace.control
    rotation = control["uav_rotation"]
    body_vector = control["root_to_tip_uav_body"]
    world_vector = control["root_to_tip_world"]
    reconstructed = np.einsum("nij,nj->ni", rotation, body_vector)
    quaternion_rotation = _quat_matrices(control["uav_quaternion"])
    quaternion_body_vector = np.einsum(
        "nji,nj->ni", quaternion_rotation, world_vector
    )

    np.testing.assert_allclose(reconstructed, world_vector, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(rotation, quaternion_rotation, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        quaternion_body_vector, body_vector, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(body_vector, axis=1),
        np.linalg.norm(world_vector, axis=1),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        body_vector[0], [0.0, 0.0, -TOTAL_ARM_LENGTH_M], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        control["uav_angular_velocity_world"],
        np.einsum(
            "nij,nj->ni", rotation, control["uav_angular_velocity_body"]
        ),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_system_com_and_total_linear_momentum_are_conserved_from_initial(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    metrics = conservation_metrics(trace)
    diagnostics = _diagnostics(trace)

    assert metrics["maximum_com_drift_m"] <= 1e-6, diagnostics
    assert metrics["com_tail_speed_m_s"] <= 2e-7, diagnostics
    assert metrics["maximum_delta_p_ns"] <= 2e-8, diagnostics
    assert metrics["delta_p_tail_positive_slope_n"] <= 1e-9, diagnostics


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_total_angular_momentum_about_instantaneous_system_com_is_conserved(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    metrics = conservation_metrics(trace)
    diagnostics = _diagnostics(trace)

    assert metrics["maximum_delta_h_nms"] <= 1e-9, diagnostics
    assert metrics["delta_h_tail_positive_slope_nm"] <= 1e-10, diagnostics


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_uav_arm_component_momenta_reconstruct_total_with_nonzero_activity(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    activity = activity_metrics(trace)
    conservation = conservation_metrics(trace)
    p_activity = activity["linear_component_activity_ns"]
    h_activity = activity["angular_component_activity_nms"]
    p_residual = conservation["maximum_p_partition_residual_ns"]
    h_residual = conservation["maximum_h_partition_residual_nms"]
    diagnostics = _diagnostics(trace)

    assert p_activity >= 1e-8, diagnostics
    assert h_activity >= 1e-10, diagnostics
    assert p_residual <= max(1e-12, 1e-6 * p_activity), diagnostics
    assert h_residual <= max(1e-14, 1e-6 * h_activity), diagnostics
    assert p_activity / max(p_residual, np.finfo(float).tiny) >= 1e4
    assert h_activity / max(h_residual, np.finfo(float).tiny) >= 1e4


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_actuator_transmission_moment_force_and_power_identities_hold(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    physics = trace.physics
    safety = trace.safety
    diagnostics = _diagnostics(trace)

    np.testing.assert_allclose(
        physics["actuator_force"], physics["command"], rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        physics["actuator_length"], -physics["tendon_length"], atol=1e-12
    )
    np.testing.assert_allclose(
        physics["actuator_velocity"], -physics["tendon_velocity"], atol=1e-12
    )
    np.testing.assert_allclose(
        physics["generalized_power"], physics["actuator_power"], atol=1e-12
    )
    np.testing.assert_allclose(
        physics["actuator_power"], physics["tendon_power"], atol=1e-12
    )
    np.testing.assert_allclose(
        physics["tendon_power"], physics["command_tendon_power"], atol=1e-12
    )
    assert safety.maximum_force_command_error_n <= 1e-15, diagnostics
    assert safety.maximum_moment_residual <= 1e-12, diagnostics
    assert safety.maximum_generalized_power_residual_w <= 1e-12, diagnostics
    assert safety.maximum_actuator_tendon_power_residual_w <= 1e-12
    assert safety.maximum_command_tendon_power_residual_w <= 1e-12


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_zero_rotor_wrench_generalized_force_power_and_work(
    c1_b2_results,
    sim_freq,
):
    trace = c1_b2_results[sim_freq]
    control, physics = trace.control, trace.physics
    diagnostics = _diagnostics(trace)

    np.testing.assert_array_equal(control["rotor_rpm"], 0.0)
    np.testing.assert_array_equal(control["rotor_wrench"], 0.0)
    assert np.max(np.abs(control["rotor_qfrc"])) <= 1e-15, diagnostics
    assert np.max(np.abs(physics["rotor_power"])) <= 1e-15, diagnostics
    assert np.max(np.abs(physics["rotor_work"])) <= 1e-15, diagnostics
    assert trace.safety.maximum_abs_rotor_generalized_force <= 1e-15
    assert trace.safety.maximum_abs_rotor_power_w <= 1e-15


def test_c1_b2_mechanical_energy_tendon_work_and_damping_ledger_closes(
    c1_b2_results,
):
    metrics = {
        sim_freq: energy_metrics(c1_b2_results[sim_freq])
        for sim_freq in SIM_FREQUENCIES
    }
    for sim_freq, item in metrics.items():
        trace = c1_b2_results[sim_freq]
        diagnostics = _diagnostics(trace)
        assert item["maximum_abs_ledger_residual_j"] <= item["ledger_limit_j"], (
            diagnostics
        )
        assert item["final_tendon_work_j"] > 0.0, diagnostics
        assert item["final_damping_dissipation_j"] >= 0.0, diagnostics
        assert abs(item["final_rotor_work_j"]) <= 1e-15, diagnostics
        assert np.min(trace.physics["damping_power"]) >= -1e-15


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b2_physics_rate_safety_contract(c1_b2_results, sim_freq):
    trace = c1_b2_results[sim_freq]
    safety = trace.safety
    diagnostics = _diagnostics(trace)

    assert trace.status == "completed", diagnostics
    assert safety.first_failure is None, diagnostics
    assert safety.completed_steps == safety.requested_steps == sim_freq
    assert len(trace.physics["time"]) == sim_freq + 1
    assert len(trace.control["time"]) == 241
    assert safety.maximum_contacts == 0
    assert safety.minimum_tendon_margin_m >= -1e-9, diagnostics
    assert safety.maximum_abs_qvel_native < 1000.0, diagnostics
    assert safety.maximum_abs_qacc_native < 2e5, diagnostics
    assert safety.maximum_root_to_tip_length_m <= TOTAL_ARM_LENGTH_M + 1e-9
    assert np.all(trace.tendon_limited)
    assert np.all(trace.physics["warning_counts"] == 0)
    assert np.all(trace.physics["contacts"] == 0)
    assert trace.control["time"][-1] == pytest.approx(1.0, abs=1e-12)
    for values in (*trace.physics.values(), *trace.control.values()):
        assert not values.flags.writeable
        assert np.all(np.isfinite(values))


def test_c1_b2_2400_converges_to_4800_on_exact_control_grid(c1_b2_results):
    candidate = c1_b2_results[2400]
    reference = c1_b2_results[4800]
    metrics = convergence_metrics(candidate, reference)
    diagnostics = {
        "metrics": dict(metrics),
        "candidate": _diagnostics(candidate),
        "reference": _diagnostics(reference),
    }

    np.testing.assert_array_equal(candidate.control["tick"], reference.control["tick"])
    np.testing.assert_allclose(
        candidate.control["time"], reference.control["time"], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        candidate.control["command"], reference.control["command"], rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        candidate.control["actuator_force"],
        reference.control["actuator_force"],
        rtol=0.0,
        atol=1e-15,
    )
    assert metrics["passed"], diagnostics
