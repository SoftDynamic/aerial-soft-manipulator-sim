"""B4 whole-vehicle passive coupling acceptance tests."""

import math

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_coupled import (
    ANGULAR_MOMENTUM_MAX,
    ANGULAR_MOMENTUM_TAIL_SLOPE,
    COM_MAX_DRIFT,
    COM_TAIL_SPEED,
    E_SPRING0,
    E_TOL,
    LINEAR_MOMENTUM_MAX,
    LINEAR_MOMENTUM_TAIL_SLOPE,
    QVEL_HARD_LIMIT,
    TOTAL_ARM_LENGTH,
    conservation_metrics,
    convergence_metrics,
    reaction_metrics,
    run_hover,
    run_zero_gravity_release,
    symmetry_metrics,
)


DIRECTIONS = ("+x", "-x", "+y", "-y")


@pytest.fixture(scope="module")
def hover_result():
    return run_hover(2400)


@pytest.fixture(scope="module")
def coupled_results():
    return {
        (sim_freq, direction): run_zero_gravity_release(sim_freq, direction)
        for sim_freq in (2400, 4800)
        for direction in DIRECTIONS
    }


def test_b4_hover_compiles_formal_passive_configuration(hover_result):
    trace = hover_result

    assert trace.timestep == pytest.approx(1.0 / 2400)
    assert trace.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert trace.topology["counts"][6] == 4
    assert trace.topology["counts"][7] == 0
    np.testing.assert_allclose(trace.topology["gravity"], [0.0, 0.0, -9.81])
    assert trace.topology["floor_count"] == 1
    assert trace.status == "completed", trace.failure
    assert trace.completed_steps == 2400
    assert trace.force_overwrite_ok


def test_b4_hover_zero_disturbance_contract(hover_result):
    trace = hover_result
    first_vertical_acceleration = (
        trace.uav_linear_velocity[1, 2] - trace.uav_linear_velocity[0, 2]
    ) / trace.timestep
    position_drift = np.max(
        np.linalg.norm(trace.uav_position - trace.uav_position[0], axis=1)
    )
    linear_speed = np.max(np.linalg.norm(trace.uav_linear_velocity, axis=1))
    quaternion_dots = np.abs(
        np.sum(trace.uav_quaternion * trace.uav_quaternion[0], axis=1)
    )
    attitude_drift = np.max(
        2 * np.arccos(np.clip(quaternion_dots, -1.0, 1.0))
    )
    angular_speed = np.max(np.linalg.norm(trace.uav_angular_velocity, axis=1))
    com_drift = np.max(
        np.linalg.norm(trace.composite_com - trace.composite_com[0], axis=1)
    )

    assert abs(first_vertical_acceleration) <= 2.0e-3
    assert position_drift <= 1.0e-4
    assert linear_speed <= 1.0e-4
    assert attitude_drift <= 1.0e-4
    assert angular_speed <= 1.0e-4
    assert com_drift <= 1.0e-4
    assert trace.warning_events == ()
    assert np.max(trace.contacts) == 0
    assert np.all(np.isfinite(trace.time))
    assert np.all(np.isfinite(trace.uav_position))
    assert np.all(np.isfinite(trace.composite_com))
    assert np.all(np.isfinite(trace.tendon_lengths))


def test_b4_free_space_transform_changes_only_gravity_and_floor(coupled_results):
    for trace in coupled_results.values():
        proof = trace.topology
        assert proof["count_match"]
        assert proof["named_topology_match"]
        assert proof["mass_inertia_routing_match"]
        assert proof["timestep_match"]
        assert proof["integrator_match"]
        np.testing.assert_allclose(proof["production_gravity"], [0.0, 0.0, -9.81])
        np.testing.assert_array_equal(proof["free_space_gravity"], [0.0, 0.0, 0.0])
        assert proof["production_counts"][4] - proof["free_space_counts"][4] == 1
        assert proof["free_space_counts"][6] == 4
        assert proof["free_space_counts"][7] == 0


@pytest.mark.parametrize("sim_freq", (2400, 4800))
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_b4_zero_gravity_release_completes_passive_contract(
    coupled_results, sim_freq, direction
):
    trace = coupled_results[(sim_freq, direction)]

    assert trace.status == "completed", trace.failure
    assert trace.completed_steps == 5 * sim_freq
    assert len(trace.time) == 5 * sim_freq + 1
    assert len(trace.control("time")) == 5 * 240 + 1
    assert trace.timestep == pytest.approx(1.0 / sim_freq)
    assert trace.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert trace.warning_events == ()
    assert np.max(trace.contacts) == 0
    assert trace.arm_qvel.shape[1] == 24
    assert np.all(np.isfinite(trace.uav_position))
    assert np.all(np.isfinite(trace.uav_quaternion))
    assert np.all(np.isfinite(trace.uav_linear_velocity))
    assert np.all(np.isfinite(trace.uav_angular_velocity))
    assert np.all(np.isfinite(trace.composite_com))
    assert np.all(np.isfinite(trace.total_linear_momentum))
    assert np.all(np.isfinite(trace.total_angular_momentum))
    assert np.all(np.isfinite(trace.joint_rotation))
    assert np.all(np.isfinite(trace.arm_qvel))
    assert np.all(np.isfinite(trace.tendon_lengths))
    assert np.all(np.isfinite(trace.total_energy))
    assert trace.maximum_qvel["value"] < QVEL_HARD_LIMIT
    assert trace.maximum_qvel["category"] == "arm_rotational"
    assert trace.maximum_qvel["joint_name"].endswith("segment_08_ball_joint")
    assert trace.maximum_qvel["local_axis"] in ("x", "y")


def test_b4_release_initialization_and_energy_semantics(coupled_results):
    for trace in coupled_results.values():
        joint_axis = 0 if trace.config.direction.endswith("x") else 1
        sign = -1.0 if trace.config.direction.startswith("-") else 1.0
        np.testing.assert_allclose(
            trace.joint_rotation[0, :, joint_axis],
            sign * 0.025,
            atol=1.0e-12,
        )
        non_axes = [axis for axis in range(3) if axis != joint_axis]
        np.testing.assert_allclose(
            trace.joint_rotation[0, :, non_axes], 0.0, atol=1.0e-12
        )
        assert trace.signed_bend[0] == pytest.approx(sign * 0.2, abs=1.0e-12)
        assert trace.initial_energy_error <= 1.0e-10
        assert trace.total_energy[0] == pytest.approx(E_SPRING0, abs=1.0e-10)
        delta = trace.total_energy - trace.total_energy[0]
        assert np.max(delta) <= 5 * E_TOL
        assert delta[-1] <= E_TOL


def test_b4_tendon_and_geometry_bounds_are_checked_per_step(coupled_results):
    for trace in coupled_results.values():
        tolerance = np.maximum(
            1.0e-9,
            1.0e-8
            * np.maximum(
                np.max(np.abs(trace.tendon_ranges), axis=1),
                np.max(np.abs(trace.tendon_lengths), axis=0),
            ),
        )
        assert np.all(
            trace.tendon_lengths
            >= trace.tendon_ranges[:, 0] - tolerance
        )
        assert np.all(
            trace.tendon_lengths
            <= trace.tendon_ranges[:, 1] + tolerance
        )
        root_tip_length = np.linalg.norm(trace.root_to_tip, axis=1)
        assert np.max(root_tip_length) <= TOTAL_ARM_LENGTH + 1.0e-9


def test_b4_free_space_com_and_momentum_are_bounded(coupled_results):
    for trace in coupled_results.values():
        metrics = conservation_metrics(trace)
        assert metrics["max_com_drift"] <= COM_MAX_DRIFT
        assert metrics["com_tail_speed"] <= COM_TAIL_SPEED
        assert metrics["max_linear_momentum"] <= LINEAR_MOMENTUM_MAX
        assert (
            metrics["linear_momentum_tail_positive_slope"]
            <= LINEAR_MOMENTUM_TAIL_SLOPE
        )
        assert metrics["max_angular_momentum"] <= ANGULAR_MOMENTUM_MAX
        assert (
            metrics["angular_momentum_tail_positive_slope"]
            <= ANGULAR_MOMENTUM_TAIL_SLOPE
        )


def test_b4_uav_and_arm_show_equal_opposite_internal_reaction(coupled_results):
    for trace in coupled_results.values():
        metrics = reaction_metrics(trace)
        assert metrics["passed"], metrics


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_b4_2400_converges_to_4800_reference(coupled_results, direction):
    metrics = convergence_metrics(
        coupled_results[(2400, direction)],
        coupled_results[(4800, direction)],
    )
    assert metrics["peak_count"]["candidate"] >= 3
    assert metrics["peak_count"]["candidate"] == metrics["peak_count"]["reference"]
    assert metrics["passed"], metrics


@pytest.mark.parametrize("sim_freq", (2400, 4800))
@pytest.mark.parametrize("direction", ("-x", "+y", "-y"))
def test_b4_directional_rotation_and_sign_symmetry(
    coupled_results, sim_freq, direction
):
    metrics = symmetry_metrics(
        coupled_results[(sim_freq, "+x")],
        coupled_results[(sim_freq, direction)],
    )
    assert metrics["passed"], metrics


def test_b4_control_samples_are_exact_physics_stride_views(coupled_results):
    for trace in coupled_results.values():
        indices = np.arange(0, len(trace.time), trace.config.stride)
        np.testing.assert_array_equal(trace.control("time"), trace.time[indices])
        np.testing.assert_array_equal(
            trace.control("joint_rotation"), trace.joint_rotation[indices]
        )
        np.testing.assert_array_equal(
            trace.control("tendon_lengths"), trace.tendon_lengths[indices]
        )
        assert math.isclose(trace.control("time")[-1], 5.0, abs_tol=1.0e-9)
