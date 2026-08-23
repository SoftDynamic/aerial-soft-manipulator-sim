"""C1-A2 fixed-root single-tendon active dynamics acceptance tests."""

from types import MappingProxyType

import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    CORNERS,
    CTRL_FREQ,
    MAX_TENSION_N,
    RAMP_DURATION_S,
    STEADY_WINDOW_START_S,
    TOTAL_DURATION_S,
    ActiveTendonRunConfig,
    half_cosine_tension_at_tick,
    mixed_limit,
    run_single_tendon_characterization,
)


TENSION_LEVELS_N = (0.002, 0.005, 0.01)
EXPECTED_QUADRANTS = {
    # Empirical world-frame mapping frozen by the C1-A2 numerical runs.
    "xp_yp": (1, 1),
    "xp_yn": (1, -1),
    "xn_yp": (-1, 1),
    "xn_yn": (-1, -1),
}
DIAGONAL_PAIRS = (
    ("xp_yp", "xn_yn"),
    ("xp_yn", "xn_yp"),
)
ROTATE_Z_PI = np.diag([-1.0, -1.0, 1.0])
DIAGONAL_CHANNEL_PERMUTATION = np.asarray([3, 2, 1, 0], dtype=int)


def _trace_diagnostics(trace):
    steady_tip = trace.steady_tip_displacement_m
    steady_mask = trace.steady_mask
    steady_mean = np.mean(trace.tip_displacement_m[steady_mask], axis=0)
    tip_drift = float(
        np.max(
            np.linalg.norm(
                trace.tip_displacement_m[steady_mask] - steady_mean,
                axis=1,
            )
        )
    )
    return (
        f"corner={trace.config.corner}, tension="
        f"{trace.config.final_tension_n}, sim_freq={trace.config.sim_freq}, "
        f"steady_tip={steady_tip}, steady_bend="
        f"{trace.steady_total_bend_rad}, steady_shortening="
        f"{trace.steady_target_tendon_excursion_m}, "
        f"tip_drift={tip_drift}, steady_qvel="
        f"{np.max(trace.maximum_arm_qvel_rad_s[steady_mask])}, "
        f"safety={trace.safety}"
    )


@pytest.fixture(scope="module")
def active_traces_2400():
    traces = {
        (corner, tension): run_single_tendon_characterization(
            ActiveTendonRunConfig(
                corner=corner,
                final_tension_n=tension,
                sim_freq=2400,
            )
        )
        for tension in TENSION_LEVELS_N
        for corner in CORNERS
    }
    return MappingProxyType(traces)


@pytest.fixture(scope="module")
def active_traces_4800():
    traces = {
        corner: run_single_tendon_characterization(
            ActiveTendonRunConfig(
                corner=corner,
                final_tension_n=MAX_TENSION_N,
                sim_freq=4800,
            )
        )
        for corner in CORNERS
    }
    return MappingProxyType(traces)


@pytest.mark.parametrize("final_tension_n", TENSION_LEVELS_N)
def test_half_cosine_ramp_has_exact_endpoints_and_no_overshoot(
    final_tension_n,
):
    config = ActiveTendonRunConfig("xp_yp", final_tension_n)
    values = np.asarray(
        [
            half_cosine_tension_at_tick(
                final_tension_n,
                tick,
                config.ramp_ticks,
            )
            for tick in range(2 * config.ramp_ticks + 1)
        ]
    )

    assert config.ramp_ticks == 60
    assert values[0] == 0.0
    assert values[config.ramp_ticks] == final_tension_n
    assert np.all(values[config.ramp_ticks:] == final_tension_n)
    assert np.all(values >= 0.0)
    assert np.all(values <= final_tension_n)
    assert np.all(np.diff(values) >= 0.0)


def test_active_run_config_rejects_ramp_not_aligned_to_control_ticks():
    with pytest.raises(ValueError, match="control tick grid"):
        ActiveTendonRunConfig(
            "xp_yp",
            0.002,
            ramp_duration_s=0.251,
        )


@pytest.mark.parametrize("corner", CORNERS)
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_single_tendon_trace_has_complete_read_only_control_grid(
    active_traces_2400,
    corner,
    tension_n,
):
    trace = active_traces_2400[(corner, tension_n)]
    config = trace.config
    expected_ticks = np.arange(config.total_ticks + 1)
    expected_time = expected_ticks / config.ctrl_freq
    expected_steps = expected_ticks * config.sample_stride
    diagnostics = _trace_diagnostics(trace)

    assert config.ctrl_freq == CTRL_FREQ
    assert config.ramp_duration_s == RAMP_DURATION_S
    assert config.total_duration_s == TOTAL_DURATION_S
    assert config.steady_window_start_s == STEADY_WINDOW_START_S
    assert config.total_ticks == 840
    assert config.requested_steps == 8400
    assert trace.safety.completed_steps == config.requested_steps
    assert trace.model_counts == (4, 4, 0, 4, 4)
    np.testing.assert_array_equal(trace.control_tick, expected_ticks)
    np.testing.assert_array_equal(trace.physics_step, expected_steps)
    np.testing.assert_allclose(
        trace.time_s,
        expected_time,
        rtol=0.0,
        atol=1e-12,
    )
    assert len(trace.time_s) == 841
    assert trace.time_s[-1] == pytest.approx(3.5, abs=1e-12)

    for values in (
        trace.time_s,
        trace.control_tick,
        trace.physics_step,
        trace.tension_command_n,
        trace.actuator_force_n,
        trace.tip_world_m,
        trace.tip_displacement_m,
        trace.joint_rotation_vectors_rad,
        trace.composite_rotation_vectors_rad,
        trace.tendon_length_m,
        trace.tendon_velocity_m_s,
        trace.actuator_length_m,
        trace.actuator_velocity_m_s,
        trace.qpos,
        trace.qvel_rad_s,
        trace.qacc_rad_s2,
        trace.qfrc_actuator_nm,
        trace.generalized_actuator_power_w,
        trace.actuator_space_power_w,
        trace.tendon_coordinate_power_w,
        trace.warning_counts,
        trace.contacts,
        trace.tendon_range_margin_m,
        trace.position_energy_j,
        trace.kinetic_energy_j,
        trace.total_energy_j,
    ):
        assert not values.flags.writeable
        assert np.all(np.isfinite(values)), diagnostics


@pytest.mark.parametrize("corner", CORNERS)
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_single_tendon_command_force_response_and_steady_state(
    active_traces_2400,
    corner,
    tension_n,
):
    trace = active_traces_2400[(corner, tension_n)]
    target = trace.target_index
    non_target = np.asarray(
        [index for index in range(4) if index != target],
        dtype=int,
    )
    expected_target = np.asarray(
        [
            half_cosine_tension_at_tick(
                tension_n,
                int(tick),
                trace.config.ramp_ticks,
            )
            for tick in trace.control_tick
        ]
    )
    diagnostics = _trace_diagnostics(trace)

    np.testing.assert_array_equal(
        trace.tension_command_n[:, target],
        expected_target,
    )
    np.testing.assert_array_equal(
        trace.tension_command_n[:, non_target],
        0.0,
    )
    np.testing.assert_allclose(
        trace.actuator_force_n,
        trace.tension_command_n,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        trace.actuator_force_n[:, non_target],
        0.0,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.min(trace.actuator_force_n[:, target]) >= -1e-15
    assert np.max(trace.actuator_force_n[:, target]) <= tension_n + 1e-15
    np.testing.assert_allclose(
        trace.actuator_force_n[trace.control_tick >= trace.config.ramp_ticks],
        trace.tension_command_n[
            trace.control_tick >= trace.config.ramp_ticks
        ],
        rtol=0.0,
        atol=1e-15,
    )

    steady_tip = trace.steady_tip_displacement_m
    assert abs(steady_tip[0]) >= 1e-4, diagnostics
    assert abs(steady_tip[1]) >= 1e-4, diagnostics
    assert trace.steady_target_tendon_excursion_m <= -1e-6, diagnostics

    steady_values = trace.tip_displacement_m[trace.steady_mask]
    steady_mean = np.mean(steady_values, axis=0)
    tip_window_drift = float(
        np.max(np.linalg.norm(steady_values - steady_mean, axis=1))
    )
    steady_excursion = float(np.linalg.norm(steady_mean))
    assert tip_window_drift <= mixed_limit(
        1e-6,
        0.005,
        steady_excursion,
    ), diagnostics
    assert np.max(
        trace.maximum_arm_qvel_rad_s[trace.steady_mask]
    ) <= 5e-4, diagnostics


@pytest.mark.parametrize("corner", CORNERS)
def test_single_tendon_response_increases_strictly_with_tension(
    active_traces_2400,
    corner,
):
    traces = [
        active_traces_2400[(corner, tension)]
        for tension in TENSION_LEVELS_N
    ]
    radial_tip = [
        float(np.linalg.norm(trace.steady_tip_displacement_m[:2]))
        for trace in traces
    ]
    total_bend = [trace.steady_total_bend_rad for trace in traces]
    shortening = [
        -trace.steady_target_tendon_excursion_m for trace in traces
    ]
    diagnostics = (
        f"corner={corner}, radial_tip={radial_tip}, bend={total_bend}, "
        f"shortening={shortening}"
    )

    assert radial_tip[0] < radial_tip[1] < radial_tip[2], diagnostics
    assert total_bend[0] < total_bend[1] < total_bend[2], diagnostics
    assert shortening[0] < shortening[1] < shortening[2], diagnostics


@pytest.mark.parametrize("corner", CORNERS)
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_single_tendon_world_xy_quadrant_is_empirically_locked(
    active_traces_2400,
    corner,
    tension_n,
):
    trace = active_traces_2400[(corner, tension_n)]
    steady_xy = trace.steady_tip_displacement_m[:2]
    expected_signs = EXPECTED_QUADRANTS[corner]
    diagnostics = _trace_diagnostics(trace)

    assert np.all(np.abs(steady_xy) >= 1e-4), diagnostics
    assert tuple(np.sign(steady_xy).astype(int)) == expected_signs


@pytest.mark.parametrize("corner", CORNERS)
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_single_tendon_twist_stays_below_mixed_bending_limit(
    active_traces_2400,
    corner,
    tension_n,
):
    trace = active_traces_2400[(corner, tension_n)]
    joints = trace.joint_rotation_vectors_rad
    composite = trace.composite_rotation_vectors_rad
    maximum_joint_twist = np.max(np.abs(joints[:, :, 2]), axis=0)
    maximum_joint_bend = np.max(
        np.linalg.norm(joints[:, :, :2], axis=2),
        axis=0,
    )
    joint_limits = np.maximum(1e-6, 0.01 * maximum_joint_bend)
    maximum_composite_twist = float(np.max(np.abs(composite[:, 2])))
    maximum_total_bend = float(
        np.max(np.linalg.norm(composite[:, :2], axis=1))
    )
    composite_limit = max(1e-6, 0.01 * maximum_total_bend)
    diagnostics = _trace_diagnostics(trace)

    assert np.all(maximum_joint_twist <= joint_limits), (
        diagnostics,
        maximum_joint_twist,
        joint_limits,
    )
    assert maximum_composite_twist <= composite_limit, (
        diagnostics,
        maximum_composite_twist,
        composite_limit,
    )
    assert np.all(
        np.isfinite(
            np.mean(joints[trace.steady_mask, :, 2], axis=0)
        )
    )
    assert np.isfinite(
        np.mean(composite[trace.steady_mask, 2])
    )


@pytest.mark.parametrize("corner", CORNERS)
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_single_tendon_physics_rate_safety_moment_and_power_gates(
    active_traces_2400,
    corner,
    tension_n,
):
    trace = active_traces_2400[(corner, tension_n)]
    safety = trace.safety
    diagnostics = _trace_diagnostics(trace)

    assert safety.first_failure is None, diagnostics
    assert safety.warning_events == (), diagnostics
    assert safety.maximum_contacts == 0, diagnostics
    assert safety.completed_steps == safety.requested_steps == 8400
    assert safety.minimum_tendon_range_margin_m >= -1e-9, diagnostics
    assert safety.maximum_force_command_error_n <= 1e-15, diagnostics
    assert safety.maximum_moment_residual <= 1e-12, diagnostics
    assert (
        safety.maximum_generalized_actuator_power_residual_w <= 1e-12
    ), diagnostics
    assert (
        safety.maximum_actuator_tendon_power_residual_w <= 1e-12
    ), diagnostics
    assert safety.maximum_command_tendon_power_residual_w <= 1e-12
    assert safety.maximum_actuator_length_sign_residual_m <= 1e-12
    assert safety.maximum_actuator_velocity_sign_residual_m_s <= 1e-12
    assert np.all(trace.tendon_limited), diagnostics
    assert np.all(trace.contacts == 0), diagnostics
    assert np.all(trace.warning_counts == 0), diagnostics
    assert np.all(trace.tendon_range_margin_m >= -1e-9), diagnostics


def _maximum_vector_error(candidate, reference) -> float:
    return float(
        np.max(np.linalg.norm(candidate - reference, axis=-1))
    )


def _relative_scalar_limit(left: float, right: float, relative: float):
    return mixed_limit(1e-12, relative, max(abs(left), abs(right)))


@pytest.mark.parametrize("left_corner,right_corner", DIAGONAL_PAIRS)
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_diagonal_tendons_are_full_trajectory_z_pi_mirrors(
    active_traces_2400,
    left_corner,
    right_corner,
    tension_n,
):
    left = active_traces_2400[(left_corner, tension_n)]
    right = active_traces_2400[(right_corner, tension_n)]
    permutation = DIAGONAL_CHANNEL_PERMUTATION

    np.testing.assert_array_equal(left.control_tick, right.control_tick)
    np.testing.assert_allclose(left.time_s, right.time_s, atol=1e-15)
    np.testing.assert_allclose(
        left.tension_command_n,
        right.tension_command_n[:, permutation],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        left.actuator_force_n,
        right.actuator_force_n[:, permutation],
        rtol=0.0,
        atol=1e-15,
    )

    transformed_tip = left.tip_displacement_m @ ROTATE_Z_PI.T
    tip_error = _maximum_vector_error(
        transformed_tip,
        right.tip_displacement_m,
    )
    tip_scale = max(
        float(np.max(np.linalg.norm(left.tip_displacement_m, axis=1))),
        float(np.max(np.linalg.norm(right.tip_displacement_m, axis=1))),
    )
    tip_limit = mixed_limit(1e-8, 0.005, tip_scale)

    transformed_joints = (
        left.joint_rotation_vectors_rad @ ROTATE_Z_PI.T
    )
    joint_error = _maximum_vector_error(
        transformed_joints,
        right.joint_rotation_vectors_rad,
    )
    joint_scale = max(
        float(
            np.max(
                np.linalg.norm(
                    left.joint_rotation_vectors_rad,
                    axis=2,
                )
            )
        ),
        float(
            np.max(
                np.linalg.norm(
                    right.joint_rotation_vectors_rad,
                    axis=2,
                )
            )
        ),
    )
    joint_limit = mixed_limit(1e-7, 0.005, joint_scale)

    transformed_composite = (
        left.composite_rotation_vectors_rad @ ROTATE_Z_PI.T
    )
    composite_error = _maximum_vector_error(
        transformed_composite,
        right.composite_rotation_vectors_rad,
    )
    composite_scale = max(
        float(
            np.max(
                np.linalg.norm(
                    left.composite_rotation_vectors_rad,
                    axis=1,
                )
            )
        ),
        float(
            np.max(
                np.linalg.norm(
                    right.composite_rotation_vectors_rad,
                    axis=1,
                )
            )
        ),
    )
    composite_limit = mixed_limit(1e-7, 0.005, composite_scale)

    left_length_excursion = (
        left.tendon_length_m - left.tendon_length_m[0]
    )
    right_length_excursion = (
        right.tendon_length_m - right.tendon_length_m[0]
    )[:, permutation]
    tendon_length_error = float(
        np.max(
            np.abs(left_length_excursion - right_length_excursion)
        )
    )
    tendon_length_scale = max(
        float(np.max(np.abs(left_length_excursion))),
        float(np.max(np.abs(right_length_excursion))),
    )
    tendon_length_limit = mixed_limit(
        1e-10,
        0.005,
        tendon_length_scale,
    )

    right_tendon_velocity = right.tendon_velocity_m_s[:, permutation]
    tendon_velocity_error = float(
        np.max(
            np.abs(left.tendon_velocity_m_s - right_tendon_velocity)
        )
    )
    tendon_velocity_scale = max(
        float(np.max(np.abs(left.tendon_velocity_m_s))),
        float(np.max(np.abs(right_tendon_velocity))),
    )
    tendon_velocity_limit = mixed_limit(
        1e-8,
        0.005,
        tendon_velocity_scale,
    )

    bend_error = float(
        np.max(np.abs(left.total_bend_rad - right.total_bend_rad))
    )
    bend_limit = mixed_limit(
        1e-7,
        0.005,
        max(
            float(np.max(left.total_bend_rad)),
            float(np.max(right.total_bend_rad)),
        ),
    )
    steady_tip_error = float(
        np.linalg.norm(
            left.steady_tip_displacement_m @ ROTATE_Z_PI.T
            - right.steady_tip_displacement_m
        )
    )
    left_peak_qvel = float(np.max(left.maximum_arm_qvel_rad_s))
    right_peak_qvel = float(np.max(right.maximum_arm_qvel_rad_s))
    peak_qvel_error = abs(left_peak_qvel - right_peak_qvel)
    peak_qvel_limit = _relative_scalar_limit(
        left_peak_qvel,
        right_peak_qvel,
        0.005,
    )
    left_peak_power = float(
        np.max(np.abs(left.actuator_space_power_w))
    )
    right_peak_power = float(
        np.max(np.abs(right.actuator_space_power_w))
    )
    peak_power_error = abs(left_peak_power - right_peak_power)
    peak_power_limit = _relative_scalar_limit(
        left_peak_power,
        right_peak_power,
        0.005,
    )
    diagnostics = {
        "pair": (left_corner, right_corner),
        "tension_n": tension_n,
        "tip": (tip_error, tip_limit),
        "joint": (joint_error, joint_limit),
        "composite": (composite_error, composite_limit),
        "tendon_length": (tendon_length_error, tendon_length_limit),
        "tendon_velocity": (tendon_velocity_error, tendon_velocity_limit),
        "bend": (bend_error, bend_limit),
        "steady_tip": (steady_tip_error, tip_limit),
        "peak_qvel": (peak_qvel_error, peak_qvel_limit),
        "peak_power": (peak_power_error, peak_power_limit),
    }

    assert tip_error <= tip_limit, diagnostics
    assert joint_error <= joint_limit, diagnostics
    assert composite_error <= composite_limit, diagnostics
    assert tendon_length_error <= tendon_length_limit, diagnostics
    assert tendon_velocity_error <= tendon_velocity_limit, diagnostics
    assert bend_error <= bend_limit, diagnostics
    assert steady_tip_error <= tip_limit, diagnostics
    assert peak_qvel_error <= peak_qvel_limit, diagnostics
    assert peak_power_error <= peak_power_limit, diagnostics


@pytest.mark.parametrize("corner", CORNERS)
def test_active_2400_hz_converges_to_4800_hz_control_grid_trace(
    active_traces_2400,
    active_traces_4800,
    corner,
):
    candidate = active_traces_2400[(corner, MAX_TENSION_N)]
    reference = active_traces_4800[corner]
    np.testing.assert_array_equal(
        candidate.control_tick,
        reference.control_tick,
    )
    np.testing.assert_allclose(
        candidate.time_s,
        reference.time_s,
        rtol=0.0,
        atol=1e-12,
    )
    assert candidate.safety.completed_steps == 8400
    assert reference.safety.completed_steps == 16800

    tip_error = _maximum_vector_error(
        candidate.tip_displacement_m,
        reference.tip_displacement_m,
    )
    reference_tip_excursion = float(
        np.max(np.linalg.norm(reference.tip_displacement_m, axis=1))
    )
    tip_limit = mixed_limit(2e-5, 0.02, reference_tip_excursion)

    joint_error = _maximum_vector_error(
        candidate.joint_rotation_vectors_rad,
        reference.joint_rotation_vectors_rad,
    )
    joint_amplitude = float(
        np.max(
            np.linalg.norm(
                reference.joint_rotation_vectors_rad,
                axis=2,
            )
        )
    )
    joint_limit = mixed_limit(1e-4, 0.02, joint_amplitude)

    composite_error = _maximum_vector_error(
        candidate.composite_rotation_vectors_rad,
        reference.composite_rotation_vectors_rad,
    )
    composite_amplitude = float(
        np.max(
            np.linalg.norm(
                reference.composite_rotation_vectors_rad,
                axis=1,
            )
        )
    )
    composite_limit = mixed_limit(
        1e-4,
        0.02,
        composite_amplitude,
    )

    reference_tendon_excursion = (
        reference.tendon_length_m - reference.tendon_length_m[0]
    )
    tendon_length_error = float(
        np.max(
            np.abs(
                candidate.tendon_length_m - reference.tendon_length_m
            )
        )
    )
    tendon_length_limit = mixed_limit(
        2e-8,
        0.02,
        float(np.max(np.abs(reference_tendon_excursion))),
    )

    candidate_velocity_extrema = np.stack(
        (
            np.min(candidate.tendon_velocity_m_s, axis=0),
            np.max(candidate.tendon_velocity_m_s, axis=0),
        )
    )
    reference_velocity_extrema = np.stack(
        (
            np.min(reference.tendon_velocity_m_s, axis=0),
            np.max(reference.tendon_velocity_m_s, axis=0),
        )
    )
    tendon_velocity_error = float(
        np.max(
            np.abs(
                candidate_velocity_extrema - reference_velocity_extrema
            )
        )
    )
    tendon_velocity_limit = mixed_limit(
        1e-6,
        0.02,
        float(np.max(np.abs(reference_velocity_extrema))),
    )

    steady_tip_error = float(
        np.linalg.norm(
            candidate.steady_tip_displacement_m
            - reference.steady_tip_displacement_m
        )
    )
    steady_tip_limit = mixed_limit(
        1e-8,
        0.02,
        float(np.linalg.norm(reference.steady_tip_displacement_m)),
    )
    steady_bend_error = abs(
        candidate.steady_total_bend_rad
        - reference.steady_total_bend_rad
    )
    steady_bend_limit = mixed_limit(
        1e-8,
        0.02,
        reference.steady_total_bend_rad,
    )
    candidate_peak_qvel = float(
        np.max(candidate.maximum_arm_qvel_rad_s)
    )
    reference_peak_qvel = float(
        np.max(reference.maximum_arm_qvel_rad_s)
    )
    peak_qvel_error = abs(candidate_peak_qvel - reference_peak_qvel)
    peak_qvel_limit = mixed_limit(
        1e-8,
        0.02,
        reference_peak_qvel,
    )
    candidate_peak_power = float(
        np.max(np.abs(candidate.actuator_space_power_w))
    )
    reference_peak_power = float(
        np.max(np.abs(reference.actuator_space_power_w))
    )
    peak_power_error = abs(candidate_peak_power - reference_peak_power)
    peak_power_limit = mixed_limit(
        1e-12,
        0.02,
        reference_peak_power,
    )

    final_qvel_error = np.abs(
        candidate.qvel_rad_s[-1] - reference.qvel_rad_s[-1]
    )
    reference_qvel_amplitude = np.max(
        np.abs(reference.qvel_rad_s),
        axis=0,
    )
    final_qvel_limits = np.maximum(
        1e-5,
        0.02 * reference_qvel_amplitude,
    )
    diagnostics = {
        "corner": corner,
        "tip": (tip_error, tip_limit),
        "joint": (joint_error, joint_limit),
        "composite": (composite_error, composite_limit),
        "tendon_length": (tendon_length_error, tendon_length_limit),
        "tendon_velocity": (
            tendon_velocity_error,
            tendon_velocity_limit,
        ),
        "steady_tip": (steady_tip_error, steady_tip_limit),
        "steady_bend": (steady_bend_error, steady_bend_limit),
        "peak_qvel": (peak_qvel_error, peak_qvel_limit),
        "peak_power": (peak_power_error, peak_power_limit),
        "final_qvel_max_ratio": float(
            np.max(final_qvel_error / final_qvel_limits)
        ),
    }

    assert tip_error <= tip_limit, diagnostics
    assert joint_error <= joint_limit, diagnostics
    assert composite_error <= composite_limit, diagnostics
    assert tendon_length_error <= tendon_length_limit, diagnostics
    assert tendon_velocity_error <= tendon_velocity_limit, diagnostics
    assert steady_tip_error <= steady_tip_limit, diagnostics
    assert steady_bend_error <= steady_bend_limit, diagnostics
    assert peak_qvel_error <= peak_qvel_limit, diagnostics
    assert peak_power_error <= peak_power_limit, diagnostics
    assert np.all(final_qvel_error <= final_qvel_limits), diagnostics
    np.testing.assert_allclose(
        candidate.actuator_force_n,
        reference.actuator_force_n,
        rtol=0.0,
        atol=1e-15,
    )
