"""C1-A4a fixed-root common-mode static tip-compliance acceptance tests."""

from types import MappingProxyType

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_compliance import (
    COMMON_TENSION_LEVELS_N,
    DIRECTIONS,
    FORCE_MAGNITUDES_N,
    MAX_TIP_FORCE_N,
    CommonModeComplianceRunConfig,
    classify_common_mode_effect,
    fit_axis_compliance,
    run_common_mode_compliance,
    signed_axis_ols,
)


def _key(tc, direction, magnitude):
    return float(tc), direction, float(magnitude)


def _diagnostics(trace):
    return {
        "tc": trace.config.common_tension_n,
        "direction": trace.config.direction,
        "force": trace.config.force_magnitude_n,
        "delta_tip": trace.delta_tip_world_m,
        "drift": trace.loaded_window_drift_m,
        "noise": trace.loaded_window_noise_m,
        "steady_qvel": trace.loaded_window_max_arm_qvel_rad_s,
        "window_sensitivity": trace.loaded_window_sensitivity,
        "preload_shift": trace.preload_tip_shift_m,
        "tendon_excursion": trace.incremental_tendon_excursion_m,
        "safety": trace.safety,
    }


@pytest.fixture(scope="module")
def compliance_traces_2400():
    return MappingProxyType(
        {
            _key(tc, direction, magnitude): run_common_mode_compliance(
                CommonModeComplianceRunConfig(tc, direction, magnitude)
            )
            for tc in COMMON_TENSION_LEVELS_N
            for direction in DIRECTIONS
            for magnitude in FORCE_MAGNITUDES_N
        }
    )


@pytest.fixture(scope="module")
def compliance_fits(compliance_traces_2400):
    traces = tuple(compliance_traces_2400.values())
    return MappingProxyType(
        {
            (tc, axis): fit_axis_compliance(traces, tc, axis)
            for tc in COMMON_TENSION_LEVELS_N
            for axis in ("X", "Y")
        }
    )


@pytest.fixture(scope="module")
def repeated_endpoint_trace():
    return run_common_mode_compliance(
        CommonModeComplianceRunConfig(0.01, "+X", MAX_TIP_FORCE_N)
    )


@pytest.fixture(scope="module")
def convergence_traces_4800():
    return MappingProxyType(
        {
            tc: run_common_mode_compliance(
                CommonModeComplianceRunConfig(
                    tc, "+X", MAX_TIP_FORCE_N, sim_freq=4800
                )
            )
            for tc in (0.0, 0.01)
        }
    )


@pytest.fixture(scope="module")
def convergence_relative_error(
    compliance_traces_2400,
    convergence_traces_4800,
):
    errors = []
    for tc, reference in convergence_traces_4800.items():
        candidate = compliance_traces_2400[_key(tc, "+X", MAX_TIP_FORCE_N)]
        candidate_compliance = candidate.one_sided_compliance_m_n
        reference_compliance = reference.one_sided_compliance_m_n
        errors.extend(
            (
                abs(candidate_compliance - reference_compliance)
                / abs(reference_compliance),
                abs(1 / candidate_compliance - 1 / reference_compliance)
                / abs(1 / reference_compliance),
            )
        )
    return max(errors)


@pytest.fixture(scope="module")
def common_mode_effects(compliance_fits, convergence_relative_error):
    baseline_x = compliance_fits[(0.0, "X")]
    baseline_y = compliance_fits[(0.0, "Y")]
    return MappingProxyType(
        {
            tc: classify_common_mode_effect(
                tc,
                compliance_fits[(tc, "X")],
                compliance_fits[(tc, "Y")],
                baseline_x,
                baseline_y,
                convergence_relative_error,
            )
            for tc in COMMON_TENSION_LEVELS_N
        }
    )


@pytest.mark.parametrize(
    "args",
    [
        (True, "+X", 1e-4),
        (np.nan, "+X", 1e-4),
        (np.inf, "+X", 1e-4),
        (-1e-6, "+X", 1e-4),
        (0.010001, "+X", 1e-4),
        (0.0, "X", 1e-4),
        (0.0, "+Z", 1e-4),
        (0.0, "+X", True),
        (0.0, "+X", np.nan),
        (0.0, "+X", 0.0),
        (0.0, "+X", MAX_TIP_FORCE_N + 1e-8),
    ],
)
def test_compliance_config_rejects_invalid_loads(args):
    with pytest.raises(ValueError):
        CommonModeComplianceRunConfig(*args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sim_freq": True},
        {"ctrl_freq": 0},
        {"sim_freq": 2500},
        {"integrator": "RK4"},
        {"tension_ramp_duration_s": 0.251},
        {"preload_window_start_s": 0.2},
        {"force_start_s": 1.501},
        {"force_ramp_duration_s": 2.6},
        {"loaded_window_start_s": 1.7},
        {"total_duration_s": 4.0},
    ],
)
def test_compliance_config_rejects_invalid_grid(overrides):
    with pytest.raises(ValueError):
        CommonModeComplianceRunConfig(0.0, "+X", 1e-4, **overrides)


def test_compliance_config_accepts_zero_common_mode_and_exact_grid():
    config = CommonModeComplianceRunConfig(0.0, "+X", 1e-4)

    assert config.sample_stride == 10
    assert config.tension_ramp_ticks == 60
    assert config.force_start_tick == 360
    assert config.force_ramp_ticks == 60
    assert config.total_ticks == 1080
    assert config.requested_steps == 10800
    np.testing.assert_array_equal(config.direction_vector, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(config.final_force_world_n, [1e-4, 0.0, 0.0])


def test_signed_axis_ols_keeps_negative_force_and_displacement():
    force = np.asarray((-2e-4, -1e-4, -5e-5, 5e-5, 1e-4, 2e-4))
    expected_compliance = 3.25
    expected_intercept = 2e-8
    displacement = expected_compliance * force + expected_intercept

    slope, intercept, r_squared, residual = signed_axis_ols(
        force, displacement
    )

    assert slope == pytest.approx(expected_compliance, rel=1e-14)
    assert intercept == pytest.approx(expected_intercept, abs=1e-18)
    assert r_squared == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(residual, 0.0, atol=1e-18)
    folded_slope = np.linalg.lstsq(
        np.column_stack((force, np.ones(len(force)))),
        np.abs(displacement - expected_intercept),
        rcond=None,
    )[0][0]
    assert abs(folded_slope) <= 1e-12


def test_formal_matrix_contains_exactly_48_independent_trajectories(
    compliance_traces_2400,
):
    assert len(compliance_traces_2400) == 48
    assert set(compliance_traces_2400) == {
        _key(tc, direction, magnitude)
        for tc in COMMON_TENSION_LEVELS_N
        for direction in DIRECTIONS
        for magnitude in FORCE_MAGNITUDES_N
    }
    assert len({id(item) for item in compliance_traces_2400.values()}) == 48


@pytest.mark.parametrize("tc", COMMON_TENSION_LEVELS_N)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("magnitude", FORCE_MAGNITUDES_N)
def test_trace_has_complete_read_only_phase_and_control_grid(
    compliance_traces_2400,
    tc,
    direction,
    magnitude,
):
    trace = compliance_traces_2400[_key(tc, direction, magnitude)]
    config = trace.config
    ticks = np.arange(config.total_ticks + 1)
    diagnostics = _diagnostics(trace)

    np.testing.assert_array_equal(trace.control_tick, ticks)
    np.testing.assert_array_equal(
        trace.physics_step, ticks * config.sample_stride
    )
    np.testing.assert_allclose(
        trace.time_s, ticks / config.ctrl_freq, rtol=0.0, atol=1e-12
    )
    assert trace.safety.actuation.completed_steps == 10800, diagnostics
    assert trace.safety.actuation.requested_steps == 10800, diagnostics
    assert np.count_nonzero(trace.preload_mask) == 121
    assert np.count_nonzero(trace.loaded_mask) == 121
    assert not trace.preload_mask.flags.writeable
    assert not trace.loaded_mask.flags.writeable
    assert not trace.total_bend_rad.flags.writeable
    assert trace.phase[0] == "tension_ramp"
    assert trace.phase[60] == "preload_settle"
    assert trace.phase[240] == "preload_measure"
    assert trace.phase[360] == "force_ramp"
    assert trace.phase[420] == "loaded_settle"
    assert trace.phase[960] == "loaded_measure"
    for values in (
        trace.phase,
        trace.control_tick,
        trace.physics_step,
        trace.time_s,
        trace.tension_command_n,
        trace.actuator_force_n,
        trace.tip_force_world_n,
        trace.application_point_world_m,
        trace.qfrc_applied,
        trace.root_world_m,
        trace.tip_world_m,
        trace.tip_velocity_world_m_s,
        trace.qpos,
        trace.qvel_rad_s,
        trace.qacc_rad_s2,
        trace.arm_qvel_rad_s,
        trace.joint_rotation_vectors_rad,
        trace.composite_rotation_vectors_rad,
        trace.tendon_length_m,
        trace.tendon_velocity_m_s,
        trace.qfrc_actuator_nm,
        trace.generalized_actuator_power_w,
        trace.actuator_space_power_w,
        trace.tendon_coordinate_power_w,
        trace.command_tendon_power_w,
        trace.external_generalized_power_w,
        trace.external_tip_power_w,
        trace.warning_counts,
        trace.contacts,
        trace.tendon_range_margin_m,
    ):
        assert not values.flags.writeable, diagnostics


def test_tension_and_tip_force_ramps_have_exact_endpoints_without_overshoot(
    compliance_traces_2400,
):
    trace = compliance_traces_2400[_key(0.01, "+X", MAX_TIP_FORCE_N)]
    target_command = np.full(4, 0.01)

    np.testing.assert_array_equal(trace.tension_command_n[0], 0.0)
    np.testing.assert_array_equal(trace.tension_command_n[60], target_command)
    assert np.all(trace.tension_command_n[60:] == target_command)
    assert np.all(np.diff(trace.tension_command_n[:61], axis=0) >= 0.0)
    np.testing.assert_array_equal(trace.tip_force_world_n[360], 0.0)
    np.testing.assert_array_equal(
        trace.tip_force_world_n[420], [MAX_TIP_FORCE_N, 0.0, 0.0]
    )
    assert np.all(
        trace.tip_force_world_n[420:] == [MAX_TIP_FORCE_N, 0.0, 0.0]
    )
    assert np.all(np.diff(trace.tip_force_world_n[360:421, 0]) >= 0.0)


def test_manifest_resolves_true_tip_site_and_final_segment_body(
    compliance_traces_2400,
):
    trace = compliance_traces_2400[_key(0.0, "+X", 5e-5)]

    assert trace.root_site_name.endswith("_root_site")
    assert trace.tip_site_name.endswith("_tip_site")
    assert trace.tip_body_name.endswith("_segment_08")
    assert trace.root_site_id >= 0
    assert trace.tip_site_id >= 0
    assert trace.tip_body_id >= 0
    np.testing.assert_allclose(
        trace.application_point_world_m,
        trace.tip_world_m,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("tc", COMMON_TENSION_LEVELS_N)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("magnitude", FORCE_MAGNITUDES_N)
def test_command_force_wrench_power_and_safety_hold_at_physics_rate(
    compliance_traces_2400,
    tc,
    direction,
    magnitude,
):
    trace = compliance_traces_2400[_key(tc, direction, magnitude)]
    safety = trace.safety
    actuation = safety.actuation
    diagnostics = _diagnostics(trace)

    np.testing.assert_allclose(
        trace.actuator_force_n,
        trace.tension_command_n,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        trace.generalized_actuator_power_w,
        trace.actuator_space_power_w,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace.actuator_space_power_w,
        trace.tendon_coordinate_power_w,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace.tendon_coordinate_power_w,
        trace.command_tendon_power_w,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace.external_generalized_power_w,
        trace.external_tip_power_w,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace.tip_force_world_n[-1],
        magnitude * trace.config.direction_vector,
        rtol=0.0,
        atol=1e-15,
    )
    assert actuation.warning_events == (), diagnostics
    assert actuation.first_failure is None, diagnostics
    assert actuation.maximum_contacts == 0, diagnostics
    assert actuation.minimum_tendon_range_margin_m >= -1e-9, diagnostics
    assert actuation.maximum_force_command_error_n <= 1e-15, diagnostics
    assert actuation.maximum_moment_residual <= 1e-12, diagnostics
    assert actuation.maximum_generalized_actuator_power_residual_w <= 1e-12
    assert actuation.maximum_actuator_tendon_power_residual_w <= 1e-12
    assert actuation.maximum_command_tendon_power_residual_w <= 1e-12
    assert safety.maximum_external_generalized_force_residual <= 1e-12
    assert safety.maximum_external_power_residual_w <= 1e-12
    assert safety.maximum_abs_applied_force_n <= magnitude + 1e-15
    assert safety.final_qfrc_applied_zero
    assert safety.final_xfrc_applied_zero
    assert safety.first_failure is None
    assert np.all(trace.warning_counts == 0), diagnostics
    assert np.all(trace.contacts == 0), diagnostics


@pytest.mark.parametrize("tc", COMMON_TENSION_LEVELS_N)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("magnitude", FORCE_MAGNITUDES_N)
def test_per_trajectory_equilibria_steady_state_and_local_response(
    compliance_traces_2400,
    tc,
    direction,
    magnitude,
):
    trace = compliance_traces_2400[_key(tc, direction, magnitude)]
    diagnostics = _diagnostics(trace)
    displacement = float(np.linalg.norm(trace.delta_tip_world_m))
    maximum_bend = float(np.max(trace.total_bend_rad))

    np.testing.assert_allclose(
        trace.preload_tip_world_m,
        np.mean(trace.tip_world_m[trace.preload_mask], axis=0),
    )
    np.testing.assert_allclose(
        trace.loaded_tip_world_m,
        np.mean(trace.tip_world_m[trace.loaded_mask], axis=0),
    )
    assert trace.delta_along_force_m > 0.0, diagnostics
    assert trace.one_sided_compliance_m_n > 0.0, diagnostics
    assert trace.loaded_window_drift_m <= max(
        1e-6, 0.005 * displacement
    ), diagnostics
    assert trace.loaded_window_max_arm_qvel_rad_s <= 1e-4, diagnostics
    assert trace.delta_along_force_m >= 50 * trace.loaded_window_noise_m
    assert trace.loaded_window_sensitivity is not None, diagnostics
    assert displacement <= 0.01 * 0.08, diagnostics
    assert maximum_bend <= 0.03, diagnostics
    assert trace.safety.actuation.maximum_joint_bend_rad <= 0.01, diagnostics
    assert trace.safety.actuation.maximum_abs_composite_twist_rad <= max(
        1e-6, 0.01 * maximum_bend
    ), diagnostics
    assert trace.safety.actuation.maximum_abs_joint_twist_rad <= max(
        1e-6,
        0.01 * trace.safety.actuation.maximum_joint_bend_rad,
    ), diagnostics
    assert np.all(np.isfinite(trace.preload_tip_shift_m)), diagnostics
    assert np.all(np.isfinite(trace.preload_tendon_excursion_m)), diagnostics
    assert np.all(np.isfinite(trace.incremental_tendon_excursion_m)), diagnostics


@pytest.mark.parametrize("tc", COMMON_TENSION_LEVELS_N)
@pytest.mark.parametrize("axis", ("X", "Y"))
def test_fixed_axis_signed_fit_central_difference_and_linearity(
    compliance_fits,
    tc,
    axis,
):
    fit = compliance_fits[(tc, axis)]
    diagnostics = fit

    np.testing.assert_array_equal(
        fit.force_axis_n,
        [-2e-4, -1e-4, -5e-5, 5e-5, 1e-4, 2e-4],
    )
    assert np.all(fit.delta_axis_m[:3] < 0), diagnostics
    assert np.all(fit.delta_axis_m[3:] > 0), diagnostics
    assert fit.valid, diagnostics
    assert fit.slope_compliance_m_n > 0, diagnostics
    assert fit.stiffness_n_m is not None and fit.stiffness_n_m > 0
    assert abs(fit.intercept_m) <= 1e-7, diagnostics
    assert fit.maximum_fit_residual_m <= max(
        5e-8, 2e-4 * np.max(np.abs(fit.delta_axis_m))
    ), diagnostics
    assert fit.r_squared >= 0.999999, diagnostics
    assert fit.central_relative_spread <= 0.0005, diagnostics
    assert fit.one_sided_relative_asymmetry <= 0.001, diagnostics
    assert fit.maximum_cross_axis_residual_m <= max(
        1e-9, 0.001 * np.max(np.abs(fit.delta_axis_m))
    ), diagnostics
    assert fit.maximum_window_sensitivity < 0.001, diagnostics
    assert fit.minimum_signal_to_noise >= 50, diagnostics


@pytest.mark.parametrize("tc", COMMON_TENSION_LEVELS_N)
def test_x_y_fitted_compliance_is_symmetric(compliance_fits, tc):
    x_fit = compliance_fits[(tc, "X")]
    y_fit = compliance_fits[(tc, "Y")]
    relative = abs(
        x_fit.slope_compliance_m_n - y_fit.slope_compliance_m_n
    ) / max(x_fit.slope_compliance_m_n, y_fit.slope_compliance_m_n)

    assert relative <= 0.001, (tc, x_fit, y_fit)


def test_effect_floor_uses_only_dimensionless_errors_and_classifies_endpoint(
    common_mode_effects,
):
    baseline = common_mode_effects[0.0]
    endpoint = common_mode_effects[0.01]

    assert baseline.stiffness_ratio == pytest.approx(1.0, abs=1e-15)
    assert baseline.classification == "no resolved change"
    for effect in common_mode_effects.values():
        assert np.isfinite(effect.uncertainty_relative)
        assert effect.effect_floor_relative == pytest.approx(
            max(0.001, 3 * effect.uncertainty_relative)
        )
        assert effect.classification in {
            "effective hardening",
            "no resolved change",
            "effective softening",
            "possible instability",
        }
    assert endpoint.stiffness_ratio < 1 - endpoint.effect_floor_relative
    assert endpoint.classification == "effective softening"


def test_endpoint_repeat_is_deterministic_over_complete_key_trace(
    compliance_traces_2400,
    repeated_endpoint_trace,
):
    reference = compliance_traces_2400[_key(0.01, "+X", MAX_TIP_FORCE_N)]
    candidate = repeated_endpoint_trace
    fields = (
        "phase",
        "control_tick",
        "physics_step",
        "time_s",
        "tension_command_n",
        "actuator_force_n",
        "tip_force_world_n",
        "application_point_world_m",
        "qfrc_applied",
        "root_world_m",
        "tip_world_m",
        "tip_velocity_world_m_s",
        "qpos",
        "qvel_rad_s",
        "qacc_rad_s2",
        "joint_rotation_vectors_rad",
        "composite_rotation_vectors_rad",
        "tendon_length_m",
        "tendon_velocity_m_s",
        "generalized_actuator_power_w",
        "external_generalized_power_w",
    )
    errors = {}
    for field in fields:
        left = getattr(reference, field)
        right = getattr(candidate, field)
        if left.dtype.kind in "USO":
            np.testing.assert_array_equal(left, right)
        else:
            errors[field] = float(np.max(np.abs(left - right)))
    assert all(value == 0.0 for value in errors.values()), errors
    assert reference.safety == candidate.safety


@pytest.mark.parametrize("tc", (0.0, 0.01))
def test_representative_2400_hz_converges_to_4800_hz(
    compliance_traces_2400,
    convergence_traces_4800,
    tc,
):
    candidate = compliance_traces_2400[_key(tc, "+X", MAX_TIP_FORCE_N)]
    reference = convergence_traces_4800[tc]
    np.testing.assert_array_equal(candidate.control_tick, reference.control_tick)
    np.testing.assert_array_equal(candidate.phase, reference.phase)
    np.testing.assert_allclose(candidate.time_s, reference.time_s, atol=1e-12)
    np.testing.assert_allclose(
        candidate.tension_command_n,
        reference.tension_command_n,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        candidate.actuator_force_n,
        reference.actuator_force_n,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        candidate.tip_force_world_n,
        reference.tip_force_world_n,
        rtol=0.0,
        atol=1e-15,
    )

    tip_error = float(
        np.max(np.linalg.norm(candidate.tip_world_m - reference.tip_world_m, axis=1))
    )
    tip_scale = float(
        np.max(np.linalg.norm(reference.tip_world_m - reference.tip_world_m[0], axis=1))
    )
    joint_error = float(
        np.max(
            np.linalg.norm(
                candidate.joint_rotation_vectors_rad
                - reference.joint_rotation_vectors_rad,
                axis=2,
            )
        )
    )
    joint_scale = float(
        np.max(np.linalg.norm(reference.joint_rotation_vectors_rad, axis=2))
    )
    candidate_excursion = candidate.tendon_length_m - candidate.tendon_length_m[0]
    reference_excursion = reference.tendon_length_m - reference.tendon_length_m[0]
    tendon_error = float(np.max(np.abs(candidate_excursion - reference_excursion)))
    compliance_error = abs(
        candidate.one_sided_compliance_m_n - reference.one_sided_compliance_m_n
    ) / abs(reference.one_sided_compliance_m_n)
    stiffness_error = abs(
        1 / candidate.one_sided_compliance_m_n
        - 1 / reference.one_sided_compliance_m_n
    ) / abs(1 / reference.one_sided_compliance_m_n)
    candidate_peak_qvel = candidate.safety.actuation.maximum_abs_qvel_rad_s
    reference_peak_qvel = reference.safety.actuation.maximum_abs_qvel_rad_s
    qvel_error = abs(candidate_peak_qvel - reference_peak_qvel) / reference_peak_qvel
    actuator_scale = max(
        candidate.safety.maximum_abs_actuator_power_w,
        reference.safety.maximum_abs_actuator_power_w,
        1e-12,
    )
    actuator_error = abs(
        candidate.safety.maximum_abs_actuator_power_w
        - reference.safety.maximum_abs_actuator_power_w
    ) / actuator_scale
    external_error = abs(
        candidate.safety.maximum_abs_external_power_w
        - reference.safety.maximum_abs_external_power_w
    ) / reference.safety.maximum_abs_external_power_w
    diagnostics = {
        "tc": tc,
        "tip_error": tip_error,
        "joint_error": joint_error,
        "tendon_error": tendon_error,
        "compliance_error": compliance_error,
        "stiffness_error": stiffness_error,
        "qvel_error": qvel_error,
        "actuator_error": actuator_error,
        "external_error": external_error,
    }

    assert tip_error <= max(1e-6, 0.005 * tip_scale), diagnostics
    assert np.linalg.norm(
        candidate.preload_tip_world_m - reference.preload_tip_world_m
    ) <= 1e-9, diagnostics
    assert np.linalg.norm(
        candidate.loaded_tip_world_m - reference.loaded_tip_world_m
    ) <= 1e-6, diagnostics
    assert joint_error <= max(5e-6, 0.005 * joint_scale), diagnostics
    assert tendon_error <= max(
        5e-8, 0.005 * float(np.max(np.abs(reference_excursion)))
    ), diagnostics
    assert compliance_error <= 0.0005, diagnostics
    assert stiffness_error <= 0.0005, diagnostics
    assert qvel_error <= 0.005, diagnostics
    assert actuator_error <= 0.005, diagnostics
    assert external_error <= 0.005, diagnostics


def test_formal_characterization_values_are_finite_and_reportable(
    compliance_fits,
    common_mode_effects,
):
    rows = []
    for tc in COMMON_TENSION_LEVELS_N:
        x_fit = compliance_fits[(tc, "X")]
        y_fit = compliance_fits[(tc, "Y")]
        effect = common_mode_effects[tc]
        rows.append(
            (
                tc,
                x_fit.slope_compliance_m_n,
                y_fit.slope_compliance_m_n,
                x_fit.stiffness_n_m,
                y_fit.stiffness_n_m,
                effect.stiffness_ratio,
                effect.effect_floor_relative,
            )
        )
    assert np.all(np.isfinite(np.asarray(rows, dtype=float)))
