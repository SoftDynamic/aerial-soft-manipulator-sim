"""C1-A4b common-mode modal-decay and disturbance-recovery tests."""

from types import MappingProxyType
import math

import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_recovery import (
    COMMON_TENSION_LEVELS_N,
    DIRECTIONS,
    ForceReleaseRecoveryRunConfig,
    MatchedStateDecayRunConfig,
    analyze_recovery_signal,
    classify_frequency_effect,
    run_force_release_recovery,
    run_matched_state_decay,
)


FD_SYNTHETIC_HZ = 3.49
ALPHA_SYNTHETIC_S = 2.59


def _synthetic_signal(
    sim_freq,
    *,
    duration_s=3.0,
    amplitude=0.004,
    offset=0.0,
    phase=0.0,
    perturbation=0.0,
    secondary_amplitude=0.0,
):
    time = np.arange(int(round(duration_s * sim_freq)) + 1) / sim_freq
    signal = offset + amplitude * np.exp(-ALPHA_SYNTHETIC_S * time) * np.cos(
        2 * math.pi * FD_SYNTHETIC_HZ * time + phase
    )
    if perturbation:
        signal += perturbation * (
            np.sin(2 * math.pi * 17.0 * time)
            + 0.5 * np.cos(2 * math.pi * 31.0 * time)
        )
    if secondary_amplitude:
        signal += (
            secondary_amplitude
            * amplitude
            * np.exp(-3.1 * time)
            * np.cos(2 * math.pi * 8.2 * time + 0.4)
        )
    velocity = np.gradient(signal, time)
    arm_qvel = 25.0 * np.abs(velocity)
    return time, signal, velocity, arm_qvel


@pytest.mark.parametrize("sim_freq", (2400, 4800))
@pytest.mark.parametrize("phase", (0.0, math.pi, 0.31, -0.44))
def test_synthetic_single_mode_frequency_bias_is_below_one_basis_point(
    sim_freq,
    phase,
):
    time, signal, velocity, arm_qvel = _synthetic_signal(
        sim_freq,
        phase=phase,
        offset=2e-8,
        perturbation=2e-9,
    )
    analysis = analyze_recovery_signal(
        time,
        signal,
        velocity,
        arm_qvel,
        equilibrium_noise=1e-10,
    )
    frequencies = (
        analysis.peak_frequency_hz,
        analysis.zero_crossing_frequency_hz,
        None if analysis.fit is None else analysis.fit.frequency_hz,
    )
    biases = [abs(value / FD_SYNTHETIC_HZ - 1.0) for value in frequencies]

    assert all(value is not None for value in frequencies), analysis
    assert max(biases) <= 1e-4, (sim_freq, phase, frequencies, biases)
    assert analysis.frequency_method_spread_relative <= 1e-4
    assert analysis.fit.final_frequency_resolution_hz < 1e-4
    assert analysis.fit.frequency_search_bounds_hz[0] >= 0.5
    assert analysis.fit.frequency_search_bounds_hz[1] <= 20.0
    assert analysis.fit.alpha_search_bounds_s[0] == 0.0
    assert analysis.fit.levels == 6
    assert analysis.fit.frequency_grid_size == 11
    assert analysis.fit.alpha_grid_size == 9
    assert analysis.fit.normalized_rms_residual <= 5e-5
    assert analysis.single_mode_adequate, analysis.unresolved_reason


def test_synthetic_peaks_zero_crossings_decrement_decay_and_settling():
    time, signal, velocity, arm_qvel = _synthetic_signal(
        2400,
        amplitude=0.004,
    )
    analysis = analyze_recovery_signal(time, signal, velocity, arm_qvel)
    expected_decrement = ALPHA_SYNTHETIC_S / FD_SYNTHETIC_HZ

    assert len(analysis.positive_peak_times_s) >= 7
    assert len(analysis.negative_peak_times_s) >= 6
    assert len(analysis.zero_crossing_times_s) >= 16
    assert analysis.peak_frequency_hz == pytest.approx(
        FD_SYNTHETIC_HZ, rel=1e-5
    )
    assert analysis.zero_crossing_frequency_hz == pytest.approx(
        FD_SYNTHETIC_HZ, rel=1e-5
    )
    assert analysis.median_log_decrement == pytest.approx(
        expected_decrement, rel=2e-4
    )
    assert analysis.decay_rate_s == pytest.approx(
        ALPHA_SYNTHETIC_S, rel=2e-4
    )
    assert analysis.envelope_log_rms_residual < 1e-4
    assert analysis.settling_time_s is not None
    assert analysis.settling_time_s <= time[-1] - 0.25


@pytest.mark.parametrize("duration_s", (2.4, 2.7, 3.0))
def test_synthetic_analysis_is_robust_to_low_amplitude_tail_truncation(
    duration_s,
):
    time, signal, velocity, arm_qvel = _synthetic_signal(
        2400,
        duration_s=duration_s,
        amplitude=0.001,
        offset=-3e-8,
        perturbation=3e-9,
    )
    analysis = analyze_recovery_signal(
        time,
        signal,
        velocity,
        arm_qvel,
        equilibrium_noise=2e-10,
    )

    assert analysis.peak_frequency_hz == pytest.approx(
        FD_SYNTHETIC_HZ, rel=1e-4
    )
    assert analysis.zero_crossing_frequency_hz == pytest.approx(
        FD_SYNTHETIC_HZ, rel=1e-4
    )
    assert analysis.fit.frequency_hz == pytest.approx(
        FD_SYNTHETIC_HZ, rel=1e-4
    )


def test_synthetic_weak_secondary_mode_is_detected_without_frequency_locking():
    time, signal, velocity, arm_qvel = _synthetic_signal(
        2400,
        secondary_amplitude=0.05,
    )
    analysis = analyze_recovery_signal(time, signal, velocity, arm_qvel)

    assert analysis.secondary_frequency_hz is not None
    assert analysis.secondary_energy_ratio > 0.0
    assert analysis.peak_frequency_hz == pytest.approx(
        FD_SYNTHETIC_HZ, rel=5e-4
    )
    assert (
        not analysis.single_mode_adequate
        or analysis.fit.normalized_rms_residual > 0.002
    )


def test_frequency_effect_never_hardcodes_a_tension_classification():
    baseline = 3.49
    unresolved = classify_frequency_effect(
        0.01,
        3.485,
        baseline,
        (1e-4,),
        eligible=False,
    )
    no_change = classify_frequency_effect(
        0.01,
        baseline * (1 - 4e-4),
        baseline,
        (1e-4,),
    )
    decrease = classify_frequency_effect(
        0.002,
        baseline * (1 - 2e-3),
        baseline,
        (1e-4,),
    )

    assert unresolved.classification == "unresolved"
    assert no_change.classification == "no resolved change"
    assert decrease.classification == "resolved decrease"


# MuJoCo fixtures and acceptance tests are defined below the synthetic tests
# so estimator validation can be selected and run before any expensive trace.


def _matched_key(tc, direction, bend=0.10):
    return float(tc), direction, float(bend)


def _force_key(tc, direction, force=2e-4):
    return float(tc), direction, float(force)


def _diagnostics(trace):
    analysis = trace.analysis
    return {
        "kind": trace.experiment_kind,
        "tc": trace.config.common_tension_n,
        "direction": trace.config.direction,
        "release": (trace.release_time_s, trace.release_physics_step),
        "x0": float(trace.recovery_signal_m[0]),
        "release_qvel": float(np.max(np.abs(trace.release_qvel))),
        "release_qacc": float(np.max(np.abs(trace.release_qacc))),
        "fd": (
            analysis.peak_frequency_hz,
            analysis.post_transient_peak_frequency_hz,
            analysis.zero_crossing_frequency_hz,
            None if analysis.fit is None else analysis.fit.frequency_hz,
        ),
        "fit_residual": (
            None if analysis.fit is None else analysis.fit.normalized_rms_residual
        ),
        "mode_shape": analysis.within_trace_mode_shape_residual,
        "high_frequency": analysis.high_frequency_energy_fraction,
        "secondary": (
            analysis.secondary_frequency_hz,
            analysis.secondary_energy_ratio,
        ),
        "settling": analysis.settling_time_s,
        "adequate": analysis.single_mode_adequate,
        "reason": analysis.unresolved_reason,
        "safety": trace.safety,
    }


@pytest.fixture(scope="module")
def matched_traces_2400():
    return MappingProxyType(
        {
            _matched_key(tc, direction): run_matched_state_decay(
                MatchedStateDecayRunConfig(tc, direction)
            )
            for tc in COMMON_TENSION_LEVELS_N
            for direction in DIRECTIONS
        }
    )


@pytest.fixture(scope="module")
def matched_amplitude_traces_2400():
    return MappingProxyType(
        {
            _matched_key(tc, "+X", bend): run_matched_state_decay(
                MatchedStateDecayRunConfig(tc, "+X", total_bend_rad=bend)
            )
            for tc in (0.0, 0.01)
            for bend in (0.025, 0.05, 0.20)
        }
    )


@pytest.fixture(scope="module")
def matched_convergence_traces_4800():
    return MappingProxyType(
        {
            tc: run_matched_state_decay(
                MatchedStateDecayRunConfig(tc, "+X", sim_freq=4800)
            )
            for tc in (0.0, 0.01)
        }
    )


@pytest.fixture(scope="module")
def matched_endpoint_repeat():
    return run_matched_state_decay(MatchedStateDecayRunConfig(0.01, "+X"))


@pytest.fixture(scope="module")
def force_traces_2400():
    keys = [
        _force_key(tc, direction)
        for tc in (0.0, 0.01)
        for direction in DIRECTIONS
    ] + [
        _force_key(tc, direction)
        for tc in (0.002, 0.005)
        for direction in ("+X", "+Y")
    ]
    return MappingProxyType(
        {
            key: run_force_release_recovery(
                ForceReleaseRecoveryRunConfig(key[0], key[1], force_magnitude_n=key[2])
            )
            for key in keys
        }
    )


@pytest.fixture(scope="module")
def force_amplitude_traces_2400():
    return MappingProxyType(
        {
            force: run_force_release_recovery(
                ForceReleaseRecoveryRunConfig(
                    0.01,
                    "+X",
                    force_magnitude_n=force,
                )
            )
            for force in (1e-4, 5e-4)
        }
    )


@pytest.fixture(scope="module")
def force_convergence_traces_4800():
    return MappingProxyType(
        {
            tc: run_force_release_recovery(
                ForceReleaseRecoveryRunConfig(tc, "+X", sim_freq=4800)
            )
            for tc in (0.0, 0.01)
        }
    )


def test_recovery_configs_reject_invalid_or_unaligned_values():
    invalid_matched = (
        {"common_tension_n": True, "direction": "+X"},
        {"common_tension_n": -1e-6, "direction": "+X"},
        {"common_tension_n": 0.0, "direction": "X"},
        {"common_tension_n": 0.0, "direction": "+X", "sim_freq": 2500},
        {"common_tension_n": 0.0, "direction": "+X", "total_bend_rad": 0.0},
        {"common_tension_n": 0.0, "direction": "+X", "release_time_s": 1.501},
    )
    for values in invalid_matched:
        with pytest.raises(ValueError):
            MatchedStateDecayRunConfig(**values)

    with pytest.raises(ValueError):
        ForceReleaseRecoveryRunConfig(0.0, "+X", force_magnitude_n=0.0)
    with pytest.raises(ValueError):
        ForceReleaseRecoveryRunConfig(0.0, "+X", force_magnitude_n=5.1e-4)
    with pytest.raises(ValueError):
        ForceReleaseRecoveryRunConfig(0.0, "+X", release_time_s=4.501)


@pytest.mark.parametrize("tc", COMMON_TENSION_LEVELS_N)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_matched_release_has_fresh_physics_rate_sample_and_exact_grid(
    matched_traces_2400,
    tc,
    direction,
):
    trace = matched_traces_2400[_matched_key(tc, direction)]
    config = trace.config
    diagnostics = _diagnostics(trace)

    assert trace.experiment_kind == "virtual_constraint_matched_state"
    assert trace.release_control_tick == 360
    assert trace.release_physics_step == 3600
    assert trace.release_time_s == 1.5
    assert len(trace.physics_time_s) == config.recovery_steps + 1 == 7201
    assert trace.physics_time_s[0] == 1.5
    assert trace.recovery_time_s[0] == 0.0
    assert trace.physics_time_s[-1] == 4.5
    assert trace.recovery_time_s[-1] == 3.0
    np.testing.assert_allclose(
        np.diff(trace.physics_time_s), 1 / config.sim_freq, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        np.diff(trace.recovery_time_s), 1 / config.sim_freq, rtol=0.0, atol=1e-15
    )
    np.testing.assert_array_equal(
        trace.control_tick,
        np.arange(
            config.release_tick
            + int(round(config.recovery_duration_s * config.ctrl_freq))
            + 1
        ),
    )
    np.testing.assert_allclose(
        trace.control_time_s,
        trace.control_tick / config.ctrl_freq,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.all(trace.control_phase[config.release_tick:] == "recovery")
    assert len(np.unique(trace.physics_time_s)) == len(trace.physics_time_s)
    assert np.max(np.abs(trace.release_qvel)) == 0.0, diagnostics
    assert np.max(np.abs(trace.release_qacc)) > 0.0, diagnostics
    assert trace.safety.maximum_quaternion_norm_error <= 2e-15, diagnostics
    np.testing.assert_array_equal(trace.tip_force_world_n, 0.0)
    np.testing.assert_array_equal(trace.qfrc_applied, 0.0)
    np.testing.assert_array_equal(trace.external_power_w, 0.0)
    np.testing.assert_allclose(
        trace.tension_command_n,
        tc,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        trace.actuator_force_n,
        trace.tension_command_n,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        trace.tip_world_m[0],
        trace.control_tip_world_m[config.release_tick],
        rtol=0.0,
        atol=0.0,
    )
    assert np.linalg.norm(
        trace.tip_world_m[0] - trace.equilibrium_tip_world_m
    ) >= 0.003, diagnostics
    assert np.ptp(trace.tendon_length_m[0]) >= 1e-4, diagnostics
    assert trace.safety.active.completed_steps == config.total_steps == 10800

    for values in (
        trace.control_tick,
        trace.control_time_s,
        trace.control_phase,
        trace.physics_time_s,
        trace.recovery_time_s,
        trace.tension_command_n,
        trace.tip_world_m,
        trace.qpos,
        trace.arm_qvel_rad_s,
        trace.arm_qacc_rad_s2,
        trace.joint_rotation_vectors_rad,
        trace.tendon_length_m,
        trace.actuator_power_w,
    ):
        assert not values.flags.writeable
        assert np.all(np.isfinite(values)) if values.dtype.kind not in "USO" else True


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_matched_qpos_and_joint_energy_state_are_identical_across_tc(
    matched_traces_2400,
    direction,
):
    traces = [
        matched_traces_2400[_matched_key(tc, direction)]
        for tc in COMMON_TENSION_LEVELS_N
    ]
    reference = traces[0]
    for trace in traces[1:]:
        np.testing.assert_array_equal(trace.release_qpos, reference.release_qpos)
        np.testing.assert_array_equal(trace.release_qvel, reference.release_qvel)
        np.testing.assert_allclose(
            trace.joint_rotation_vectors_rad[0],
            reference.joint_rotation_vectors_rad[0],
            rtol=0.0,
            atol=1e-15,
        )
    # Actuator power/work is intentionally not asserted equal across Tc.
    assert any(
        np.max(np.abs(trace.actuator_power_w)) > 0.0 for trace in traces[1:]
    )


@pytest.mark.parametrize("tc", COMMON_TENSION_LEVELS_N)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_matched_main_matrix_has_safe_adequate_single_mode_decay(
    matched_traces_2400,
    tc,
    direction,
):
    trace = matched_traces_2400[_matched_key(tc, direction)]
    analysis = trace.analysis
    safety = trace.safety
    diagnostics = _diagnostics(trace)

    assert analysis.single_mode_adequate, diagnostics
    assert analysis.peak_frequency_hz > 0.0, diagnostics
    assert analysis.zero_crossing_frequency_hz > 0.0, diagnostics
    assert analysis.fit is not None, diagnostics
    assert analysis.fit.normalized_rms_residual <= 0.02, diagnostics
    assert analysis.frequency_method_spread_relative <= 5e-4, diagnostics
    assert analysis.within_trace_mode_shape_residual <= 0.01, diagnostics
    assert analysis.decay_rate_s > 0.0, diagnostics
    assert 0.0 < analysis.damping_ratio < 1.0, diagnostics
    assert analysis.natural_frequency_hz > analysis.peak_frequency_hz
    assert analysis.settling_time_s is not None, diagnostics
    assert analysis.settling_time_s <= 2.75, diagnostics
    assert safety.active.warning_events == (), diagnostics
    assert safety.active.maximum_contacts == 0, diagnostics
    assert safety.active.minimum_tendon_range_margin_m >= -1e-9, diagnostics
    assert safety.active.maximum_moment_residual <= 1e-12, diagnostics
    assert safety.active.maximum_generalized_actuator_power_residual_w <= 1e-12
    assert safety.active.maximum_actuator_tendon_power_residual_w <= 1e-12
    assert safety.active.maximum_command_tendon_power_residual_w <= 1e-12
    assert safety.maximum_post_release_force_n == 0.0
    assert safety.maximum_post_release_qfrc_applied == 0.0
    assert np.max(np.abs(trace.cross_axis_signal_m)) <= max(
        1e-8, 0.001 * np.max(np.abs(trace.recovery_signal_m))
    )


def _normalized_mode_shape(trace):
    analysis = trace.analysis
    peak_time = analysis.positive_peak_times_s[1]
    index = int(np.searchsorted(trace.recovery_time_s, peak_time))
    index = min(max(index, 1), len(trace.recovery_time_s) - 1)
    axis = 1 if trace.config.direction.endswith("X") else 0
    values = trace.joint_rotation_vectors_rad[index, :, axis]
    return np.abs(values) / np.linalg.norm(values)


def test_matched_joint_mode_shape_is_consistent_across_tc_and_directions(
    matched_traces_2400,
):
    reference = _normalized_mode_shape(
        matched_traces_2400[_matched_key(0.0, "+X")]
    )
    residuals = {
        key: float(np.linalg.norm(_normalized_mode_shape(trace) - reference))
        for key, trace in matched_traces_2400.items()
    }
    assert max(residuals.values()) <= 0.01, residuals


@pytest.mark.parametrize("tc", COMMON_TENSION_LEVELS_N)
def test_matched_positive_negative_and_x_y_frequency_symmetry(
    matched_traces_2400,
    tc,
):
    frequencies = {
        direction: matched_traces_2400[
            _matched_key(tc, direction)
        ].analysis.peak_frequency_hz
        for direction in DIRECTIONS
    }
    spread = np.ptp(tuple(frequencies.values())) / max(frequencies.values())
    assert spread <= 5e-4, (tc, frequencies, spread)


def test_matched_amplitude_sweep_preserves_frequency_and_safety(
    matched_traces_2400,
    matched_amplitude_traces_2400,
):
    for tc in (0.0, 0.01):
        traces = [matched_traces_2400[_matched_key(tc, "+X")]] + [
            matched_amplitude_traces_2400[_matched_key(tc, "+X", bend)]
            for bend in (0.025, 0.05, 0.20)
        ]
        frequencies = np.asarray(
            [trace.analysis.peak_frequency_hz for trace in traces]
        )
        assert np.ptp(frequencies) / np.max(frequencies) <= 5e-4
        assert all(trace.analysis.single_mode_adequate for trace in traces)
        assert all(trace.safety.active.warning_events == () for trace in traces)


def test_matched_repeat_is_identical_over_complete_physics_trace(
    matched_traces_2400,
    matched_endpoint_repeat,
):
    reference = matched_traces_2400[_matched_key(0.01, "+X")]
    fields = (
        "control_tick",
        "control_time_s",
        "control_phase",
        "physics_time_s",
        "recovery_time_s",
        "tension_command_n",
        "actuator_force_n",
        "tip_world_m",
        "tip_velocity_world_m_s",
        "qpos",
        "arm_qvel_rad_s",
        "arm_qacc_rad_s2",
        "joint_rotation_vectors_rad",
        "composite_rotation_vectors_rad",
        "tendon_length_m",
        "tendon_velocity_m_s",
        "actuator_power_w",
    )
    for field in fields:
        np.testing.assert_array_equal(
            getattr(reference, field), getattr(matched_endpoint_repeat, field)
        )
    analysis_arrays = (
        "positive_peak_times_s",
        "positive_peak_amplitudes",
        "negative_peak_times_s",
        "negative_peak_amplitudes",
        "zero_crossing_times_s",
        "zero_crossing_directions",
        "positive_log_decrements",
        "negative_log_decrements",
    )
    for field in analysis_arrays:
        np.testing.assert_array_equal(
            getattr(reference.analysis, field),
            getattr(matched_endpoint_repeat.analysis, field),
        )
    reference_scalars = {
        field: value
        for field, value in vars(reference.analysis).items()
        if field not in analysis_arrays
    }
    repeat_scalars = {
        field: value
        for field, value in vars(matched_endpoint_repeat.analysis).items()
        if field not in analysis_arrays
    }
    assert reference_scalars == repeat_scalars
    assert reference.safety == matched_endpoint_repeat.safety


def _frequency_window_sensitivity(analysis):
    periods = []
    for peak_times in (
        analysis.positive_peak_times_s,
        analysis.negative_peak_times_s,
    ):
        periods.extend(np.diff(peak_times))
    full = 1.0 / np.median(periods)
    candidates = [full]
    if len(periods) >= 6:
        candidates.extend(
            (
                1.0 / np.median(periods[1:]),
                1.0 / np.median(periods[:-1]),
                1.0 / np.median(periods[: len(periods) // 2]),
                1.0 / np.median(periods[len(periods) // 2 :]),
            )
        )
    return float(np.ptp(candidates) / max(candidates))


def _synthetic_estimator_max_bias():
    biases = []
    for sim_freq in (2400, 4800):
        for phase in (0.0, math.pi, 0.31, -0.44):
            time, signal, velocity, arm_qvel = _synthetic_signal(
                sim_freq,
                phase=phase,
                offset=2e-8,
                perturbation=2e-9,
            )
            analysis = analyze_recovery_signal(
                time, signal, velocity, arm_qvel, equilibrium_noise=1e-10
            )
            biases.extend(
                abs(value / FD_SYNTHETIC_HZ - 1.0)
                for value in (
                    analysis.peak_frequency_hz,
                    analysis.zero_crossing_frequency_hz,
                    analysis.fit.frequency_hz,
                )
            )
    return max(biases)


@pytest.fixture(scope="module")
def formal_frequency_effects(
    matched_traces_2400,
    matched_convergence_traces_4800,
    matched_endpoint_repeat,
):
    mean_frequency = {
        tc: float(
            np.mean(
                [
                    matched_traces_2400[
                        _matched_key(tc, direction)
                    ].analysis.peak_frequency_hz
                    for direction in DIRECTIONS
                ]
            )
        )
        for tc in COMMON_TENSION_LEVELS_N
    }
    raw_convergence = max(
        abs(
            matched_traces_2400[_matched_key(tc, "+X")].analysis.peak_frequency_hz
            - matched_convergence_traces_4800[tc].analysis.peak_frequency_hz
        )
        / matched_convergence_traces_4800[tc].analysis.peak_frequency_hz
        for tc in (0.0, 0.01)
    )
    ratio_2400 = (
        matched_traces_2400[_matched_key(0.01, "+X")].analysis.peak_frequency_hz
        / matched_traces_2400[_matched_key(0.0, "+X")].analysis.peak_frequency_hz
    )
    ratio_4800 = (
        matched_convergence_traces_4800[0.01].analysis.peak_frequency_hz
        / matched_convergence_traces_4800[0.0].analysis.peak_frequency_hz
    )
    ratio_convergence = abs(ratio_2400 - ratio_4800) / abs(ratio_4800)
    direction_variation = max(
        np.ptp(
            [
                matched_traces_2400[
                    _matched_key(tc, direction)
                ].analysis.peak_frequency_hz
                for direction in DIRECTIONS
            ]
        )
        / mean_frequency[tc]
        for tc in COMMON_TENSION_LEVELS_N
    )
    peak_zero = max(
        abs(a.peak_frequency_hz - a.zero_crossing_frequency_hz)
        / max(a.peak_frequency_hz, a.zero_crossing_frequency_hz)
        for a in (trace.analysis for trace in matched_traces_2400.values())
    )
    peak_fit = max(
        abs(a.peak_frequency_hz - a.fit.frequency_hz)
        / max(a.peak_frequency_hz, a.fit.frequency_hz)
        for a in (trace.analysis for trace in matched_traces_2400.values())
    )
    window = max(
        _frequency_window_sensitivity(trace.analysis)
        for trace in matched_traces_2400.values()
    )
    reference = matched_traces_2400[_matched_key(0.01, "+X")]
    repeat = abs(
        reference.analysis.peak_frequency_hz
        - matched_endpoint_repeat.analysis.peak_frequency_hz
    ) / reference.analysis.peak_frequency_hz
    synthetic = _synthetic_estimator_max_bias()
    components = MappingProxyType(
        {
            "raw_fd_convergence": raw_convergence,
            "frequency_ratio_convergence": ratio_convergence,
            "direction_variation": direction_variation,
            "peak_vs_zero": peak_zero,
            "peak_vs_fit": peak_fit,
            "fit_window_sensitivity": window,
            "repeat_variation": repeat,
            "synthetic_estimator_bias": synthetic,
        }
    )
    all_eligible = all(
        trace.analysis.single_mode_adequate
        and trace.safety.active.warning_events == ()
        for trace in matched_traces_2400.values()
    )
    effects = MappingProxyType(
        {
            tc: classify_frequency_effect(
                tc,
                mean_frequency[tc],
                mean_frequency[0.0],
                components.values(),
                eligible=all_eligible,
            )
            for tc in COMMON_TENSION_LEVELS_N
        }
    )
    return components, effects, ratio_2400, ratio_4800


def test_formal_frequency_floor_uses_all_required_uncertainty_components(
    formal_frequency_effects,
):
    components, effects, ratio_2400, ratio_4800 = formal_frequency_effects
    assert set(components) == {
        "raw_fd_convergence",
        "frequency_ratio_convergence",
        "direction_variation",
        "peak_vs_zero",
        "peak_vs_fit",
        "fit_window_sensitivity",
        "repeat_variation",
        "synthetic_estimator_bias",
    }
    expected_uncertainty = max(components.values())
    expected_floor = max(5e-4, 3 * expected_uncertainty)
    assert abs(ratio_2400 - ratio_4800) / abs(ratio_4800) == pytest.approx(
        components["frequency_ratio_convergence"]
    )
    for effect in effects.values():
        assert effect.uncertainty_relative == pytest.approx(expected_uncertainty)
        assert effect.effect_floor_relative == pytest.approx(expected_floor)
        assert effect.classification in {
            "resolved increase",
            "no resolved change",
            "resolved decrease",
            "unresolved",
        }
    assert effects[0.0].frequency_ratio == pytest.approx(1.0)


@pytest.mark.parametrize("tc", (0.0, 0.01))
def test_matched_2400_4800_raw_and_modal_convergence(
    matched_traces_2400,
    matched_convergence_traces_4800,
    tc,
):
    candidate = matched_traces_2400[_matched_key(tc, "+X")]
    reference = matched_convergence_traces_4800[tc]
    ca, ra = candidate.analysis, reference.analysis
    diagnostics = (_diagnostics(candidate), _diagnostics(reference))

    raw_fd_error = abs(ca.peak_frequency_hz - ra.peak_frequency_hz) / ra.peak_frequency_hz
    fn_error = abs(ca.natural_frequency_hz - ra.natural_frequency_hz) / ra.natural_frequency_hz
    zeta_error = abs(ca.damping_ratio - ra.damping_ratio) / ra.damping_ratio
    alpha_error = abs(ca.decay_rate_s - ra.decay_rate_s) / ra.decay_rate_s
    assert raw_fd_error <= 4e-4, diagnostics
    assert fn_error <= 5e-4, diagnostics
    assert zeta_error <= 5e-4, diagnostics
    assert alpha_error <= 6e-4, diagnostics
    assert abs(ca.settling_time_s - ra.settling_time_s) <= 1 / 240
    assert np.linalg.norm(
        _normalized_mode_shape(candidate) - _normalized_mode_shape(reference)
    ) <= 0.01


@pytest.mark.parametrize("key", [
    _force_key(tc, direction)
    for tc in (0.0, 0.01)
    for direction in DIRECTIONS
] + [
    _force_key(tc, direction)
    for tc in (0.002, 0.005)
    for direction in ("+X", "+Y")
])
def test_force_release_is_safe_settled_scenario_characterization(
    force_traces_2400,
    key,
):
    trace = force_traces_2400[key]
    analysis = trace.analysis
    diagnostics = _diagnostics(trace)

    assert trace.experiment_kind == "instantaneous_tip_force_release"
    assert trace.release_control_tick == 1080
    assert trace.release_physics_step == 10800
    assert trace.release_time_s == 4.5
    assert trace.physics_time_s[0] == 4.5
    assert trace.recovery_time_s[0] == 0.0
    assert np.max(np.abs(trace.release_qvel)) <= 1e-4, diagnostics
    assert np.max(np.abs(trace.release_qacc)) > 0.0, diagnostics
    np.testing.assert_array_equal(trace.tip_force_world_n, 0.0)
    np.testing.assert_array_equal(trace.qfrc_applied, 0.0)
    np.testing.assert_array_equal(trace.external_power_w, 0.0)
    assert analysis.first_complete_peak_time_s is not None, diagnostics
    assert analysis.first_complete_peak_time_s > 0.0, diagnostics
    assert analysis.settling_time_s is not None, diagnostics
    assert analysis.settling_time_s <= 2.75, diagnostics
    assert np.isfinite(analysis.high_frequency_energy_fraction)
    assert 0.0 <= analysis.high_frequency_energy_fraction <= 1.0
    assert analysis.secondary_frequency_hz is not None
    assert analysis.secondary_energy_ratio >= 0.0
    assert trace.safety.active.warning_events == (), diagnostics
    assert trace.safety.active.maximum_contacts == 0, diagnostics
    assert trace.safety.maximum_post_release_force_n == 0.0
    assert trace.safety.maximum_post_release_qfrc_applied == 0.0
    assert trace.safety.maximum_external_generalized_force_residual <= 1e-12
    assert trace.safety.maximum_external_power_residual_w <= 1e-12
    # Modal values are characterization only; adequacy controls whether they
    # may be interpreted, never the A4b1 common-mode classification.
    if analysis.single_mode_adequate:
        assert analysis.peak_frequency_hz > 0.0
        assert analysis.fit.normalized_rms_residual <= 0.02


def test_force_amplitude_sweep_reports_release_transient_without_mode_locking(
    force_traces_2400,
    force_amplitude_traces_2400,
):
    traces = [force_traces_2400[_force_key(0.01, "+X")]] + list(
        force_amplitude_traces_2400.values()
    )
    frequencies = [
        trace.analysis.peak_frequency_hz
        for trace in traces
        if trace.analysis.single_mode_adequate
    ]
    assert len(frequencies) >= 2
    assert np.ptp(frequencies) / max(frequencies) <= 5e-4
    for trace in traces:
        analysis = trace.analysis
        assert np.max(np.abs(trace.release_qacc)) > 0.0
        assert analysis.first_complete_peak_time_s > 0.0
        assert analysis.decay_rate_s > 0.0
        assert analysis.release_inclusive_decay_rate_s > 0.0
        assert analysis.post_transient_peak_frequency_hz > 0.0
        assert np.isfinite(
            analysis.release_inclusive_decay_rate_s - analysis.decay_rate_s
        )
        assert analysis.envelope_log_rms_residual <= 0.01


@pytest.mark.parametrize("tc", (0.0, 0.01))
def test_force_release_2400_4800_converges_without_becoming_modal_gate(
    force_traces_2400,
    force_convergence_traces_4800,
    tc,
):
    candidate = force_traces_2400[_force_key(tc, "+X")]
    reference = force_convergence_traces_4800[tc]
    diagnostics = (_diagnostics(candidate), _diagnostics(reference))
    frequency_error = abs(
        candidate.analysis.peak_frequency_hz
        - reference.analysis.peak_frequency_hz
    ) / reference.analysis.peak_frequency_hz

    assert frequency_error <= 4e-4, diagnostics
    assert abs(
        candidate.analysis.settling_time_s
        - reference.analysis.settling_time_s
    ) <= 1 / 240
    assert abs(
        np.max(np.abs(candidate.release_qacc))
        - np.max(np.abs(reference.release_qacc))
    ) / np.max(np.abs(reference.release_qacc)) <= 5e-4
    assert candidate.analysis.high_frequency_energy_fraction == pytest.approx(
        reference.analysis.high_frequency_energy_fraction,
        rel=0.02,
    )


def test_all_formal_characterization_values_are_reportable(
    matched_traces_2400,
    force_traces_2400,
    formal_frequency_effects,
):
    components, effects, ratio_2400, ratio_4800 = formal_frequency_effects
    rows = []
    for trace in tuple(matched_traces_2400.values()) + tuple(force_traces_2400.values()):
        analysis = trace.analysis
        rows.append(
            (
                trace.config.common_tension_n,
                analysis.peak_frequency_hz,
                analysis.zero_crossing_frequency_hz,
                analysis.fit.frequency_hz,
                analysis.fit.normalized_rms_residual,
                analysis.decay_rate_s,
                analysis.damping_ratio,
                analysis.settling_time_s,
                analysis.high_frequency_energy_fraction,
                analysis.secondary_energy_ratio,
            )
        )
    assert np.all(np.isfinite(np.asarray(rows)))
    assert np.all(np.isfinite(np.asarray(tuple(components.values()))))
    assert np.isfinite(ratio_2400) and np.isfinite(ratio_4800)
    assert all(effect.frequency_ratio is not None for effect in effects.values())
