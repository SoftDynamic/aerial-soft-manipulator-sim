"""Gate C2-B1a pure fixed-tension profile and metric contracts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_probe import (
    AlignedProbeTrace,
    FixedTensionProfileConfig,
    ProbeCondition,
    baseline_correct_probe_traces,
    build_fixed_tension_command_profile,
    compare_aligned_responses,
    compute_probe_metrics,
    compute_zero_run_activity_floor,
    integrate_interval_power_w,
    machine_resolution_floor_m,
    measure_s_probe,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    half_cosine_ramp_scale_at_tick,
)


CTRL_FREQ_HZ = 240
DURATION_S = 3.5
RAMP_DURATION_S = 0.25
MAX_TENSION_N = 0.01
ACTIVE_INDEX = 2


def _profile_config(**overrides):
    values = {
        "ctrl_freq_hz": CTRL_FREQ_HZ,
        "duration_s": DURATION_S,
        "ramp_duration_s": RAMP_DURATION_S,
        "target_tension_n": 0.002,
        "max_tension_n": MAX_TENSION_N,
        "num_tendons": 4,
        "active_channel_index": ACTIVE_INDEX,
    }
    values.update(overrides)
    return FixedTensionProfileConfig(**values)


def _trace(
    *,
    time=None,
    contraction=None,
    contraction_velocity=None,
    tip_displacement=None,
    initial_length=None,
    initial_tip=None,
):
    time = np.asarray([0.0, 1.0, 2.0, 3.0] if time is None else time)
    sample_count = len(time)
    if contraction is None:
        contraction = np.zeros((sample_count, 4))
    if contraction_velocity is None:
        contraction_velocity = np.zeros((sample_count, 4))
    if tip_displacement is None:
        tip_displacement = np.zeros((sample_count, 3))
    if initial_length is None:
        initial_length = np.ones(4)
    if initial_tip is None:
        initial_tip = np.zeros(3)
    contraction = np.asarray(contraction, dtype=float)
    contraction_velocity = np.asarray(contraction_velocity, dtype=float)
    tip_displacement = np.asarray(tip_displacement, dtype=float)
    return AlignedProbeTrace(
        time_s=time,
        tendon_length_m=np.asarray(initial_length) - contraction,
        tendon_velocity_m_s=-contraction_velocity,
        tip_position_m=np.asarray(initial_tip) + tip_displacement,
    )


def test_b1a_source_remains_pure_and_reuses_existing_half_cosine():
    helper = Path(__file__).with_name(
        "_scaled_stablesoft_acceleration_probe.py"
    ).read_text(encoding="utf-8")

    assert "half_cosine_ramp_scale_at_tick" in helper
    assert "import mujoco" not in helper
    assert "mujoco.mj_" not in helper
    assert "MjModel" not in helper
    assert "MjData" not in helper
    assert "mj_step" not in helper
    assert "argparse" not in helper
    assert "json" not in helper


def test_profile_config_normalizes_values_and_is_frozen():
    config = _profile_config(
        ctrl_freq_hz=np.int64(240),
        duration_s=np.float32(3.5),
        ramp_duration_s=np.float64(0.25),
        target_tension_n=np.float32(0.002),
        max_tension_n=np.float64(0.01),
        num_tendons=np.int32(4),
        active_channel_index=np.int64(2),
    )

    assert type(config.ctrl_freq_hz) is int
    assert type(config.duration_s) is float
    assert type(config.ramp_duration_s) is float
    assert type(config.target_tension_n) is float
    assert type(config.max_tension_n) is float
    assert type(config.num_tendons) is int
    assert type(config.active_channel_index) is int
    assert config.number_of_control_intervals == 840
    assert config.ramp_ticks == 60
    with pytest.raises(FrozenInstanceError):
        config.target_tension_n = 0.005


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"ctrl_freq_hz": True}, "must be an integer"),
        ({"ctrl_freq_hz": 0}, "must be positive"),
        ({"ctrl_freq_hz": -240}, "must be positive"),
        ({"ctrl_freq_hz": 240.0}, "must be an integer"),
        ({"ctrl_freq_hz": "240"}, "must be an integer"),
        ({"duration_s": True}, "must be a real number"),
        ({"duration_s": 0.0}, "must be positive"),
        ({"duration_s": -1.0}, "must be positive"),
        ({"duration_s": float("nan")}, "must be finite"),
        ({"duration_s": float("inf")}, "must be finite"),
        ({"duration_s": 1 + 2j}, "must be a real number"),
        ({"duration_s": "3.5"}, "must be a real number"),
        ({"ramp_duration_s": True}, "must be a real number"),
        ({"ramp_duration_s": 0.0}, "must be positive"),
        ({"ramp_duration_s": 3.6}, "must not exceed"),
        ({"duration_s": 3.5001}, "control tick grid"),
        ({"ramp_duration_s": 0.2501}, "control tick grid"),
        ({"target_tension_n": True}, "must be a real number"),
        ({"target_tension_n": -1e-6}, "must lie"),
        ({"target_tension_n": 0.011}, "must lie"),
        ({"target_tension_n": float("nan")}, "must be finite"),
        ({"max_tension_n": True}, "must be a real number"),
        ({"max_tension_n": 0.0}, "must be positive"),
        ({"max_tension_n": float("inf")}, "must be finite"),
        ({"num_tendons": True}, "must be an integer"),
        ({"num_tendons": 3}, "exactly 4"),
        ({"num_tendons": 4.0}, "must be an integer"),
        ({"active_channel_index": True}, "must be an integer"),
        ({"active_channel_index": -1}, "must be in"),
        ({"active_channel_index": 4}, "must be in"),
        ({"active_channel_index": 2.0}, "must be an integer"),
    ],
)
def test_profile_config_rejects_invalid_or_ambiguous_values(overrides, match):
    with pytest.raises(ValueError, match=match):
        _profile_config(**overrides)


def test_profile_allows_equal_ramp_and_duration_on_an_integer_grid():
    config = _profile_config(duration_s=0.25, ramp_duration_s=0.25)

    assert config.number_of_control_intervals == config.ramp_ticks == 60


def test_profile_tick_zero_fifty_nine_and_sixty_match_existing_semantics():
    config = _profile_config()
    command = build_fixed_tension_command_profile(config)
    expected_59 = 0.002 * half_cosine_ramp_scale_at_tick(59, 60)

    assert command.shape == (840, 4)
    assert command[0, ACTIVE_INDEX] == 0.0
    assert command[59, ACTIVE_INDEX] == pytest.approx(expected_59)
    assert command[59, ACTIVE_INDEX] < config.target_tension_n
    assert command[60, ACTIVE_INDEX] == config.target_tension_n
    assert command[-1, ACTIVE_INDEX] == config.target_tension_n


def test_profile_is_monotone_bounded_and_independent_of_sim_frequency():
    first = build_fixed_tension_command_profile(_profile_config())
    second = build_fixed_tension_command_profile(_profile_config())
    active = first[:, ACTIVE_INDEX]

    assert np.all(np.diff(active) >= 0.0)
    assert np.all(active >= 0.0)
    assert np.all(active <= 0.002)
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, second)


def test_zero_target_profile_is_exactly_zero_and_read_only():
    command = build_fixed_tension_command_profile(
        _profile_config(target_tension_n=0.0)
    )

    np.testing.assert_array_equal(command, 0.0)
    assert not command.flags.writeable
    with pytest.raises(ValueError):
        command[0, 0] = 1.0


@pytest.mark.parametrize("active_index", range(4))
def test_profile_writes_only_the_selected_active_channel(active_index):
    command = build_fixed_tension_command_profile(
        _profile_config(active_channel_index=active_index)
    )
    inactive = [index for index in range(4) if index != active_index]

    np.testing.assert_array_equal(command[:, inactive], 0.0)
    assert command[60, active_index] == 0.002


def test_profile_builder_rejects_the_wrong_config_type():
    with pytest.raises(TypeError, match="FixedTensionProfileConfig"):
        build_fixed_tension_command_profile(object())


def test_aligned_trace_defensively_copies_inputs_and_is_read_only():
    time = np.array([0.0, 1.0])
    length = np.ones((2, 4))
    velocity = np.zeros((2, 4))
    tip = np.zeros((2, 3))
    trace = AlignedProbeTrace(time, length, velocity, tip)
    length[:] = 7.0

    np.testing.assert_array_equal(trace.tendon_length_m, 1.0)
    for values in (
        trace.time_s,
        trace.tendon_length_m,
        trace.tendon_velocity_m_s,
        trace.tip_position_m,
        trace.contraction_m,
        trace.contraction_velocity_m_s,
        trace.tip_displacement_m,
    ):
        assert not values.flags.writeable
    with pytest.raises(ValueError):
        trace.tendon_length_m[0, 0] = 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"time_s": [[0.0, 1.0]]}, "one-dimensional"),
        ({"time_s": [0.0]}, "at least two"),
        ({"time_s": [0.0, 0.0]}, "strictly increasing"),
        ({"time_s": [1.0, 0.0]}, "strictly increasing"),
        ({"time_s": [0.0, float("nan")]}, "only finite"),
        ({"tendon_length_m": np.ones((2, 3))}, "shape"),
        ({"tendon_velocity_m_s": np.ones((2, 5))}, "shape"),
        ({"tip_position_m": np.ones((2, 2))}, "shape"),
        ({"tendon_length_m": [["x"] * 4] * 2}, "real numeric"),
        ({"tendon_velocity_m_s": np.full((2, 4), np.inf)}, "finite"),
        ({"tip_position_m": np.full((2, 3), np.nan)}, "finite"),
    ],
)
def test_aligned_trace_rejects_invalid_time_shape_or_values(kwargs, match):
    values = {
        "time_s": [0.0, 1.0],
        "tendon_length_m": np.ones((2, 4)),
        "tendon_velocity_m_s": np.zeros((2, 4)),
        "tip_position_m": np.zeros((2, 3)),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        AlignedProbeTrace(**values)


def test_identical_traces_have_exact_zero_baseline_correction():
    trace = _trace()
    response = baseline_correct_probe_traces(trace, trace)

    np.testing.assert_array_equal(response.delta_contraction_m, 0.0)
    np.testing.assert_array_equal(
        response.delta_contraction_velocity_m_s, 0.0
    )
    np.testing.assert_array_equal(response.delta_tip_displacement_m, 0.0)


def test_baseline_correction_preserves_sign_drift_and_separate_initial_lengths():
    zero_contraction = np.zeros((4, 4))
    zero_contraction[:, 2] = [0.0, 0.01, 0.02, 0.03]
    tension_contraction = np.zeros((4, 4))
    tension_contraction[:, 2] = [0.0, 0.03, -0.01, 0.08]
    zero_velocity = np.zeros((4, 4))
    zero_velocity[:, 2] = [0.0, 0.01, 0.02, 0.03]
    tension_velocity = np.zeros((4, 4))
    tension_velocity[:, 2] = [0.0, 0.04, -0.02, 0.06]
    zero_tip = np.column_stack(([0.0, 0.1, 0.2, 0.3], [0.0] * 4, [0.0] * 4))
    tension_tip = np.column_stack(([0.0, 0.15, 0.1, 0.5], [0.0] * 4, [0.0] * 4))
    zero = _trace(
        contraction=zero_contraction,
        contraction_velocity=zero_velocity,
        tip_displacement=zero_tip,
        initial_length=np.ones(4),
    )
    tension = _trace(
        contraction=tension_contraction,
        contraction_velocity=tension_velocity,
        tip_displacement=tension_tip,
        initial_length=np.full(4, 2.0),
        initial_tip=np.ones(3),
    )
    response = baseline_correct_probe_traces(zero, tension)

    np.testing.assert_allclose(
        response.delta_contraction_m[:, 2], [0.0, 0.02, -0.03, 0.05]
    )
    np.testing.assert_allclose(
        response.delta_contraction_velocity_m_s[:, 2],
        [0.0, 0.03, -0.04, 0.03],
    )
    np.testing.assert_allclose(
        response.delta_tip_displacement_m[:, 0], [0.0, 0.05, -0.1, 0.2]
    )


def test_baseline_correction_rejects_time_or_shape_mismatch():
    with pytest.raises(ValueError, match="time grids"):
        baseline_correct_probe_traces(
            _trace(time=[0.0, 1.0, 2.0, 3.0]),
            _trace(time=[0.0, 1.0, 2.0, 3.1]),
        )


def test_probe_metrics_have_exact_peak_tail_tip_and_cross_coupling_values():
    contraction = np.zeros((4, 4))
    contraction[:, 2] = [0.0, 0.1, 0.2, 0.3]
    contraction[:, 0] = [0.0, 0.01, 0.02, 0.03]
    contraction[:, 1] = [0.0, -0.01, -0.02, -0.01]
    contraction[:, 3] = [0.0, 0.005, 0.01, 0.005]
    velocity = np.zeros((4, 4))
    velocity[:, 2] = [0.0, 0.1, -0.2, 0.05]
    tension_tip = np.column_stack(
        ([0.0, 0.1, 0.2, 0.3], [0.0] * 4, [0.0] * 4)
    )
    zero_tip = np.column_stack(
        ([0.0, 0.02, 0.04, 0.06], [0.0] * 4, [0.0] * 4)
    )
    response = baseline_correct_probe_traces(
        _trace(tip_displacement=zero_tip),
        _trace(
            contraction=contraction,
            contraction_velocity=velocity,
            tip_displacement=tension_tip,
        ),
    )
    metrics = compute_probe_metrics(
        response,
        active_channel_index=2,
        tail_start_s=2.0,
        tail_end_s=3.0,
        activity_resolution_m=1e-12,
    )

    assert metrics.peak_contraction_m == pytest.approx(0.3)
    assert metrics.peak_absolute_contraction_m == pytest.approx(0.3)
    assert metrics.tail_mean_contraction_m == pytest.approx(0.25)
    assert metrics.tail_std_contraction_m == pytest.approx(0.05)
    assert metrics.peak_contraction_velocity_m_s == pytest.approx(0.1)
    assert metrics.minimum_contraction_velocity_m_s == pytest.approx(-0.2)
    assert metrics.maximum_contraction_velocity_m_s == pytest.approx(0.1)
    assert metrics.baseline_corrected_peak_contraction_m == pytest.approx(0.3)
    assert metrics.baseline_corrected_tail_contraction_m == pytest.approx(0.25)
    assert metrics.tip_peak_displacement_m == pytest.approx(0.3)
    assert metrics.baseline_corrected_tip_peak_displacement_m == pytest.approx(0.24)
    np.testing.assert_allclose(
        metrics.inactive_peak_contraction_m, [0.03, 0.02, 0.01]
    )
    assert metrics.cross_coupling_ratio == pytest.approx(0.1)


@pytest.mark.parametrize(
    "active_values,resolution",
    [
        ([0.0, 0.0, 0.0, 0.0], 0.0),
        ([0.0, -0.1, -0.2, -0.3], 0.0),
        ([0.0, 1e-9, 2e-9, 3e-9], 3.1e-9),
    ],
)
def test_cross_coupling_ratio_is_undefined_without_positive_resolved_activity(
    active_values, resolution
):
    contraction = np.zeros((4, 4))
    contraction[:, 2] = active_values
    contraction[:, 0] = [0.0, 0.01, 0.02, 0.03]
    response = baseline_correct_probe_traces(
        _trace(), _trace(contraction=contraction)
    )
    metrics = compute_probe_metrics(
        response,
        active_channel_index=2,
        tail_start_s=2.0,
        tail_end_s=3.0,
        activity_resolution_m=resolution,
    )

    assert metrics.cross_coupling_ratio is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tail_start_s": -1.0}, "within"),
        ({"tail_start_s": 3.0, "tail_end_s": 2.0}, "must not exceed"),
        ({"tail_end_s": 4.0}, "within"),
        ({"tail_start_s": 2.5, "tail_end_s": 2.5}, "at least"),
        ({"minimum_tail_samples": 0}, "must be positive"),
        ({"active_channel_index": 4}, "must be in"),
        ({"activity_resolution_m": -1.0}, "non-negative"),
        ({"activity_resolution_m": float("nan")}, "finite"),
    ],
)
def test_probe_metrics_reject_invalid_tail_channel_or_resolution(kwargs, match):
    values = {
        "active_channel_index": 2,
        "tail_start_s": 2.0,
        "tail_end_s": 3.0,
        "activity_resolution_m": 0.0,
    }
    values.update(kwargs)
    response = baseline_correct_probe_traces(_trace(), _trace())
    with pytest.raises(ValueError, match=match):
        compute_probe_metrics(response, **values)


def test_s_probe_is_signed_tail_mean_with_explicit_reference_metadata():
    contraction = np.zeros((4, 4))
    contraction[:, 2] = [0.0, -0.1, -0.2, -0.4]
    response = baseline_correct_probe_traces(
        _trace(), _trace(contraction=contraction)
    )
    condition = ProbeCondition(0.002, 2400, 240)
    measurement = measure_s_probe(
        response,
        condition,
        active_channel_index=2,
        tail_start_s=2.0,
        tail_end_s=3.0,
    )

    assert measurement.condition.target_tension_n == 0.002
    assert measurement.condition.sim_freq_hz == 2400
    assert measurement.condition.ctrl_freq_hz == 240
    assert measurement.value_m == pytest.approx(-0.3)


def test_zero_run_activity_floor_includes_passive_motion():
    contraction = np.zeros((4, 4))
    contraction[:, 1] = [0.0, 0.01, -0.02, 0.03]
    velocity = np.zeros((4, 4))
    velocity[:, 3] = [0.0, -0.4, 0.2, 0.1]
    floor = compute_zero_run_activity_floor(
        _trace(
            contraction=contraction,
            contraction_velocity=velocity,
        )
    )

    assert floor.n_s_m == pytest.approx(0.03)
    assert floor.n_v_m_s == pytest.approx(0.4)
    assert floor.rms_contraction_m == pytest.approx(
        np.sqrt((0.01**2 + 0.02**2 + 0.03**2) / 16)
    )
    assert floor.rms_contraction_velocity_m_s == pytest.approx(
        np.sqrt((0.4**2 + 0.2**2 + 0.1**2) / 16)
    )


def test_machine_resolution_floor_uses_length_scale_and_explicit_fallback():
    expected = 64 * np.finfo(float).eps * 2.0

    assert machine_resolution_floor_m([1.0, -2.0, 0.5, 0.0]) == expected
    assert machine_resolution_floor_m(
        [0.0, 0.0, 0.0, 0.0], fallback_length_scale_m=2.0
    ) == expected
    with pytest.raises(ValueError, match="fallback"):
        machine_resolution_floor_m([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="positive"):
        machine_resolution_floor_m(
            [0.0, 0.0, 0.0, 0.0], fallback_length_scale_m=0.0
        )


def test_aligned_response_difference_computes_per_channel_and_global_maxima():
    contraction_a = np.zeros((4, 4))
    contraction_b = np.zeros((4, 4))
    contraction_a[-1] = [0.1, 0.2, 0.3, 0.4]
    contraction_b[-1] = [0.11, 0.18, 0.35, 0.4]
    velocity_a = np.zeros((4, 4))
    velocity_b = np.zeros((4, 4))
    velocity_b[1] = [1.0, 2.0, 3.0, 4.0]
    first = baseline_correct_probe_traces(
        _trace(),
        _trace(
            contraction=contraction_a,
            contraction_velocity=velocity_a,
        ),
    )
    second = baseline_correct_probe_traces(
        _trace(),
        _trace(
            contraction=contraction_b,
            contraction_velocity=velocity_b,
        ),
    )
    difference = compare_aligned_responses(first, second)

    np.testing.assert_allclose(
        difference.contraction_difference_m, [0.01, 0.02, 0.05, 0.0]
    )
    np.testing.assert_allclose(
        difference.velocity_difference_m_s, [1.0, 2.0, 3.0, 4.0]
    )
    assert difference.d_s_m == pytest.approx(0.05)
    assert difference.d_v_m_s == pytest.approx(4.0)


@pytest.mark.parametrize(
    "power,dt,expected",
    [
        (np.zeros((2, 11)), 0.1, 0.0),
        (np.full((1, 11), 2.0), 0.1, 2.0),
        (np.linspace(0.0, 2.0, 11)[None, :], 0.1, 1.0),
        (np.full((1, 11), -2.0), 0.1, -2.0),
        (np.vstack((np.zeros(2), np.full(2, 10.0))), 0.1, 1.0),
    ],
)
def test_interval_power_integral_matches_analytic_values(power, dt, expected):
    assert integrate_interval_power_w(power, dt_sim_s=dt) == pytest.approx(
        expected
    )


def test_interval_power_does_not_bridge_a_command_boundary():
    power = np.asarray([[0.0, 0.0], [10.0, 10.0]])

    result = integrate_interval_power_w(power, dt_sim_s=0.1)

    assert result == pytest.approx(1.0)
    assert result != pytest.approx(0.5)


def test_ten_and_twenty_substeps_integrate_the_same_linear_power():
    ten_substeps = np.linspace(-1.0, 3.0, 11)[None, :]
    twenty_substeps = np.linspace(-1.0, 3.0, 21)[None, :]

    work_10 = integrate_interval_power_w(ten_substeps, dt_sim_s=0.1)
    work_20 = integrate_interval_power_w(twenty_substeps, dt_sim_s=0.05)

    assert work_10 == pytest.approx(1.0)
    assert work_20 == pytest.approx(1.0)
    assert work_10 == pytest.approx(work_20)


@pytest.mark.parametrize(
    "power,dt,match",
    [
        ([0.0, 1.0], 0.1, "two-dimensional"),
        (np.empty((0, 2)), 0.1, "at least one interval"),
        (np.zeros((2, 1)), 0.1, "start and end"),
        (np.asarray([[0.0, np.nan]]), 0.1, "finite"),
        (np.asarray([[0.0, np.inf]]), 0.1, "finite"),
        (np.zeros((1, 2)), 0.0, "positive"),
        (np.zeros((1, 2)), -0.1, "positive"),
        (np.zeros((1, 2)), float("nan"), "finite"),
    ],
)
def test_interval_power_rejects_invalid_input(power, dt, match):
    with pytest.raises(ValueError, match=match):
        integrate_interval_power_w(power, dt_sim_s=dt)


def test_results_are_deterministic_read_only_and_do_not_share_memory():
    zero = _trace()
    tension = _trace()
    first = baseline_correct_probe_traces(zero, tension)
    second = baseline_correct_probe_traces(zero, tension)

    for left, right in (
        (first.time_s, second.time_s),
        (first.delta_contraction_m, second.delta_contraction_m),
        (
            first.delta_contraction_velocity_m_s,
            second.delta_contraction_velocity_m_s,
        ),
        (first.delta_tip_displacement_m, second.delta_tip_displacement_m),
    ):
        np.testing.assert_array_equal(left, right)
        assert not left.flags.writeable
        assert not right.flags.writeable
        assert not np.shares_memory(left, right)
