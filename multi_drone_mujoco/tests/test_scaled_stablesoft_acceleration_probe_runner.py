"""Gate C2-B1b fixed-root direct-tension runner contracts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_fixed_root import (
    FixedRootZeroControlConfig,
    run_fixed_root_zero_control,
)
from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_probe import (
    AlignedProbeTrace,
    build_fixed_tension_command_profile,
    integrate_interval_power_w,
)
from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_probe_runner import (
    CTRL_FREQ_HZ,
    SIM_FREQUENCIES_HZ,
    FixedRootProbeRunConfig,
    resolve_probe_channel,
    run_fixed_root_tension_probe,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    CORNERS,
    TendonActuatorBinding,
)


RUN_DURATION_S = 0.025
RAMP_DURATION_S = 0.0125
TARGET_TENSION_N = 0.002
MAX_TENSION_N = 0.01
ACTIVE_CORNER = "xn_yp"


def _config(sim_freq_hz=2400, target_tension_n=TARGET_TENSION_N, **overrides):
    values = {
        "sim_freq_hz": sim_freq_hz,
        "ctrl_freq_hz": CTRL_FREQ_HZ,
        "duration_s": RUN_DURATION_S,
        "ramp_duration_s": RAMP_DURATION_S,
        "target_tension_n": target_tension_n,
        "max_tension_n": MAX_TENSION_N,
        "active_tendon_corner": ACTIVE_CORNER,
    }
    values.update(overrides)
    return FixedRootProbeRunConfig(**values)


@pytest.fixture(scope="module", params=SIM_FREQUENCIES_HZ)
def probe_traces(request):
    sim_freq = request.param
    return {
        "zero": run_fixed_root_tension_probe(_config(sim_freq, 0.0)),
        "active": run_fixed_root_tension_probe(
            _config(sim_freq, TARGET_TENSION_N)
        ),
    }


def test_runner_source_excludes_controller_cli_output_viewer_and_formal_matrix():
    helper = Path(__file__).with_name(
        "_scaled_stablesoft_acceleration_probe_runner.py"
    ).read_text(encoding="utf-8")

    assert "tendon_acceleration_controller" not in helper
    assert "TendonContractionAccelerationController" not in helper
    assert "argparse" not in helper
    assert "json" not in helper
    assert "viewer" not in helper.lower()
    assert "3.5" not in helper


def test_runner_config_is_frozen_normalized_and_has_exact_short_grid():
    config = _config(
        sim_freq_hz=np.int64(2400),
        ctrl_freq_hz=np.int32(240),
        duration_s=np.float64(RUN_DURATION_S),
        ramp_duration_s=np.float64(RAMP_DURATION_S),
        target_tension_n=np.float32(TARGET_TENSION_N),
        max_tension_n=np.float64(MAX_TENSION_N),
        active_tendon_corner=" xn_yp ",
    )

    assert type(config.sim_freq_hz) is int
    assert type(config.ctrl_freq_hz) is int
    assert type(config.duration_s) is float
    assert type(config.ramp_duration_s) is float
    assert type(config.target_tension_n) is float
    assert type(config.max_tension_n) is float
    assert config.active_tendon_corner == ACTIVE_CORNER
    assert config.number_of_control_intervals == 6
    assert config.ramp_ticks == 3
    assert config.substeps_per_control_interval == 10
    assert config.number_of_physics_steps == 60
    with pytest.raises(FrozenInstanceError):
        config.sim_freq_hz = 4800


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"sim_freq_hz": True}, "must be an integer"),
        ({"sim_freq_hz": 2400.0}, "must be an integer"),
        ({"sim_freq_hz": "2400"}, "must be an integer"),
        ({"sim_freq_hz": 0}, "must be one of"),
        ({"sim_freq_hz": 1200}, "must be one of"),
        ({"ctrl_freq_hz": True}, "must be an integer"),
        ({"ctrl_freq_hz": 240.0}, "must be an integer"),
        ({"ctrl_freq_hz": 120}, "exactly 240"),
        ({"duration_s": True}, "must be a real number"),
        ({"duration_s": 0.0}, "must be positive"),
        ({"duration_s": float("nan")}, "must be finite"),
        ({"duration_s": float("inf")}, "must be finite"),
        ({"duration_s": 1 + 2j}, "must be a real number"),
        ({"duration_s": "0.025"}, "must be a real number"),
        ({"duration_s": 0.0251}, "control tick grid"),
        ({"ramp_duration_s": True}, "must be a real number"),
        ({"ramp_duration_s": 0.0}, "must be positive"),
        ({"ramp_duration_s": 0.0251}, "must not exceed"),
        ({"ramp_duration_s": 0.0126}, "control tick grid"),
        ({"ramp_duration_s": float("nan")}, "must be finite"),
        ({"ramp_duration_s": float("inf")}, "must be finite"),
        ({"ramp_duration_s": 1 + 2j}, "must be a real number"),
        ({"ramp_duration_s": "0.0125"}, "must be a real number"),
        ({"target_tension_n": True}, "must be a real number"),
        ({"target_tension_n": -1e-6}, "must lie"),
        ({"target_tension_n": float("nan")}, "must be finite"),
        ({"target_tension_n": float("inf")}, "must be finite"),
        ({"target_tension_n": 1 + 2j}, "must be a real number"),
        ({"target_tension_n": 0.011}, "must lie"),
        ({"max_tension_n": True}, "must be a real number"),
        ({"max_tension_n": 0.0}, "must be positive"),
        ({"max_tension_n": 0.011}, "must not exceed"),
        ({"max_tension_n": float("nan")}, "must be finite"),
        ({"max_tension_n": float("inf")}, "must be finite"),
        ({"max_tension_n": 1 + 2j}, "must be a real number"),
        ({"max_tension_n": "0.01"}, "must be a real number"),
        ({"active_tendon_corner": True}, "must be a string"),
        ({"active_tendon_corner": 2}, "must be a string"),
        ({"active_tendon_corner": "  "}, "must not be empty"),
    ],
)
def test_runner_config_rejects_invalid_values(overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(**overrides)


def test_runner_requires_a_valid_config_before_building_a_fixture():
    with pytest.raises(TypeError, match="FixedRootProbeRunConfig"):
        run_fixed_root_tension_probe(object())


def test_runner_rejects_a_missing_corner_after_manifest_resolution():
    with pytest.raises(ValueError, match="exactly one manifest tendon"):
        run_fixed_root_tension_probe(
            _config(active_tendon_corner="missing_corner")
        )


@pytest.mark.parametrize("corner", CORNERS)
def test_all_formal_corners_resolve_by_manifest_name(probe_traces, corner):
    trace = probe_traces["active"]
    resolved = resolve_probe_channel(
        corner,
        trace.tendon_names,
        trace.actuator_names,
        trace.bindings,
    )

    assert resolved.corner == corner
    assert resolved.tendon_name.endswith(f"_{corner}_tendon")
    assert resolved.actuator_name.endswith(
        f"_{corner}_ideal_tension_actuator"
    )
    assert trace.bindings[resolved.binding_index].tendon_name == (
        resolved.tendon_name
    )


def test_probe_channel_rejects_missing_and_ambiguous_manifest_names():
    binding = TendonActuatorBinding("arm_x_tendon", "arm_x_act", 7, 8, 2, 3)
    with pytest.raises(ValueError, match="exactly one manifest tendon"):
        resolve_probe_channel(
            "missing", (binding.tendon_name,), (binding.actuator_name,), (binding,)
        )

    names = ("left_xn_yp_tendon", "right_xn_yp_tendon")
    actuators = ("left_act", "right_act")
    bindings = (
        TendonActuatorBinding(names[0], actuators[0], 7, 8, 2, 3),
        TendonActuatorBinding(names[1], actuators[1], 9, 10, 4, 5),
    )
    with pytest.raises(ValueError, match="exactly one manifest tendon"):
        resolve_probe_channel("xn_yp", names, actuators, bindings)


def test_probe_channel_keeps_binding_index_ids_and_addresses_distinct():
    names = ("arm_xp_yp_tendon", "arm_xn_yp_tendon")
    actuators = ("arm_xp_yp_act", "arm_xn_yp_act")
    bindings = (
        TendonActuatorBinding(names[0], actuators[0], 31, 41, 51, 61),
        TendonActuatorBinding(names[1], actuators[1], 32, 42, 52, 62),
    )
    resolved = resolve_probe_channel("xn_yp", names, actuators, bindings)

    assert resolved.binding_index == 1
    assert resolved.tendon_id == 32
    assert resolved.actuator_id == 42
    assert resolved.control_address == 52
    assert resolved.output_address == 62


def test_probe_channel_rejects_manifest_binding_actuator_mismatch():
    binding = TendonActuatorBinding(
        "arm_xn_yp_tendon", "wrong_actuator", 1, 2, 3, 4
    )
    with pytest.raises(ValueError, match="actuator names"):
        resolve_probe_channel(
            "xn_yp",
            (binding.tendon_name,),
            ("expected_actuator",),
            (binding,),
        )


def test_fixed_root_topology_and_resolved_channel_contract(probe_traces):
    trace = probe_traces["active"]
    resolved = trace.resolved_channel

    assert trace.model_counts == (11, 8, 32, 24, 10, 75, 4, 4, 0, 4, 4, 4)
    assert trace.profile_config.active_channel_index == resolved.binding_index
    assert resolved.corner == ACTIVE_CORNER
    assert resolved.tendon_name.endswith("_xn_yp_tendon")
    assert trace.bindings[resolved.binding_index].actuator_name == (
        resolved.actuator_name
    )
    np.testing.assert_array_equal(trace.tendon_limited, True)
    np.testing.assert_allclose(
        trace.actuator_gear,
        np.tile([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], (4, 1)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        trace.actuator_ctrl_range_n,
        np.tile([0.0, MAX_TENSION_N], (4, 1)),
        rtol=0.0,
        atol=1e-15,
    )


def test_state_control_and_physics_shapes_follow_three_grid_contract(
    probe_traces,
):
    for trace in probe_traces.values():
        intervals = 6
        substeps = {2400: 10, 4800: 20}[trace.config.sim_freq_hz]
        state_shape = (intervals + 1,)
        physics_shape = (intervals, substeps + 1)

        assert trace.number_of_control_intervals == intervals
        assert trace.number_of_physics_steps == intervals * substeps
        assert trace.time_s.shape == state_shape
        assert trace.qpos.shape == (intervals + 1, 32)
        assert trace.qvel.shape == (intervals + 1, 24)
        assert trace.tendon_length_m.shape == (intervals + 1, 4)
        assert trace.tendon_velocity_m_s.shape == (intervals + 1, 4)
        assert trace.tip_position_m.shape == (intervals + 1, 3)
        assert trace.joint_rotation_vectors_rad.shape == (intervals + 1, 8, 3)
        assert trace.composite_rotation_vectors_rad.shape == (intervals + 1, 3)
        assert trace.commanded_tension_n.shape == (intervals, 4)
        assert trace.physics_time_s.shape == physics_shape
        assert trace.physics_data_ctrl_n.shape == physics_shape + (4,)
        assert trace.physics_actuator_force_n.shape == physics_shape + (4,)
        assert trace.physics_actuator_velocity_m_s.shape == physics_shape + (4,)
        assert trace.physics_tendon_velocity_m_s.shape == physics_shape + (4,)
        assert trace.physics_warning_counts.shape[:2] == physics_shape
        for values in (
            trace.physics_generalized_power_w,
            trace.physics_actuator_power_w,
            trace.physics_tendon_power_w,
            trace.physics_command_power_w,
            trace.physics_contact_count,
            trace.physics_finite_state,
        ):
            assert values.shape == physics_shape


def test_control_and_physics_times_are_aligned_without_deduplicating_boundaries(
    probe_traces,
):
    for trace in probe_traces.values():
        config = trace.config
        intervals = config.number_of_control_intervals
        substeps = config.substeps_per_control_interval
        expected_state = np.arange(intervals + 1) / config.ctrl_freq_hz
        expected_start = np.arange(intervals) / config.ctrl_freq_hz
        expected_end = np.arange(1, intervals + 1) / config.ctrl_freq_hz
        expected_physics = (
            expected_start[:, None]
            + np.arange(substeps + 1)[None, :] / config.sim_freq_hz
        )

        np.testing.assert_array_equal(trace.time_s, expected_state)
        np.testing.assert_array_equal(trace.interval_start_time_s, expected_start)
        np.testing.assert_array_equal(trace.interval_end_time_s, expected_end)
        np.testing.assert_allclose(
            trace.physics_time_s,
            expected_physics,
            rtol=0.0,
            atol=8 * np.finfo(float).eps * trace.number_of_physics_steps,
        )
        np.testing.assert_allclose(
            trace.physics_time_s[:-1, -1],
            trace.physics_time_s[1:, 0],
            rtol=0.0,
            atol=8 * np.finfo(float).eps * trace.number_of_physics_steps,
        )
        np.testing.assert_array_equal(trace.physics_substeps, substeps)


def test_2400_and_4800_runs_use_identical_control_profiles(probe_traces):
    # The parametrized fixture exposes one frequency per invocation; rebuild
    # the other pure profile without running another model.
    trace = probe_traces["active"]
    expected = build_fixed_tension_command_profile(trace.profile_config)

    np.testing.assert_array_equal(trace.commanded_tension_n, expected)


def test_runner_profiles_are_identical_between_2400_and_4800_hz():
    trace_2400 = run_fixed_root_tension_probe(
        _config(2400, TARGET_TENSION_N)
    )
    trace_4800 = run_fixed_root_tension_probe(
        _config(4800, TARGET_TENSION_N)
    )

    np.testing.assert_array_equal(
        trace_2400.commanded_tension_n,
        trace_4800.commanded_tension_n,
    )


def test_zero_and_active_profiles_write_all_channels_and_isolate_inactive(
    probe_traces,
):
    zero = probe_traces["zero"]
    active = probe_traces["active"]
    target = active.resolved_channel.binding_index
    inactive = [index for index in range(4) if index != target]

    np.testing.assert_array_equal(zero.commanded_tension_n, 0.0)
    np.testing.assert_array_equal(zero.data_ctrl_n, 0.0)
    np.testing.assert_allclose(zero.actuator_force_n, 0.0, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(active.commanded_tension_n[:, inactive], 0.0)
    np.testing.assert_array_equal(active.data_ctrl_n[:, inactive], 0.0)
    np.testing.assert_allclose(
        active.actuator_force_n[:, inactive], 0.0, rtol=0.0, atol=1e-15
    )
    np.testing.assert_array_equal(
        active.physics_data_ctrl_n[:, :, inactive], 0.0
    )
    np.testing.assert_allclose(
        active.physics_actuator_force_n[:, :, inactive],
        0.0,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        active.physics_actuator_force_n,
        active.physics_data_ctrl_n,
        rtol=0.0,
        atol=1e-15,
    )
    assert active.commanded_tension_n[0, target] == 0.0
    assert active.commanded_tension_n[2, target] < TARGET_TENSION_N
    assert active.commanded_tension_n[3, target] == TARGET_TENSION_N


def test_physics_transmission_moment_and_four_power_identities(probe_traces):
    for trace in probe_traces.values():
        safety = trace.safety
        np.testing.assert_allclose(
            trace.physics_actuator_velocity_m_s,
            -trace.physics_tendon_velocity_m_s,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trace.physics_generalized_power_w,
            trace.physics_actuator_power_w,
            rtol=1e-9,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trace.physics_actuator_power_w,
            trace.physics_tendon_power_w,
            rtol=1e-9,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trace.physics_tendon_power_w,
            trace.physics_command_power_w,
            rtol=1e-9,
            atol=1e-12,
        )
        assert safety.maximum_force_command_error_n <= 1e-15
        assert safety.maximum_moment_residual <= 1e-12
        assert safety.maximum_generalized_actuator_power_residual_w <= 1e-12
        assert safety.maximum_actuator_tendon_power_residual_w <= 1e-12
        assert safety.maximum_command_tendon_power_residual_w <= 1e-12
        assert safety.maximum_actuator_length_sign_residual_m <= 1e-12
        assert safety.maximum_actuator_velocity_sign_residual_m_s <= 1e-12


def test_interval_partitioned_work_is_finite_and_preserves_power_identities(
    probe_traces,
):
    for trace in probe_traces.values():
        dt = 1.0 / trace.config.sim_freq_hz
        actuator_work = integrate_interval_power_w(
            trace.physics_actuator_power_w, dt_sim_s=dt
        )

        assert np.isfinite(actuator_work)
        assert trace.actuator_work_j == actuator_work
        assert trace.generalized_actuator_work_j == pytest.approx(
            actuator_work, rel=1e-9, abs=1e-12
        )
        assert trace.tendon_work_j == pytest.approx(
            actuator_work, rel=1e-9, abs=1e-12
        )
        assert trace.command_work_j == pytest.approx(
            actuator_work, rel=1e-9, abs=1e-12
        )


def test_run_is_safe_finite_contact_free_and_within_tendon_ranges(probe_traces):
    for trace in probe_traces.values():
        assert trace.safety.warning_events == ()
        assert trace.safety.maximum_contacts == 0
        assert trace.safety.completed_physics_steps == trace.number_of_physics_steps
        assert trace.safety.minimum_tendon_range_margin_m >= -1e-9
        np.testing.assert_array_equal(trace.contact_count, 0)
        np.testing.assert_array_equal(trace.physics_contact_count, 0)
        np.testing.assert_array_equal(trace.warning_counts, 0)
        np.testing.assert_array_equal(trace.physics_warning_counts, 0)
        np.testing.assert_array_equal(trace.physics_finite_state, True)
        lower = trace.tendon_range_m[:, 0]
        upper = trace.tendon_range_m[:, 1]
        assert np.all(trace.tendon_length_m >= lower - 1e-9)
        assert np.all(trace.tendon_length_m <= upper + 1e-9)
        for values in (
            trace.qpos,
            trace.qvel,
            trace.tendon_length_m,
            trace.tendon_velocity_m_s,
            trace.tip_position_m,
            trace.joint_rotation_vectors_rad,
            trace.composite_rotation_vectors_rad,
            trace.spring_energy_j,
            trace.position_energy_j,
            trace.kinetic_energy_j,
            trace.total_energy_j,
        ):
            assert np.all(np.isfinite(values))
        assert np.all(trace.spring_energy_j >= 0.0)


def test_trace_and_adapter_arrays_are_read_only_and_defensively_copied(
    probe_traces,
):
    trace = probe_traces["active"]
    arrays = [
        value
        for value in vars(trace).values()
        if isinstance(value, np.ndarray)
    ]
    assert arrays
    assert all(not values.flags.writeable for values in arrays)
    with pytest.raises(ValueError):
        trace.qpos[0, 0] = 1.0

    aligned = trace.to_aligned_probe_trace()
    assert isinstance(aligned, AlignedProbeTrace)
    np.testing.assert_array_equal(aligned.time_s, trace.time_s)
    np.testing.assert_array_equal(
        aligned.tendon_length_m, trace.tendon_length_m
    )
    assert not np.shares_memory(aligned.time_s, trace.time_s)
    assert not np.shares_memory(
        aligned.tendon_length_m, trace.tendon_length_m
    )
    assert not aligned.time_s.flags.writeable


def test_repeated_runs_are_deterministic_and_do_not_share_numpy_state():
    config = _config(2400, TARGET_TENSION_N)
    first = run_fixed_root_tension_probe(config)
    second = run_fixed_root_tension_probe(config)

    for left, right in (
        (first.time_s, second.time_s),
        (first.qpos, second.qpos),
        (first.qvel, second.qvel),
        (first.tendon_length_m, second.tendon_length_m),
        (first.commanded_tension_n, second.commanded_tension_n),
        (first.physics_actuator_force_n, second.physics_actuator_force_n),
        (first.physics_actuator_power_w, second.physics_actuator_power_w),
    ):
        np.testing.assert_array_equal(left, right)
        assert not np.shares_memory(left, right)


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES_HZ)
def test_zero_profile_is_exactly_compatible_with_b0(sim_freq):
    b0 = run_fixed_root_zero_control(
        FixedRootZeroControlConfig(
            sim_freq=sim_freq,
            ctrl_freq=CTRL_FREQ_HZ,
            duration_s=RUN_DURATION_S,
        )
    )
    b1 = run_fixed_root_tension_probe(_config(sim_freq, 0.0))
    substeps = b1.config.substeps_per_control_interval

    for left, right in (
        (b1.time_s, b0.time_s),
        (b1.qpos, b0.qpos),
        (b1.qvel, b0.qvel),
        (b1.tendon_length_m, b0.tendon_length_m),
        (b1.tendon_velocity_m_s, b0.tendon_velocity_m_s),
        (b1.tip_position_m, b0.tip_position_m),
        (b1.commanded_tension_n, b0.commanded_tension_n),
        (b1.data_ctrl_n, b0.data_ctrl_n),
        (b1.actuator_force_n, b0.actuator_force_n),
        (b1.contact_count, b0.contact_count),
        (b1.warning_counts, b0.warning_counts),
    ):
        np.testing.assert_array_equal(left, right)

    assert b1.number_of_control_intervals == b0.number_of_control_intervals
    assert b1.number_of_physics_steps == b0.number_of_physics_steps
    np.testing.assert_array_equal(
        b1.physics_time_s[:, 1:].reshape(-1), b0.physics_time_s
    )
    np.testing.assert_array_equal(
        b1.physics_data_ctrl_n[:, 1:].reshape(-1, 4),
        b0.physics_data_ctrl_n,
    )
    np.testing.assert_array_equal(
        b1.physics_actuator_force_n[:, 1:].reshape(-1, 4),
        b0.physics_actuator_force_n,
    )
    np.testing.assert_array_equal(
        b1.physics_contact_count[:, 1:].reshape(-1),
        b0.physics_contact_count,
    )
    np.testing.assert_array_equal(
        b1.physics_warning_counts[:, 1:].reshape(
            -1, b1.physics_warning_counts.shape[-1]
        ),
        b0.physics_warning_counts,
    )
    assert b1.physics_time_s.shape == (6, substeps + 1)
