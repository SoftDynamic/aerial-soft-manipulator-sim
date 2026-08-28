"""Gate C2-B0 contracts for the fixed-root zero-control runner."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_fixed_root import (
    CORNERS,
    CTRL_FREQ,
    MAX_TENSION_N,
    SIM_FREQUENCIES,
    FixedRootZeroControlConfig,
    run_fixed_root_zero_control,
)


RUN_DURATION_S = 0.025


@pytest.fixture(scope="module", params=SIM_FREQUENCIES)
def zero_control_trace(request):
    return run_fixed_root_zero_control(
        FixedRootZeroControlConfig(
            sim_freq=request.param,
            ctrl_freq=CTRL_FREQ,
            duration_s=RUN_DURATION_S,
        )
    )


def test_b0_source_does_not_import_or_instantiate_acceleration_controller():
    helper = Path(__file__).with_name(
        "_scaled_stablesoft_acceleration_fixed_root.py"
    ).read_text(encoding="utf-8")

    assert "tendon_acceleration_controller" not in helper
    assert "TendonContractionAccelerationController" not in helper
    assert "controller.reset" not in helper
    assert "controller.update" not in helper


def test_zero_control_config_is_frozen_and_has_exact_integer_grid():
    config = FixedRootZeroControlConfig(2400, 240, RUN_DURATION_S)

    assert config.number_of_control_intervals == 6
    assert config.substeps_per_control_interval == 10
    assert config.number_of_physics_steps == 60
    with pytest.raises(FrozenInstanceError):
        config.sim_freq = 4800


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"sim_freq": True}, "sim_freq must be an integer"),
        ({"sim_freq": 0}, "sim_freq must be positive"),
        ({"sim_freq": -2400}, "sim_freq must be positive"),
        ({"sim_freq": 2400.0}, "sim_freq must be an integer"),
        ({"sim_freq": "2400"}, "sim_freq must be an integer"),
        ({"ctrl_freq": True}, "ctrl_freq must be an integer"),
        ({"ctrl_freq": 0}, "ctrl_freq must be positive"),
        ({"ctrl_freq": -240}, "ctrl_freq must be positive"),
        ({"ctrl_freq": 240.0}, "ctrl_freq must be an integer"),
        ({"sim_freq": 2400, "ctrl_freq": 241}, "divisible"),
        ({"sim_freq": 2400, "ctrl_freq": 120}, "exactly 240"),
        ({"sim_freq": 1200}, "sim_freq must be one of"),
        ({"duration_s": True}, "duration_s must be a real number"),
        ({"duration_s": "0.1"}, "duration_s must be a real number"),
        ({"duration_s": float("nan")}, "duration_s must be finite"),
        ({"duration_s": float("inf")}, "duration_s must be finite"),
        ({"duration_s": 0.0}, "duration_s must be positive"),
        ({"duration_s": -0.1}, "duration_s must be positive"),
        ({"duration_s": 0.01001}, "control tick grid"),
    ],
)
def test_zero_control_config_rejects_invalid_or_ambiguous_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FixedRootZeroControlConfig(**kwargs)


def test_runner_requires_valid_config_before_constructing_a_fixture():
    with pytest.raises(TypeError, match="FixedRootZeroControlConfig"):
        run_fixed_root_zero_control(object())


def test_fixed_root_topology_and_manifest_binding_contract(zero_control_trace):
    trace = zero_control_trace
    topology = trace.topology

    assert topology.model_counts == (11, 8, 32, 24, 10, 75, 4, 4, 0, 4, 4, 4)
    assert topology.fixture_mount_parent_body_id == 0
    assert topology.fixture_mount_joint_count == 0
    assert topology.root_segment_parent_body_id == topology.fixture_mount_body_id
    assert topology.root_segment_joint_count == 0
    assert topology.joint_types == (int(mujoco.mjtJoint.mjJNT_BALL),) * 8
    assert topology.tendon_names == tuple(
        f"fixture_scaled_stablesoft_arm_{corner}_tendon"
        for corner in CORNERS
    )
    assert topology.actuator_names == tuple(
        name.removesuffix("_tendon") + "_ideal_tension_actuator"
        for name in topology.tendon_names
    )
    assert tuple(item.tendon_name for item in trace.bindings) == topology.tendon_names
    assert (
        tuple(item.actuator_name for item in trace.bindings)
        == topology.actuator_names
    )
    assert tuple(item.tendon_id for item in trace.bindings) == topology.tendon_ids
    assert tuple(item.actuator_id for item in trace.bindings) == topology.actuator_ids
    assert (
        tuple(item.control_address for item in trace.bindings)
        == topology.control_addresses
    )
    assert (
        tuple(item.output_address for item in trace.bindings)
        == topology.output_addresses
    )

    np.testing.assert_allclose(
        topology.actuator_gear,
        np.tile([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], (4, 1)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(topology.actuator_ctrl_limited, True)
    np.testing.assert_allclose(
        topology.actuator_ctrl_range_n,
        np.tile([0.0, MAX_TENSION_N], (4, 1)),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(topology.actuator_force_limited, False)
    np.testing.assert_allclose(topology.actuator_force_range_n, 0.0)
    np.testing.assert_array_equal(topology.tendon_limited, True)


def test_reset_records_one_finite_zero_control_initial_sample(zero_control_trace):
    trace = zero_control_trace

    assert trace.number_of_reset_writes == 1
    assert trace.time_s[0] == 0.0
    np.testing.assert_array_equal(trace.qvel[0], 0.0)
    assert trace.contact_count[0] == 0
    assert np.all(trace.warning_counts[0] == 0)
    for values in (
        trace.qpos[0],
        trace.qvel[0],
        trace.tendon_length_m[0],
        trace.tendon_velocity_m_s[0],
        trace.tip_position_m[0],
    ):
        assert np.all(np.isfinite(values))


def test_control_and_physics_grids_have_exact_counts_and_alignment(
    zero_control_trace,
):
    trace = zero_control_trace
    config = trace.config
    intervals = config.number_of_control_intervals
    substeps = config.substeps_per_control_interval
    expected_state_time = np.arange(intervals + 1) / config.ctrl_freq
    expected_start = np.arange(intervals) / config.ctrl_freq
    expected_end = np.arange(1, intervals + 1) / config.ctrl_freq
    expected_physics_time = (
        np.arange(1, config.number_of_physics_steps + 1)
        / config.sim_freq
    )

    assert trace.number_of_control_intervals == intervals
    assert trace.number_of_physics_steps == intervals * substeps
    assert trace.number_of_control_writes == intervals
    assert trace.number_of_physics_control_checks == intervals * substeps
    assert trace.substeps_per_control_interval == substeps
    assert substeps == {2400: 10, 4800: 20}[config.sim_freq]
    assert len(trace.time_s) == intervals + 1
    assert len(trace.interval_start_time_s) == intervals
    assert len(trace.physics_time_s) == intervals * substeps
    np.testing.assert_array_equal(trace.time_s, expected_state_time)
    np.testing.assert_array_equal(trace.interval_start_time_s, expected_start)
    np.testing.assert_array_equal(trace.interval_end_time_s, expected_end)
    np.testing.assert_allclose(
        trace.physics_time_s,
        expected_physics_time,
        rtol=0.0,
        atol=8 * np.finfo(float).eps * config.number_of_physics_steps,
    )
    np.testing.assert_array_equal(trace.physics_substeps, substeps)
    np.testing.assert_array_equal(
        trace.physics_interval_index,
        np.repeat(np.arange(intervals), substeps),
    )
    np.testing.assert_array_equal(
        trace.physics_substep_index,
        np.tile(np.arange(1, substeps + 1), intervals),
    )
    assert trace.maximum_control_time_error_s <= (
        8
        * np.finfo(float).eps
        * config.number_of_physics_steps
    )


def test_zero_command_control_and_force_hold_at_all_sampling_rates(
    zero_control_trace,
):
    trace = zero_control_trace

    for values in (
        trace.commanded_tension_n,
        trace.data_ctrl_n,
        trace.actuator_force_n,
        trace.physics_data_ctrl_n,
    ):
        np.testing.assert_array_equal(values, 0.0)
    np.testing.assert_allclose(
        trace.physics_actuator_force_n, 0.0, rtol=0.0, atol=1e-15
    )


def test_zero_control_run_is_safe_without_forcing_passive_state_constant(
    zero_control_trace,
):
    trace = zero_control_trace
    topology = trace.topology

    assert trace.warning_events == ()
    assert np.all(trace.contact_count == 0)
    assert np.all(trace.physics_contact_count == 0)
    assert np.all(trace.warning_counts == 0)
    assert np.all(trace.physics_warning_counts == 0)
    for values in (
        trace.qpos,
        trace.qvel,
        trace.tendon_length_m,
        trace.tendon_velocity_m_s,
        trace.tip_position_m,
        trace.actuator_force_n,
        trace.physics_actuator_force_n,
    ):
        assert np.all(np.isfinite(values))
    lower = topology.tendon_range_m[:, 0]
    upper = topology.tendon_range_m[:, 1]
    assert np.all(trace.tendon_length_m >= lower - 1e-9)
    assert np.all(trace.tendon_length_m <= upper + 1e-9)


def test_trace_arrays_and_nested_topology_arrays_are_read_only(
    zero_control_trace,
):
    trace = zero_control_trace
    arrays = (
        trace.time_s,
        trace.qpos,
        trace.qvel,
        trace.tendon_length_m,
        trace.tendon_velocity_m_s,
        trace.tip_position_m,
        trace.contact_count,
        trace.warning_counts,
        trace.interval_start_time_s,
        trace.interval_end_time_s,
        trace.commanded_tension_n,
        trace.data_ctrl_n,
        trace.actuator_force_n,
        trace.physics_substeps,
        trace.physics_time_s,
        trace.physics_interval_index,
        trace.physics_substep_index,
        trace.physics_data_ctrl_n,
        trace.physics_actuator_force_n,
        trace.physics_contact_count,
        trace.physics_warning_counts,
        trace.topology.actuator_gear,
        trace.topology.actuator_ctrl_limited,
        trace.topology.actuator_ctrl_range_n,
        trace.topology.actuator_force_limited,
        trace.topology.actuator_force_range_n,
        trace.topology.tendon_limited,
        trace.topology.tendon_range_m,
    )

    assert all(not values.flags.writeable for values in arrays)
    with pytest.raises(ValueError):
        trace.qpos[0, 0] = 1.0
    with pytest.raises(ValueError):
        trace.topology.actuator_gear[0, 0] = 1.0


def test_repeated_runs_are_deterministic_and_do_not_share_trace_state():
    config = FixedRootZeroControlConfig(2400, CTRL_FREQ, RUN_DURATION_S)
    first = run_fixed_root_zero_control(config)
    second = run_fixed_root_zero_control(config)

    assert first is not second
    assert first.bindings is not second.bindings
    for left, right in (
        (first.time_s, second.time_s),
        (first.qpos, second.qpos),
        (first.qvel, second.qvel),
        (first.tendon_length_m, second.tendon_length_m),
        (first.tendon_velocity_m_s, second.tendon_velocity_m_s),
        (first.tip_position_m, second.tip_position_m),
        (first.commanded_tension_n, second.commanded_tension_n),
        (first.physics_data_ctrl_n, second.physics_data_ctrl_n),
    ):
        np.testing.assert_array_equal(left, right)
        assert not np.shares_memory(left, right)
