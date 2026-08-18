"""Fixed-root free-decay and synchronized-energy acceptance tests."""

import math

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    WarningLatch,
    analyze_decay_signal,
    analyze_energy_trace,
    build_fixed_arm_fixture,
    composite_rotation_vector,
    find_settling_time,
    initialize_constant_curvature,
    joint_rotation_vectors,
    named_joint_addresses,
    run_decay_characterization,
)


SIM_FREQ = 4800
CTRL_FREQ = 240
INTEGRATOR = "implicitfast"
PHYSICAL_DURATION = 5.0
TOTAL_BEND = 0.2
PER_JOINT_BEND = 0.025
JOINT_STIFFNESS = 0.003086
TARGET_DAMPING_RATIO = 0.3


def _b2_spec():
    return ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )


@pytest.fixture(scope="module")
def decay_results():
    return {
        axis: run_decay_characterization(
            axis,
            total_bend=TOTAL_BEND,
            physical_duration=PHYSICAL_DURATION,
            sim_freq=SIM_FREQ,
            ctrl_freq=CTRL_FREQ,
            integrator=INTEGRATOR,
        )
        for axis in ("x", "y")
    }


def _diagnostics(result):
    trace = result.trace
    decay = result.decay
    energy = result.energy
    return (
        f"axis={trace.bend_axis}, completed={trace.completed_steps}/"
        f"{trace.requested_steps}, raw_failure={trace.raw_state_failure}, "
        f"sync_failure={trace.synchronized_state_failure}, "
        f"warnings={trace.warning_events}, contacts={int(np.max(trace.contacts))}, "
        f"peaks={None if decay is None else len(decay.peak_times)}, "
        f"frequency={None if decay is None else decay.frequency}, "
        f"decrement={None if decay is None else decay.median_log_decrement}, "
        f"zeta={None if decay is None else decay.apparent_damping_ratio}, "
        f"settling={result.settling_time}, energy={energy}"
    )


def _tendon_index(names, corner):
    matches = [
        index
        for index, name in enumerate(names)
        if name.endswith(f"_{corner}_tendon")
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("bend_axis,axis_index", [("x", 0), ("y", 1)])
def test_b2_vertical_fixture_initializes_named_constant_curvature(
    bend_axis, axis_index
):
    fixture = build_fixed_arm_fixture(
        _b2_spec(),
        "vertical",
        sim_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        integrator=INTEGRATOR,
    )
    model, data = fixture.model, fixture.data
    addresses = named_joint_addresses(model, fixture.manifest)
    qpos0_before = model.qpos0.copy()
    root_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        fixture.manifest.arm_root_site_name,
    )
    tip_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        fixture.manifest.arm_tip_site_name,
    )

    np.testing.assert_allclose(
        data.site_xpos[tip_id] - data.site_xpos[root_id],
        [0.0, 0.0, -_b2_spec().total_length],
        rtol=0.0,
        atol=1e-12,
    )
    initialize_constant_curvature(
        fixture,
        bend_axis,
        total_bend=TOTAL_BEND,
    )
    joints = joint_rotation_vectors(model, data, addresses)
    composite = composite_rotation_vector(model, data, addresses)
    non_target_index = 1 if axis_index == 0 else 0

    assert model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_ENERGY
    assert model.opt.timestep == pytest.approx(1.0 / SIM_FREQ)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert fixture.sample_stride == 20
    assert model.nu == 0
    assert data.ncon == 0
    assert data.time == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_array_equal(model.qpos0, qpos0_before)
    np.testing.assert_allclose(
        joints[:, axis_index],
        PER_JOINT_BEND,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        joints[:, non_target_index], 0.0, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        joints[:, 2], 0.0, rtol=0.0, atol=1e-12
    )
    assert composite[axis_index] == pytest.approx(TOTAL_BEND, abs=1e-12)
    assert composite[non_target_index] == pytest.approx(0.0, abs=1e-12)
    assert composite[2] == pytest.approx(0.0, abs=1e-12)
    for _, qpos_address, dof_address in addresses:
        assert np.linalg.norm(
            data.qpos[qpos_address:qpos_address + 4]
        ) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(
            data.qvel[dof_address:dof_address + 3],
            0.0,
            rtol=0.0,
            atol=1e-15,
        )
    assert np.all(np.isfinite(data.energy))


@pytest.mark.parametrize("bend_axis", ["x", "y"])
def test_b2_synchronized_trace_has_complete_monotonic_time(
    decay_results, bend_axis
):
    result = decay_results[bend_axis]
    trace = result.trace
    diagnostics = _diagnostics(result)
    expected_physics_records = int(PHYSICAL_DURATION * SIM_FREQ) + 1
    expected_control_records = int(PHYSICAL_DURATION * CTRL_FREQ) + 1

    assert trace.raw_state_failure is None, diagnostics
    assert trace.synchronized_state_failure is None, diagnostics
    assert trace.warning_events == (), diagnostics
    assert trace.completed_steps == trace.requested_steps == 24000, diagnostics
    assert len(trace.physics_time) == expected_physics_records, diagnostics
    assert len(trace.control_time) == expected_control_records, diagnostics
    assert trace.physics_time[0] == pytest.approx(0.0, abs=1e-15)
    assert trace.physics_time[-1] == pytest.approx(
        PHYSICAL_DURATION, rel=1e-12, abs=1e-12
    )
    assert np.all(np.diff(trace.physics_time) > 0)
    np.testing.assert_allclose(
        np.diff(trace.physics_time),
        1.0 / SIM_FREQ,
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        trace.control_indices,
        np.arange(0, expected_physics_records, 20),
    )
    np.testing.assert_allclose(
        np.diff(trace.control_time),
        1.0 / CTRL_FREQ,
        rtol=1e-12,
        atol=1e-15,
    )
    assert trace.control_time[-1] == pytest.approx(
        PHYSICAL_DURATION, rel=1e-12, abs=1e-12
    )

    assert trace.composite_rotation_vectors.shape == (
        expected_physics_records,
        3,
    )
    assert trace.joint_rotation_vectors.shape == (
        expected_physics_records,
        8,
        3,
    )
    assert trace.tip_relative_positions.shape == (
        expected_physics_records,
        3,
    )
    assert trace.arm_qvel.shape == (expected_physics_records, 24)
    assert trace.tendon_lengths.shape == (expected_physics_records, 4)
    assert trace.position_energy.shape == trace.physics_time.shape
    assert trace.kinetic_energy.shape == trace.physics_time.shape
    assert trace.total_energy.shape == trace.physics_time.shape
    assert trace.all_synchronized_values_finite, diagnostics
    assert np.max(trace.contacts) == 0, diagnostics
    assert np.min(trace.kinetic_energy) >= -1e-12, diagnostics


@pytest.mark.parametrize("bend_axis,axis_index", [("x", 0), ("y", 1)])
def test_b2_decay_reaches_approved_envelope_and_settling(
    decay_results, bend_axis, axis_index
):
    result = decay_results[bend_axis]
    trace = result.trace
    decay = result.decay
    diagnostics = _diagnostics(result)
    assert decay is not None, diagnostics

    assert trace.control_signed_bend[0] == pytest.approx(
        TOTAL_BEND, abs=1e-12
    )
    np.testing.assert_allclose(
        trace.control_joint_rotation_vectors[0, :, axis_index],
        PER_JOINT_BEND,
        rtol=0.0,
        atol=1e-12,
    )
    assert decay.analysis_path == "peaks", diagnostics
    assert decay.frequency_source == "peaks", diagnostics
    assert len(decay.peak_times) >= 3, diagnostics
    assert decay.envelope_valid, diagnostics
    assert decay.frequency is not None and decay.frequency > 0, diagnostics
    assert decay.zero_crossing_frequency is not None, diagnostics
    assert decay.frequency_consistent is True, diagnostics
    assert decay.median_log_decrement is not None
    assert decay.median_log_decrement > 0, diagnostics
    assert decay.decay_rate is not None and decay.decay_rate > 0, diagnostics
    assert decay.apparent_damping_ratio is not None
    assert 0 < decay.apparent_damping_ratio < 1, diagnostics
    assert result.settling_time is not None, diagnostics
    assert result.settling_time <= PHYSICAL_DURATION - 0.25, diagnostics
    assert np.max(np.abs(trace.control_non_target_bend)) <= 0.002, diagnostics
    assert np.max(np.abs(trace.control_twist)) <= 0.002, diagnostics

    lower = trace.tendon_ranges[:, 0]
    upper = trace.tendon_ranges[:, 1]
    assert np.all(trace.control_tendon_lengths >= lower)
    assert np.all(trace.control_tendon_lengths <= upper)


@pytest.mark.parametrize("bend_axis", ["x", "y"])
def test_b2_synchronized_energy_has_no_sustained_injection(
    decay_results, bend_axis
):
    result = decay_results[bend_axis]
    energy = result.energy
    diagnostics = _diagnostics(result)
    assert energy is not None, diagnostics

    assert result.initial_spring_energy == pytest.approx(
        0.5 * 8 * JOINT_STIFFNESS * PER_JOINT_BEND ** 2,
        rel=1e-12,
    )
    assert energy.tolerance == pytest.approx(
        max(1e-9, 0.01 * result.initial_spring_energy),
        rel=1e-12,
    )
    assert energy.passed, diagnostics
    assert not energy.hard_limit_failed, diagnostics
    assert energy.longest_above_tolerance_run < 20, diagnostics
    assert not energy.sustained_growth_failed, diagnostics
    assert energy.final_delta <= energy.tolerance, diagnostics


def test_b2_x_y_decay_is_mirror_symmetric(decay_results):
    x_result = decay_results["x"]
    y_result = decay_results["y"]
    x_trace, y_trace = x_result.trace, y_result.trace
    x_decay, y_decay = x_result.decay, y_result.decay
    assert x_decay is not None and y_decay is not None

    np.testing.assert_allclose(
        x_trace.physics_time,
        y_trace.physics_time,
        rtol=0.0,
        atol=1e-15,
    )
    assert abs(
        x_trace.control_signed_bend[0] - y_trace.control_signed_bend[0]
    ) <= 1e-12
    np.testing.assert_allclose(
        x_trace.control_signed_bend,
        y_trace.control_signed_bend,
        rtol=0.0,
        atol=max(1e-5, 0.005 * TOTAL_BEND),
    )

    assert x_decay.analysis_path == y_decay.analysis_path
    assert abs(x_decay.frequency - y_decay.frequency) / max(
        x_decay.frequency, y_decay.frequency
    ) <= 0.02
    assert abs(
        x_decay.median_log_decrement - y_decay.median_log_decrement
    ) / max(
        x_decay.median_log_decrement, y_decay.median_log_decrement
    ) <= 0.05
    assert abs(x_decay.decay_rate - y_decay.decay_rate) / max(
        x_decay.decay_rate, y_decay.decay_rate
    ) <= 0.05
    assert abs(x_result.settling_time - y_result.settling_time) <= 0.02

    for x_values, y_values in (
        (x_trace.control_non_target_bend, y_trace.control_non_target_bend),
        (x_trace.control_twist, y_trace.control_twist),
    ):
        x_max = float(np.max(np.abs(x_values)))
        y_max = float(np.max(np.abs(y_values)))
        assert x_max <= 0.002
        assert y_max <= 0.002
        assert abs(x_max - y_max) <= max(1e-5, 0.005 * max(x_max, y_max))

    rotate_z_positive_90 = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_x_tip = (
        rotate_z_positive_90 @ x_trace.control_tip_relative_positions.T
    ).T
    tip_error = float(
        np.max(
            np.linalg.norm(
                rotated_x_tip - y_trace.control_tip_relative_positions,
                axis=1,
            )
        )
    )
    maximum_tip_displacement = max(
        float(
            np.max(
                np.linalg.norm(
                    x_trace.control_tip_relative_positions
                    - x_trace.control_tip_relative_positions[0],
                    axis=1,
                )
            )
        ),
        float(
            np.max(
                np.linalg.norm(
                    y_trace.control_tip_relative_positions
                    - y_trace.control_tip_relative_positions[0],
                    axis=1,
                )
            )
        ),
    )
    assert tip_error <= max(1e-6, 0.005 * maximum_tip_displacement)

    corner_mapping = {
        "xp_yp": "xn_yp",
        "xp_yn": "xp_yp",
        "xn_yp": "xn_yn",
        "xn_yn": "xp_yn",
    }
    tendon_error = 0.0
    maximum_tendon_change = 0.0
    for x_corner, y_corner in corner_mapping.items():
        x_index = _tendon_index(x_trace.tendon_names, x_corner)
        y_index = _tendon_index(y_trace.tendon_names, y_corner)
        x_lengths = x_trace.control_tendon_lengths[:, x_index]
        y_lengths = y_trace.control_tendon_lengths[:, y_index]
        tendon_error = max(
            tendon_error,
            float(np.max(np.abs(x_lengths - y_lengths))),
        )
        maximum_tendon_change = max(
            maximum_tendon_change,
            float(np.max(np.abs(x_lengths - x_lengths[0]))),
            float(np.max(np.abs(y_lengths - y_lengths[0]))),
        )
    assert tendon_error <= max(1e-9, 0.005 * maximum_tendon_change)


def test_decay_analysis_recovers_signed_synthetic_frequency_and_decrement():
    time = np.arange(0.0, 5.0 + 0.5 / CTRL_FREQ, 1.0 / CTRL_FREQ)
    decay_rate = 0.5
    frequency = 4.0
    signal = TOTAL_BEND * np.exp(-decay_rate * time) * np.cos(
        2 * math.pi * frequency * time
    )

    analysis = analyze_decay_signal(time, signal)

    assert analysis.analysis_path == "peaks"
    assert analysis.frequency == pytest.approx(frequency, rel=0.02)
    assert analysis.zero_crossing_frequency == pytest.approx(
        frequency, rel=0.02
    )
    assert analysis.frequency_consistent is True
    assert analysis.envelope_valid
    assert analysis.median_log_decrement == pytest.approx(
        decay_rate / frequency,
        rel=0.03,
    )
    assert analysis.decay_rate == pytest.approx(decay_rate, rel=0.03)
    assert np.any(signal < 0), "signed signal must retain its negative half-cycles"


def test_decay_analysis_uses_settling_fallback_when_peaks_are_insufficient():
    time = np.arange(0.0, 2.0 + 0.5 / CTRL_FREQ, 1.0 / CTRL_FREQ)
    signal = TOTAL_BEND * np.exp(-8.0 * time)
    max_qvel = 1.6 * np.exp(-8.0 * time)

    analysis = analyze_decay_signal(time, signal)
    settling = find_settling_time(time, signal, max_qvel)

    assert analysis.analysis_path == "settling_fallback"
    assert len(analysis.peak_times) < 3
    assert analysis.median_log_decrement is None
    assert settling is not None
    assert settling <= time[-1] - 0.25


def test_decay_analysis_collapses_sign_changing_zero_plateau():
    time = np.arange(5, dtype=float) / CTRL_FREQ
    analysis = analyze_decay_signal(time, [1.0, 0.0, 0.0, 0.0, -1.0])

    np.testing.assert_allclose(
        analysis.zero_crossing_times,
        [0.5 * (time[1] + time[3])],
    )
    np.testing.assert_array_equal(analysis.zero_crossing_directions, [-1])


def test_decay_analysis_ignores_same_sign_zero_plateau():
    time = np.arange(4, dtype=float) / CTRL_FREQ
    analysis = analyze_decay_signal(time, [1.0, 0.0, 0.0, 1.0])

    assert len(analysis.zero_crossing_times) == 0
    assert analysis.zero_crossing_frequency is None


def test_decay_analysis_ignores_trailing_zero_plateau():
    time = np.arange(5, dtype=float) / CTRL_FREQ
    analysis = analyze_decay_signal(time, [1.0, 0.5, 0.0, 0.0, 0.0])

    assert len(analysis.zero_crossing_times) == 0
    assert analysis.zero_crossing_frequency is None


def test_decay_analysis_all_zero_signal_has_no_frequency():
    time = np.arange(5, dtype=float) / CTRL_FREQ
    analysis = analyze_decay_signal(time, np.zeros_like(time))

    assert len(analysis.zero_crossing_times) == 0
    assert analysis.zero_crossing_frequency is None
    assert analysis.frequency is None


def test_decay_analysis_preserves_adjacent_linear_zero_interpolation():
    time = np.array([0.0, 0.1, 0.2])
    analysis = analyze_decay_signal(time, [1.0, -3.0, -2.0])

    np.testing.assert_allclose(analysis.zero_crossing_times, [0.025])
    np.testing.assert_array_equal(analysis.zero_crossing_directions, [-1])


def test_decay_analysis_zero_plateau_periodic_signal_is_not_240_hz():
    signal = np.array(
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0] * 3 + [1.0]
    )
    time = np.arange(len(signal), dtype=float) / CTRL_FREQ
    analysis = analyze_decay_signal(time, signal)

    assert analysis.zero_crossing_frequency == pytest.approx(40.0)
    assert analysis.zero_crossing_frequency != pytest.approx(CTRL_FREQ)


def test_energy_analysis_distinguishes_spikes_and_sustained_growth():
    spring_energy = 7.715e-6
    tolerance = max(1e-9, 0.01 * spring_energy)

    isolated_spike = np.zeros(100)
    isolated_spike[30] = 2 * tolerance
    isolated = analyze_energy_trace(isolated_spike, spring_energy)
    assert isolated.passed
    assert isolated.maximum_after_release_delta == pytest.approx(2 * tolerance)
    assert isolated.longest_above_tolerance_run == 1

    persistent = np.zeros(100)
    persistent[20:40] = 2 * tolerance
    persistent_result = analyze_energy_trace(persistent, spring_energy)
    assert persistent_result.persistent_upper_bound_failed
    assert not persistent_result.passed

    hard_spike = np.zeros(100)
    hard_spike[20] = 6 * tolerance
    hard_result = analyze_energy_trace(hard_spike, spring_energy)
    assert hard_result.hard_limit_failed
    assert not hard_result.passed

    sustained = np.zeros(100)
    sustained[10:31] = np.linspace(-2 * tolerance, 0.0, 21)
    sustained[31:] = 0.0
    sustained_result = analyze_energy_trace(sustained, spring_energy)
    assert sustained_result.longest_positive_increment_run >= 20
    assert sustained_result.maximum_positive_run_increase > tolerance
    assert sustained_result.sustained_growth_failed
    assert not sustained_result.passed


def test_warning_latch_survives_counter_reset():
    latch = WarningLatch()
    latch.snapshot_counts(
        (0, 0, 0), stage="initial_forward", step_index=0, time=0.0
    )
    latch.snapshot_counts(
        (0, 1, 0), stage="after_step", step_index=1, time=1 / SIM_FREQ
    )
    latch.snapshot_counts(
        (0, 0, 0), stage="after_forward", step_index=1, time=1 / SIM_FREQ
    )

    assert latch.triggered
    assert len(latch.events) == 1
    assert latch.events[0].stage == "after_step"
    assert latch.events[0].step_index == 1
    assert latch.events[0].raised_indices == (1,)


def test_raw_nonfinite_state_stops_before_post_step_forward(monkeypatch):
    original_forward = mujoco.mj_forward
    forward_calls = []

    def tracking_forward(model, data):
        forward_calls.append(float(data.time))
        return original_forward(model, data)

    def nonfinite_step(model, data):
        data.time += model.opt.timestep
        data.qpos[:] = np.nan

    monkeypatch.setattr(mujoco, "mj_forward", tracking_forward)
    monkeypatch.setattr(mujoco, "mj_step", nonfinite_step)

    result = run_decay_characterization(
        "x",
        physical_duration=1.0 / SIM_FREQ,
        sim_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        integrator=INTEGRATOR,
    )

    assert len(forward_calls) == 2  # fixture forward and initialized-state forward
    assert result.trace.raw_state_failure is not None
    assert result.trace.synchronized_state_failure is None
    assert result.trace.raw_state_failure.step_index == 1
    assert not result.trace.raw_state_failure.qpos_finite
    assert result.trace.completed_steps == 0
    assert len(result.trace.physics_time) == 1
    assert result.decay is None
    assert result.energy is None
