"""B4 whole-vehicle passive coupling acceptance tests."""

import math

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.examples import scaled_stablesoft_passive_demo
from multi_drone_mujoco.examples.scaled_stablesoft_passive_demo import (
    _VIEWER_HOLD_REFRESH_SECONDS,
    _VIEWER_SHUTDOWN_GRACE_SECONDS,
    _build_parser,
    _build_viewer_overlay,
    _closing_viewer,
    _hold_final_viewer_frame,
    _try_update_viewer_overlay,
)
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


class _FakeViewer:
    def __init__(self, error=None, close_error=None, events=None):
        self.error = error
        self.close_error = close_error
        self.events = events
        self.overlays = []
        self.close_calls = 0

    def set_texts(self, overlay):
        self.overlays.append(overlay)
        if self.error is not None:
            raise self.error

    def close(self):
        self.close_calls += 1
        if self.events is not None:
            self.events.append("close")
        if self.close_error is not None:
            raise self.close_error


class _FakeHoldingViewer:
    def __init__(self, running_states, sync_error=None):
        self.running_states = iter(running_states)
        self.sync_error = sync_error
        self.is_running_calls = 0
        self.sync_calls = 0

    def is_running(self):
        self.is_running_calls += 1
        return next(self.running_states)

    def sync(self):
        self.sync_calls += 1
        if self.sync_error is not None:
            raise self.sync_error


def test_b4_viewer_overlay_matches_mujoco_tuple_contract():
    overlay = _build_viewer_overlay("hover", 0.1254, 2)

    assert isinstance(overlay, tuple)
    assert not isinstance(overlay, dict)
    assert len(overlay) == 4
    assert overlay[0] == mujoco.mjtFontScale.mjFONTSCALE_150
    assert overlay[1] == mujoco.mjtGridPos.mjGRID_TOPLEFT
    assert overlay[2] == "scenario\ntime\ncontacts"
    assert overlay[3] == "hover\n0.125 s\n2"


def test_b4_viewer_overlay_update_succeeds_with_fake_viewer():
    viewer = _FakeViewer()

    enabled = _try_update_viewer_overlay(
        viewer, "open-loop-release", 1.25, 0
    )

    assert enabled
    assert len(viewer.overlays) == 1
    assert isinstance(viewer.overlays[0], tuple)
    assert not isinstance(viewer.overlays[0], dict)


def test_b4_viewer_overlay_value_error_disables_future_updates(capsys):
    viewer = _FakeViewer(ValueError("unsupported overlay"))
    simulation_continued = []

    enabled = _try_update_viewer_overlay(viewer, "hover", 0.1, 0)
    simulation_continued.append("step_after_overlay_failure")
    enabled = _try_update_viewer_overlay(
        viewer, "hover", 0.2, 0, enabled=enabled
    )
    simulation_continued.append("next_step")

    assert not enabled
    assert len(viewer.overlays) == 1
    assert simulation_continued == [
        "step_after_overlay_failure",
        "next_step",
    ]
    captured = capsys.readouterr()
    assert captured.err.count("viewer overlay disabled") == 1
    assert "unsupported overlay" in captured.err


def test_b4_viewer_lifecycle_closes_then_waits_on_normal_exit():
    events = []
    viewer = _FakeViewer(events=events)

    def fake_sleep(seconds):
        events.append(("sleep", seconds))

    with _closing_viewer(viewer, sleep_fn=fake_sleep):
        pass

    assert viewer.close_calls == 1
    assert events == [
        "close",
        ("sleep", _VIEWER_SHUTDOWN_GRACE_SECONDS),
    ]
    assert _VIEWER_SHUTDOWN_GRACE_SECONDS == 2.0


def test_b4_viewer_lifecycle_closes_and_waits_before_propagating_exception():
    events = []
    viewer = _FakeViewer(events=events)

    def fake_sleep(seconds):
        events.append(("sleep", seconds))

    with pytest.raises(RuntimeError, match="stepping failed"):
        with _closing_viewer(viewer, sleep_fn=fake_sleep):
            raise RuntimeError("stepping failed")

    assert viewer.close_calls == 1
    assert events == [
        "close",
        ("sleep", _VIEWER_SHUTDOWN_GRACE_SECONDS),
    ]


def test_b4_viewer_lifecycle_waits_when_close_raises():
    events = []
    viewer = _FakeViewer(
        close_error=RuntimeError("close failed"), events=events
    )

    def fake_sleep(seconds):
        events.append(("sleep", seconds))

    with pytest.raises(RuntimeError, match="close failed"):
        with _closing_viewer(viewer, sleep_fn=fake_sleep):
            pass

    assert viewer.close_calls == 1
    assert events == [
        "close",
        ("sleep", _VIEWER_SHUTDOWN_GRACE_SECONDS),
    ]


def test_b4_viewer_lifecycle_cleans_up_before_keyboard_interrupt_propagates():
    events = []
    viewer = _FakeViewer(events=events)

    def fake_sleep(seconds):
        events.append(("sleep", seconds))

    with pytest.raises(KeyboardInterrupt):
        with _closing_viewer(viewer, sleep_fn=fake_sleep):
            raise KeyboardInterrupt

    assert viewer.close_calls == 1
    assert events == [
        "close",
        ("sleep", _VIEWER_SHUTDOWN_GRACE_SECONDS),
    ]


def test_b4_viewer_parser_disables_final_frame_hold_by_default():
    args = _build_parser().parse_args([])

    assert args.hold_viewer_open is False


def test_b4_viewer_parser_accepts_final_frame_hold_flag():
    args = _build_parser().parse_args(["--hold-viewer-open"])

    assert args.hold_viewer_open is True


def test_b4_final_frame_hold_disabled_does_not_poll_or_wait():
    viewer = _FakeHoldingViewer([True])
    sleep_calls = []

    _hold_final_viewer_frame(
        viewer,
        enabled=False,
        sleep_fn=sleep_calls.append,
    )

    assert viewer.is_running_calls == 0
    assert viewer.sync_calls == 0
    assert sleep_calls == []


def test_b4_final_frame_hold_refreshes_until_closed_without_stepping(
    monkeypatch,
):
    viewer = _FakeHoldingViewer([True, True, False])
    sleep_calls = []

    def fail_if_stepped(*args, **kwargs):
        raise AssertionError("mj_step must not run while holding final frame")

    monkeypatch.setattr(
        scaled_stablesoft_passive_demo.mujoco,
        "mj_step",
        fail_if_stepped,
    )

    _hold_final_viewer_frame(
        viewer,
        enabled=True,
        sleep_fn=sleep_calls.append,
    )

    assert viewer.is_running_calls == 3
    assert viewer.sync_calls == 2
    assert sleep_calls == [
        _VIEWER_HOLD_REFRESH_SECONDS,
        _VIEWER_HOLD_REFRESH_SECONDS,
    ]


def test_b4_final_frame_hold_stops_cleanly_on_keyboard_interrupt():
    viewer = _FakeHoldingViewer([True], sync_error=KeyboardInterrupt())
    sleep_calls = []

    _hold_final_viewer_frame(
        viewer,
        enabled=True,
        sleep_fn=sleep_calls.append,
    )

    assert viewer.is_running_calls == 1
    assert viewer.sync_calls == 1
    assert sleep_calls == []


def test_b4_final_frame_hold_without_viewer_does_not_wait():
    sleep_calls = []

    _hold_final_viewer_frame(
        None,
        enabled=True,
        sleep_fn=sleep_calls.append,
    )

    assert sleep_calls == []


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
