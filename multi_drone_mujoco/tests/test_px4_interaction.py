"""Headless contract and physics tests for P11 interaction."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import pytest

from multi_drone_mujoco.backends.interaction import (
    ArmInteractionCatalog,
    DeterministicInteractionTimeline,
    ForceInteractionConfig,
    InteractionCommand,
    InteractionFrame,
    InteractionMode,
    KeyboardInteractionState,
    JointTorqueInteractionState,
    MouseForceInteractor,
    MujocoArmForceApplier,
    MujocoArmInteractionApplier,
    TimedInteractionEvent,
)
from multi_drone_mujoco.backends.px4_interactive_viewer import (
    Px4InteractionViewer,
)


def _command(**overrides):
    values = dict(
        timestamp_us=1000,
        sequence=1,
        mode=InteractionMode.EXTERNAL_FORCE,
        target_body="tip",
        target_site="tip_site",
        frame=InteractionFrame.WORLD_ENU,
        force_n=(0.002, 0.0, 0.0),
        point_world_m=(0.0, 0.0, -0.1),
        active=True,
        source="test",
    )
    values.update(overrides)
    return InteractionCommand(**values)


def test_interaction_command_is_typed_complete_and_serializable():
    command = _command(clipped=True)
    payload = command.as_dict()
    assert payload["timestamp_us"] == 1000
    assert payload["sequence"] == 1
    assert payload["mode"] == "external_force"
    assert payload["frame"] == "world_enu"
    assert payload["target_body"] == "tip"
    assert payload["target_site"] == "tip_site"
    assert payload["force_n"] == (0.002, 0.0, 0.0)
    assert payload["active"] and payload["clipped"]


def test_command_rejects_mixed_force_and_tendon_semantics():
    with pytest.raises(ValueError, match="cannot contain tendon"):
        _command(tendon_acceleration_m_s2=(0.01, 0.0))
    with pytest.raises(ValueError, match="cannot contain an external force"):
        _command(
            mode=InteractionMode.TENDON_MODAL,
            frame=InteractionFrame.TENDON_MODAL,
            force_n=(0.001, 0.0, 0.0),
        )


def test_mouse_virtual_spring_damper_clips_and_releases_exactly():
    interactor = MouseForceInteractor(ForceInteractionConfig(
        max_force_n=0.005,
        spring_n_per_m=1.0,
        damping_n_s_per_m=0.1,
    ))
    interactor.begin((0.0, 0.0, 0.0))
    interactor.drag_to((0.1, 0.0, 0.0))
    force, clipped = interactor.calculate((0.0, 0.0, 0.0), (0.01, 0.0, 0.0))
    assert clipped
    np.testing.assert_allclose(force, (0.005, 0.0, 0.0), atol=1e-15)
    interactor.release()
    force, clipped = interactor.calculate((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert force == (0.0, 0.0, 0.0)
    assert not clipped


def test_keyboard_deadman_axes_diagonal_reverse_and_zero():
    state = KeyboardInteractionState(mode=InteractionMode.EXTERNAL_FORCE)
    state.set_pressed("x_pos", True)
    assert state.vector()[0] == pytest.approx((0.002, 0.0))
    state.set_pressed("y_neg", True)
    assert np.linalg.norm(state.vector()[0]) == pytest.approx(0.002)
    state.set_pressed("x_pos", False)
    assert state.vector()[0] == pytest.approx((0.0, -0.002))
    state.set_pressed("y_neg", False)
    assert state.vector()[0] == (0.0, 0.0)
    state.set_pressed("x_neg", True)
    assert state.vector()[0][0] < 0.0
    state.zero()
    assert state.vector()[0] == (0.0, 0.0)


def test_keyboard_force_adjustment_is_hard_bounded_and_emergency_latched():
    state = KeyboardInteractionState(
        mode=InteractionMode.EXTERNAL_FORCE,
        force_step_n=0.002,
        force_limit_n=0.005,
    )
    state.adjust_force_magnitude(20)
    state.set_pressed("x_pos", True)
    assert state.vector()[0] == pytest.approx((0.005, 0.0))
    state.emergency_stop()
    state.set_pressed("x_pos", True)
    assert state.vector()[0] == (0.0, 0.0)
    state.clear_emergency_stop()
    state.set_pressed("x_pos", True)
    assert state.vector()[0] == pytest.approx((0.005, 0.0))


def test_tendon_keyboard_uses_two_modal_axes_and_frozen_acceleration_limit():
    state = KeyboardInteractionState(
        mode=InteractionMode.TENDON_MODAL,
        tendon_step_m_s2=0.04,
        tendon_limit_m_s2=0.04,
    )
    state.set_pressed("x_pos", True)
    state.set_pressed("y_pos", True)
    vector, clipped = state.vector()
    assert np.linalg.norm(vector) == pytest.approx(0.04)
    assert not clipped
    state.set_pressed("x_pos", False)
    state.set_pressed("x_neg", True)
    assert state.vector()[0][0] < 0.0


def test_timeline_is_simulation_time_deterministic_and_release_is_one_sample(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"events": [{
        "start_s": 1.0,
        "duration_s": 0.2,
        "mode": "external_force",
        "force_n": [0.002, 0.0, 0.0],
    }]}))
    timeline = DeterministicInteractionTimeline.from_json(path)
    kwargs = dict(target_body="tip", target_site="site", point_world_m=(0, 0, 0))
    assert not timeline.sample(0.999, **kwargs).active
    assert timeline.sample(1.0, **kwargs).active
    assert timeline.sample(1.199999, **kwargs).active
    released = timeline.sample(1.2, **kwargs)
    assert not released.active and released.released
    assert not timeline.sample(1.3, **kwargs).released
    timeline.reset()
    assert timeline.sample(1.0, **kwargs).sequence == 1


def test_timeline_rejects_overlap():
    with pytest.raises(ValueError, match="must not overlap"):
        DeterministicInteractionTimeline((
            TimedInteractionEvent(1.0, 0.5, InteractionMode.EXTERNAL_FORCE, (0.002, 0, 0)),
            TimedInteractionEvent(1.4, 0.2, InteractionMode.EXTERNAL_FORCE, (0, 0.002, 0)),
        ))


def test_point_force_generates_equivalent_torque_and_clear_prevents_residue():
    xml = """
    <mujoco><option gravity="0 0 0" timestep="0.001"/>
      <worldbody><body name="tip" pos="0 0 0">
        <freejoint/><geom type="sphere" size="0.01" mass="0.001"/>
        <site name="tip_site" pos="0 0 -0.1"/>
      </body></worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    applier = MujocoArmForceApplier(model, data, "tip", "tip_site")
    command = _command()
    applier.apply(command)
    np.testing.assert_allclose(data.xfrc_applied[applier.body_id, :3], command.force_n)
    expected_torque = np.cross(
        np.asarray(command.point_world_m) - data.xipos[applier.body_id],
        np.asarray(command.force_n),
    )
    np.testing.assert_allclose(data.xfrc_applied[applier.body_id, 3:], expected_torque)
    mujoco.mj_step(model, data)
    applier.clear()
    np.testing.assert_array_equal(data.xfrc_applied[applier.body_id], np.zeros(6))
    before = data.qvel.copy()
    mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qvel, before, atol=1e-15)


def _arm_catalog_fixture():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><option gravity="0 0 0"/><worldbody>
        <body name="uav"><freejoint name="uav_free"/>
          <geom type="sphere" size="0.01" mass="0.02"/>
          <body name="arm_0" pos="0 0 -0.02">
            <geom type="sphere" size="0.005" mass="0.001"/>
            <site name="arm_0_site"/>
            <body name="arm_1" pos="0 0 -0.02">
              <joint name="arm_joint_1" type="ball" stiffness="0.003086"/>
              <geom type="sphere" size="0.005" mass="0.001"/>
              <site name="arm_1_site" pos="0.003 0 0"/>
              <body name="arm_2" pos="0 0 -0.02">
                <joint name="arm_joint_2" type="ball" stiffness="0.003086"/>
                <geom type="sphere" size="0.005" mass="0.001"/>
                <site name="arm_tip_site" pos="0.004 0 0"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    manifest = SimpleNamespace(
        segment_body_names=("arm_0", "arm_1", "arm_2"),
        ball_joint_names=("arm_joint_1", "arm_joint_2"),
        arm_tip_site_name="arm_tip_site",
    )
    return model, data, ArmInteractionCatalog.discover(model, manifest)


def test_manifest_catalog_discovers_only_arm_chain_sites_joints_and_ball_dofs():
    model, _, catalog = _arm_catalog_fixture()
    assert [item.body_name for item in catalog.segments] == ["arm_0", "arm_1", "arm_2"]
    assert [item.joint_name for item in catalog.joints] == ["arm_joint_1", "arm_joint_2"]
    assert catalog.segments[1].site_names == ("arm_1_site",)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "uav_free")
    free_dofs = set(range(int(model.jnt_dofadr[free_id]), int(model.jnt_dofadr[free_id]) + 6))
    assert free_dofs.isdisjoint(catalog.dof_indices)
    assert catalog.joints[0].dof_indices == (6, 7, 8)
    with pytest.raises(ValueError, match="not an arm segment"):
        catalog.segment_for_body("uav")
    with pytest.raises(ValueError, match="not an arm segment"):
        catalog.segment_for_body(0)


def test_dynamic_point_force_enriches_exact_com_torque_and_rejects_uav():
    model, data, catalog = _arm_catalog_fixture()
    applier = MujocoArmInteractionApplier(model, data, catalog)
    point = tuple(data.xpos[catalog.segments[1].body_id] + np.asarray((0.003, 0.002, 0)))
    command = _command(
        target_body="arm_1", target_site=None, point_world_m=point,
        force_n=(0.001, -0.0005, 0.0),
    )
    enriched = applier.enrich(command)
    expected = np.cross(np.asarray(point) - data.xipos[enriched.target_body_id], command.force_n)
    np.testing.assert_allclose(enriched.generated_torque_nm, expected)
    applier.apply(enriched)
    np.testing.assert_allclose(data.xfrc_applied[enriched.target_body_id, 3:], expected)
    applier.clear()
    np.testing.assert_array_equal(data.xfrc_applied[list(catalog.body_ids)], 0.0)
    with pytest.raises(ValueError, match="not an arm segment"):
        applier.enrich(_command(target_body="uav", target_site=None))


def test_joint_torque_diagnostic_maps_axis_bounds_pulses_and_clears_all_arm_dofs():
    model, data, catalog = _arm_catalog_fixture()
    applier = MujocoArmInteractionApplier(model, data, catalog)
    state = JointTorqueInteractionState(2, step_nm=10e-6, limit_nm=50e-6)
    state.cycle_joint(1)
    state.select_axis(2)
    state.adjust_magnitude(99)
    state.pulse(-1)
    torque = state.consume()
    assert torque == pytest.approx(-50e-6)
    assert state.consume() == 0.0
    joint = catalog.joints[state.joint_index]
    command = InteractionCommand(
        timestamp_us=1, sequence=1, mode=InteractionMode.JOINT_TORQUE,
        target_body=joint.body_name, target_site=None,
        target_joint=joint.joint_name, target_dof_axis=state.axis,
        frame=InteractionFrame.JOINT_DOF, joint_torque_nm=torque,
        active=True, source="test",
    )
    enriched = applier.enrich(command)
    assert enriched.target_dof_index == joint.dof_indices[2]
    applier.apply(enriched)
    assert data.qfrc_applied[enriched.target_dof_index] == pytest.approx(-50e-6)
    applier.clear()
    np.testing.assert_array_equal(data.qfrc_applied[list(catalog.dof_indices)], 0.0)
    np.testing.assert_array_equal(data.qfrc_applied[:6], 0.0)


class _FakePassiveViewer:
    def __init__(self, data, body_id, native_force=(0.0, 0.0, 0.0)):
        self.data = data
        self.body_id = body_id
        self.native_force = np.asarray(native_force, dtype=float)
        self.perturb = SimpleNamespace(select=body_id)
        self.running = True
        self.close_calls = 0
        self.overlays = []

    def is_running(self):
        return self.running

    def sync(self):
        self.data.xfrc_applied[self.body_id, :3] = self.native_force

    def set_texts(self, texts):
        self.overlays.append(texts)

    def close(self):
        self.close_calls += 1
        self.running = False


def _viewer_fixture(monkeypatch, mode, native_force=(0.0, 0.0, 0.0)):
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body name="tip"><freejoint/>
        <geom type="sphere" size="0.01" mass="0.001"/>
        <site name="tip_site" pos="0 0 -0.1"/>
      </body></worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    applier = MujocoArmForceApplier(model, data, "tip", "tip_site")
    fake = _FakePassiveViewer(data, applier.body_id, native_force)
    callback = {}

    def launch(*args, **kwargs):
        callback["key"] = kwargs["key_callback"]
        return fake

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    config = SimpleNamespace(
        interaction_mode=mode,
        interaction_force_step_n=0.002,
        interaction_max_force_n=0.005,
        tendon_acceleration_step_m_s2=0.005,
        tendon_acceleration_limit_m_s2=0.04,
        interaction_overlay=True,
    )
    aviary = SimpleNamespace(model=model, data=data)
    adapter = Px4InteractionViewer(aviary, applier, config)
    return adapter, fake, callback


def test_viewer_accepts_only_target_mouse_force_clamps_then_clears_native(monkeypatch):
    adapter, fake, _ = _viewer_fixture(
        monkeypatch, "mouse_force", native_force=(0.020, 0.0, 0.0)
    )
    command = adapter.poll(1.0)
    assert command.active and command.clipped
    assert command.force_n == pytest.approx((0.005, 0.0, 0.0))
    np.testing.assert_array_equal(fake.data.xfrc_applied, np.zeros_like(fake.data.xfrc_applied))
    fake.perturb.select = 0
    released = adapter.poll(1.001)
    assert not released.active and released.released
    adapter.close()
    adapter.close()
    assert fake.close_calls == 1


def test_viewer_keyboard_press_is_one_step_deadman_pulse_and_pause_steps(monkeypatch):
    adapter, _, callback = _viewer_fixture(monkeypatch, "keyboard_force")
    callback["key"](glfw.KEY_W)
    command = adapter.poll(2.0)
    assert command.active and command.force_n == pytest.approx((0.002, 0.0, 0.0))
    adapter.after_step()
    released = adapter.poll(2.001)
    assert not released.active and released.released
    callback["key"](glfw.KEY_SPACE)
    adapter.poll(2.001)
    assert adapter.paused and not adapter.consume_advance_permission()
    callback["key"](glfw.KEY_PERIOD)
    adapter.poll(2.001)
    assert adapter.consume_advance_permission()
    assert not adapter.consume_advance_permission()
    adapter.close()


def test_viewer_switches_arm_segments_only_after_drag_release_and_uses_pick_point(
    monkeypatch,
):
    model, data, catalog = _arm_catalog_fixture()
    applier = MujocoArmInteractionApplier(
        model, data, catalog, default_site_name="arm_tip_site"
    )
    first, middle, tip = catalog.segments

    class SegmentViewer(_FakePassiveViewer):
        def __init__(self):
            super().__init__(data, first.body_id)
            self.force_body = first.body_id
            self.native_force = np.asarray((0.001, 0.0, 0.0))
            self.perturb = SimpleNamespace(
                select=first.body_id, localpos=np.asarray((0.002, 0.003, 0.0))
            )

        def sync(self):
            self.data.xfrc_applied[self.force_body, :3] = self.native_force

    fake = SegmentViewer()
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *a, **k: fake)
    config = SimpleNamespace(
        interaction_mode="mouse_force", interaction_force_step_n=0.0005,
        interaction_max_force_n=0.001, tendon_acceleration_step_m_s2=0.005,
        tendon_acceleration_limit_m_s2=0.04, interaction_overlay=False,
    )
    adapter = Px4InteractionViewer(SimpleNamespace(model=model, data=data), applier, config)
    first_command = adapter.poll(1.0)
    assert first_command.target_body == first.body_name
    expected_point = data.xpos[first.body_id] + data.xmat[first.body_id].reshape(3, 3) @ fake.perturb.localpos
    np.testing.assert_allclose(first_command.point_world_m, expected_point)

    # Selection changes during a nonzero drag, but the force remains locked.
    fake.perturb.select = middle.body_id
    locked = adapter.poll(1.001)
    assert locked.target_body == first.body_name

    # Release accepts the pending selection; the next drag targets the middle.
    fake.native_force[:] = 0.0
    released = adapter.poll(1.002)
    assert released.released and not released.active
    fake.force_body = middle.body_id
    fake.native_force[:] = (0.0005, 0.0, 0.0)
    switched = adapter.poll(1.003)
    assert switched.target_body == middle.body_name

    # UAV selection is rejected and cannot replace the selected arm body.
    fake.native_force[:] = 0.0
    fake.perturb.select = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "uav")
    adapter.poll(1.004)
    assert "rejected non-arm" in adapter._selection_rejection
    fake.perturb.select = tip.body_id
    adapter.poll(1.005)
    assert adapter._selected_body_id == tip.body_id
    adapter.close()
    np.testing.assert_array_equal(data.xfrc_applied, 0.0)
    np.testing.assert_array_equal(data.qfrc_applied, 0.0)


def test_viewer_joint_mode_selects_parent_axis_pulses_and_estop_clears(monkeypatch):
    model, data, catalog = _arm_catalog_fixture()
    applier = MujocoArmInteractionApplier(model, data, catalog)
    target = catalog.segments[1]
    fake = _FakePassiveViewer(data, target.body_id)
    callback = {}

    def launch(*args, **kwargs):
        callback["key"] = kwargs["key_callback"]
        return fake

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    config = SimpleNamespace(
        interaction_mode="joint_torque", interaction_force_step_n=0.0005,
        interaction_max_force_n=0.001, tendon_acceleration_step_m_s2=0.005,
        tendon_acceleration_limit_m_s2=0.04, interaction_overlay=False,
        joint_torque_step_nm=10e-6, joint_torque_limit_nm=50e-6,
    )
    adapter = Px4InteractionViewer(SimpleNamespace(model=model, data=data), applier, config)
    callback["key"](glfw.KEY_2)
    callback["key"](glfw.KEY_W)
    command = adapter.poll(1.0)
    assert command.mode is InteractionMode.JOINT_TORQUE
    assert command.target_joint == catalog.joints[0].joint_name
    assert command.target_dof_axis == 1
    assert command.target_dof_index == catalog.joints[0].dof_indices[1]
    assert command.joint_torque_nm == pytest.approx(10e-6)
    applier.apply(command)
    callback["key"](glfw.KEY_BACKSPACE)
    stopped = adapter.poll(1.001)
    assert not stopped.active
    applier.clear()
    np.testing.assert_array_equal(data.qfrc_applied, 0.0)
    adapter.close()


def test_viewer_overlay_includes_px4_pose_actuator_and_frequency_state(monkeypatch):
    adapter, fake, _ = _viewer_fixture(monkeypatch, "keyboard_force")
    adapter.config.target_position_enu_m = (0.0, 0.0, 1.0)
    adapter.config.sim_freq_hz = 2400
    adapter.config.plant_step_freq_hz = 200
    adapter.config.control_freq_hz = 200
    adapter.aviary.pos = np.asarray([[0.0, 0.0, 1.0]])
    adapter.aviary.rpy = np.zeros((1, 3))
    backend = SimpleNamespace(
        transport=SimpleNamespace(
            health=SimpleNamespace(state=SimpleNamespace(value="connected"))
        ),
        pwm_to_rotor_speed=SimpleNamespace(
            config=SimpleNamespace(rpm_max=20_000.0)
        ),
        last_normalized_command=SimpleNamespace(values=(0.1, 0.2, 0.3, 0.4)),
        last_pwm_command=SimpleNamespace(
            pulse_width_us=(1100.0, 1200.0, 1300.0, 1400.0)
        ),
    )
    adapter.status_source = SimpleNamespace(
        backend=backend,
        flight_control=None,
        _last_rpm=np.asarray((10_000.0,) * 4),
        _final_actuator_armed=True,
        _tendon_controller=None,
    )
    adapter.poll(1.0)
    labels = fake.overlays[-1][0][2]
    values = fake.overlays[-1][0][3]
    assert "PX4 link / armed" in labels
    assert "normalized" in labels and "PWM us" in labels and "RPM" in labels
    assert "connected; armed=True" in values
    assert "sim/plant/PX4/tendon=2400/200/200/240 Hz" in values
    adapter.close()


@pytest.mark.parametrize(
    ("preset", "mode"),
    (("mouse", "mouse_force"), ("tendon", "tendon_modal"),
     ("keyboard-force", "keyboard_force"), ("joint", "joint_torque")),
)
def test_short_viewer_launcher_dry_run_presets_and_safe_passthrough(
    tmp_path, preset, mode
):
    root = Path(__file__).resolve().parents[2]
    autopilot = tmp_path / "PX4 checkout with spaces"
    autopilot.mkdir()
    env = {
        "PATH": "/usr/bin:/bin",
        "MUJOCO_UAV_PYTHON": "/bin/true",
        "PX4_AUTOPILOT_DIR": str(autopilot),
        "PX4_VIEWER_LOG_DIR": str(tmp_path / "logs with spaces"),
    }
    result = subprocess.run(
        [
            "bash", str(root / "scripts/run_px4_viewer.sh"), preset,
            "--dry-run", "--", "--duration", "17", "--log-dir", "a b",
        ],
        cwd="/", env=env, text=True, capture_output=True, check=True,
    )
    assert f"--interaction-mode {mode}" in result.stdout
    assert "--duration 17" in result.stdout
    assert "a\\ b" in result.stdout
    assert "PX4\\ checkout\\ with\\ spaces" in result.stdout


def test_short_viewer_launcher_is_valid_bash():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["bash", "-n", str(root / "scripts/run_px4_viewer.sh")], check=True
    )
