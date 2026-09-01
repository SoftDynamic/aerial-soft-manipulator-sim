"""Optional MuJoCo passive-viewer adapter for P11 interaction.

This module is imported only for an explicitly requested interactive viewer,
so headless and ordinary environment imports do not depend on GLFW/GLX.
MuJoCo's native selection/perturb UI supplies mouse intent; the adapter clears
all native wrenches, accepts only the configured terminal body, hard-clamps
the force, and returns an ordinary :class:`InteractionCommand`.
"""

from __future__ import annotations

import queue
import time

import numpy as np

from multi_drone_mujoco.backends.interaction import (
    InteractionCommand,
    InteractionFrame,
    InteractionMode,
    JointTorqueInteractionState,
    KeyboardInteractionState,
    clamp_norm,
)


class Px4InteractionViewer:
    """Small input/render adapter; it never advances physics."""

    def __init__(self, aviary, force_applier, config, status_source=None):
        import glfw
        import mujoco
        import mujoco.viewer

        self._glfw = glfw
        self._mujoco = mujoco
        self.aviary = aviary
        self.force_applier = force_applier
        self.config = config
        self.status_source = status_source
        self._events = queue.SimpleQueue()
        self._viewer = mujoco.viewer.launch_passive(
            aviary.model,
            aviary.data,
            key_callback=lambda keycode: self._events.put(int(keycode)),
        )
        self._closed = False
        self._sequence = 0
        self.paused = False
        self._single_step = False
        self.exit_requested = False
        self.emergency_stopped = False
        mode = (
            InteractionMode.TENDON_MODAL
            if config.interaction_mode == "tendon_modal"
            else InteractionMode.EXTERNAL_FORCE
        )
        self.keyboard = KeyboardInteractionState(
            mode=mode,
            force_step_n=config.interaction_force_step_n,
            force_limit_n=config.interaction_max_force_n,
            tendon_step_m_s2=config.tendon_acceleration_step_m_s2,
            tendon_limit_m_s2=config.tendon_acceleration_limit_m_s2,
        )
        self.catalog = getattr(force_applier, "catalog", None)
        self.allowed_body_ids = tuple(getattr(
            force_applier, "allowed_body_ids", (force_applier.body_id,)
        ))
        self._selected_body_id = force_applier.body_id
        self._last_viewer_select = 0
        self._drag_body_id = None
        self._drag_localpos = None
        self._selection_rejection = None
        self.joint_keyboard = None
        self._joint_target_enabled = False
        if config.interaction_mode == "joint_torque":
            if self.catalog is None or not self.catalog.joints:
                raise ValueError("joint_torque viewer requires an arm joint catalog")
            self.joint_keyboard = JointTorqueInteractionState(
                len(self.catalog.joints),
                step_nm=config.joint_torque_step_nm,
                limit_nm=config.joint_torque_limit_nm,
            )
            self._joint_target_enabled = True
        self._last_command = None
        self._released_pending = False

    @property
    def is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def consume_advance_permission(self) -> bool:
        if not self.paused:
            return True
        if self._single_step:
            self._single_step = False
            return True
        return False

    def _drain_keys(self) -> None:
        glfw = self._glfw
        while True:
            try:
                key = self._events.get_nowait()
            except queue.Empty:
                break
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_L:
                self.exit_requested = True
            elif key == glfw.KEY_SPACE:
                self.paused = not self.paused
                self._single_step = False
            elif key == glfw.KEY_PERIOD and self.paused:
                self._single_step = True
            elif key == glfw.KEY_BACKSPACE:
                self.keyboard.emergency_stop()
                if self.joint_keyboard is not None:
                    self.joint_keyboard.emergency_stop()
                self.emergency_stopped = True
                self._released_pending = True
            elif key in (glfw.KEY_F8, glfw.KEY_F9):
                self.keyboard.zero()
                if self.joint_keyboard is not None:
                    self.joint_keyboard.zero()
                self._released_pending = True
            elif key == glfw.KEY_UP:
                if self.joint_keyboard is not None:
                    self.joint_keyboard.adjust_magnitude(1)
                else:
                    self.keyboard.adjust_force_magnitude(1)
            elif key == glfw.KEY_DOWN:
                if self.joint_keyboard is not None:
                    self.joint_keyboard.adjust_magnitude(-1)
                else:
                    self.keyboard.adjust_force_magnitude(-1)
            elif self.joint_keyboard is not None:
                if key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
                    self.joint_keyboard.select_axis(key - glfw.KEY_1)
                elif key == glfw.KEY_A:
                    self.joint_keyboard.cycle_joint(-1)
                    self._joint_target_enabled = True
                elif key == glfw.KEY_D:
                    self.joint_keyboard.cycle_joint(1)
                    self._joint_target_enabled = True
                elif key == glfw.KEY_W:
                    self.joint_keyboard.pulse(1)
                elif key == glfw.KEY_S:
                    self.joint_keyboard.pulse(-1)
            else:
                directions = {
                    glfw.KEY_W: "x_pos",
                    glfw.KEY_S: "x_neg",
                    glfw.KEY_A: "y_pos",
                    glfw.KEY_D: "y_neg",
                }
                direction = directions.get(key)
                if direction is not None:
                    self.keyboard.set_pressed(direction, True)

    def _body_name(self, body_id: int):
        if body_id is None:
            return None
        if self.catalog is not None:
            return self.catalog.segment_for_body(body_id).body_name
        return self.force_applier.body_name

    def _site_name(self, body_id: int):
        if body_id is None:
            return None
        if body_id == self.force_applier.body_id:
            return self.force_applier.site_name
        return None

    def _update_selection(self, selected: int) -> None:
        if self._drag_body_id is not None:
            return
        if selected in self.allowed_body_ids:
            self._selected_body_id = selected
            self._selection_rejection = None
            if self.joint_keyboard is not None:
                joint = self.catalog.joint_for_body(selected)
                if joint is None:
                    self._joint_target_enabled = False
                    self._selection_rejection = (
                        f"segment {self._body_name(selected)} has no internal joint"
                    )
                else:
                    self.joint_keyboard.select_joint(joint.index)
                    self._joint_target_enabled = True
        elif selected == 0 and self._last_viewer_select in self.allowed_body_ids:
            self._selected_body_id = None
            self._joint_target_enabled = False
            self._selection_rejection = "selection cancelled on background"
        elif selected != 0:
            self._selection_rejection = f"rejected non-arm body id {selected}"
        self._last_viewer_select = selected

    def _picked_point(self, body_id: int):
        localpos = getattr(self._viewer.perturb, "localpos", None)
        if localpos is None:
            return np.asarray(self.force_applier.point_world_m, dtype=float), None
        localpos = np.asarray(localpos, dtype=float).copy()
        rotation = np.asarray(self.aviary.data.xmat[body_id], dtype=float).reshape(3, 3)
        point = np.asarray(self.aviary.data.xpos[body_id], dtype=float) + rotation @ localpos
        return point, localpos

    def _locked_point(self):
        if self._drag_body_id is None or self._drag_localpos is None:
            return np.asarray(self.force_applier.point_world_m, dtype=float)
        rotation = np.asarray(
            self.aviary.data.xmat[self._drag_body_id], dtype=float
        ).reshape(3, 3)
        return (
            np.asarray(self.aviary.data.xpos[self._drag_body_id], dtype=float)
            + rotation @ self._drag_localpos
        )

    def poll(self, simulation_time_s: float) -> InteractionCommand:
        native = np.zeros_like(self.aviary.data.xfrc_applied)
        if not self.is_running:
            self.exit_requested = True
        else:
            self._viewer.sync()
            native = self.aviary.data.xfrc_applied.copy()
        # Native perturbations are input intent only.  Clear every body in all
        # interaction modes so the viewer can never bypass target filtering,
        # clipping, pause, timeout, or the explicit point-force adapter.
        self.aviary.data.xfrc_applied[:] = 0.0
        self._drain_keys()
        selected = int(self._viewer.perturb.select) if self.is_running else 0
        self._update_selection(selected)
        self._sequence += 1
        timestamp_us = round(simulation_time_s * 1_000_000)
        released = self._released_pending
        self._released_pending = False
        command = None
        if self.config.interaction_mode == "mouse_force" and self.is_running:
            candidate = (
                self._drag_body_id
                if self._drag_body_id is not None else self._selected_body_id
            )
            raw_force = (
                np.zeros(3, dtype=float)
                if candidate is None else native[candidate, :3]
            )
            if selected == 0:
                raw_force = np.zeros(3, dtype=float)
            dragging = candidate in self.allowed_body_ids and np.linalg.norm(raw_force) > 0.0
            if dragging:
                if self._drag_body_id is None:
                    self._drag_body_id = candidate
                    _, self._drag_localpos = self._picked_point(candidate)
                force, clipped = clamp_norm(
                    raw_force, self.config.interaction_max_force_n
                )
                point = self._locked_point()
                command = InteractionCommand(
                    timestamp_us=timestamp_us,
                    sequence=self._sequence,
                    mode=InteractionMode.EXTERNAL_FORCE,
                    target_body=self._body_name(self._drag_body_id),
                    target_site=self._site_name(self._drag_body_id),
                    frame=InteractionFrame.WORLD_ENU,
                    force_n=tuple(float(value) for value in force),
                    point_world_m=tuple(float(value) for value in point),
                    active=True,
                    clipped=clipped,
                    source="mouse_perturb",
                )
            elif self._drag_body_id is not None:
                self._drag_body_id = None
                self._drag_localpos = None
                released = True
                self._update_selection(selected)
        elif self.config.interaction_mode in ("keyboard_force", "tendon_modal"):
            vector, clipped = self.keyboard.vector()
            active = bool(np.linalg.norm(vector)) and not self.emergency_stopped
            if (
                self.config.interaction_mode == "keyboard_force"
                and active and self._selected_body_id is not None
            ):
                command = InteractionCommand(
                    timestamp_us=timestamp_us,
                    sequence=self._sequence,
                    mode=InteractionMode.EXTERNAL_FORCE,
                    target_body=self._body_name(self._selected_body_id),
                    target_site=self._site_name(self._selected_body_id),
                    frame=InteractionFrame.WORLD_ENU,
                    force_n=(vector[0], vector[1], 0.0),
                    point_world_m=tuple(
                        float(value) for value in self.aviary.data.xpos[
                            self._selected_body_id
                        ]
                    ),
                    active=True,
                    clipped=clipped,
                    source="keyboard_deadman_pulse",
                )
            elif self.config.interaction_mode == "tendon_modal" and active:
                command = InteractionCommand(
                    timestamp_us=timestamp_us,
                    sequence=self._sequence,
                    mode=InteractionMode.TENDON_MODAL,
                    target_body=self.force_applier.body_name,
                    target_site=self.force_applier.site_name,
                    target_body_id=self.force_applier.body_id,
                    frame=InteractionFrame.TENDON_MODAL,
                    tendon_acceleration_m_s2=vector,
                    active=True,
                    clipped=clipped,
                    source="keyboard_deadman_pulse",
                )
        elif self.config.interaction_mode == "joint_torque":
            torque = self.joint_keyboard.consume()
            joint = self.catalog.joints[self.joint_keyboard.joint_index]
            if torque and not self.emergency_stopped and self._joint_target_enabled:
                command = InteractionCommand(
                    timestamp_us=timestamp_us,
                    sequence=self._sequence,
                    mode=InteractionMode.JOINT_TORQUE,
                    target_body=joint.body_name,
                    target_site=None,
                    target_joint=joint.joint_name,
                    target_dof_axis=self.joint_keyboard.axis,
                    frame=InteractionFrame.JOINT_DOF,
                    joint_torque_nm=torque,
                    active=True,
                    source="keyboard_joint_diagnostic_pulse",
                )
        if command is None:
            command = InteractionCommand(
                timestamp_us=timestamp_us,
                sequence=self._sequence,
                mode=InteractionMode.DISABLED,
                target_body=self._body_name(self._selected_body_id),
                target_site=self._site_name(self._selected_body_id),
                target_body_id=self._selected_body_id,
                frame=InteractionFrame.WORLD_ENU,
                released=released or bool(
                    self._last_command is not None and self._last_command.active
                ),
                source="viewer",
            )
        enrich = getattr(self.force_applier, "enrich", None)
        if callable(enrich):
            command = enrich(command)
        self._last_command = command
        self._update_overlay(command)
        return command

    def after_step(self) -> None:
        # launch_passive exposes presses but not releases.  A press is one
        # simulation-step pulse; OS key repeat refreshes it while held.  Thus a
        # stopped repeat can never leave a persistent wrench.
        if self.keyboard.vector()[0] != (0.0, 0.0):
            self.keyboard.zero()
            self._released_pending = True

    def _update_overlay(self, command: InteractionCommand) -> None:
        if not self.config.interaction_overlay or not self.is_running:
            return
        set_texts = getattr(self._viewer, "set_texts", None)
        if not callable(set_texts):
            return
        force_mn = 1000.0 * np.asarray(command.force_n)
        tendon = command.tendon_acceleration_m_s2
        point = command.point_world_m or self.force_applier.point_world_m
        target_name = (
            command.target_body or self._body_name(self._selected_body_id) or "none"
        )
        association = "joint=-; sites=-"
        if self.catalog is not None and target_name != "none":
            segment = self.catalog.segment_for_body(target_name)
            parent_joint = self.catalog.joint_for_body(segment.body_id)
            association = (
                f"joint={parent_joint.joint_name if parent_joint else 'welded-root'}; "
                f"sites={','.join(segment.site_names) if segment.site_names else '-'}"
            )
        generated_torque_unm = 1e6 * np.asarray(command.generated_torque_nm)
        joint_qvel = (
            None if command.target_dof_index is None
            else float(self.aviary.data.qvel[command.target_dof_index])
        )
        joint_qvel_text = "-" if joint_qvel is None else f"{joint_qvel:+.5f} rad/s"
        rejection = self._selection_rejection or "none"
        status = "PAUSED" if self.paused else "RUNNING"
        if self.emergency_stopped:
            status = "INTERACTION E-STOP"
        source = self.status_source
        if source is None:
            values = (
                f"{status}\n{self.config.interaction_mode}\n"
                f"{target_name} (id={command.target_body_id})\n"
                f"{association}\n"
                f"{force_mn[0]:+.2f} {force_mn[1]:+.2f} {force_mn[2]:+.2f}\n"
                f"{tendon[0]:+.4f} {tendon[1]:+.4f}\n"
                f"{command.target_joint or '-'} axis={command.target_dof_axis}; "
                f"{1e6 * command.joint_torque_nm:+.1f} uN m; {joint_qvel_text}\n"
                f"{rejection}\n"
                f"{'YES' if command.clipped else 'no'}"
            )
            labels = (
                "PX4 interaction\nmode\ntarget\nassociated joint/sites\nforce mN ENU\n"
                "tendon accel m/s2\njoint diagnostic\nselection\nclipped"
            )
        else:
            backend = source.backend
            transport = backend.transport.health
            flight = (
                source.flight_control.health
                if source.flight_control is not None else None
            )
            position = np.asarray(self.aviary.pos[0], dtype=float)
            target = np.asarray(self.config.target_position_enu_m, dtype=float)
            rpy = np.asarray(self.aviary.rpy[0], dtype=float)
            normalized = backend.last_normalized_command
            pwm = backend.last_pwm_command
            normalized_text = (
                "-" if normalized is None else
                " ".join(f"{value:+.3f}" for value in normalized.values)
            )
            pwm_text = (
                "-" if pwm is None else
                " ".join(f"{value:.0f}" for value in pwm.pulse_width_us)
            )
            rpm = np.asarray(source._last_rpm, dtype=float)
            calibration = source.backend.pwm_to_rotor_speed.config
            saturated = bool(np.any(
                rpm >= calibration.rpm_max * (1.0 - 1e-9)
            ))
            tendon_state = (
                source._tendon_controller.state
                if source._tendon_controller is not None else None
            )
            tendon_text = (
                "-" if tendon_state is None else
                "q="
                + ",".join(
                    f"{1e6 * value:+.1f}"
                    for value in tendon_state.reference.bending_contraction_reference_m
                )
                + " um; T="
                + ",".join(
                    f"{1e3 * value:.2f}"
                    for value in tendon_state.tension_command_n
                )
                + " mN"
            )
            estimator = (
                "n/a" if flight is None else
                f"health={flight.estimator_health_flags} "
                f"timeout={flight.estimator_timeout_flags}"
            )
            values = (
                f"{status}; {self.config.interaction_mode}\n"
                f"{transport.state.value}; armed={source._final_actuator_armed}\n"
                f"{estimator}\n"
                f"{position[0]:+.3f} {position[1]:+.3f} {position[2]:+.3f}; "
                f"err={np.linalg.norm(position-target):.3f}\n"
                f"{np.degrees(rpy[0]):+.2f} {np.degrees(rpy[1]):+.2f} "
                f"{np.degrees(rpy[2]):+.2f}; tilt={np.degrees(np.linalg.norm(rpy[:2])):.2f}\n"
                f"{target_name} (id={command.target_body_id})\n"
                f"{association}\n"
                f"{force_mn[0]:+.2f} {force_mn[1]:+.2f} {force_mn[2]:+.2f}; "
                f"p={point[0]:+.3f},{point[1]:+.3f},{point[2]:+.3f}\n"
                f"tau@COM={generated_torque_unm[0]:+.1f},"
                f"{generated_torque_unm[1]:+.1f},"
                f"{generated_torque_unm[2]:+.1f} uN m\n"
                f"DIAGNOSTIC JOINT TORQUE (qfrc_applied): {command.target_joint or '-'} "
                f"axis={command.target_dof_axis} dof={command.target_dof_index}; "
                f"{1e6 * command.joint_torque_nm:+.1f} uN m; {joint_qvel_text}\n"
                f"selection: {rejection}\n"
                f"{tendon[0]:+.4f} {tendon[1]:+.4f}; {tendon_text}\n"
                f"{normalized_text}\n{pwm_text}\n"
                f"{' '.join(f'{value:.0f}' for value in rpm)}; sat={'YES' if saturated else 'no'}\n"
                f"sim/plant/PX4/tendon={self.config.sim_freq_hz}/"
                f"{self.config.plant_step_freq_hz}/{self.config.control_freq_hz}/240 Hz\n"
                f"{'YES' if command.clipped else 'no'}"
            )
            labels = (
                "status / interaction\nPX4 link / armed\nestimator\n"
                "position ENU / error m\nRPY / tilt deg\ntarget body\n"
                "associated joint/sites\n"
                "force mN / point m\ngenerated torque at COM\n"
                "DIAGNOSTIC JOINT TORQUE only\nselection filter\n"
                "modal accel / reference,tension\n"
                "normalized\nPWM us\nRPM / saturated\nfrequencies\nclipped"
            )
        set_texts([(
            self._mujoco.mjtFontScale.mjFONTSCALE_100,
            self._mujoco.mjtGridPos.mjGRID_TOPLEFT,
            labels,
            values,
        ), (
            self._mujoco.mjtFontScale.mjFONTSCALE_100,
            self._mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "keys",
            "Space pause | . step | Esc/L land+exit | Backspace e-stop\n"
            "force: W/S +/-X, A/D +/-Y | Up/Down magnitude | F8/F9 zero\n"
            "joint diagnostic: 1/2/3 axis, W/S +/-tau, A/D joint\n"
            "mouse: double-click target, Ctrl+right-drag translate",
        )])

    def close(self) -> None:
        if self._closed:
            return
        self.keyboard.emergency_stop()
        if self.joint_keyboard is not None:
            self.joint_keyboard.emergency_stop()
        self.force_applier.clear()
        self.aviary.data.xfrc_applied[:] = 0.0
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            # Give the native UI thread a chance to destroy its context before
            # the model/data owner closes.  X11/GLX failures remain visible.
            time.sleep(0.05)
        self._closed = True

    def render_only(self) -> None:
        if self.is_running:
            self._viewer.sync()
        self.force_applier.clear()
        self.aviary.data.xfrc_applied[:] = 0.0

    def reset(self) -> None:
        self.keyboard.zero()
        self.keyboard.clear_emergency_stop()
        if self.joint_keyboard is not None:
            self.joint_keyboard.zero()
            self.joint_keyboard.clear_emergency_stop()
        self.emergency_stopped = False
        self.paused = False
        self._single_step = False
        self._released_pending = False
        self._last_command = None
        self._drag_body_id = None
        self._drag_localpos = None
        self._selection_rejection = None
        self._last_viewer_select = 0
        self._selected_body_id = self.force_applier.body_id
        self._joint_target_enabled = self.joint_keyboard is not None
