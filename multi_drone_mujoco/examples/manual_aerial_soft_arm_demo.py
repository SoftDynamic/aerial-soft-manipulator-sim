"""Manually drive one CF2X and its active scaled StableSoft arm.

This is an interactive, open-loop demonstration.  It intentionally contains
no feedback controller, rotor mixer, or tendon allocation logic.  The MuJoCo
passive viewer remains responsible for camera control, body selection, and
mouse perturbations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import queue
import sys
import time
from typing import Optional

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.examples.scaled_stablesoft_passive_demo import (
    _closing_viewer,
    _spec,
)
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftVisualSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


# This is the fixed, already-characterized active C1 configuration.  It is a
# model parameter, not a command-line option.  The optional --tension-limit is
# only a stricter manual-command clamp and never changes this drive spec.
VERIFIED_ACTIVE_MAX_TENSION_N = 0.01
DEFAULT_SIM_FREQ = 2400
DEFAULT_CTRL_FREQ = 240
PAUSED_REFRESH_PERIOD_S = 1.0 / 60.0
OVERLAY_REFRESH_PERIOD_S = 1.0 / 20.0

GROUP_ROTOR = "rotor"
GROUP_TENDON = "tendon"

KEY_HELP = (
    "Space run/pause | . single-step (paused) | Enter rotor/tendon\n"
    "Up/Down select channel | Left/Right -/+ | F8 zero rotors | "
    "F9 zero tendons | Backspace reset\n"
    "Double-click select | Ctrl+drag rotate | Ctrl+right-drag translate | "
    "F1 MuJoCo help | Esc free camera"
)


@dataclass
class ManualCommandState:
    """Mutable open-loop command and interaction state."""

    rotor_rpm: np.ndarray
    tendon_tension: np.ndarray
    selected_group: str
    selected_channel: int
    paused: bool
    pending_single_step: bool
    initial_mode: str
    hover_rpm: float
    max_rpm: float
    tension_limit: float
    rpm_step: float
    tension_step: float
    notice: str = ""

    @classmethod
    def create(
        cls,
        *,
        initial_mode: str,
        hover_rpm: float,
        max_rpm: float,
        tension_limit: float,
        rpm_step: float,
        tension_step: float,
    ) -> "ManualCommandState":
        state = cls(
            rotor_rpm=np.zeros(4, dtype=float),
            tendon_tension=np.zeros(4, dtype=float),
            selected_group=GROUP_ROTOR,
            selected_channel=0,
            paused=True,
            pending_single_step=False,
            initial_mode=initial_mode,
            hover_rpm=float(hover_rpm),
            max_rpm=float(max_rpm),
            tension_limit=float(tension_limit),
            rpm_step=float(rpm_step),
            tension_step=float(tension_step),
        )
        state.reset()
        return state

    def reset(self) -> None:
        if self.initial_mode == "zero":
            self.rotor_rpm[:] = 0.0
        elif self.initial_mode == "hover":
            self.rotor_rpm[:] = self.hover_rpm
        else:
            raise ValueError(f"unsupported initial RPM mode: {self.initial_mode}")
        self.tendon_tension[:] = 0.0
        self.selected_group = GROUP_ROTOR
        self.selected_channel = 0
        self.paused = True
        self.pending_single_step = False
        self.notice = "reset to CLI initial state"
        self.clamp_commands()

    def clamp_commands(self) -> None:
        np.clip(self.rotor_rpm, 0.0, self.max_rpm, out=self.rotor_rpm)
        np.clip(
            self.tendon_tension,
            0.0,
            self.tension_limit,
            out=self.tendon_tension,
        )

    def select_relative(self, offset: int) -> None:
        self.selected_channel = (self.selected_channel + offset) % 4
        self.notice = self.selected_label

    def adjust_selected(self, direction: int) -> None:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if self.selected_group == GROUP_ROTOR:
            self.rotor_rpm[self.selected_channel] += direction * self.rpm_step
        else:
            self.tendon_tension[self.selected_channel] += (
                direction * self.tension_step
            )
        self.clamp_commands()
        self.notice = f"updated {self.selected_label}"

    def toggle_group(self) -> None:
        self.selected_group = (
            GROUP_TENDON
            if self.selected_group == GROUP_ROTOR
            else GROUP_ROTOR
        )
        self.notice = self.selected_label

    def zero_rotors(self) -> None:
        self.rotor_rpm[:] = 0.0
        self.notice = "all rotor RPM commands zeroed"

    def zero_tendons(self) -> None:
        self.tendon_tension[:] = 0.0
        self.notice = "all tendon tension commands zeroed"

    @property
    def selected_label(self) -> str:
        prefix = "R" if self.selected_group == GROUP_ROTOR else "T"
        return f"{self.selected_group} {prefix}{self.selected_channel + 1}"

    @property
    def status(self) -> str:
        if self.pending_single_step:
            return "SINGLE-STEP"
        return "PAUSED" if self.paused else "RUNNING"


@dataclass(frozen=True)
class _ResolvedEntities:
    drone_body_id: int
    b8_body_id: int
    tip_site_id: int
    tendon_ids: tuple[int, ...]
    tendon_names: tuple[str, ...]
    actuator_names: tuple[str, ...]


class _ManualPerturbationAviary(BaseAviary):
    """Demo-local bridge preserving native UAV perturbations across substeps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._manual_drone_body_ids = tuple(
            _required_id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"drone{index}",
            )
            for index in range(self.NUM_DRONES)
        )
        self._manual_uav_viewer_wrenches = np.zeros(
            (self.NUM_DRONES, 6), dtype=float
        )

    def set_uav_viewer_wrench(self, nth_drone: int, wrench) -> None:
        values = np.asarray(wrench, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("native UAV viewer wrench must contain 6 finite values")
        self._manual_uav_viewer_wrenches[nth_drone] = values

    def snapshot_uav_viewer_wrenches(self) -> None:
        for index, body_id in enumerate(self._manual_drone_body_ids):
            self.set_uav_viewer_wrench(
                index,
                self.data.xfrc_applied[body_id].copy(),
            )

    def clear_uav_viewer_wrenches(self) -> None:
        self._manual_uav_viewer_wrenches[:] = 0.0

    def _physics(self, rpm, nth_drone):
        """Apply rotor physics once, then add the cached native UAV wrench."""
        super()._physics(rpm, nth_drone)
        body_id = self._manual_drone_body_ids[nth_drone]
        self.data.xfrc_applied[body_id] += (
            self._manual_uav_viewer_wrenches[nth_drone]
        )


def _required_id(model, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(object_id)


def _resolve_entities(env) -> _ResolvedEntities:
    manifest = env.SCALED_STABLESOFT_ARM_MANIFEST
    bindings = tuple(env.SCALED_STABLESOFT_TENDON_BINDINGS)
    if manifest is None or len(bindings) != 4:
        raise ValueError("manual demo requires four active StableSoft tendons")
    tendon_names = tuple(manifest.tendon_names)
    actuator_names = tuple(manifest.tendon_actuator_names)
    if tuple(item.tendon_name for item in bindings) != tendon_names:
        raise ValueError("runtime tendon bindings do not follow manifest order")
    if tuple(item.actuator_name for item in bindings) != actuator_names:
        raise ValueError("runtime actuator bindings do not follow manifest order")
    return _ResolvedEntities(
        drone_body_id=_required_id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "drone0"
        ),
        b8_body_id=_required_id(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            manifest.segment_body_names[-1],
        ),
        tip_site_id=_required_id(
            env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            manifest.arm_tip_site_name,
        ),
        tendon_ids=tuple(item.tendon_id for item in bindings),
        tendon_names=tendon_names,
        actuator_names=actuator_names,
    )


def _make_key_callback(event_queue):
    """Return a viewer-thread callback which only enqueues one keycode."""

    def key_callback(keycode):
        event_queue.put(int(keycode))

    return key_callback


def _drain_key_queue(event_queue) -> tuple[int, ...]:
    keycodes = []
    while True:
        try:
            keycodes.append(int(event_queue.get_nowait()))
        except queue.Empty:
            return tuple(keycodes)


def _apply_key(state: ManualCommandState, keycode: int) -> bool:
    """Apply one discrete press in the main thread; return reset requested."""
    if keycode == glfw.KEY_SPACE:
        state.paused = not state.paused
        state.pending_single_step = False
        state.notice = "paused" if state.paused else "running"
    elif keycode == glfw.KEY_PERIOD:
        if state.paused:
            state.pending_single_step = True
            state.notice = "single-step requested"
        else:
            state.notice = "single-step ignored: pause first"
    elif keycode == glfw.KEY_ENTER:
        state.toggle_group()
    elif keycode == glfw.KEY_UP:
        state.select_relative(-1)
    elif keycode == glfw.KEY_DOWN:
        state.select_relative(1)
    elif keycode == glfw.KEY_LEFT:
        state.adjust_selected(-1)
    elif keycode == glfw.KEY_RIGHT:
        state.adjust_selected(1)
    elif keycode == glfw.KEY_F8:
        state.zero_rotors()
    elif keycode == glfw.KEY_F9:
        state.zero_tendons()
    elif keycode == glfw.KEY_BACKSPACE:
        return True
    return False


def _warning_summary(data) -> str:
    warning_items = getattr(data, "warning", ())
    active = []
    for index, item in enumerate(warning_items):
        count = int(item.number)
        if count:
            try:
                name = mujoco.mjtWarning(index).name.removeprefix("mjWARN_")
            except ValueError:
                name = f"W{index}"
            active.append(f"{name}:{count}")
    return ", ".join(active) if active else "none"


def _build_overlay(
    env,
    state: ManualCommandState,
    entities: _ResolvedEntities,
    visual_mode: str,
    *,
    status_override: Optional[str] = None,
):
    status = status_override or state.status
    initial_rpm_text = (
        "OPEN-LOOP HOVER BIAS" if state.initial_mode == "hover" else "zero"
    )
    rpm_text = " ".join(
        f"R{index + 1}={value:.1f}"
        for index, value in enumerate(state.rotor_rpm)
    )
    tension_text = " ".join(
        f"T{index + 1}={value:.6f}"
        for index, value in enumerate(state.tendon_tension)
    )
    tendon_lengths = np.asarray(env.data.ten_length)[list(entities.tendon_ids)]
    length_text = " ".join(
        f"L{index + 1}={value:.6f}"
        for index, value in enumerate(tendon_lengths)
    )
    position = np.asarray(env.pos[0])
    rpy = np.asarray(env.rpy[0])
    tip = np.asarray(env.data.site_xpos[entities.tip_site_id])
    left_labels = (
        "mode\nsafety\ninitial RPM\ntime\nfrequency\nvisual\nselected\nrotors\ntendons"
    )
    left_values = (
        f"{status}\n"
        "OPEN-LOOP — NO FEEDBACK CONTROLLER\n"
        f"{initial_rpm_text}\n"
        f"{float(env.data.time):.6f} s\n"
        f"{env.SIM_FREQ}/{env.CTRL_FREQ} Hz; "
        f"{env.SIM_STEPS_PER_CTRL} substeps\n"
        f"{visual_mode}\n"
        f"{state.selected_label}\n"
        f"{rpm_text}\n"
        f"{tension_text}"
    )
    right_labels = "UAV world xyz\nUAV RPY rad\narm-tip world xyz\ntendon lengths\nwarnings\nnotice"
    right_values = (
        f"{position[0]:+.5f} {position[1]:+.5f} {position[2]:+.5f}\n"
        f"{rpy[0]:+.5f} {rpy[1]:+.5f} {rpy[2]:+.5f}\n"
        f"{tip[0]:+.5f} {tip[1]:+.5f} {tip[2]:+.5f}\n"
        f"{length_text}\n"
        f"{_warning_summary(env.data)}\n"
        f"{state.notice or '-'}"
    )
    return [
        (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            left_labels,
            left_values,
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            right_labels,
            right_values,
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "manual controls",
            KEY_HELP,
        ),
    ]


def _try_update_overlay(viewer, texts, *, enabled: bool) -> bool:
    if not enabled:
        return False
    try:
        viewer.set_texts(texts)
    except (AttributeError, TypeError, ValueError) as exc:
        print(
            "viewer overlay disabled after set_texts compatibility "
            f"error: {exc}",
            file=sys.stderr,
        )
        return False
    return True


def _sleep_until(deadline: float, *, clock, sleep_fn) -> float:
    now = clock()
    remaining = deadline - now
    if remaining > 0:
        sleep_fn(remaining)
        return deadline
    return now


def _run_manual_loop(
    env,
    viewer,
    state: ManualCommandState,
    entities: _ResolvedEntities,
    visual_mode: str,
    event_queue,
    *,
    clock=time.perf_counter,
    sleep_fn=time.sleep,
) -> None:
    overlay_enabled = callable(getattr(viewer, "set_texts", None))
    if not overlay_enabled:
        print("viewer overlay disabled: set_texts is unavailable", file=sys.stderr)
    next_deadline = clock()
    next_overlay = next_deadline
    was_paused = True

    try:
        while viewer.is_running():
            viewer.sync()

            reset_requested = False
            keycodes = _drain_key_queue(event_queue)
            for keycode in keycodes:
                reset_requested = _apply_key(state, keycode) or reset_requested

            if reset_requested:
                env.clear_uav_viewer_wrenches()
                env.reset()
                state.reset()
                state.notice = "environment and commands reset"
                next_deadline = clock()

            should_step = (not state.paused) or state.pending_single_step
            single_step = state.paused and state.pending_single_step
            if should_step:
                if was_paused and not single_step:
                    next_deadline = clock()
                env.snapshot_uav_viewer_wrenches()
                state.clamp_commands()
                try:
                    env.set_scaled_stablesoft_tension_command(
                        tuple(float(value) for value in state.tendon_tension)
                    )
                    env.step(state.rotor_rpm.copy())
                finally:
                    # A native wrench is valid for exactly this control period.
                    env.clear_uav_viewer_wrenches()
                if single_step:
                    state.pending_single_step = False
                    state.notice = "advanced exactly one control period"
            else:
                env.clear_uav_viewer_wrenches()

            status_override = "SINGLE-STEP" if single_step else None
            now = clock()
            if now >= next_overlay or keycodes or single_step or reset_requested:
                overlay_enabled = _try_update_overlay(
                    viewer,
                    _build_overlay(
                        env,
                        state,
                        entities,
                        visual_mode,
                        status_override=status_override,
                    ),
                    enabled=overlay_enabled,
                )
                next_overlay = now + OVERLAY_REFRESH_PERIOD_S

            if state.paused:
                next_deadline = now + PAUSED_REFRESH_PERIOD_S
                _sleep_until(next_deadline, clock=clock, sleep_fn=sleep_fn)
            else:
                control_period = float(env.CTRL_TIMESTEP)
                next_deadline += control_period
                after_step = clock()
                if next_deadline < after_step - control_period:
                    next_deadline = after_step + control_period
                _sleep_until(next_deadline, clock=clock, sleep_fn=sleep_fn)
            was_paused = state.paused
    finally:
        env.clear_uav_viewer_wrenches()


def _make_env(*, sim_freq: int, ctrl_freq: int, visual_mode: str):
    """Build the same fixed active model used by the accepted C1 baseline."""
    return _ManualPerturbationAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.MJC,
        sim_freq=sim_freq,
        ctrl_freq=ctrl_freq,
        gui=False,
        record=False,
        obstacles=False,
        vision_attributes=False,
        act_type=ActionType.RPM,
        render_mode=None,
        scaled_stablesoft_arm=_spec(),
        scaled_stablesoft_visual=ScaledStableSoftVisualSpec(mode=visual_mode),
        scaled_stablesoft_tendon_drive=IdealTendonTensionSpec(
            max_tension=VERIFIED_ACTIVE_MAX_TENSION_N
        ),
        integrator="implicitfast",
    )


def _positive_finite(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys:\n" + KEY_HELP,
    )
    parser.add_argument(
        "--initial-rpm",
        choices=("zero", "hover"),
        default="zero",
        help="Initial raw RPM command; hover is a fixed open-loop bias.",
    )
    parser.add_argument("--rpm-step", type=_positive_finite, required=True)
    parser.add_argument("--tension-step", type=_positive_finite, required=True)
    parser.add_argument(
        "--tension-limit",
        type=_positive_finite,
        default=None,
        help=(
            "Optional manual command clamp, never greater than the compiled "
            "0.01 N actuator limit and never changing the model."
        ),
    )
    parser.add_argument(
        "--visual-mode",
        choices=("mesh", "cylinder"),
        default=ScaledStableSoftVisualSpec().mode,
    )
    parser.add_argument(
        "--sim-freq",
        type=int,
        choices=(DEFAULT_SIM_FREQ,),
        default=DEFAULT_SIM_FREQ,
    )
    parser.add_argument(
        "--ctrl-freq",
        type=int,
        choices=(DEFAULT_CTRL_FREQ,),
        default=DEFAULT_CTRL_FREQ,
    )
    return parser


def _runtime_tension_limit(requested, drive) -> float:
    compiled_maximum = float(drive.max_tension)
    if requested is None:
        return compiled_maximum
    value = float(requested)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tension_limit must be finite and positive")
    if value > compiled_maximum:
        raise ValueError(
            "tension_limit cannot exceed compiled drive max_tension "
            f"({compiled_maximum:g} N)"
        )
    return value


def run_viewer(
    args,
    *,
    env_factory=_make_env,
    launch_viewer=mujoco.viewer.launch_passive,
    clock=time.perf_counter,
    sleep_fn=time.sleep,
) -> None:
    env = env_factory(
        sim_freq=args.sim_freq,
        ctrl_freq=args.ctrl_freq,
        visual_mode=args.visual_mode,
    )
    try:
        drive = env.SCALED_STABLESOFT_TENDON_DRIVE_SPEC
        if drive is None or drive.max_tension != VERIFIED_ACTIVE_MAX_TENSION_N:
            raise ValueError("manual demo environment is not the verified active model")
        tension_limit = _runtime_tension_limit(args.tension_limit, drive)
        if args.initial_rpm == "hover" and env.HOVER_RPM > env.MAX_RPM:
            raise ValueError("open-loop hover bias exceeds the formal RPM limit")
        env.reset()
        env.clear_uav_viewer_wrenches()
        entities = _resolve_entities(env)
        state = ManualCommandState.create(
            initial_mode=args.initial_rpm,
            hover_rpm=env.HOVER_RPM,
            max_rpm=env.MAX_RPM,
            tension_limit=tension_limit,
            rpm_step=args.rpm_step,
            tension_step=args.tension_step,
        )
        event_queue = queue.SimpleQueue()
        viewer = launch_viewer(
            env.model,
            env.data,
            key_callback=_make_key_callback(event_queue),
        )
        with _closing_viewer(viewer, sleep_fn=sleep_fn):
            _run_manual_loop(
                env,
                viewer,
                state,
                entities,
                args.visual_mode,
                event_queue,
                clock=clock,
                sleep_fn=sleep_fn,
            )
    finally:
        env.clear_uav_viewer_wrenches()
        env.close()


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run_viewer(args)
    except KeyboardInterrupt:
        print("manual demo interrupted; resources closed", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
