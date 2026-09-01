"""Simulation-time interaction commands for the optional PX4 runner.

The command/state layer deliberately has no GLFW or MuJoCo import.  A viewer,
headless replay, or another input device can therefore produce the same
immutable command stream, while the plant adapter remains the only code that
turns a command into a MuJoCo wrench.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


class InteractionMode(str, Enum):
    DISABLED = "disabled"
    EXTERNAL_FORCE = "external_force"
    TENDON_MODAL = "tendon_modal"
    JOINT_TORQUE = "joint_torque"


class InteractionFrame(str, Enum):
    WORLD_ENU = "world_enu"
    TENDON_MODAL = "tendon_modal"
    JOINT_DOF = "joint_dof"


def _vector(name: str, values, length: int) -> Tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite values") from exc
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class InteractionCommand:
    """One fully described interaction sample at simulation time."""

    timestamp_us: int
    sequence: int
    mode: InteractionMode
    target_body: Optional[str]
    target_site: Optional[str]
    frame: InteractionFrame
    target_body_id: Optional[int] = None
    target_joint: Optional[str] = None
    target_joint_id: Optional[int] = None
    target_dof_axis: Optional[int] = None
    target_dof_index: Optional[int] = None
    force_n: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    point_world_m: Optional[Tuple[float, float, float]] = None
    body_com_world_m: Optional[Tuple[float, float, float]] = None
    generated_torque_nm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tendon_acceleration_m_s2: Tuple[float, float] = (0.0, 0.0)
    joint_torque_nm: float = 0.0
    active: bool = False
    released: bool = False
    clipped: bool = False
    source: str = "none"

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_us, bool) or self.timestamp_us < 0:
            raise ValueError("timestamp_us must be a non-negative integer")
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        object.__setattr__(self, "mode", InteractionMode(self.mode))
        object.__setattr__(self, "frame", InteractionFrame(self.frame))
        object.__setattr__(self, "force_n", _vector("force_n", self.force_n, 3))
        object.__setattr__(
            self,
            "generated_torque_nm",
            _vector("generated_torque_nm", self.generated_torque_nm, 3),
        )
        object.__setattr__(
            self,
            "tendon_acceleration_m_s2",
            _vector(
                "tendon_acceleration_m_s2",
                self.tendon_acceleration_m_s2,
                2,
            ),
        )
        if self.point_world_m is not None:
            object.__setattr__(
                self,
                "point_world_m",
                _vector("point_world_m", self.point_world_m, 3),
            )
        if self.body_com_world_m is not None:
            object.__setattr__(
                self,
                "body_com_world_m",
                _vector("body_com_world_m", self.body_com_world_m, 3),
            )
        torque = float(self.joint_torque_nm)
        if not math.isfinite(torque):
            raise ValueError("joint_torque_nm must be finite")
        object.__setattr__(self, "joint_torque_nm", torque)
        for name in (
            "target_body_id", "target_joint_id", "target_dof_axis",
            "target_dof_index",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        if self.mode is InteractionMode.EXTERNAL_FORCE:
            if self.frame is not InteractionFrame.WORLD_ENU:
                raise ValueError("external force commands must use world_enu")
            if not self.target_body:
                raise ValueError("external force commands require a target body")
            if any(self.tendon_acceleration_m_s2) or self.joint_torque_nm:
                raise ValueError("an external force command cannot contain tendon input")
        elif self.mode is InteractionMode.TENDON_MODAL:
            if self.frame is not InteractionFrame.TENDON_MODAL:
                raise ValueError("tendon commands must use tendon_modal frame")
            if any(self.force_n) or self.point_world_m is not None or self.joint_torque_nm:
                raise ValueError("a tendon command cannot contain an external force")
        elif self.mode is InteractionMode.JOINT_TORQUE:
            if self.frame is not InteractionFrame.JOINT_DOF:
                raise ValueError("joint torque commands must use joint_dof frame")
            if (
                not self.target_joint
                or self.target_dof_axis is None
            ):
                raise ValueError("joint torque commands require joint and axis")
            if any(self.force_n) or any(self.tendon_acceleration_m_s2):
                raise ValueError("a joint torque command cannot contain force/tendon input")
        elif self.active:
            raise ValueError("a disabled command cannot be active")

    def as_dict(self) -> dict:
        result = {
            "timestamp_us": self.timestamp_us,
            "sequence": self.sequence,
            "mode": self.mode.value,
            "target_body": self.target_body,
            "target_site": self.target_site,
            "target_body_id": self.target_body_id,
            "target_joint": self.target_joint,
            "target_joint_id": self.target_joint_id,
            "target_dof_axis": self.target_dof_axis,
            "target_dof_index": self.target_dof_index,
            "frame": self.frame.value,
            "force_n": self.force_n,
            "point_world_m": self.point_world_m,
            "body_com_world_m": self.body_com_world_m,
            "generated_torque_nm": self.generated_torque_nm,
            "tendon_acceleration_m_s2": self.tendon_acceleration_m_s2,
            "joint_torque_nm": self.joint_torque_nm,
            "diagnostic_label": (
                "DIAGNOSTIC JOINT TORQUE"
                if self.mode is InteractionMode.JOINT_TORQUE else None
            ),
            "active": self.active,
            "released": self.released,
            "clipped": self.clipped,
            "source": self.source,
        }
        return result


@dataclass(frozen=True)
class ForceInteractionConfig:
    step_n: float = 0.002
    max_force_n: float = 0.005
    spring_n_per_m: float = 0.25
    damping_n_s_per_m: float = 0.01

    def __post_init__(self) -> None:
        for name in ("step_n", "max_force_n", "spring_n_per_m", "damping_n_s_per_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.step_n <= 0.0 or self.max_force_n <= 0.0 or self.spring_n_per_m <= 0.0:
            raise ValueError("force step, limit, and spring gain must be positive")


def clamp_norm(values: Sequence[float], maximum: float) -> Tuple[np.ndarray, bool]:
    vector = np.asarray(tuple(values), dtype=float)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("values must be a finite vector")
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm == 0.0:
        return vector, False
    return vector * (float(maximum) / norm), True


class MouseForceInteractor:
    """Bounded virtual spring-damper calculation, independent of a viewer."""

    def __init__(self, config: ForceInteractionConfig):
        self.config = config
        self._anchor = None
        self._target = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def begin(self, point_world_m) -> None:
        point = np.asarray(_vector("point_world_m", point_world_m, 3))
        self._anchor = point.copy()
        self._target = point.copy()
        self._active = True

    def drag_to(self, target_world_m) -> None:
        if not self._active:
            raise RuntimeError("begin must be called before drag_to")
        self._target = np.asarray(_vector("target_world_m", target_world_m, 3))

    def release(self) -> None:
        self._active = False
        self._anchor = None
        self._target = None

    def calculate(self, point_world_m, velocity_world_m_s) -> Tuple[Tuple[float, ...], bool]:
        if not self._active:
            return (0.0, 0.0, 0.0), False
        point = np.asarray(_vector("point_world_m", point_world_m, 3))
        velocity = np.asarray(_vector("velocity_world_m_s", velocity_world_m_s, 3))
        force = (
            self.config.spring_n_per_m * (self._target - point)
            - self.config.damping_n_s_per_m * velocity
        )
        force, clipped = clamp_norm(force, self.config.max_force_n)
        return tuple(float(value) for value in force), clipped


class KeyboardInteractionState:
    """Press/release state machine with dead-man zero semantics."""

    _DIRECTIONS = {
        "x_pos": (1.0, 0.0),
        "x_neg": (-1.0, 0.0),
        "y_pos": (0.0, 1.0),
        "y_neg": (0.0, -1.0),
    }

    def __init__(
        self,
        *,
        mode: InteractionMode,
        force_step_n: float = 0.002,
        force_limit_n: float = 0.005,
        tendon_step_m_s2: float = 0.005,
        tendon_limit_m_s2: float = 0.04,
    ):
        self.mode = InteractionMode(mode)
        if self.mode not in (InteractionMode.EXTERNAL_FORCE, InteractionMode.TENDON_MODAL):
            raise ValueError("keyboard mode must be external_force or tendon_modal")
        self.force_step_n = float(force_step_n)
        self.force_limit_n = float(force_limit_n)
        self.tendon_step_m_s2 = float(tendon_step_m_s2)
        self.tendon_limit_m_s2 = float(tendon_limit_m_s2)
        values = (
            self.force_step_n,
            self.force_limit_n,
            self.tendon_step_m_s2,
            self.tendon_limit_m_s2,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("keyboard steps and limits must be finite and positive")
        self.force_magnitude_n = min(self.force_step_n, self.force_limit_n)
        self._held = set()
        self._emergency_stopped = False

    def set_pressed(self, direction: str, pressed: bool) -> None:
        if direction not in self._DIRECTIONS:
            raise ValueError(f"unknown direction: {direction}")
        if self._emergency_stopped and pressed:
            return
        if pressed:
            self._held.add(direction)
        else:
            self._held.discard(direction)

    def adjust_force_magnitude(self, steps: int) -> None:
        value = self.force_magnitude_n + int(steps) * self.force_step_n
        self.force_magnitude_n = min(self.force_limit_n, max(0.0, value))

    def zero(self) -> None:
        self._held.clear()

    def emergency_stop(self) -> None:
        self.zero()
        self._emergency_stopped = True

    def clear_emergency_stop(self) -> None:
        self._emergency_stopped = False

    def vector(self) -> Tuple[Tuple[float, float], bool]:
        direction = np.zeros(2)
        for name in self._held:
            direction += self._DIRECTIONS[name]
        norm = float(np.linalg.norm(direction))
        if norm > 1.0:
            direction /= norm
        maximum = (
            self.force_magnitude_n
            if self.mode is InteractionMode.EXTERNAL_FORCE
            else self.tendon_limit_m_s2
        )
        if self.mode is InteractionMode.TENDON_MODAL:
            direction *= self.tendon_step_m_s2
        else:
            direction *= self.force_magnitude_n
        direction, clipped = clamp_norm(direction, maximum)
        return (float(direction[0]), float(direction[1])), clipped


class JointTorqueInteractionState:
    """One-step, bounded diagnostic torque selection state.

    This state deliberately represents generalized joint effort, not tendon
    actuation.  A viewer press creates one plant-step pulse; it is never held
    implicitly after the UI stops delivering key-repeat events.
    """

    def __init__(
        self,
        joint_count: int,
        *,
        step_nm: float = 10e-6,
        limit_nm: float = 50e-6,
    ):
        if isinstance(joint_count, bool) or int(joint_count) <= 0:
            raise ValueError("joint_count must be a positive integer")
        self.joint_count = int(joint_count)
        self.step_nm = float(step_nm)
        self.limit_nm = float(limit_nm)
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.step_nm, self.limit_nm)
        ):
            raise ValueError("joint torque step and limit must be finite and positive")
        if self.step_nm > self.limit_nm:
            raise ValueError("joint torque step cannot exceed its limit")
        self.joint_index = 0
        self.axis = 0
        self.magnitude_nm = self.step_nm
        self._pulse_sign = 0
        self._emergency_stopped = False

    def select_joint(self, index: int) -> None:
        self.joint_index = int(index) % self.joint_count

    def cycle_joint(self, steps: int) -> None:
        self.select_joint(self.joint_index + int(steps))

    def select_axis(self, axis: int) -> None:
        if isinstance(axis, bool) or int(axis) not in (0, 1, 2):
            raise ValueError("joint DOF axis must be 0, 1, or 2")
        self.axis = int(axis)

    def adjust_magnitude(self, steps: int) -> None:
        value = self.magnitude_nm + int(steps) * self.step_nm
        self.magnitude_nm = min(self.limit_nm, max(self.step_nm, value))

    def pulse(self, sign: int) -> None:
        if sign not in (-1, 1):
            raise ValueError("joint torque pulse sign must be -1 or +1")
        if not self._emergency_stopped:
            self._pulse_sign = int(sign)

    def consume(self) -> float:
        torque = self._pulse_sign * self.magnitude_nm
        self._pulse_sign = 0
        return float(torque)

    def zero(self) -> None:
        self._pulse_sign = 0

    def emergency_stop(self) -> None:
        self.zero()
        self._emergency_stopped = True

    def clear_emergency_stop(self) -> None:
        self._emergency_stopped = False


@dataclass(frozen=True)
class ArmSegmentBinding:
    index: int
    body_id: int
    body_name: str
    parent_body_id: int
    site_ids: Tuple[int, ...]
    site_names: Tuple[str, ...]


@dataclass(frozen=True)
class ArmJointBinding:
    index: int
    joint_id: int
    joint_name: str
    body_id: int
    body_name: str
    dof_indices: Tuple[int, int, int]
    stiffness_nm_per_rad: float


@dataclass(frozen=True)
class ArmInteractionCatalog:
    """Manifest-derived allowlist for arm-only interactive writes."""

    segments: Tuple[ArmSegmentBinding, ...]
    joints: Tuple[ArmJointBinding, ...]

    @classmethod
    def discover(cls, model, manifest) -> "ArmInteractionCatalog":
        import mujoco

        if manifest is None:
            raise ValueError("arm interaction requires an arm manifest")
        body_names = tuple(manifest.segment_body_names)
        joint_names = tuple(manifest.ball_joint_names)
        if not body_names:
            raise ValueError("arm manifest contains no segment bodies")
        segments = []
        body_ids = []
        for index, body_name in enumerate(body_names):
            body_id = int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            ))
            if body_id <= 0:
                raise ValueError(f"MuJoCo arm body not found: {body_name}")
            body_ids.append(body_id)
            site_ids = tuple(
                site_id for site_id in range(model.nsite)
                if int(model.site_bodyid[site_id]) == body_id
            )
            site_names = tuple(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
                for site_id in site_ids
            )
            if any(name is None for name in site_names):
                raise ValueError(f"arm segment {body_name} contains an unnamed site")
            segments.append(ArmSegmentBinding(
                index=index,
                body_id=body_id,
                body_name=body_name,
                parent_body_id=int(model.body_parentid[body_id]),
                site_ids=site_ids,
                site_names=site_names,
            ))
        for previous, current in zip(body_ids, body_ids[1:]):
            if int(model.body_parentid[current]) != previous:
                raise ValueError("arm manifest segment bodies are not one descendant chain")
        if len(joint_names) != len(body_names) - 1:
            raise ValueError("arm manifest ball-joint count does not match segment chain")
        joints = []
        for index, (joint_name, expected_body_id) in enumerate(
            zip(joint_names, body_ids[1:])
        ):
            joint_id = int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            ))
            if joint_id < 0:
                raise ValueError(f"MuJoCo arm joint not found: {joint_name}")
            if int(model.jnt_bodyid[joint_id]) != expected_body_id:
                raise ValueError(f"arm joint {joint_name} is attached to the wrong body")
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_BALL):
                raise ValueError(f"arm joint {joint_name} is not a ball joint")
            dof_start = int(model.jnt_dofadr[joint_id])
            dofs = (dof_start, dof_start + 1, dof_start + 2)
            joints.append(ArmJointBinding(
                index=index,
                joint_id=joint_id,
                joint_name=joint_name,
                body_id=expected_body_id,
                body_name=body_names[index + 1],
                dof_indices=dofs,
                stiffness_nm_per_rad=float(model.jnt_stiffness[joint_id]),
            ))
        return cls(tuple(segments), tuple(joints))

    @property
    def terminal_segment(self) -> ArmSegmentBinding:
        return self.segments[-1]

    @property
    def body_ids(self) -> Tuple[int, ...]:
        return tuple(segment.body_id for segment in self.segments)

    @property
    def dof_indices(self) -> Tuple[int, ...]:
        return tuple(dof for joint in self.joints for dof in joint.dof_indices)

    def segment_for_body(self, value) -> ArmSegmentBinding:
        for segment in self.segments:
            if value == segment.body_id or value == segment.body_name:
                return segment
        raise ValueError(f"interaction target is not an arm segment: {value}")

    def joint_for(self, value) -> ArmJointBinding:
        for joint in self.joints:
            if value == joint.joint_id or value == joint.joint_name:
                return joint
        raise ValueError(f"interaction target is not an arm ball joint: {value}")

    def joint_for_body(self, body_id: int) -> Optional[ArmJointBinding]:
        return next((joint for joint in self.joints if joint.body_id == body_id), None)


@dataclass(frozen=True)
class TimedInteractionEvent:
    start_s: float
    duration_s: float
    mode: InteractionMode
    force_n: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tendon_acceleration_m_s2: Tuple[float, float] = (0.0, 0.0)
    target_body: Optional[str] = None
    target_site: Optional[str] = None
    point_world_m: Optional[Tuple[float, float, float]] = None
    target_joint: Optional[str] = None
    target_dof_axis: Optional[int] = None
    joint_torque_nm: float = 0.0

    def __post_init__(self) -> None:
        start = float(self.start_s)
        duration = float(self.duration_s)
        if not math.isfinite(start) or start < 0.0:
            raise ValueError("event start_s must be finite and non-negative")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("event duration_s must be finite and positive")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "mode", InteractionMode(self.mode))
        object.__setattr__(self, "force_n", _vector("force_n", self.force_n, 3))
        object.__setattr__(
            self,
            "tendon_acceleration_m_s2",
            _vector("tendon_acceleration_m_s2", self.tendon_acceleration_m_s2, 2),
        )
        if self.point_world_m is not None:
            object.__setattr__(
                self, "point_world_m", _vector("point_world_m", self.point_world_m, 3)
            )
        torque = float(self.joint_torque_nm)
        if not math.isfinite(torque):
            raise ValueError("joint_torque_nm must be finite")
        object.__setattr__(self, "joint_torque_nm", torque)
        if self.target_dof_axis is not None and (
            isinstance(self.target_dof_axis, bool)
            or int(self.target_dof_axis) not in (0, 1, 2)
        ):
            raise ValueError("target_dof_axis must be 0, 1, or 2")
        if self.target_dof_axis is not None:
            object.__setattr__(self, "target_dof_axis", int(self.target_dof_axis))
        if self.mode is InteractionMode.EXTERNAL_FORCE:
            if any(self.tendon_acceleration_m_s2) or self.joint_torque_nm:
                raise ValueError("force event cannot contain tendon/joint input")
        elif self.mode is InteractionMode.TENDON_MODAL:
            if any(self.force_n) or self.joint_torque_nm:
                raise ValueError("tendon event cannot contain force/joint input")
        elif self.mode is InteractionMode.JOINT_TORQUE:
            if any(self.force_n) or any(self.tendon_acceleration_m_s2):
                raise ValueError("joint event cannot contain force/tendon input")
            if not self.target_joint or self.target_dof_axis is None:
                raise ValueError("joint event requires target_joint and target_dof_axis")

    @property
    def stop_s(self) -> float:
        return self.start_s + self.duration_s


class DeterministicInteractionTimeline:
    """Select a unique event using simulation time only."""

    def __init__(self, events: Iterable[TimedInteractionEvent]):
        self.events = tuple(sorted(events, key=lambda event: event.start_s))
        for previous, current in zip(self.events, self.events[1:]):
            if current.start_s < previous.stop_s - 1e-15:
                raise ValueError("interaction events must not overlap")
        self._last_active = False
        self._sequence = 0

    @classmethod
    def from_json(cls, path: Path) -> "DeterministicInteractionTimeline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("events") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("interaction JSON must be a list or contain an events list")
        events = []
        for row in rows:
            events.append(TimedInteractionEvent(
                start_s=row["start_s"],
                duration_s=row["duration_s"],
                mode=row["mode"],
                force_n=row.get("force_n", (0.0, 0.0, 0.0)),
                tendon_acceleration_m_s2=row.get(
                    "tendon_acceleration_m_s2", (0.0, 0.0)
                ),
                target_body=row.get("target_body"),
                target_site=row.get("target_site"),
                point_world_m=row.get("point_world_m"),
                target_joint=row.get("target_joint"),
                target_dof_axis=row.get("target_dof_axis"),
                joint_torque_nm=row.get("joint_torque_nm", 0.0),
            ))
        return cls(events)

    def sample(
        self,
        simulation_time_s: float,
        *,
        target_body: Optional[str],
        target_site: Optional[str],
        point_world_m: Optional[Sequence[float]] = None,
    ) -> InteractionCommand:
        now = float(simulation_time_s)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("simulation_time_s must be finite and non-negative")
        event = next(
            (item for item in self.events if item.start_s <= now < item.stop_s),
            None,
        )
        active = event is not None
        released = self._last_active and not active
        self._last_active = active
        self._sequence += 1
        if event is None:
            return InteractionCommand(
                timestamp_us=round(now * 1_000_000),
                sequence=self._sequence,
                mode=InteractionMode.DISABLED,
                target_body=target_body,
                target_site=target_site,
                frame=InteractionFrame.WORLD_ENU,
                active=False,
                released=released,
                source="replay",
            )
        if event.mode is InteractionMode.EXTERNAL_FORCE:
            return InteractionCommand(
                timestamp_us=round(now * 1_000_000),
                sequence=self._sequence,
                mode=event.mode,
                target_body=event.target_body or target_body,
                target_site=(
                    event.target_site
                    if event.target_site is not None
                    else (target_site if event.target_body is None else None)
                ),
                frame=InteractionFrame.WORLD_ENU,
                force_n=event.force_n,
                point_world_m=(
                    event.point_world_m
                    if event.point_world_m is not None
                    else (point_world_m if event.target_body is None else None)
                ),
                active=True,
                source="replay",
            )
        if event.mode is InteractionMode.JOINT_TORQUE:
            return InteractionCommand(
                timestamp_us=round(now * 1_000_000),
                sequence=self._sequence,
                mode=event.mode,
                target_body=event.target_body or target_body,
                target_site=event.target_site or target_site,
                target_joint=event.target_joint,
                target_dof_axis=event.target_dof_axis,
                frame=InteractionFrame.JOINT_DOF,
                joint_torque_nm=event.joint_torque_nm,
                active=True,
                source="replay",
            )
        return InteractionCommand(
            timestamp_us=round(now * 1_000_000),
            sequence=self._sequence,
            mode=event.mode,
            target_body=target_body,
            target_site=target_site,
            frame=InteractionFrame.TENDON_MODAL,
            tendon_acceleration_m_s2=event.tendon_acceleration_m_s2,
            active=True,
            source="replay",
        )

    def reset(self) -> None:
        self._last_active = False
        self._sequence = 0


class MujocoArmForceApplier:
    """Apply a world-frame point force through equivalent body wrench."""

    def __init__(self, model, data, body_name: str, site_name: str):
        import mujoco

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {body_name}")
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {site_name}")
        if int(model.site_bodyid[site_id]) != int(body_id):
            raise ValueError("target site must belong to target body")
        self.model = model
        self.data = data
        self.body_name = body_name
        self.site_name = site_name
        self.body_id = int(body_id)
        self.site_id = int(site_id)

    @property
    def point_world_m(self) -> Tuple[float, float, float]:
        return tuple(float(value) for value in self.data.site_xpos[self.site_id])

    def apply(self, command: InteractionCommand) -> None:
        self.clear()
        if not command.active or command.mode is not InteractionMode.EXTERNAL_FORCE:
            return
        if command.target_body != self.body_name or command.target_site != self.site_name:
            raise ValueError("interaction command target does not match force applier")
        force = np.asarray(command.force_n, dtype=float)
        point = np.asarray(
            self.point_world_m if command.point_world_m is None else command.point_world_m,
            dtype=float,
        )
        center = np.asarray(self.data.xipos[self.body_id], dtype=float)
        self.data.xfrc_applied[self.body_id, :3] = force
        self.data.xfrc_applied[self.body_id, 3:] = np.cross(point - center, force)

    def clear(self) -> None:
        self.data.xfrc_applied[self.body_id] = 0.0


class MujocoArmInteractionApplier:
    """Arm-only point-force and diagnostic generalized-effort adapter."""

    def __init__(
        self,
        model,
        data,
        catalog: ArmInteractionCatalog,
        *,
        allowed_body_ids: Optional[Sequence[int]] = None,
        default_site_name: Optional[str] = None,
    ):
        import mujoco

        self.model = model
        self.data = data
        self.catalog = catalog
        allowed = catalog.body_ids if allowed_body_ids is None else tuple(
            int(value) for value in allowed_body_ids
        )
        unknown = set(allowed) - set(catalog.body_ids)
        if unknown or not allowed:
            raise ValueError(f"invalid arm interaction body allowlist: {sorted(unknown)}")
        self.allowed_body_ids = allowed
        self.default_segment = catalog.terminal_segment
        if self.default_segment.body_id not in allowed:
            self.default_segment = catalog.segment_for_body(allowed[-1])
        self.default_site_id = None
        self.default_site_name = default_site_name
        if default_site_name is not None:
            site_id = int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, default_site_name
            ))
            if site_id < 0:
                raise ValueError(f"MuJoCo interaction site not found: {default_site_name}")
            site_body = int(model.site_bodyid[site_id])
            if site_body not in allowed:
                raise ValueError("default interaction site is outside the body allowlist")
            self.default_site_id = site_id

    # Compatibility properties used by P11 callers and overlays.
    @property
    def body_id(self) -> int:
        return self.default_segment.body_id

    @property
    def body_name(self) -> str:
        return self.default_segment.body_name

    @property
    def site_id(self) -> Optional[int]:
        return self.default_site_id

    @property
    def site_name(self) -> Optional[str]:
        return self.default_site_name

    @property
    def point_world_m(self) -> Tuple[float, float, float]:
        if self.default_site_id is not None:
            values = self.data.site_xpos[self.default_site_id]
        else:
            values = self.data.xpos[self.default_segment.body_id]
        return tuple(float(value) for value in values)

    def _point_for(self, command: InteractionCommand, segment) -> np.ndarray:
        import mujoco

        if command.point_world_m is not None:
            return np.asarray(command.point_world_m, dtype=float)
        if command.target_site:
            site_id = int(mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, command.target_site
            ))
            if site_id < 0 or int(self.model.site_bodyid[site_id]) != segment.body_id:
                raise ValueError("interaction target site does not belong to target segment")
            return np.asarray(self.data.site_xpos[site_id], dtype=float)
        return np.asarray(self.data.xpos[segment.body_id], dtype=float)

    def enrich(self, command: InteractionCommand) -> InteractionCommand:
        if command.mode is InteractionMode.EXTERNAL_FORCE:
            segment = self.catalog.segment_for_body(command.target_body)
            if segment.body_id not in self.allowed_body_ids:
                raise ValueError("interaction target body is outside configured allowlist")
            point = self._point_for(command, segment)
            center = np.asarray(self.data.xipos[segment.body_id], dtype=float)
            generated = np.cross(point - center, np.asarray(command.force_n, dtype=float))
            return replace(
                command,
                target_body=segment.body_name,
                target_body_id=segment.body_id,
                point_world_m=tuple(float(value) for value in point),
                body_com_world_m=tuple(float(value) for value in center),
                generated_torque_nm=tuple(float(value) for value in generated),
            )
        if command.mode is InteractionMode.JOINT_TORQUE:
            joint = self.catalog.joint_for(command.target_joint)
            axis = int(command.target_dof_axis)
            return replace(
                command,
                target_body=joint.body_name,
                target_body_id=joint.body_id,
                target_joint=joint.joint_name,
                target_joint_id=joint.joint_id,
                target_dof_index=joint.dof_indices[axis],
            )
        return command

    def apply(self, command: InteractionCommand) -> None:
        self.clear()
        if not command.active:
            return
        command = self.enrich(command)
        if command.mode is InteractionMode.EXTERNAL_FORCE:
            self.data.xfrc_applied[command.target_body_id, :3] = command.force_n
            self.data.xfrc_applied[
                command.target_body_id, 3:
            ] = command.generated_torque_nm
        elif command.mode is InteractionMode.JOINT_TORQUE:
            self.data.qfrc_applied[command.target_dof_index] = command.joint_torque_nm

    def clear(self) -> None:
        self.data.xfrc_applied[np.asarray(self.catalog.body_ids, dtype=int)] = 0.0
        self.data.qfrc_applied[np.asarray(self.catalog.dof_indices, dtype=int)] = 0.0
