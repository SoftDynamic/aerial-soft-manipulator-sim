"""Test-only zero-control runner for the fixed-root active soft arm."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Tuple

import mujoco
import numpy as np

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    CORNERS,
    JOINT_STIFFNESS,
    MAX_TENSION_N,
    TARGET_DAMPING_RATIO,
    TendonActuatorBinding,
    resolve_tendon_actuator_bindings,
    write_tension_command,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    WarningEvent,
    WarningLatch,
    build_fixed_arm_fixture,
)


CTRL_FREQ = 240
SIM_FREQUENCIES = (2400, 4800)
INTEGRATOR = "implicitfast"
ZERO_TENSION_N = (0.0, 0.0, 0.0, 0.0)


def _finite_float(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _aligned_intervals(duration_s: float, ctrl_freq: int) -> int:
    raw_intervals = duration_s * ctrl_freq
    intervals = int(round(raw_intervals))
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(raw_intervals))
    if abs(raw_intervals - intervals) > tolerance:
        raise ValueError("duration_s must align with the control tick grid")
    if intervals <= 0:
        raise ValueError("duration_s must contain at least one control interval")
    return intervals


@dataclass(frozen=True)
class FixedRootZeroControlConfig:
    """Timing configuration for one fixed-root zero-tension run."""

    sim_freq: int = 2400
    ctrl_freq: int = CTRL_FREQ
    duration_s: float = 0.05

    def __post_init__(self):
        for name in ("sim_freq", "ctrl_freq"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sim_freq % self.ctrl_freq != 0:
            raise ValueError("sim_freq must be divisible by ctrl_freq")
        if self.ctrl_freq != CTRL_FREQ:
            raise ValueError(f"ctrl_freq must be exactly {CTRL_FREQ}")
        if self.sim_freq not in SIM_FREQUENCIES:
            raise ValueError(
                f"sim_freq must be one of {SIM_FREQUENCIES}"
            )
        duration = _finite_float("duration_s", self.duration_s)
        if duration <= 0.0:
            raise ValueError("duration_s must be positive")
        object.__setattr__(self, "duration_s", duration)
        _ = self.number_of_control_intervals

    @property
    def number_of_control_intervals(self) -> int:
        return _aligned_intervals(self.duration_s, self.ctrl_freq)

    @property
    def substeps_per_control_interval(self) -> int:
        return self.sim_freq // self.ctrl_freq

    @property
    def number_of_physics_steps(self) -> int:
        return (
            self.number_of_control_intervals
            * self.substeps_per_control_interval
        )


@dataclass(frozen=True)
class FixedRootTopologyProof:
    """Resolved topology and actuator metadata for the B0 fixture."""

    model_counts: Tuple[int, ...]
    fixture_mount_body_id: int
    fixture_mount_parent_body_id: int
    fixture_mount_joint_count: int
    root_segment_body_id: int
    root_segment_parent_body_id: int
    root_segment_joint_count: int
    joint_types: Tuple[int, ...]
    tendon_names: Tuple[str, ...]
    actuator_names: Tuple[str, ...]
    tendon_ids: Tuple[int, ...]
    actuator_ids: Tuple[int, ...]
    control_addresses: Tuple[int, ...]
    output_addresses: Tuple[int, ...]
    actuator_gear: np.ndarray
    actuator_ctrl_limited: np.ndarray
    actuator_ctrl_range_n: np.ndarray
    actuator_force_limited: np.ndarray
    actuator_force_range_n: np.ndarray
    tendon_limited: np.ndarray
    tendon_range_m: np.ndarray


@dataclass(frozen=True)
class FixedRootZeroControlTrace:
    """Immutable control-grid, interval, and physics-step B0 evidence."""

    config: FixedRootZeroControlConfig
    topology: FixedRootTopologyProof
    bindings: Tuple[TendonActuatorBinding, ...]
    time_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    tendon_length_m: np.ndarray
    tendon_velocity_m_s: np.ndarray
    tip_position_m: np.ndarray
    contact_count: np.ndarray
    warning_counts: np.ndarray
    interval_start_time_s: np.ndarray
    interval_end_time_s: np.ndarray
    commanded_tension_n: np.ndarray
    data_ctrl_n: np.ndarray
    actuator_force_n: np.ndarray
    physics_substeps: np.ndarray
    physics_time_s: np.ndarray
    physics_interval_index: np.ndarray
    physics_substep_index: np.ndarray
    physics_data_ctrl_n: np.ndarray
    physics_actuator_force_n: np.ndarray
    physics_contact_count: np.ndarray
    physics_warning_counts: np.ndarray
    number_of_control_intervals: int
    number_of_physics_steps: int
    number_of_control_writes: int
    number_of_reset_writes: int
    number_of_physics_control_checks: int
    substeps_per_control_interval: int
    maximum_control_time_error_s: float
    warning_events: Tuple[WarningEvent, ...]


def _freeze(values, *, dtype=None):
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _required_id(model, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise AssertionError(f"MuJoCo object not found: {name}")
    return int(object_id)


def _warning_counts(data) -> Tuple[int, ...]:
    return tuple(int(warning.number) for warning in data.warning)


def _time_tolerance(expected_time: float, completed_steps: int, timestep: float):
    scale = max(1.0, abs(expected_time), abs(timestep))
    return 8 * np.finfo(float).eps * max(1, completed_steps) * scale


def _range_tolerance(lower: float, upper: float, length: float) -> float:
    scale = max(abs(lower), abs(upper), abs(length))
    return max(1e-9, 1e-8 * scale)


def _require_synchronized_safe(
    fixture,
    bindings,
    tendon_ranges,
    tendon_limited,
    warning_latch,
    *,
    stage: str,
    step_index: int,
) -> None:
    model, data = fixture.model, fixture.data
    warning_latch.snapshot(data, stage=stage, step_index=step_index)
    if warning_latch.triggered:
        raise AssertionError(f"MuJoCo warning: {warning_latch.events[-1]}")

    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    control_addresses = np.asarray(
        [item.control_address for item in bindings], dtype=int
    )
    output_addresses = np.asarray(
        [item.output_address for item in bindings], dtype=int
    )
    fields = {
        "time": np.asarray([data.time]),
        "qpos": data.qpos,
        "qvel": data.qvel,
        "qacc": data.qacc,
        "site_xpos": data.site_xpos,
        "tendon_length": data.ten_length[tendon_ids],
        "tendon_velocity": data.ten_velocity[tendon_ids],
        "data_ctrl": data.ctrl[control_addresses],
        "actuator_force": data.actuator_force[output_addresses],
    }
    for name, values in fields.items():
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"non-finite {name} at {stage}:{step_index}")
    if int(data.ncon) != 0:
        raise AssertionError(f"contact at {stage}:{step_index}: {data.ncon}")
    if not np.all(tendon_limited):
        raise AssertionError("all four production tendons must remain limited")
    for index, length in enumerate(data.ten_length[tendon_ids]):
        lower, upper = tendon_ranges[index]
        tolerance = _range_tolerance(float(lower), float(upper), float(length))
        if not lower - tolerance <= length <= upper + tolerance:
            raise AssertionError(
                f"tendon range violation at {stage}:{step_index}:{index}"
            )
    if not np.array_equal(data.ctrl[control_addresses], np.zeros(4)):
        raise AssertionError(f"nonzero tendon control at {stage}:{step_index}")
    if not np.allclose(
        data.actuator_force[output_addresses], 0.0, rtol=0.0, atol=1e-15
    ):
        raise AssertionError(f"nonzero actuator force at {stage}:{step_index}")
    if model.nu != len(bindings):
        raise AssertionError("unexpected controls outside tendon bindings")


def _topology_proof(fixture, bindings) -> FixedRootTopologyProof:
    model, manifest = fixture.model, fixture.manifest
    fixture_mount = _required_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "fixture_mount"
    )
    root_segment = _required_id(
        model, mujoco.mjtObj.mjOBJ_BODY, manifest.segment_body_names[0]
    )
    actuator_ids = np.asarray([item.actuator_id for item in bindings], dtype=int)
    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    return FixedRootTopologyProof(
        model_counts=(
            int(model.nbody),
            int(model.njnt),
            int(model.nq),
            int(model.nv),
            int(model.ngeom),
            int(model.nsite),
            int(model.ntendon),
            int(model.nu),
            int(model.na),
            int(model.nout),
            int(getattr(model, "nactuator", model.nu)),
            int(model.nsensor),
        ),
        fixture_mount_body_id=fixture_mount,
        fixture_mount_parent_body_id=int(model.body_parentid[fixture_mount]),
        fixture_mount_joint_count=int(model.body_jntnum[fixture_mount]),
        root_segment_body_id=root_segment,
        root_segment_parent_body_id=int(model.body_parentid[root_segment]),
        root_segment_joint_count=int(model.body_jntnum[root_segment]),
        joint_types=tuple(int(value) for value in model.jnt_type),
        tendon_names=tuple(manifest.tendon_names),
        actuator_names=tuple(manifest.tendon_actuator_names),
        tendon_ids=tuple(int(value) for value in tendon_ids),
        actuator_ids=tuple(int(value) for value in actuator_ids),
        control_addresses=tuple(item.control_address for item in bindings),
        output_addresses=tuple(item.output_address for item in bindings),
        actuator_gear=_freeze(model.actuator_gear[actuator_ids]),
        actuator_ctrl_limited=_freeze(
            model.actuator_ctrllimited[actuator_ids], dtype=bool
        ),
        actuator_ctrl_range_n=_freeze(model.actuator_ctrlrange[actuator_ids]),
        actuator_force_limited=_freeze(
            model.actuator_forcelimited[actuator_ids], dtype=bool
        ),
        actuator_force_range_n=_freeze(
            model.actuator_forcerange[actuator_ids]
        ),
        tendon_limited=_freeze(model.tendon_limited[tendon_ids], dtype=bool),
        tendon_range_m=_freeze(model.tendon_range[tendon_ids]),
    )


def _state_sample(fixture, bindings, tip_id):
    data = fixture.data
    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    return {
        "time": float(data.time),
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "tendon_length": np.asarray(data.ten_length[tendon_ids]).copy(),
        "tendon_velocity": np.asarray(data.ten_velocity[tendon_ids]).copy(),
        "tip": np.asarray(data.site_xpos[tip_id]).copy(),
        "contacts": int(data.ncon),
        "warnings": _warning_counts(data),
    }


def run_fixed_root_zero_control(
    config: FixedRootZeroControlConfig,
) -> FixedRootZeroControlTrace:
    """Run one vertical fixed-root active arm with four exact zero inputs."""
    if not isinstance(config, FixedRootZeroControlConfig):
        raise TypeError("config must be a FixedRootZeroControlConfig")

    arm_spec = ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )
    drive_spec = IdealTendonTensionSpec(max_tension=MAX_TENSION_N)
    fixture = build_fixed_arm_fixture(
        arm_spec,
        "vertical",
        sim_freq=config.sim_freq,
        ctrl_freq=config.ctrl_freq,
        integrator=INTEGRATOR,
        tendon_drive=drive_spec,
    )
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    bindings = tuple(resolve_tendon_actuator_bindings(fixture))
    if tuple(item.tendon_name for item in bindings) != tuple(
        manifest.tendon_names
    ):
        raise AssertionError("tendon bindings do not preserve manifest order")
    if tuple(item.actuator_name for item in bindings) != tuple(
        manifest.tendon_actuator_names
    ):
        raise AssertionError("actuator bindings do not preserve manifest order")

    topology = _topology_proof(fixture, bindings)
    expected_suffixes = tuple(f"_{corner}_tendon" for corner in CORNERS)
    if not all(
        name.endswith(suffix)
        for name, suffix in zip(topology.tendon_names, expected_suffixes)
    ):
        raise AssertionError("formal tendon manifest order changed")
    if topology.fixture_mount_parent_body_id != 0:
        raise AssertionError("fixture mount is not attached directly to world")
    if topology.fixture_mount_joint_count != 0:
        raise AssertionError("fixture mount must have no joint")
    if topology.root_segment_parent_body_id != topology.fixture_mount_body_id:
        raise AssertionError("arm root is not parented by fixture mount")
    if topology.root_segment_joint_count != 0:
        raise AssertionError("arm root segment must be welded")
    if len(topology.joint_types) != 8 or any(
        value != int(mujoco.mjtJoint.mjJNT_BALL)
        for value in topology.joint_types
    ):
        raise AssertionError("fixture must contain exactly eight ball joints")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone0") >= 0:
        raise AssertionError("fixed-root fixture unexpectedly contains a UAV")
    if not np.allclose(
        model.opt.gravity, (0.0, 0.0, -9.81), rtol=0.0, atol=0.0
    ):
        raise AssertionError("fixed-root gravity changed")
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise AssertionError("fixed-root integrator changed")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0:
        raise AssertionError("fixed-root fixture unexpectedly contains a floor")
    if np.any(model.geom_contype) or np.any(model.geom_conaffinity):
        raise AssertionError("fixed-root arm collision must remain disabled")

    tendon_ids = np.asarray(topology.tendon_ids, dtype=int)
    control_addresses = np.asarray(topology.control_addresses, dtype=int)
    output_addresses = np.asarray(topology.output_addresses, dtype=int)
    tendon_ranges = np.asarray(model.tendon_range[tendon_ids]).copy()
    tendon_limited = np.asarray(model.tendon_limited[tendon_ids]).copy()
    tip_id = _required_id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name
    )

    mujoco.mj_resetData(model, data)
    if not np.array_equal(data.qpos, model.qpos0):
        raise AssertionError("mj_resetData did not restore qpos0")
    if not np.array_equal(data.qvel, np.zeros(model.nv)):
        raise AssertionError("initial qvel is not exactly zero")
    reset_command, reset_bindings = write_tension_command(
        fixture, ZERO_TENSION_N, drive_spec, bindings=bindings
    )
    if reset_bindings != bindings or reset_command != ZERO_TENSION_N:
        raise AssertionError("reset tendon binding or command changed")
    mujoco.mj_forward(model, data)
    warning_latch = WarningLatch()
    _require_synchronized_safe(
        fixture,
        bindings,
        tendon_ranges,
        tendon_limited,
        warning_latch,
        stage="reset_forward",
        step_index=0,
    )

    state_records = [_state_sample(fixture, bindings, tip_id)]
    interval_records = []
    physics_records = []
    completed_steps = 0
    control_writes = 0
    maximum_time_error = 0.0
    timestep = float(model.opt.timestep)

    for interval_index in range(config.number_of_control_intervals):
        expected_start = interval_index / config.ctrl_freq
        if float(data.time) != expected_start:
            raise AssertionError("control interval did not start on exact grid")
        command, current_bindings = write_tension_command(
            fixture, ZERO_TENSION_N, drive_spec, bindings=bindings
        )
        control_writes += 1
        if current_bindings != bindings or command != ZERO_TENSION_N:
            raise AssertionError("control binding or zero command changed")
        mujoco.mj_forward(model, data)
        _require_synchronized_safe(
            fixture,
            bindings,
            tendon_ranges,
            tendon_limited,
            warning_latch,
            stage="control_forward",
            step_index=completed_steps,
        )
        interval_records.append(
            {
                "start": expected_start,
                "end": (interval_index + 1) / config.ctrl_freq,
                "command": np.asarray(command).copy(),
                "ctrl": np.asarray(data.ctrl[control_addresses]).copy(),
                "force": np.asarray(
                    data.actuator_force[output_addresses]
                ).copy(),
                "substeps": config.substeps_per_control_interval,
            }
        )

        for substep_index in range(config.substeps_per_control_interval):
            if not np.array_equal(data.ctrl[control_addresses], np.zeros(4)):
                raise AssertionError("data.ctrl changed before a physics step")
            mujoco.mj_step(model, data)
            completed_steps += 1
            if not (
                math.isfinite(float(data.time))
                and np.all(np.isfinite(data.qpos))
                and np.all(np.isfinite(data.qvel))
            ):
                raise AssertionError("non-finite raw physics state")
            mujoco.mj_forward(model, data)
            _require_synchronized_safe(
                fixture,
                bindings,
                tendon_ranges,
                tendon_limited,
                warning_latch,
                stage="physics_forward",
                step_index=completed_steps,
            )
            physics_records.append(
                {
                    "time": float(data.time),
                    "interval": interval_index,
                    "substep": substep_index + 1,
                    "ctrl": np.asarray(data.ctrl[control_addresses]).copy(),
                    "force": np.asarray(
                        data.actuator_force[output_addresses]
                    ).copy(),
                    "contacts": int(data.ncon),
                    "warnings": _warning_counts(data),
                }
            )

        expected_end = (interval_index + 1) / config.ctrl_freq
        time_error = abs(float(data.time) - expected_end)
        maximum_time_error = max(maximum_time_error, time_error)
        tolerance = _time_tolerance(expected_end, completed_steps, timestep)
        if time_error > tolerance:
            raise AssertionError(
                f"control time error {time_error} exceeds {tolerance}"
            )
        # Reuse the established active-run convention: after validating the
        # accumulated MuJoCo clock, snap only time to the exact control grid.
        data.time = expected_end
        state_records.append(_state_sample(fixture, bindings, tip_id))

    if completed_steps != config.number_of_physics_steps:
        raise AssertionError("fixed-root run ended before all physics steps")

    def state_values(name, *, dtype=None):
        return _freeze([item[name] for item in state_records], dtype=dtype)

    def interval_values(name, *, dtype=None):
        return _freeze([item[name] for item in interval_records], dtype=dtype)

    def physics_values(name, *, dtype=None):
        return _freeze([item[name] for item in physics_records], dtype=dtype)

    return FixedRootZeroControlTrace(
        config=config,
        topology=topology,
        bindings=bindings,
        time_s=state_values("time"),
        qpos=state_values("qpos"),
        qvel=state_values("qvel"),
        tendon_length_m=state_values("tendon_length"),
        tendon_velocity_m_s=state_values("tendon_velocity"),
        tip_position_m=state_values("tip"),
        contact_count=state_values("contacts", dtype=int),
        warning_counts=state_values("warnings", dtype=int),
        interval_start_time_s=interval_values("start"),
        interval_end_time_s=interval_values("end"),
        commanded_tension_n=interval_values("command"),
        data_ctrl_n=interval_values("ctrl"),
        actuator_force_n=interval_values("force"),
        physics_substeps=interval_values("substeps", dtype=int),
        physics_time_s=physics_values("time"),
        physics_interval_index=physics_values("interval", dtype=int),
        physics_substep_index=physics_values("substep", dtype=int),
        physics_data_ctrl_n=physics_values("ctrl"),
        physics_actuator_force_n=physics_values("force"),
        physics_contact_count=physics_values("contacts", dtype=int),
        physics_warning_counts=physics_values("warnings", dtype=int),
        number_of_control_intervals=len(interval_records),
        number_of_physics_steps=completed_steps,
        number_of_control_writes=control_writes,
        number_of_reset_writes=1,
        number_of_physics_control_checks=len(physics_records),
        substeps_per_control_interval=fixture.sample_stride,
        maximum_control_time_error_s=maximum_time_error,
        warning_events=tuple(warning_latch.events),
    )
