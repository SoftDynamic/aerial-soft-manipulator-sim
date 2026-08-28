"""Headless fixed-root MuJoCo runner for Gate C2-B1b probes."""

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Optional, Tuple

import mujoco
import numpy as np

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)
from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_probe import (
    AlignedProbeTrace,
    FixedTensionProfileConfig,
    build_fixed_tension_command_profile,
    integrate_interval_power_w,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    CORNERS,
    JOINT_STIFFNESS,
    MAX_TENSION_N,
    TARGET_DAMPING_RATIO,
    TendonActuatorBinding,
    resolve_tendon_actuator_bindings,
    sample_ideal_tendon_transmission,
    write_tension_command,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    WarningEvent,
    WarningLatch,
    build_fixed_arm_fixture,
    composite_rotation_vector,
    joint_rotation_vectors,
    named_joint_addresses,
)


CTRL_FREQ_HZ = 240
SIM_FREQUENCIES_HZ = (2400, 4800)
INTEGRATOR = "implicitfast"
ZERO_COMMAND_N = (0.0, 0.0, 0.0, 0.0)


def _integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class FixedRootProbeRunConfig:
    """Validated timing, tension, and selector for one B1b run."""

    sim_freq_hz: int
    ctrl_freq_hz: int
    duration_s: float
    ramp_duration_s: float
    target_tension_n: float
    max_tension_n: float
    active_tendon_corner: str

    def __post_init__(self) -> None:
        sim_freq = _integer("sim_freq_hz", self.sim_freq_hz)
        ctrl_freq = _integer("ctrl_freq_hz", self.ctrl_freq_hz)
        if sim_freq not in SIM_FREQUENCIES_HZ:
            raise ValueError(
                f"sim_freq_hz must be one of {SIM_FREQUENCIES_HZ}"
            )
        if ctrl_freq != CTRL_FREQ_HZ:
            raise ValueError(
                f"ctrl_freq_hz must be exactly {CTRL_FREQ_HZ}"
            )
        if not isinstance(self.active_tendon_corner, str):
            raise ValueError("active_tendon_corner must be a string")
        corner = self.active_tendon_corner.strip()
        if not corner:
            raise ValueError("active_tendon_corner must not be empty")

        # B1a remains the sole profile-value and integer-grid validator.  The
        # sentinel index is replaced by the manifest-resolved index before a
        # command profile is built.
        validated = FixedTensionProfileConfig(
            ctrl_freq_hz=ctrl_freq,
            duration_s=self.duration_s,
            ramp_duration_s=self.ramp_duration_s,
            target_tension_n=self.target_tension_n,
            max_tension_n=self.max_tension_n,
            num_tendons=4,
            active_channel_index=0,
        )
        if validated.max_tension_n > MAX_TENSION_N:
            raise ValueError(
                f"max_tension_n must not exceed {MAX_TENSION_N}"
            )
        object.__setattr__(self, "sim_freq_hz", sim_freq)
        object.__setattr__(self, "ctrl_freq_hz", ctrl_freq)
        object.__setattr__(self, "duration_s", validated.duration_s)
        object.__setattr__(
            self, "ramp_duration_s", validated.ramp_duration_s
        )
        object.__setattr__(
            self, "target_tension_n", validated.target_tension_n
        )
        object.__setattr__(self, "max_tension_n", validated.max_tension_n)
        object.__setattr__(self, "active_tendon_corner", corner)

    @property
    def number_of_control_intervals(self) -> int:
        return self.profile_config_for_channel(0).number_of_control_intervals

    @property
    def ramp_ticks(self) -> int:
        return self.profile_config_for_channel(0).ramp_ticks

    @property
    def substeps_per_control_interval(self) -> int:
        return self.sim_freq_hz // self.ctrl_freq_hz

    @property
    def number_of_physics_steps(self) -> int:
        return (
            self.number_of_control_intervals
            * self.substeps_per_control_interval
        )

    def profile_config_for_channel(
        self, active_channel_index: int
    ) -> FixedTensionProfileConfig:
        return FixedTensionProfileConfig(
            ctrl_freq_hz=self.ctrl_freq_hz,
            duration_s=self.duration_s,
            ramp_duration_s=self.ramp_duration_s,
            target_tension_n=self.target_tension_n,
            max_tension_n=self.max_tension_n,
            num_tendons=4,
            active_channel_index=active_channel_index,
        )


@dataclass(frozen=True)
class ResolvedProbeChannel:
    corner: str
    tendon_name: str
    actuator_name: str
    binding_index: int
    tendon_id: int
    actuator_id: int
    control_address: int
    output_address: int


def resolve_probe_channel(
    corner: str,
    tendon_names,
    actuator_names,
    bindings,
) -> ResolvedProbeChannel:
    """Resolve one corner by names without deriving layout from numeric IDs."""
    if not isinstance(corner, str) or not corner:
        raise ValueError("corner must be a non-empty string")
    tendon_names = tuple(tendon_names)
    actuator_names = tuple(actuator_names)
    bindings = tuple(bindings)
    if len(tendon_names) != len(actuator_names):
        raise ValueError("manifest tendon and actuator counts must match")
    suffix = f"_{corner}_tendon"
    manifest_matches = [
        index
        for index, tendon_name in enumerate(tendon_names)
        if tendon_name.endswith(suffix)
    ]
    if len(manifest_matches) != 1:
        raise ValueError(
            f"corner {corner!r} must match exactly one manifest tendon"
        )
    manifest_index = manifest_matches[0]
    tendon_name = tendon_names[manifest_index]
    actuator_name = actuator_names[manifest_index]
    binding_matches = [
        (index, binding)
        for index, binding in enumerate(bindings)
        if binding.tendon_name == tendon_name
    ]
    if len(binding_matches) != 1:
        raise ValueError(
            f"tendon {tendon_name!r} must match exactly one binding"
        )
    binding_index, binding = binding_matches[0]
    if binding.actuator_name != actuator_name:
        raise ValueError("manifest and binding actuator names do not match")
    return ResolvedProbeChannel(
        corner=corner,
        tendon_name=tendon_name,
        actuator_name=actuator_name,
        binding_index=binding_index,
        tendon_id=binding.tendon_id,
        actuator_id=binding.actuator_id,
        control_address=binding.control_address,
        output_address=binding.output_address,
    )


@dataclass(frozen=True)
class ProbeRunFailure:
    stage: str
    control_interval: int
    physics_substep: int
    physics_step: int
    time_s: float
    field: str
    value: object


class ProbeRunSafetyError(AssertionError):
    """Fail a B1b run while retaining the first unsafe sample context."""

    def __init__(self, failure: ProbeRunFailure):
        self.failure = failure
        super().__init__(
            f"{failure.field} failed at stage={failure.stage}, "
            f"interval={failure.control_interval}, "
            f"substep={failure.physics_substep}, "
            f"step={failure.physics_step}, time={failure.time_s}: "
            f"{failure.value}"
        )


@dataclass(frozen=True)
class ProbeSafetySummary:
    completed_physics_steps: int
    warning_events: Tuple[WarningEvent, ...]
    maximum_contacts: int
    minimum_tendon_range_margin_m: float
    maximum_force_command_error_n: float
    maximum_moment_residual: float
    maximum_generalized_actuator_power_residual_w: float
    maximum_actuator_tendon_power_residual_w: float
    maximum_command_tendon_power_residual_w: float
    maximum_actuator_length_sign_residual_m: float
    maximum_actuator_velocity_sign_residual_m_s: float


@dataclass
class _SafetyAccumulator:
    completed_physics_steps: int = 0
    maximum_contacts: int = 0
    minimum_tendon_range_margin_m: float = math.inf
    maximum_force_command_error_n: float = 0.0
    maximum_moment_residual: float = 0.0
    maximum_generalized_actuator_power_residual_w: float = 0.0
    maximum_actuator_tendon_power_residual_w: float = 0.0
    maximum_command_tendon_power_residual_w: float = 0.0
    maximum_actuator_length_sign_residual_m: float = 0.0
    maximum_actuator_velocity_sign_residual_m_s: float = 0.0


def _freeze(values, *, dtype=None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FixedRootProbeTrace:
    """Immutable B1b state, control-interval, and physics-rate evidence."""

    config: FixedRootProbeRunConfig
    profile_config: FixedTensionProfileConfig
    resolved_channel: ResolvedProbeChannel
    bindings: Tuple[TendonActuatorBinding, ...]
    model_counts: Tuple[int, ...]
    tendon_names: Tuple[str, ...]
    actuator_names: Tuple[str, ...]
    tendon_range_m: np.ndarray
    tendon_limited: np.ndarray
    actuator_gear: np.ndarray
    actuator_ctrl_range_n: np.ndarray
    time_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    tendon_length_m: np.ndarray
    tendon_velocity_m_s: np.ndarray
    tip_position_m: np.ndarray
    joint_rotation_vectors_rad: np.ndarray
    composite_rotation_vectors_rad: np.ndarray
    spring_energy_j: np.ndarray
    position_energy_j: np.ndarray
    kinetic_energy_j: np.ndarray
    total_energy_j: np.ndarray
    contact_count: np.ndarray
    warning_counts: np.ndarray
    interval_start_time_s: np.ndarray
    interval_end_time_s: np.ndarray
    commanded_tension_n: np.ndarray
    data_ctrl_n: np.ndarray
    actuator_force_n: np.ndarray
    physics_substeps: np.ndarray
    physics_time_s: np.ndarray
    physics_data_ctrl_n: np.ndarray
    physics_actuator_force_n: np.ndarray
    physics_actuator_velocity_m_s: np.ndarray
    physics_tendon_velocity_m_s: np.ndarray
    physics_generalized_power_w: np.ndarray
    physics_actuator_power_w: np.ndarray
    physics_tendon_power_w: np.ndarray
    physics_command_power_w: np.ndarray
    physics_contact_count: np.ndarray
    physics_warning_counts: np.ndarray
    physics_finite_state: np.ndarray
    number_of_control_intervals: int
    number_of_physics_steps: int
    maximum_control_time_error_s: float
    safety: ProbeSafetySummary

    def to_aligned_probe_trace(self) -> AlignedProbeTrace:
        return AlignedProbeTrace(
            time_s=self.time_s,
            tendon_length_m=self.tendon_length_m,
            tendon_velocity_m_s=self.tendon_velocity_m_s,
            tip_position_m=self.tip_position_m,
        )

    @property
    def generalized_actuator_work_j(self) -> float:
        return integrate_interval_power_w(
            self.physics_generalized_power_w,
            dt_sim_s=1.0 / self.config.sim_freq_hz,
        )

    @property
    def actuator_work_j(self) -> float:
        return integrate_interval_power_w(
            self.physics_actuator_power_w,
            dt_sim_s=1.0 / self.config.sim_freq_hz,
        )

    @property
    def tendon_work_j(self) -> float:
        return integrate_interval_power_w(
            self.physics_tendon_power_w,
            dt_sim_s=1.0 / self.config.sim_freq_hz,
        )

    @property
    def command_work_j(self) -> float:
        return integrate_interval_power_w(
            self.physics_command_power_w,
            dt_sim_s=1.0 / self.config.sim_freq_hz,
        )


def _required_id(model, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise AssertionError(f"MuJoCo object not found: {name}")
    return int(object_id)


def _warning_counts(data) -> Tuple[int, ...]:
    return tuple(int(warning.number) for warning in data.warning)


def _range_tolerance(lower: float, upper: float, length: float) -> float:
    scale = max(abs(lower), abs(upper), abs(length))
    return max(1e-9, 1e-8 * scale)


def _time_tolerance(expected_time: float, completed_steps: int, timestep: float):
    scale = max(1.0, abs(expected_time), abs(timestep))
    return 8 * np.finfo(float).eps * max(1, completed_steps) * scale


def _fail(
    *,
    stage,
    control_interval,
    physics_substep,
    physics_step,
    time_s,
    field,
    value,
):
    raise ProbeRunSafetyError(
        ProbeRunFailure(
            stage=str(stage),
            control_interval=int(control_interval),
            physics_substep=int(physics_substep),
            physics_step=int(physics_step),
            time_s=float(time_s),
            field=str(field),
            value=value,
        )
    )


def _require(condition, *, field, value, **context) -> None:
    if not bool(condition):
        _fail(field=field, value=value, **context)


def _validate_topology(fixture, bindings, config):
    model, manifest = fixture.model, fixture.manifest
    if tuple(item.tendon_name for item in bindings) != tuple(
        manifest.tendon_names
    ):
        raise AssertionError("bindings do not preserve manifest tendon order")
    if tuple(item.actuator_name for item in bindings) != tuple(
        manifest.tendon_actuator_names
    ):
        raise AssertionError("bindings do not preserve manifest actuator order")
    counts = (
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
    )
    if counts != (11, 8, 32, 24, 10, 75, 4, 4, 0, 4, 4, 4):
        raise AssertionError(f"fixed-root topology changed: {counts}")
    mount_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "fixture_mount")
    root_id = _required_id(
        model, mujoco.mjtObj.mjOBJ_BODY, manifest.segment_body_names[0]
    )
    checks = {
        "fixture_mount_parent": int(model.body_parentid[mount_id]) == 0,
        "fixture_mount_welded": int(model.body_jntnum[mount_id]) == 0,
        "root_parent": int(model.body_parentid[root_id]) == mount_id,
        "root_welded": int(model.body_jntnum[root_id]) == 0,
        "gravity": np.array_equal(model.opt.gravity, [0.0, 0.0, -9.81]),
        "integrator": (
            model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        ),
        "no_uav": (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone0") < 0
        ),
        "no_floor": (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") < 0
        ),
        "collision_disabled": (
            not np.any(model.geom_contype)
            and not np.any(model.geom_conaffinity)
        ),
        "sample_stride": (
            fixture.sample_stride == config.substeps_per_control_interval
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"fixed-root topology checks failed: {failed}")
    if len(bindings) != 4:
        raise AssertionError("fixed-root probe requires four tendon bindings")
    return counts


def _spring_energy(model, joint_vectors, addresses) -> float:
    return float(
        math.fsum(
            0.5
            * float(model.jnt_stiffness[joint_id])
            * float(np.dot(joint_vectors[index], joint_vectors[index]))
            for index, (joint_id, _, _) in enumerate(addresses)
        )
    )


def _state_sample(fixture, bindings, addresses, tip_id):
    model, data = fixture.model, fixture.data
    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    joints = joint_rotation_vectors(model, data, addresses)
    composite = composite_rotation_vector(model, data, addresses)
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    result = {
        "time": float(data.time),
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "tendon_length": np.asarray(data.ten_length[tendon_ids]).copy(),
        "tendon_velocity": np.asarray(data.ten_velocity[tendon_ids]).copy(),
        "tip": np.asarray(data.site_xpos[tip_id]).copy(),
        "joints": joints,
        "composite": composite,
        "spring_energy": _spring_energy(model, joints, addresses),
        "position_energy": float(data.energy[0]),
        "kinetic_energy": float(data.energy[1]),
        "total_energy": float(np.sum(data.energy)),
        "contacts": int(data.ncon),
        "warnings": _warning_counts(data),
    }
    if not all(
        np.all(np.isfinite(value))
        for key, value in result.items()
        if key not in ("contacts", "warnings")
    ):
        raise AssertionError("non-finite control-rate state sample")
    return result


def _check_synchronized_sample(
    fixture,
    bindings,
    command,
    active_index,
    tendon_ranges,
    tendon_limited,
    warning_latch,
    accumulator,
    *,
    stage,
    control_interval,
    physics_substep,
    physics_step,
):
    model, data = fixture.model, fixture.data
    context = {
        "stage": stage,
        "control_interval": control_interval,
        "physics_substep": physics_substep,
        "physics_step": physics_step,
        "time_s": data.time,
    }
    warning_latch.snapshot(data, stage=stage, step_index=physics_step)
    _require(
        not warning_latch.triggered,
        field="mujoco_warning",
        value=(warning_latch.events[-1] if warning_latch.events else None),
        **context,
    )
    command = np.asarray(command, dtype=float)
    transmission = sample_ideal_tendon_transmission(
        model, data, bindings, command
    )
    control_addresses = np.asarray(
        [item.control_address for item in bindings], dtype=int
    )
    ctrl = np.asarray(data.ctrl[control_addresses]).copy()
    finite_fields = {
        "time": np.asarray([data.time]),
        "qpos": data.qpos,
        "qvel": data.qvel,
        "qacc": data.qacc,
        "site_xpos": data.site_xpos,
        "tendon_length": transmission.tendon_length,
        "tendon_velocity": transmission.tendon_velocity,
        "actuator_length": transmission.actuator_length,
        "actuator_velocity": transmission.actuator_velocity,
        "actuator_force": transmission.actuator_force,
        "qfrc_actuator": data.qfrc_actuator,
        "ctrl": ctrl,
        "power": np.asarray(
            [
                transmission.generalized_power,
                transmission.actuator_power,
                transmission.tendon_power,
                transmission.command_tendon_power,
            ]
        ),
    }
    for name, values in finite_fields.items():
        _require(
            np.all(np.isfinite(values)),
            field=f"finite:{name}",
            value=np.asarray(values).shape,
            **context,
        )
    accumulator.maximum_contacts = max(
        accumulator.maximum_contacts, int(data.ncon)
    )
    _require(
        int(data.ncon) == 0,
        field="contacts",
        value=int(data.ncon),
        **context,
    )
    _require(
        np.all(tendon_limited),
        field="tendon_limited",
        value=tendon_limited.tolist(),
        **context,
    )
    margins = np.minimum(
        transmission.tendon_length - tendon_ranges[:, 0],
        tendon_ranges[:, 1] - transmission.tendon_length,
    )
    for index, length in enumerate(transmission.tendon_length):
        lower, upper = tendon_ranges[index]
        margin = float(margins[index])
        tolerance = _range_tolerance(
            float(lower), float(upper), float(length)
        )
        accumulator.minimum_tendon_range_margin_m = min(
            accumulator.minimum_tendon_range_margin_m, margin
        )
        _require(
            margin >= -tolerance,
            field=f"tendon_range:{bindings[index].tendon_name}",
            value={"length": float(length), "margin": margin},
            **context,
        )

    _require(
        np.array_equal(ctrl, command),
        field="data_ctrl",
        value={"ctrl": ctrl.tolist(), "command": command.tolist()},
        **context,
    )
    inactive = np.ones(4, dtype=bool)
    inactive[active_index] = False
    _require(
        np.array_equal(command[inactive], np.zeros(3)),
        field="inactive_command",
        value=command[inactive].tolist(),
        **context,
    )
    _require(
        np.array_equal(ctrl[inactive], np.zeros(3)),
        field="inactive_ctrl",
        value=ctrl[inactive].tolist(),
        **context,
    )
    force_error = float(
        np.max(np.abs(transmission.actuator_force - command))
    )
    accumulator.maximum_force_command_error_n = max(
        accumulator.maximum_force_command_error_n, force_error
    )
    _require(
        force_error <= 1e-15,
        field="force_command_error_n",
        value=force_error,
        **context,
    )
    _require(
        np.all(np.abs(transmission.actuator_force[inactive]) <= 1e-15),
        field="inactive_actuator_force_n",
        value=transmission.actuator_force[inactive].tolist(),
        **context,
    )

    moment_residual = float(
        np.max(np.abs(data.qfrc_actuator - transmission.expected_qfrc))
    )
    moment_scale = max(
        1.0,
        float(np.max(np.abs(data.qfrc_actuator))),
        float(np.max(np.abs(transmission.expected_qfrc))),
    )
    accumulator.maximum_moment_residual = max(
        accumulator.maximum_moment_residual, moment_residual
    )
    _require(
        moment_residual <= 1e-12 + 1e-10 * moment_scale,
        field="moment_residual",
        value=moment_residual,
        **context,
    )

    power_scale = max(
        1e-12,
        abs(transmission.generalized_power),
        abs(transmission.actuator_power),
        abs(transmission.tendon_power),
        abs(transmission.command_tendon_power),
    )
    power_tolerance = 1e-12 + 1e-9 * power_scale
    residuals = (
        (
            "generalized_actuator_power_residual_w",
            abs(transmission.generalized_power - transmission.actuator_power),
            "maximum_generalized_actuator_power_residual_w",
        ),
        (
            "actuator_tendon_power_residual_w",
            abs(transmission.actuator_power - transmission.tendon_power),
            "maximum_actuator_tendon_power_residual_w",
        ),
        (
            "command_tendon_power_residual_w",
            abs(
                transmission.tendon_power
                - transmission.command_tendon_power
            ),
            "maximum_command_tendon_power_residual_w",
        ),
    )
    for field, residual, accumulator_field in residuals:
        setattr(
            accumulator,
            accumulator_field,
            max(getattr(accumulator, accumulator_field), residual),
        )
        _require(
            residual <= power_tolerance,
            field=field,
            value=residual,
            **context,
        )

    length_residual = float(
        np.max(
            np.abs(transmission.actuator_length + transmission.tendon_length)
        )
    )
    velocity_residual = float(
        np.max(
            np.abs(
                transmission.actuator_velocity
                + transmission.tendon_velocity
            )
        )
    )
    accumulator.maximum_actuator_length_sign_residual_m = max(
        accumulator.maximum_actuator_length_sign_residual_m,
        length_residual,
    )
    accumulator.maximum_actuator_velocity_sign_residual_m_s = max(
        accumulator.maximum_actuator_velocity_sign_residual_m_s,
        velocity_residual,
    )
    _require(
        length_residual
        <= 1e-12
        + 1e-10 * float(np.max(np.abs(transmission.tendon_length))),
        field="actuator_length_sign_residual_m",
        value=length_residual,
        **context,
    )
    _require(
        velocity_residual
        <= 1e-12
        + 1e-10
        * max(1e-12, float(np.max(np.abs(transmission.tendon_velocity)))),
        field="actuator_velocity_sign_residual_m_s",
        value=velocity_residual,
        **context,
    )
    return {
        "time": float(data.time),
        "ctrl": ctrl,
        "actuator_force": transmission.actuator_force,
        "actuator_velocity": transmission.actuator_velocity,
        "tendon_velocity": transmission.tendon_velocity,
        "generalized_power": transmission.generalized_power,
        "actuator_power": transmission.actuator_power,
        "tendon_power": transmission.tendon_power,
        "command_power": transmission.command_tendon_power,
        "contacts": int(data.ncon),
        "warnings": _warning_counts(data),
        "finite": True,
    }


def _records_array(records, name, *, dtype=None):
    return _freeze([record[name] for record in records], dtype=dtype)


def _grouped_physics_array(records, name, *, dtype=None):
    return _freeze(
        [[sample[name] for sample in interval] for interval in records],
        dtype=dtype,
    )


def run_fixed_root_tension_probe(
    config: FixedRootProbeRunConfig,
) -> FixedRootProbeTrace:
    """Run one zero or single-tendon B1b command profile headlessly."""
    if not isinstance(config, FixedRootProbeRunConfig):
        raise TypeError("config must be a FixedRootProbeRunConfig")
    arm_spec = ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )
    drive_spec = IdealTendonTensionSpec(max_tension=config.max_tension_n)
    fixture = build_fixed_arm_fixture(
        arm_spec,
        "vertical",
        sim_freq=config.sim_freq_hz,
        ctrl_freq=config.ctrl_freq_hz,
        integrator=INTEGRATOR,
        tendon_drive=drive_spec,
    )
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    bindings = tuple(resolve_tendon_actuator_bindings(fixture))
    model_counts = _validate_topology(fixture, bindings, config)
    resolved = resolve_probe_channel(
        config.active_tendon_corner,
        manifest.tendon_names,
        manifest.tendon_actuator_names,
        bindings,
    )
    profile_config = config.profile_config_for_channel(
        resolved.binding_index
    )
    command_profile = build_fixed_tension_command_profile(profile_config)
    addresses = named_joint_addresses(model, manifest)
    tip_id = _required_id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name
    )
    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    actuator_ids = np.asarray(
        [item.actuator_id for item in bindings], dtype=int
    )
    tendon_ranges = np.asarray(model.tendon_range[tendon_ids]).copy()
    tendon_limited = np.asarray(model.tendon_limited[tendon_ids]).copy()

    mujoco.mj_resetData(model, data)
    if not np.array_equal(data.qpos, model.qpos0):
        raise AssertionError("mj_resetData did not restore qpos0")
    if not np.array_equal(data.qvel, np.zeros(model.nv)):
        raise AssertionError("initial qvel is not exactly zero")
    reset_command, reset_bindings = write_tension_command(
        fixture, ZERO_COMMAND_N, drive_spec, bindings=bindings
    )
    if reset_bindings != bindings or reset_command != ZERO_COMMAND_N:
        raise AssertionError("reset command or binding changed")
    mujoco.mj_forward(model, data)
    warning_latch = WarningLatch()
    accumulator = _SafetyAccumulator()
    _check_synchronized_sample(
        fixture,
        bindings,
        reset_command,
        resolved.binding_index,
        tendon_ranges,
        tendon_limited,
        warning_latch,
        accumulator,
        stage="reset_forward",
        control_interval=0,
        physics_substep=0,
        physics_step=0,
    )

    state_records = [_state_sample(fixture, bindings, addresses, tip_id)]
    interval_records = []
    physics_records = []
    maximum_time_error = 0.0
    timestep = float(model.opt.timestep)

    for interval_index, requested_command in enumerate(command_profile):
        expected_start = interval_index / config.ctrl_freq_hz
        if float(data.time) != expected_start:
            raise AssertionError("control interval did not start on exact grid")
        command, current_bindings = write_tension_command(
            fixture,
            requested_command,
            drive_spec,
            bindings=bindings,
        )
        if current_bindings != bindings:
            raise AssertionError("tendon bindings changed during run")
        mujoco.mj_forward(model, data)
        samples = [
            _check_synchronized_sample(
                fixture,
                bindings,
                command,
                resolved.binding_index,
                tendon_ranges,
                tendon_limited,
                warning_latch,
                accumulator,
                stage="control_forward",
                control_interval=interval_index,
                physics_substep=0,
                physics_step=accumulator.completed_physics_steps,
            )
        ]
        interval_records.append(
            {
                "start": expected_start,
                "end": (interval_index + 1) / config.ctrl_freq_hz,
                "command": np.asarray(command).copy(),
                "ctrl": samples[0]["ctrl"],
                "force": samples[0]["actuator_force"],
                "substeps": config.substeps_per_control_interval,
            }
        )

        for substep_index in range(1, config.substeps_per_control_interval + 1):
            control_addresses = np.asarray(
                [item.control_address for item in bindings], dtype=int
            )
            if not np.array_equal(data.ctrl[control_addresses], command):
                raise AssertionError("data.ctrl changed before physics step")
            next_step = accumulator.completed_physics_steps + 1
            mujoco.mj_step(model, data)
            raw_finite = bool(
                math.isfinite(float(data.time))
                and np.all(np.isfinite(data.qpos))
                and np.all(np.isfinite(data.qvel))
            )
            if not raw_finite:
                _fail(
                    stage="after_step",
                    control_interval=interval_index,
                    physics_substep=substep_index,
                    physics_step=next_step,
                    time_s=data.time,
                    field="raw_finite_state",
                    value=False,
                )
            warning_latch.snapshot(
                data, stage="after_step", step_index=next_step
            )
            if warning_latch.triggered:
                _fail(
                    stage="after_step",
                    control_interval=interval_index,
                    physics_substep=substep_index,
                    physics_step=next_step,
                    time_s=data.time,
                    field="mujoco_warning",
                    value=warning_latch.events[-1],
                )
            mujoco.mj_forward(model, data)
            samples.append(
                _check_synchronized_sample(
                    fixture,
                    bindings,
                    command,
                    resolved.binding_index,
                    tendon_ranges,
                    tendon_limited,
                    warning_latch,
                    accumulator,
                    stage="physics_forward",
                    control_interval=interval_index,
                    physics_substep=substep_index,
                    physics_step=next_step,
                )
            )
            accumulator.completed_physics_steps = next_step

        expected_end = (interval_index + 1) / config.ctrl_freq_hz
        time_error = abs(float(data.time) - expected_end)
        maximum_time_error = max(maximum_time_error, time_error)
        tolerance = _time_tolerance(
            expected_end,
            accumulator.completed_physics_steps,
            timestep,
        )
        if time_error > tolerance:
            raise AssertionError(
                f"control time error {time_error} exceeds {tolerance}"
            )
        data.time = expected_end
        state_records.append(_state_sample(fixture, bindings, addresses, tip_id))
        physics_records.append(samples)

    if accumulator.completed_physics_steps != config.number_of_physics_steps:
        raise AssertionError("probe ended before all physics steps")
    safety = ProbeSafetySummary(
        completed_physics_steps=accumulator.completed_physics_steps,
        warning_events=tuple(warning_latch.events),
        maximum_contacts=accumulator.maximum_contacts,
        minimum_tendon_range_margin_m=(
            accumulator.minimum_tendon_range_margin_m
        ),
        maximum_force_command_error_n=(
            accumulator.maximum_force_command_error_n
        ),
        maximum_moment_residual=accumulator.maximum_moment_residual,
        maximum_generalized_actuator_power_residual_w=(
            accumulator.maximum_generalized_actuator_power_residual_w
        ),
        maximum_actuator_tendon_power_residual_w=(
            accumulator.maximum_actuator_tendon_power_residual_w
        ),
        maximum_command_tendon_power_residual_w=(
            accumulator.maximum_command_tendon_power_residual_w
        ),
        maximum_actuator_length_sign_residual_m=(
            accumulator.maximum_actuator_length_sign_residual_m
        ),
        maximum_actuator_velocity_sign_residual_m_s=(
            accumulator.maximum_actuator_velocity_sign_residual_m_s
        ),
    )
    return FixedRootProbeTrace(
        config=config,
        profile_config=profile_config,
        resolved_channel=resolved,
        bindings=bindings,
        model_counts=model_counts,
        tendon_names=tuple(manifest.tendon_names),
        actuator_names=tuple(manifest.tendon_actuator_names),
        tendon_range_m=_freeze(tendon_ranges),
        tendon_limited=_freeze(tendon_limited, dtype=bool),
        actuator_gear=_freeze(model.actuator_gear[actuator_ids]),
        actuator_ctrl_range_n=_freeze(model.actuator_ctrlrange[actuator_ids]),
        time_s=_records_array(state_records, "time"),
        qpos=_records_array(state_records, "qpos"),
        qvel=_records_array(state_records, "qvel"),
        tendon_length_m=_records_array(state_records, "tendon_length"),
        tendon_velocity_m_s=_records_array(state_records, "tendon_velocity"),
        tip_position_m=_records_array(state_records, "tip"),
        joint_rotation_vectors_rad=_records_array(state_records, "joints"),
        composite_rotation_vectors_rad=_records_array(
            state_records, "composite"
        ),
        spring_energy_j=_records_array(state_records, "spring_energy"),
        position_energy_j=_records_array(state_records, "position_energy"),
        kinetic_energy_j=_records_array(state_records, "kinetic_energy"),
        total_energy_j=_records_array(state_records, "total_energy"),
        contact_count=_records_array(state_records, "contacts", dtype=int),
        warning_counts=_records_array(state_records, "warnings", dtype=int),
        interval_start_time_s=_records_array(interval_records, "start"),
        interval_end_time_s=_records_array(interval_records, "end"),
        commanded_tension_n=_records_array(interval_records, "command"),
        data_ctrl_n=_records_array(interval_records, "ctrl"),
        actuator_force_n=_records_array(interval_records, "force"),
        physics_substeps=_records_array(
            interval_records, "substeps", dtype=int
        ),
        physics_time_s=_grouped_physics_array(physics_records, "time"),
        physics_data_ctrl_n=_grouped_physics_array(physics_records, "ctrl"),
        physics_actuator_force_n=_grouped_physics_array(
            physics_records, "actuator_force"
        ),
        physics_actuator_velocity_m_s=_grouped_physics_array(
            physics_records, "actuator_velocity"
        ),
        physics_tendon_velocity_m_s=_grouped_physics_array(
            physics_records, "tendon_velocity"
        ),
        physics_generalized_power_w=_grouped_physics_array(
            physics_records, "generalized_power"
        ),
        physics_actuator_power_w=_grouped_physics_array(
            physics_records, "actuator_power"
        ),
        physics_tendon_power_w=_grouped_physics_array(
            physics_records, "tendon_power"
        ),
        physics_command_power_w=_grouped_physics_array(
            physics_records, "command_power"
        ),
        physics_contact_count=_grouped_physics_array(
            physics_records, "contacts", dtype=int
        ),
        physics_warning_counts=_grouped_physics_array(
            physics_records, "warnings", dtype=int
        ),
        physics_finite_state=_grouped_physics_array(
            physics_records, "finite", dtype=bool
        ),
        number_of_control_intervals=len(interval_records),
        number_of_physics_steps=accumulator.completed_physics_steps,
        maximum_control_time_error_s=maximum_time_error,
        safety=safety,
    )
