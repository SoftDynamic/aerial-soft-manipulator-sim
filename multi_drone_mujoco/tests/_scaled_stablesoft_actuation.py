"""Test-only active dynamics helpers for the scaled StableSoft arm."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Tuple, Union

import mujoco
import numpy as np

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
    ScaledStableSoftVisualSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
    validate_ideal_tendon_tension_command,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    WarningEvent,
    WarningLatch,
    build_fixed_arm_fixture,
    composite_rotation_vector,
    joint_rotation_vectors,
    named_joint_addresses,
)


CORNERS = ("xp_yp", "xp_yn", "xn_yp", "xn_yn")
JOINT_STIFFNESS = 0.003086
TARGET_DAMPING_RATIO = 0.3
MAX_TENSION_N = 0.01  # Numerical diagnostic limit, not hardware rating.
CTRL_FREQ = 240
RAMP_DURATION_S = 0.25
TOTAL_DURATION_S = 3.5
STEADY_WINDOW_START_S = 3.0
INTEGRATOR = "implicitfast"


def _finite_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _aligned_ticks(name: str, duration_s: float, frequency: int) -> int:
    raw_ticks = duration_s * frequency
    ticks = int(round(raw_ticks))
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(raw_ticks))
    if abs(raw_ticks - ticks) > tolerance:
        raise ValueError(f"{name} must align with the control tick grid")
    return ticks


class _ActiveRunGrid:
    """Shared timing properties for fixed-root active test configs."""

    sim_freq: int
    ctrl_freq: int
    ramp_duration_s: float
    total_duration_s: float
    steady_window_start_s: float
    integrator: str
    max_tension_n: float

    @property
    def ramp_ticks(self) -> int:
        return _aligned_ticks(
            "ramp_duration_s",
            self.ramp_duration_s,
            self.ctrl_freq,
        )

    @property
    def total_ticks(self) -> int:
        return _aligned_ticks(
            "total_duration_s",
            self.total_duration_s,
            self.ctrl_freq,
        )

    @property
    def sample_stride(self) -> int:
        return self.sim_freq // self.ctrl_freq

    @property
    def requested_steps(self) -> int:
        return self.total_ticks * self.sample_stride


def _normalize_active_run_grid(config: _ActiveRunGrid) -> None:
    """Normalize and validate shared active-run timing fields in place."""
    for name in ("sim_freq", "ctrl_freq"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.sim_freq % config.ctrl_freq != 0:
        raise ValueError("sim_freq must be divisible by ctrl_freq")
    if config.integrator != "implicitfast":
        raise ValueError("active characterization requires implicitfast")

    for name in (
        "ramp_duration_s",
        "total_duration_s",
        "steady_window_start_s",
        "max_tension_n",
    ):
        object.__setattr__(
            config,
            name,
            _finite_float(name, getattr(config, name)),
        )
    if config.max_tension_n <= 0:
        raise ValueError("max_tension_n must be positive")
    if config.max_tension_n > MAX_TENSION_N:
        raise ValueError(
            f"max_tension_n must not exceed the {MAX_TENSION_N} N "
            "C1 active-dynamics diagnostic limit"
        )
    if config.ramp_duration_s <= 0:
        raise ValueError("ramp_duration_s must be positive")
    if config.total_duration_s <= config.ramp_duration_s:
        raise ValueError("total_duration_s must exceed ramp_duration_s")
    if not (
        config.ramp_duration_s
        < config.steady_window_start_s
        < config.total_duration_s
    ):
        raise ValueError(
            "steady_window_start_s must lie after the ramp and before "
            "the end"
        )
    _ = config.ramp_ticks
    _ = config.total_ticks
    _aligned_ticks(
        "steady_window_start_s",
        config.steady_window_start_s,
        config.ctrl_freq,
    )


@dataclass(frozen=True)
class TendonActuatorBinding:
    """One manifest-resolved tendon actuator and its storage addresses."""

    tendon_name: str
    actuator_name: str
    tendon_id: int
    actuator_id: int
    control_address: int
    output_address: int


@dataclass(frozen=True)
class ActiveTendonRunConfig(_ActiveRunGrid):
    """Deterministic fixed-root single-tendon experiment configuration."""

    corner: str
    final_tension_n: float
    sim_freq: int = 2400
    ctrl_freq: int = CTRL_FREQ
    ramp_duration_s: float = RAMP_DURATION_S
    total_duration_s: float = TOTAL_DURATION_S
    steady_window_start_s: float = STEADY_WINDOW_START_S
    integrator: str = INTEGRATOR
    max_tension_n: float = MAX_TENSION_N

    def __post_init__(self):
        if self.corner not in CORNERS:
            raise ValueError(f"corner must be one of {CORNERS}")
        _normalize_active_run_grid(self)
        object.__setattr__(
            self,
            "final_tension_n",
            _finite_float("final_tension_n", self.final_tension_n),
        )
        if not 0 < self.final_tension_n <= self.max_tension_n:
            raise ValueError(
                "final_tension_n must be positive and not exceed max_tension_n"
            )


@dataclass(frozen=True)
class TendonCommandRunConfig(_ActiveRunGrid):
    """Deterministic fixed-root four-tendon command configuration."""

    final_tension_command_n: Tuple[float, float, float, float]
    sim_freq: int = 2400
    ctrl_freq: int = CTRL_FREQ
    ramp_duration_s: float = RAMP_DURATION_S
    total_duration_s: float = TOTAL_DURATION_S
    steady_window_start_s: float = STEADY_WINDOW_START_S
    integrator: str = INTEGRATOR
    max_tension_n: float = MAX_TENSION_N

    def __post_init__(self):
        _normalize_active_run_grid(self)
        try:
            raw_command = tuple(self.final_tension_command_n)
        except TypeError as exc:
            raise ValueError(
                "final_tension_command_n must be a four-value sequence"
            ) from exc
        if len(raw_command) != len(CORNERS):
            raise ValueError(
                "final_tension_command_n must contain exactly four values"
            )
        command = tuple(
            _finite_float(f"final_tension_command_n[{index}]", value)
            for index, value in enumerate(raw_command)
        )
        if any(value < 0 for value in command):
            raise ValueError(
                "final_tension_command_n values must be non-negative"
            )
        if any(value > self.max_tension_n for value in command):
            raise ValueError(
                "final_tension_command_n values must not exceed "
                "max_tension_n"
            )
        if not any(value > 0 for value in command):
            raise ValueError(
                "final_tension_command_n must activate at least one tendon"
            )
        object.__setattr__(self, "final_tension_command_n", command)

    @property
    def active_indices(self) -> Tuple[int, ...]:
        return tuple(
            index
            for index, value in enumerate(self.final_tension_command_n)
            if value > 0
        )


ActiveTraceConfig = Union[ActiveTendonRunConfig, TendonCommandRunConfig]


@dataclass(frozen=True)
class ActiveRunFailure:
    stage: str
    control_tick: int
    physics_step: int
    time_s: float
    field: str
    value: object


class ActiveRunSafetyError(AssertionError):
    """Fail one run while retaining the first unsafe state context."""

    def __init__(self, failure: ActiveRunFailure):
        self.failure = failure
        super().__init__(
            f"{failure.field} failed at stage={failure.stage}, "
            f"tick={failure.control_tick}, step={failure.physics_step}, "
            f"time={failure.time_s}: {failure.value}"
        )


@dataclass(frozen=True)
class ActiveSafetySummary:
    requested_steps: int
    completed_steps: int
    warning_events: Tuple[WarningEvent, ...]
    first_failure: Optional[ActiveRunFailure]
    maximum_contacts: int
    minimum_tendon_range_margin_m: float
    maximum_abs_qvel_rad_s: float
    maximum_abs_qacc_rad_s2: float
    maximum_force_command_error_n: float
    maximum_moment_residual: float
    maximum_generalized_actuator_power_residual_w: float
    maximum_actuator_tendon_power_residual_w: float
    maximum_command_tendon_power_residual_w: float
    maximum_actuator_length_sign_residual_m: float
    maximum_actuator_velocity_sign_residual_m_s: float
    maximum_abs_joint_twist_rad: float
    maximum_joint_bend_rad: float
    maximum_abs_composite_twist_rad: float
    maximum_composite_bend_rad: float


@dataclass(frozen=True)
class ActiveTendonTrace:
    """Control-rate samples plus physics-rate safety extrema."""

    config: ActiveTraceConfig
    tendon_names: Tuple[str, ...]
    actuator_names: Tuple[str, ...]
    bindings: Tuple[TendonActuatorBinding, ...]
    active_indices: Tuple[int, ...]
    active_mask: np.ndarray
    target_index: Optional[int]
    model_counts: Tuple[int, ...]
    tendon_ranges_m: np.ndarray
    tendon_limited: np.ndarray
    initial_tip_world_m: np.ndarray
    time_s: np.ndarray
    control_tick: np.ndarray
    physics_step: np.ndarray
    tension_command_n: np.ndarray
    actuator_force_n: np.ndarray
    tip_world_m: np.ndarray
    tip_displacement_m: np.ndarray
    joint_rotation_vectors_rad: np.ndarray
    composite_rotation_vectors_rad: np.ndarray
    tendon_length_m: np.ndarray
    tendon_velocity_m_s: np.ndarray
    actuator_length_m: np.ndarray
    actuator_velocity_m_s: np.ndarray
    qpos: np.ndarray
    qvel_rad_s: np.ndarray
    qacc_rad_s2: np.ndarray
    qfrc_actuator_nm: np.ndarray
    generalized_actuator_power_w: np.ndarray
    actuator_space_power_w: np.ndarray
    tendon_coordinate_power_w: np.ndarray
    command_tendon_power_w: np.ndarray
    warning_counts: np.ndarray
    contacts: np.ndarray
    tendon_range_margin_m: np.ndarray
    position_energy_j: np.ndarray
    kinetic_energy_j: np.ndarray
    total_energy_j: np.ndarray
    safety: ActiveSafetySummary

    @property
    def steady_mask(self) -> np.ndarray:
        return self.time_s >= self.config.steady_window_start_s

    @property
    def steady_tip_displacement_m(self) -> np.ndarray:
        return np.mean(self.tip_displacement_m[self.steady_mask], axis=0)

    @property
    def total_bend_rad(self) -> np.ndarray:
        return np.linalg.norm(
            self.composite_rotation_vectors_rad[:, :2],
            axis=1,
        )

    @property
    def steady_total_bend_rad(self) -> float:
        return float(np.mean(self.total_bend_rad[self.steady_mask]))

    @property
    def tendon_excursion_m(self) -> np.ndarray:
        return _freeze(self.tendon_length_m - self.tendon_length_m[0])

    @property
    def steady_tendon_excursion_m(self) -> np.ndarray:
        return _freeze(
            np.mean(self.tendon_excursion_m[self.steady_mask], axis=0)
        )

    @property
    def active_tendon_excursion_m(self) -> np.ndarray:
        return _freeze(
            self.tendon_excursion_m[:, np.asarray(self.active_indices)]
        )

    @property
    def steady_active_tendon_excursion_m(self) -> np.ndarray:
        return _freeze(
            self.steady_tendon_excursion_m[np.asarray(self.active_indices)]
        )

    def _require_target_index(self) -> int:
        if self.target_index is None:
            raise ValueError(
                "single-tendon excursion properties are undefined for a "
                "multi-tendon trace"
            )
        return self.target_index

    @property
    def target_tendon_excursion_m(self) -> np.ndarray:
        target_index = self._require_target_index()
        return _freeze(self.tendon_excursion_m[:, target_index])

    @property
    def steady_target_tendon_excursion_m(self) -> float:
        target_index = self._require_target_index()
        return float(
            self.steady_tendon_excursion_m[target_index]
        )

    @property
    def maximum_arm_qvel_rad_s(self) -> np.ndarray:
        return np.max(np.abs(self.qvel_rad_s), axis=1)


@dataclass
class _SafetyAccumulator:
    completed_steps: int = 0
    maximum_contacts: int = 0
    minimum_tendon_range_margin_m: float = math.inf
    maximum_abs_qvel_rad_s: float = 0.0
    maximum_abs_qacc_rad_s2: float = 0.0
    maximum_force_command_error_n: float = 0.0
    maximum_moment_residual: float = 0.0
    maximum_generalized_actuator_power_residual_w: float = 0.0
    maximum_actuator_tendon_power_residual_w: float = 0.0
    maximum_command_tendon_power_residual_w: float = 0.0
    maximum_actuator_length_sign_residual_m: float = 0.0
    maximum_actuator_velocity_sign_residual_m_s: float = 0.0
    maximum_abs_joint_twist_rad: float = 0.0
    maximum_joint_bend_rad: float = 0.0
    maximum_abs_composite_twist_rad: float = 0.0
    maximum_composite_bend_rad: float = 0.0


def _named_id(model, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise AssertionError(name)
    return int(object_id)


def resolve_tendon_actuator_bindings(fixture):
    """Resolve manifest order without treating actuator IDs as addresses."""
    model = fixture.model
    tendon_names = tuple(fixture.manifest.tendon_names)
    actuator_names = tuple(fixture.manifest.tendon_actuator_names)
    if len(tendon_names) != len(actuator_names):
        raise AssertionError("tendon and actuator manifest lengths differ")

    bindings = []
    for tendon_name, actuator_name in zip(tendon_names, actuator_names):
        tendon_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tendon_name,
        )
        actuator_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if int(model.actuator_ctrlnum[actuator_id]) != 1:
            raise AssertionError(f"{actuator_name} must have one control")
        if int(model.actuator_outnum[actuator_id]) != 1:
            raise AssertionError(f"{actuator_name} must have one output")
        bindings.append(
            TendonActuatorBinding(
                tendon_name=tendon_name,
                actuator_name=actuator_name,
                tendon_id=tendon_id,
                actuator_id=actuator_id,
                control_address=int(model.actuator_ctrladr[actuator_id]),
                output_address=int(model.actuator_outadr[actuator_id]),
            )
        )
    return tuple(bindings)


def write_tension_command(fixture, command, drive_spec, *, bindings=None):
    """Validate newtons and write each value through its control address."""
    values = validate_ideal_tendon_tension_command(command, drive_spec)
    if bindings is None:
        bindings = resolve_tendon_actuator_bindings(fixture)
    else:
        bindings = tuple(bindings)
    if len(values) != len(bindings):
        raise AssertionError("command and actuator binding lengths differ")
    for value, binding in zip(values, bindings):
        fixture.data.ctrl[binding.control_address] = value
    return values, bindings


def half_cosine_ramp_scale_at_tick(
    control_tick: int,
    ramp_ticks: int,
) -> float:
    """Return an endpoint-exact dimensionless ramp scale in [0, 1]."""
    if isinstance(control_tick, bool) or not isinstance(control_tick, int):
        raise ValueError("control_tick must be an integer")
    if isinstance(ramp_ticks, bool) or not isinstance(ramp_ticks, int):
        raise ValueError("ramp_ticks must be an integer")
    if control_tick < 0:
        raise ValueError("control_tick must be non-negative")
    if ramp_ticks <= 0:
        raise ValueError("ramp_ticks must be positive")
    if control_tick == 0:
        return 0.0
    if control_tick >= ramp_ticks:
        return 1.0
    return 0.5 * (
        1.0 - math.cos(math.pi * control_tick / ramp_ticks)
    )


def half_cosine_tension_at_tick(
    final_tension_n: float,
    control_tick: int,
    ramp_ticks: int,
) -> float:
    """Return an endpoint-exact half-cosine tension ramp in newtons."""
    final_tension_n = _finite_float("final_tension_n", final_tension_n)
    if final_tension_n < 0:
        raise ValueError("final_tension_n must be non-negative")
    return final_tension_n * half_cosine_ramp_scale_at_tick(
        control_tick,
        ramp_ticks,
    )


def mixed_limit(absolute_floor: float, relative: float, scale: float) -> float:
    """Return a non-negative mixed absolute/relative comparison limit."""
    values = tuple(
        _finite_float(name, value)
        for name, value in (
            ("absolute_floor", absolute_floor),
            ("relative", relative),
            ("scale", scale),
        )
    )
    if any(value < 0 for value in values):
        raise ValueError("mixed limit inputs must be non-negative")
    return max(values[0], values[1] * values[2])


def _failure(
    config,
    *,
    stage,
    control_tick,
    physics_step,
    time_s,
    field,
    value,
):
    raise ActiveRunSafetyError(
        ActiveRunFailure(
            stage=stage,
            control_tick=int(control_tick),
            physics_step=int(physics_step),
            time_s=float(time_s),
            field=field,
            value=value,
        )
    )


def _require(
    condition,
    config,
    *,
    stage,
    control_tick,
    physics_step,
    time_s,
    field,
    value,
):
    if not bool(condition):
        _failure(
            config,
            stage=stage,
            control_tick=control_tick,
            physics_step=physics_step,
            time_s=time_s,
            field=field,
            value=value,
        )


def _warning_counts(data) -> Tuple[int, ...]:
    return tuple(int(warning.number) for warning in data.warning)


def _range_tolerance(lower, upper, length) -> float:
    scale = max(abs(float(lower)), abs(float(upper)), abs(float(length)))
    return max(1e-9, 1e-8 * scale)


def _synchronized_values(
    fixture,
    bindings,
    addresses,
    tip_id,
    tendon_ranges,
    command,
):
    model, data = fixture.model, fixture.data
    tendon_ids = np.asarray(
        [binding.tendon_id for binding in bindings],
        dtype=int,
    )
    output_addresses = np.asarray(
        [binding.output_address for binding in bindings],
        dtype=int,
    )
    tendon_length = np.asarray(data.ten_length)[tendon_ids].copy()
    tendon_velocity = np.asarray(data.ten_velocity)[tendon_ids].copy()
    actuator_length = np.asarray(data.actuator_length)[
        output_addresses
    ].copy()
    actuator_velocity = np.asarray(data.actuator_velocity)[
        output_addresses
    ].copy()
    actuator_force = np.asarray(data.actuator_force)[
        output_addresses
    ].copy()
    joints = joint_rotation_vectors(model, data, addresses)
    composite = composite_rotation_vector(model, data, addresses)
    actuator_moment = np.asarray(data.actuator_moment).reshape(
        model.nout,
        model.nv,
    )
    expected_qfrc = actuator_moment[output_addresses].T @ actuator_force
    generalized_power = float(data.qfrc_actuator @ data.qvel)
    actuator_power = float(actuator_force @ actuator_velocity)
    tendon_power = float(-actuator_force @ tendon_velocity)
    command_tendon_power = float(-np.asarray(command) @ tendon_velocity)
    margins = np.minimum(
        tendon_length - tendon_ranges[:, 0],
        tendon_ranges[:, 1] - tendon_length,
    )
    return {
        "tendon_length": tendon_length,
        "tendon_velocity": tendon_velocity,
        "actuator_length": actuator_length,
        "actuator_velocity": actuator_velocity,
        "actuator_force": actuator_force,
        "joints": joints,
        "composite": composite,
        "tip": np.asarray(data.site_xpos[tip_id]).copy(),
        "expected_qfrc": expected_qfrc,
        "generalized_power": generalized_power,
        "actuator_power": actuator_power,
        "tendon_power": tendon_power,
        "command_tendon_power": command_tendon_power,
        "margins": margins,
    }


def _check_synchronized_state(
    fixture,
    config,
    bindings,
    addresses,
    tip_id,
    tendon_ranges,
    tendon_limited,
    command,
    final_command,
    active_mask,
    accumulator,
    *,
    stage,
    control_tick,
    physics_step,
):
    model, data = fixture.model, fixture.data
    values = _synchronized_values(
        fixture,
        bindings,
        addresses,
        tip_id,
        tendon_ranges,
        command,
    )
    finite_fields = {
        "qpos": data.qpos,
        "qvel": data.qvel,
        "qacc": data.qacc,
        "site_xpos": data.site_xpos,
        "tendon_length": values["tendon_length"],
        "tendon_velocity": values["tendon_velocity"],
        "actuator_length": values["actuator_length"],
        "actuator_velocity": values["actuator_velocity"],
        "actuator_force": values["actuator_force"],
        "qfrc_actuator": data.qfrc_actuator,
        "joint_rotation_vectors": values["joints"],
        "composite_rotation_vector": values["composite"],
        "power": np.asarray(
            [
                values["generalized_power"],
                values["actuator_power"],
                values["tendon_power"],
                values["command_tendon_power"],
            ]
        ),
    }
    for field, field_values in finite_fields.items():
        _require(
            np.all(np.isfinite(field_values)),
            config,
            stage=stage,
            control_tick=control_tick,
            physics_step=physics_step,
            time_s=data.time,
            field=f"finite:{field}",
            value=np.asarray(field_values).shape,
        )

    accumulator.maximum_contacts = max(
        accumulator.maximum_contacts,
        int(data.ncon),
    )
    _require(
        int(data.ncon) == 0,
        config,
        stage=stage,
        control_tick=control_tick,
        physics_step=physics_step,
        time_s=data.time,
        field="contacts",
        value=int(data.ncon),
    )

    _require(
        np.all(tendon_limited),
        config,
        stage=stage,
        control_tick=control_tick,
        physics_step=physics_step,
        time_s=data.time,
        field="tendon_limited",
        value=tendon_limited.tolist(),
    )
    for index, length in enumerate(values["tendon_length"]):
        lower, upper = tendon_ranges[index]
        tolerance = _range_tolerance(lower, upper, length)
        margin = float(values["margins"][index])
        accumulator.minimum_tendon_range_margin_m = min(
            accumulator.minimum_tendon_range_margin_m,
            margin,
        )
        _require(
            margin >= -tolerance,
            config,
            stage=stage,
            control_tick=control_tick,
            physics_step=physics_step,
            time_s=data.time,
            field=f"tendon_range:{bindings[index].tendon_name}",
            value={"length": float(length), "margin": margin},
        )

    command_array = np.asarray(command, dtype=float)
    force = values["actuator_force"]
    force_error = float(np.max(np.abs(force - command_array)))
    accumulator.maximum_force_command_error_n = max(
        accumulator.maximum_force_command_error_n,
        force_error,
    )
    _require(
        force_error <= 1e-15,
        config,
        stage=stage,
        control_tick=control_tick,
        physics_step=physics_step,
        time_s=data.time,
        field="force_command_error_n",
        value=force_error,
    )
    inactive_force = force[~active_mask]
    _require(
        np.all(np.abs(inactive_force) <= 1e-15),
        config,
        stage=stage,
        control_tick=control_tick,
        physics_step=physics_step,
        time_s=data.time,
        field="inactive_force_n",
        value=inactive_force.tolist(),
    )
    active_force = force[active_mask]
    active_final = np.asarray(final_command, dtype=float)[active_mask]
    _require(
        np.all(active_force >= -1e-15)
        and np.all(active_force <= active_final + 1e-15),
        config,
        stage=stage,
        control_tick=control_tick,
        physics_step=physics_step,
        time_s=data.time,
        field="active_force_n",
        value={
            "force": active_force.tolist(),
            "final": active_final.tolist(),
        },
    )

    moment_residual = float(
        np.max(np.abs(data.qfrc_actuator - values["expected_qfrc"]))
    )
    moment_scale = max(
        1.0,
        float(np.max(np.abs(data.qfrc_actuator))),
        float(np.max(np.abs(values["expected_qfrc"]))),
    )
    accumulator.maximum_moment_residual = max(
        accumulator.maximum_moment_residual,
        moment_residual,
    )
    _require(
        moment_residual <= 1e-12 + 1e-10 * moment_scale,
        config,
        stage=stage,
        control_tick=control_tick,
        physics_step=physics_step,
        time_s=data.time,
        field="moment_residual",
        value=moment_residual,
    )

    power_scale = max(
        1e-12,
        abs(values["generalized_power"]),
        abs(values["actuator_power"]),
        abs(values["tendon_power"]),
        abs(values["command_tendon_power"]),
    )
    power_tolerance = 1e-12 + 1e-9 * power_scale
    generalized_residual = abs(
        values["generalized_power"] - values["actuator_power"]
    )
    tendon_residual = abs(
        values["actuator_power"] - values["tendon_power"]
    )
    command_residual = abs(
        values["tendon_power"] - values["command_tendon_power"]
    )
    accumulator.maximum_generalized_actuator_power_residual_w = max(
        accumulator.maximum_generalized_actuator_power_residual_w,
        generalized_residual,
    )
    accumulator.maximum_actuator_tendon_power_residual_w = max(
        accumulator.maximum_actuator_tendon_power_residual_w,
        tendon_residual,
    )
    accumulator.maximum_command_tendon_power_residual_w = max(
        accumulator.maximum_command_tendon_power_residual_w,
        command_residual,
    )
    for field, residual in (
        ("generalized_actuator_power_residual_w", generalized_residual),
        ("actuator_tendon_power_residual_w", tendon_residual),
        ("command_tendon_power_residual_w", command_residual),
    ):
        _require(
            residual <= power_tolerance,
            config,
            stage=stage,
            control_tick=control_tick,
            physics_step=physics_step,
            time_s=data.time,
            field=field,
            value=residual,
        )

    length_residual = float(
        np.max(
            np.abs(
                values["actuator_length"] + values["tendon_length"]
            )
        )
    )
    velocity_residual = float(
        np.max(
            np.abs(
                values["actuator_velocity"]
                + values["tendon_velocity"]
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
        length_residual <= 1e-12 + 1e-10 * float(
            np.max(np.abs(values["tendon_length"]))
        ),
        config,
        stage=stage,
        control_tick=control_tick,
        physics_step=physics_step,
        time_s=data.time,
        field="actuator_length_sign_residual_m",
        value=length_residual,
    )
    _require(
        velocity_residual <= 1e-12 + 1e-10 * max(
            1e-12,
            float(np.max(np.abs(values["tendon_velocity"]))),
        ),
        config,
        stage=stage,
        control_tick=control_tick,
        physics_step=physics_step,
        time_s=data.time,
        field="actuator_velocity_sign_residual_m_s",
        value=velocity_residual,
    )

    accumulator.maximum_abs_qvel_rad_s = max(
        accumulator.maximum_abs_qvel_rad_s,
        float(np.max(np.abs(data.qvel))),
    )
    accumulator.maximum_abs_qacc_rad_s2 = max(
        accumulator.maximum_abs_qacc_rad_s2,
        float(np.max(np.abs(data.qacc))),
    )
    joint_bend = np.linalg.norm(values["joints"][:, :2], axis=1)
    composite_bend = float(np.linalg.norm(values["composite"][:2]))
    accumulator.maximum_abs_joint_twist_rad = max(
        accumulator.maximum_abs_joint_twist_rad,
        float(np.max(np.abs(values["joints"][:, 2]))),
    )
    accumulator.maximum_joint_bend_rad = max(
        accumulator.maximum_joint_bend_rad,
        float(np.max(joint_bend)),
    )
    accumulator.maximum_abs_composite_twist_rad = max(
        accumulator.maximum_abs_composite_twist_rad,
        abs(float(values["composite"][2])),
    )
    accumulator.maximum_composite_bend_rad = max(
        accumulator.maximum_composite_bend_rad,
        composite_bend,
    )
    return values


def _freeze(values, *, dtype=None):
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _build_trace(
    config,
    fixture,
    bindings,
    active_indices,
    active_mask,
    target_index,
    tendon_ranges,
    tendon_limited,
    initial_tip,
    records,
    warning_latch,
    accumulator,
):
    def values(name, *, dtype=None):
        return _freeze([record[name] for record in records], dtype=dtype)

    position_energy = values("position_energy")
    kinetic_energy = values("kinetic_energy")
    safety = ActiveSafetySummary(
        requested_steps=config.requested_steps,
        completed_steps=accumulator.completed_steps,
        warning_events=tuple(warning_latch.events),
        first_failure=None,
        maximum_contacts=accumulator.maximum_contacts,
        minimum_tendon_range_margin_m=(
            accumulator.minimum_tendon_range_margin_m
        ),
        maximum_abs_qvel_rad_s=accumulator.maximum_abs_qvel_rad_s,
        maximum_abs_qacc_rad_s2=accumulator.maximum_abs_qacc_rad_s2,
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
        maximum_abs_joint_twist_rad=(
            accumulator.maximum_abs_joint_twist_rad
        ),
        maximum_joint_bend_rad=accumulator.maximum_joint_bend_rad,
        maximum_abs_composite_twist_rad=(
            accumulator.maximum_abs_composite_twist_rad
        ),
        maximum_composite_bend_rad=(
            accumulator.maximum_composite_bend_rad
        ),
    )
    model = fixture.model
    return ActiveTendonTrace(
        config=config,
        tendon_names=tuple(fixture.manifest.tendon_names),
        actuator_names=tuple(fixture.manifest.tendon_actuator_names),
        bindings=tuple(bindings),
        active_indices=tuple(active_indices),
        active_mask=_freeze(active_mask, dtype=bool),
        target_index=target_index,
        model_counts=(
            model.ntendon,
            model.nu,
            model.na,
            model.nout,
            getattr(model, "nactuator", model.nu),
        ),
        tendon_ranges_m=_freeze(tendon_ranges),
        tendon_limited=_freeze(tendon_limited, dtype=bool),
        initial_tip_world_m=_freeze(initial_tip),
        time_s=values("time"),
        control_tick=values("control_tick", dtype=int),
        physics_step=values("physics_step", dtype=int),
        tension_command_n=values("command"),
        actuator_force_n=values("actuator_force"),
        tip_world_m=values("tip"),
        tip_displacement_m=values("tip_displacement"),
        joint_rotation_vectors_rad=values("joints"),
        composite_rotation_vectors_rad=values("composite"),
        tendon_length_m=values("tendon_length"),
        tendon_velocity_m_s=values("tendon_velocity"),
        actuator_length_m=values("actuator_length"),
        actuator_velocity_m_s=values("actuator_velocity"),
        qpos=values("qpos"),
        qvel_rad_s=values("qvel"),
        qacc_rad_s2=values("qacc"),
        qfrc_actuator_nm=values("qfrc_actuator"),
        generalized_actuator_power_w=values("generalized_power"),
        actuator_space_power_w=values("actuator_power"),
        tendon_coordinate_power_w=values("tendon_power"),
        command_tendon_power_w=values("command_tendon_power"),
        warning_counts=values("warning_counts", dtype=int),
        contacts=values("contacts", dtype=int),
        tendon_range_margin_m=values("margins"),
        position_energy_j=position_energy,
        kinetic_energy_j=kinetic_energy,
        total_energy_j=_freeze(position_energy + kinetic_energy),
        safety=safety,
    )


def _record_control_sample(
    fixture,
    config,
    bindings,
    addresses,
    tip_id,
    tendon_ranges,
    tendon_limited,
    initial_tip,
    command,
    final_command,
    active_mask,
    accumulator,
    *,
    control_tick,
    physics_step,
):
    model, data = fixture.model, fixture.data
    synchronized = _check_synchronized_state(
        fixture,
        config,
        bindings,
        addresses,
        tip_id,
        tendon_ranges,
        tendon_limited,
        command,
        final_command,
        active_mask,
        accumulator,
        stage="control_sample",
        control_tick=control_tick,
        physics_step=physics_step,
    )
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    return {
        "time": float(data.time),
        "control_tick": control_tick,
        "physics_step": physics_step,
        "command": np.asarray(command).copy(),
        "actuator_force": synchronized["actuator_force"],
        "tip": synchronized["tip"],
        "tip_displacement": synchronized["tip"] - initial_tip,
        "joints": synchronized["joints"],
        "composite": synchronized["composite"],
        "tendon_length": synchronized["tendon_length"],
        "tendon_velocity": synchronized["tendon_velocity"],
        "actuator_length": synchronized["actuator_length"],
        "actuator_velocity": synchronized["actuator_velocity"],
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "qacc": np.asarray(data.qacc).copy(),
        "qfrc_actuator": np.asarray(data.qfrc_actuator).copy(),
        "generalized_power": synchronized["generalized_power"],
        "actuator_power": synchronized["actuator_power"],
        "tendon_power": synchronized["tendon_power"],
        "command_tendon_power": synchronized["command_tendon_power"],
        "warning_counts": _warning_counts(data),
        "contacts": int(data.ncon),
        "margins": synchronized["margins"],
        "position_energy": float(data.energy[0]),
        "kinetic_energy": float(data.energy[1]),
    }


def _run_tendon_command_characterization(
    config: ActiveTraceConfig,
    final_command,
    *,
    target_corner: Optional[str],
    visual: Optional[ScaledStableSoftVisualSpec],
) -> ActiveTendonTrace:
    """Run the sole fixed-root active simulation loop."""
    final_command = np.asarray(final_command, dtype=float)
    active_mask = final_command > 0.0
    active_indices = tuple(np.flatnonzero(active_mask).tolist())
    arm_spec = ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )
    drive_spec = IdealTendonTensionSpec(
        max_tension=config.max_tension_n,
    )
    fixture = build_fixed_arm_fixture(
        arm_spec,
        "vertical",
        sim_freq=config.sim_freq,
        ctrl_freq=config.ctrl_freq,
        integrator=config.integrator,
        tendon_drive=drive_spec,
        visual=visual,
    )
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    bindings = resolve_tendon_actuator_bindings(fixture)
    addresses = named_joint_addresses(model, manifest)
    for corner, tendon_name in zip(CORNERS, manifest.tendon_names):
        if not tendon_name.endswith(f"_{corner}_tendon"):
            raise AssertionError("manifest corner order changed")
    tip_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        manifest.arm_tip_site_name,
    )
    root_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        manifest.arm_root_site_name,
    )
    target_index = None
    if target_corner is not None:
        target_matches = [
            index
            for index, name in enumerate(manifest.tendon_names)
            if name.endswith(f"_{target_corner}_tendon")
        ]
        if len(target_matches) != 1:
            raise AssertionError(
                f"unable to resolve target corner {target_corner}"
            )
        target_index = target_matches[0]
        if target_index != CORNERS.index(target_corner):
            raise AssertionError("manifest corner order changed")

    tendon_ids = np.asarray(
        [binding.tendon_id for binding in bindings],
        dtype=int,
    )
    tendon_ranges = np.asarray(model.tendon_range)[tendon_ids].copy()
    tendon_limited = np.asarray(model.tendon_limited)[tendon_ids].copy()

    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    warning_latch = WarningLatch()
    warning_latch.snapshot(data, stage="initial_forward", step_index=0)
    accumulator = _SafetyAccumulator()

    initial_tip = np.asarray(data.site_xpos[tip_id]).copy()
    initial_tip_relative = np.asarray(
        data.site_xpos[tip_id] - data.site_xpos[root_id]
    )
    initial_joints = joint_rotation_vectors(model, data, addresses)
    initial_composite = composite_rotation_vector(model, data, addresses)
    initial_checks = {
        "initial_warning": not warning_latch.triggered,
        "initial_contact": int(data.ncon) == 0,
        "initial_joint_rotation": np.allclose(
            initial_joints,
            0.0,
            rtol=0.0,
            atol=1e-15,
        ),
        "initial_composite_rotation": np.allclose(
            initial_composite,
            0.0,
            rtol=0.0,
            atol=1e-15,
        ),
        "initial_qvel": np.allclose(
            data.qvel,
            0.0,
            rtol=0.0,
            atol=1e-15,
        ),
        "gravity": np.allclose(
            model.opt.gravity,
            [0.0, 0.0, -9.81],
            rtol=0.0,
            atol=1e-15,
        ),
        "vertical_tip": np.allclose(
            initial_tip_relative,
            [0.0, 0.0, -arm_spec.total_length],
            rtol=0.0,
            atol=1e-12,
        ),
        "no_floor": (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "floor",
            )
            < 0
        ),
        "sample_stride": fixture.sample_stride == config.sample_stride,
    }
    for field, passed in initial_checks.items():
        _require(
            passed,
            config,
            stage="initial_forward",
            control_tick=0,
            physics_step=0,
            time_s=data.time,
            field=field,
            value=passed,
        )

    records = []
    for control_tick in range(config.total_ticks + 1):
        expected_time = control_tick / config.ctrl_freq
        ramp_scale = half_cosine_ramp_scale_at_tick(
            control_tick,
            config.ramp_ticks,
        )
        requested_command = ramp_scale * final_command
        command, current_bindings = write_tension_command(
            fixture,
            requested_command,
            drive_spec,
            bindings=bindings,
        )
        if current_bindings != bindings:
            raise AssertionError("actuator binding changed during run")
        mujoco.mj_forward(model, data)
        warning_latch.snapshot(
            data,
            stage="control_forward",
            step_index=accumulator.completed_steps,
        )
        if warning_latch.triggered:
            _failure(
                config,
                stage="control_forward",
                control_tick=control_tick,
                physics_step=accumulator.completed_steps,
                time_s=data.time,
                field="mujoco_warning",
                value=warning_latch.events[-1],
            )
        _require(
            abs(float(data.time) - expected_time) <= 1e-12,
            config,
            stage="control_forward",
            control_tick=control_tick,
            physics_step=accumulator.completed_steps,
            time_s=data.time,
            field="control_time_alignment",
            value={"actual": float(data.time), "expected": expected_time},
        )
        records.append(
            _record_control_sample(
                fixture,
                config,
                bindings,
                addresses,
                tip_id,
                tendon_ranges,
                tendon_limited,
                initial_tip,
                command,
                final_command,
                active_mask,
                accumulator,
                control_tick=control_tick,
                physics_step=accumulator.completed_steps,
            )
        )
        if control_tick == config.total_ticks:
            break

        for _ in range(config.sample_stride):
            next_step = accumulator.completed_steps + 1
            mujoco.mj_step(model, data)
            raw_flags = {
                "time": math.isfinite(float(data.time)),
                "qpos": bool(np.all(np.isfinite(data.qpos))),
                "qvel": bool(np.all(np.isfinite(data.qvel))),
            }
            warning_latch.snapshot(
                data,
                stage="after_step",
                step_index=next_step,
            )
            for field, passed in raw_flags.items():
                _require(
                    passed,
                    config,
                    stage="after_step",
                    control_tick=control_tick,
                    physics_step=next_step,
                    time_s=data.time,
                    field=f"finite:{field}",
                    value=passed,
                )
            if warning_latch.triggered:
                _failure(
                    config,
                    stage="after_step",
                    control_tick=control_tick,
                    physics_step=next_step,
                    time_s=data.time,
                    field="mujoco_warning",
                    value=warning_latch.events[-1],
                )

            mujoco.mj_forward(model, data)
            warning_latch.snapshot(
                data,
                stage="after_forward",
                step_index=next_step,
            )
            if warning_latch.triggered:
                _failure(
                    config,
                    stage="after_forward",
                    control_tick=control_tick,
                    physics_step=next_step,
                    time_s=data.time,
                    field="mujoco_warning",
                    value=warning_latch.events[-1],
                )
            _check_synchronized_state(
                fixture,
                config,
                bindings,
                addresses,
                tip_id,
                tendon_ranges,
                tendon_limited,
                command,
                final_command,
                active_mask,
                accumulator,
                stage="after_forward",
                control_tick=control_tick,
                physics_step=next_step,
            )
            accumulator.completed_steps = next_step

        # MuJoCo advances ``data.time`` by repeated binary floating-point
        # additions.  Snap only that clock to the exact integer control grid
        # after checking the just-completed interval; no physical state is
        # changed, and this model has no time-dependent forces or callbacks.
        next_control_time = (control_tick + 1) / config.ctrl_freq
        _require(
            abs(float(data.time) - next_control_time) <= 1e-12,
            config,
            stage="control_interval_complete",
            control_tick=control_tick,
            physics_step=accumulator.completed_steps,
            time_s=data.time,
            field="control_time_alignment",
            value={
                "actual": float(data.time),
                "expected": next_control_time,
            },
        )
        data.time = next_control_time

    if accumulator.completed_steps != config.requested_steps:
        raise AssertionError(
            f"completed {accumulator.completed_steps}/"
            f"{config.requested_steps} physics steps"
        )
    return _build_trace(
        config,
        fixture,
        bindings,
        active_indices,
        active_mask,
        target_index,
        tendon_ranges,
        tendon_limited,
        initial_tip,
        records,
        warning_latch,
        accumulator,
    )


def run_tendon_command_characterization(
    config: TendonCommandRunConfig,
    *,
    visual: Optional[ScaledStableSoftVisualSpec] = None,
) -> ActiveTendonTrace:
    """Run one fixed-root, vertical four-tendon command trajectory."""
    if not isinstance(config, TendonCommandRunConfig):
        raise TypeError("config must be TendonCommandRunConfig")
    return _run_tendon_command_characterization(
        config,
        config.final_tension_command_n,
        target_corner=None,
        visual=visual,
    )


def run_single_tendon_characterization(
    config: ActiveTendonRunConfig,
    *,
    visual: Optional[ScaledStableSoftVisualSpec] = None,
) -> ActiveTendonTrace:
    """Run the backward-compatible single-tendon active trajectory."""
    if not isinstance(config, ActiveTendonRunConfig):
        raise TypeError("config must be ActiveTendonRunConfig")
    final_command = [0.0] * len(CORNERS)
    final_command[CORNERS.index(config.corner)] = config.final_tension_n
    return _run_tendon_command_characterization(
        config,
        final_command,
        target_corner=config.corner,
        visual=visual,
    )
