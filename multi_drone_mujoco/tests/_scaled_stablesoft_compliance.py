"""Test-only static tip-compliance helpers for the scaled StableSoft arm."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Tuple

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
    ActiveSafetySummary,
    _SafetyAccumulator,
    _check_synchronized_state,
    _freeze,
    _warning_counts,
    half_cosine_ramp_scale_at_tick,
    resolve_tendon_actuator_bindings,
    write_tension_command,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    WarningLatch,
    build_fixed_arm_fixture,
    composite_rotation_vector,
    joint_rotation_vectors,
    named_joint_addresses,
)


CTRL_FREQ = 240
SIM_FREQ = 2400
INTEGRATOR = "implicitfast"
TENSION_RAMP_DURATION_S = 0.25
PRELOAD_WINDOW_START_S = 1.0
FORCE_START_S = 1.5
FORCE_RAMP_DURATION_S = 0.25
LOADED_WINDOW_START_S = 4.0
TOTAL_DURATION_S = 4.5
MAX_TIP_FORCE_N = 2e-4  # Numerical diagnostic limit, not hardware rating.
FORCE_MAGNITUDES_N = (5e-5, 1e-4, 2e-4)
COMMON_TENSION_LEVELS_N = (0.0, 0.002, 0.005, 0.01)
DIRECTIONS = ("+X", "-X", "+Y", "-Y")
_DIRECTION_VECTORS = {
    "+X": np.asarray((1.0, 0.0, 0.0)),
    "-X": np.asarray((-1.0, 0.0, 0.0)),
    "+Y": np.asarray((0.0, 1.0, 0.0)),
    "-Y": np.asarray((0.0, -1.0, 0.0)),
}


def _finite_float(name, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _aligned_ticks(name, duration_s, frequency) -> int:
    raw = duration_s * frequency
    ticks = int(round(raw))
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(raw))
    if abs(raw - ticks) > tolerance:
        raise ValueError(f"{name} must align with the control tick grid")
    return ticks


@dataclass(frozen=True)
class CommonModeComplianceRunConfig:
    """One fixed-root common-mode preload and lateral tip-force run."""

    common_tension_n: float
    direction: str
    force_magnitude_n: float
    sim_freq: int = SIM_FREQ
    ctrl_freq: int = CTRL_FREQ
    tension_ramp_duration_s: float = TENSION_RAMP_DURATION_S
    preload_window_start_s: float = PRELOAD_WINDOW_START_S
    force_start_s: float = FORCE_START_S
    force_ramp_duration_s: float = FORCE_RAMP_DURATION_S
    loaded_window_start_s: float = LOADED_WINDOW_START_S
    total_duration_s: float = TOTAL_DURATION_S
    integrator: str = INTEGRATOR
    max_tension_n: float = MAX_TENSION_N
    max_tip_force_n: float = MAX_TIP_FORCE_N

    def __post_init__(self):
        for name in ("sim_freq", "ctrl_freq"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sim_freq % self.ctrl_freq:
            raise ValueError("sim_freq must be divisible by ctrl_freq")
        if self.integrator != INTEGRATOR:
            raise ValueError("compliance characterization requires implicitfast")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")

        for name in (
            "common_tension_n",
            "force_magnitude_n",
            "tension_ramp_duration_s",
            "preload_window_start_s",
            "force_start_s",
            "force_ramp_duration_s",
            "loaded_window_start_s",
            "total_duration_s",
            "max_tension_n",
            "max_tip_force_n",
        ):
            object.__setattr__(self, name, _finite_float(name, getattr(self, name)))
        if not 0.0 <= self.common_tension_n <= self.max_tension_n:
            raise ValueError("common_tension_n must lie in [0, max_tension_n]")
        if self.max_tension_n <= 0 or self.max_tension_n > MAX_TENSION_N:
            raise ValueError("max_tension_n exceeds the C1 diagnostic limit")
        if not 0.0 < self.force_magnitude_n <= self.max_tip_force_n:
            raise ValueError("force_magnitude_n must lie in (0, max_tip_force_n]")
        if self.max_tip_force_n <= 0 or self.max_tip_force_n > MAX_TIP_FORCE_N:
            raise ValueError("max_tip_force_n exceeds the C1-A4a diagnostic limit")
        if not (
            0.0 < self.tension_ramp_duration_s
            < self.preload_window_start_s
            < self.force_start_s
        ):
            raise ValueError("preload window must follow the tension ramp")
        force_ramp_end = self.force_start_s + self.force_ramp_duration_s
        if not (
            self.force_ramp_duration_s > 0.0
            and force_ramp_end < self.loaded_window_start_s
            < self.total_duration_s
        ):
            raise ValueError("loaded window must follow force ramp and settling")
        for name in (
            "tension_ramp_duration_s",
            "preload_window_start_s",
            "force_start_s",
            "force_ramp_duration_s",
            "loaded_window_start_s",
            "total_duration_s",
        ):
            _aligned_ticks(name, getattr(self, name), self.ctrl_freq)

    @property
    def sample_stride(self) -> int:
        return self.sim_freq // self.ctrl_freq

    @property
    def total_ticks(self) -> int:
        return _aligned_ticks("total_duration_s", self.total_duration_s, self.ctrl_freq)

    @property
    def requested_steps(self) -> int:
        return self.total_ticks * self.sample_stride

    @property
    def tension_ramp_ticks(self) -> int:
        return _aligned_ticks(
            "tension_ramp_duration_s", self.tension_ramp_duration_s, self.ctrl_freq
        )

    @property
    def force_start_tick(self) -> int:
        return _aligned_ticks("force_start_s", self.force_start_s, self.ctrl_freq)

    @property
    def force_ramp_ticks(self) -> int:
        return _aligned_ticks(
            "force_ramp_duration_s", self.force_ramp_duration_s, self.ctrl_freq
        )

    @property
    def direction_vector(self) -> np.ndarray:
        return _DIRECTION_VECTORS[self.direction].copy()

    @property
    def final_force_world_n(self) -> np.ndarray:
        return self.force_magnitude_n * self.direction_vector


@dataclass(frozen=True)
class ComplianceSafetySummary:
    actuation: ActiveSafetySummary
    maximum_external_generalized_force_residual: float
    maximum_external_power_residual_w: float
    maximum_abs_external_power_w: float
    maximum_abs_actuator_power_w: float
    maximum_abs_applied_force_n: float
    final_qfrc_applied_zero: bool
    final_xfrc_applied_zero: bool
    first_failure: Optional["ComplianceRunFailure"]


@dataclass(frozen=True)
class ComplianceRunFailure:
    stage: str
    control_tick: int
    physics_step: int
    time_s: float
    field: str
    value: object


class ComplianceRunSafetyError(AssertionError):
    """Retain the first external-force failure with grid context."""

    def __init__(self, failure: ComplianceRunFailure):
        self.failure = failure
        super().__init__(
            f"{failure.field} failed at stage={failure.stage}, "
            f"tick={failure.control_tick}, step={failure.physics_step}, "
            f"time={failure.time_s}: {failure.value}"
        )


@dataclass
class _ExternalAccumulator:
    maximum_generalized_force_residual: float = 0.0
    maximum_power_residual_w: float = 0.0
    maximum_abs_external_power_w: float = 0.0
    maximum_abs_actuator_power_w: float = 0.0
    maximum_abs_applied_force_n: float = 0.0


def apply_tip_force_and_check(
    fixture,
    tip_site_id,
    tip_body_id,
    force_world_n,
    accumulator,
    *,
    jacp,
    jacr,
    fail,
):
    """Apply one world-frame tip force and validate its power identity.

    This is a test-only primitive shared by the static compliance and
    disturbance-recovery runners.  It deliberately owns clearing the two
    MuJoCo external-force arrays so a zero-force call is an exact release.
    """
    model, data = fixture.model, fixture.data
    force = np.asarray(force_world_n, dtype=float)
    if force.shape != (3,):
        raise ValueError("force_world_n must contain three values")

    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    point = np.asarray(data.site_xpos[tip_site_id]).copy()
    mujoco.mj_applyFT(
        model,
        data,
        force,
        np.zeros(3),
        point,
        tip_body_id,
        data.qfrc_applied,
    )
    mujoco.mj_forward(model, data)
    mujoco.mj_jacSite(model, data, jacp, jacr, tip_site_id)
    tip_velocity = jacp @ data.qvel
    expected_qfrc = jacp.T @ force
    generalized_force_residual = float(
        np.max(np.abs(data.qfrc_applied - expected_qfrc))
    )
    force_scale = max(
        1.0,
        float(np.max(np.abs(data.qfrc_applied))),
        float(np.max(np.abs(expected_qfrc))),
    )
    if generalized_force_residual > 1e-12 + 1e-10 * force_scale:
        fail("tip_generalized_force_residual", generalized_force_residual)

    external_generalized_power = float(data.qfrc_applied @ data.qvel)
    external_tip_power = float(force @ tip_velocity)
    power_residual = abs(external_generalized_power - external_tip_power)
    power_scale = max(
        1e-12,
        abs(external_generalized_power),
        abs(external_tip_power),
    )
    if power_residual > 1e-12 + 1e-9 * power_scale:
        fail("external_power_residual_w", power_residual)

    if not all(
        np.all(np.isfinite(values))
        for values in (
            force,
            point,
            tip_velocity,
            data.qfrc_applied,
        )
    ):
        fail("finite_external_force_state", False)

    accumulator.maximum_generalized_force_residual = max(
        accumulator.maximum_generalized_force_residual,
        generalized_force_residual,
    )
    accumulator.maximum_power_residual_w = max(
        accumulator.maximum_power_residual_w,
        power_residual,
    )
    accumulator.maximum_abs_external_power_w = max(
        accumulator.maximum_abs_external_power_w,
        abs(external_generalized_power),
    )
    accumulator.maximum_abs_applied_force_n = max(
        accumulator.maximum_abs_applied_force_n,
        float(np.max(np.abs(force))),
    )
    return {
        "force": force.copy(),
        "point": point,
        "tip_velocity": tip_velocity.copy(),
        "external_generalized_power": external_generalized_power,
        "external_tip_power": external_tip_power,
    }


@dataclass(frozen=True)
class CommonModeComplianceTrace:
    """Read-only control-rate trace with physics-rate safety extrema."""

    config: CommonModeComplianceRunConfig
    tendon_names: Tuple[str, ...]
    actuator_names: Tuple[str, ...]
    root_site_name: str
    tip_site_name: str
    tip_body_name: str
    root_site_id: int
    tip_site_id: int
    tip_body_id: int
    tendon_ranges_m: np.ndarray
    phase: np.ndarray
    control_tick: np.ndarray
    physics_step: np.ndarray
    time_s: np.ndarray
    tension_command_n: np.ndarray
    actuator_force_n: np.ndarray
    tip_force_world_n: np.ndarray
    application_point_world_m: np.ndarray
    qfrc_applied: np.ndarray
    root_world_m: np.ndarray
    tip_world_m: np.ndarray
    tip_velocity_world_m_s: np.ndarray
    qpos: np.ndarray
    qvel_rad_s: np.ndarray
    qacc_rad_s2: np.ndarray
    arm_qvel_rad_s: np.ndarray
    joint_rotation_vectors_rad: np.ndarray
    composite_rotation_vectors_rad: np.ndarray
    tendon_length_m: np.ndarray
    tendon_velocity_m_s: np.ndarray
    qfrc_actuator_nm: np.ndarray
    generalized_actuator_power_w: np.ndarray
    actuator_space_power_w: np.ndarray
    tendon_coordinate_power_w: np.ndarray
    command_tendon_power_w: np.ndarray
    external_generalized_power_w: np.ndarray
    external_tip_power_w: np.ndarray
    warning_counts: np.ndarray
    contacts: np.ndarray
    tendon_range_margin_m: np.ndarray
    safety: ComplianceSafetySummary

    @property
    def preload_mask(self):
        return _freeze(
            (self.time_s >= self.config.preload_window_start_s)
            & (self.time_s <= self.config.force_start_s),
            dtype=bool,
        )

    @property
    def loaded_mask(self):
        return _freeze(
            self.time_s >= self.config.loaded_window_start_s,
            dtype=bool,
        )

    @property
    def preload_tip_world_m(self):
        return _freeze(np.mean(self.tip_world_m[self.preload_mask], axis=0))

    @property
    def loaded_tip_world_m(self):
        return _freeze(np.mean(self.tip_world_m[self.loaded_mask], axis=0))

    @property
    def delta_tip_world_m(self):
        return _freeze(self.loaded_tip_world_m - self.preload_tip_world_m)

    @property
    def delta_along_force_m(self) -> float:
        return float(self.config.direction_vector @ self.delta_tip_world_m)

    @property
    def one_sided_compliance_m_n(self) -> float:
        return self.delta_along_force_m / self.config.force_magnitude_n

    @property
    def preload_tendon_length_m(self):
        return _freeze(np.mean(self.tendon_length_m[self.preload_mask], axis=0))

    @property
    def loaded_tendon_length_m(self):
        return _freeze(np.mean(self.tendon_length_m[self.loaded_mask], axis=0))

    @property
    def incremental_tendon_excursion_m(self):
        return _freeze(self.loaded_tendon_length_m - self.preload_tendon_length_m)

    @property
    def preload_tip_shift_m(self):
        return _freeze(self.preload_tip_world_m - self.tip_world_m[0])

    @property
    def preload_tendon_excursion_m(self):
        return _freeze(self.preload_tendon_length_m - self.tendon_length_m[0])

    @property
    def preload_composite_rotation_rad(self):
        return _freeze(
            np.mean(self.composite_rotation_vectors_rad[self.preload_mask], axis=0)
        )

    @property
    def loaded_window_noise_m(self) -> float:
        values = self.tip_world_m[self.loaded_mask]
        center = np.mean(values, axis=0)
        return float(np.max(np.linalg.norm(values - center, axis=1)))

    @property
    def loaded_window_drift_m(self) -> float:
        values = self.tip_world_m[self.loaded_mask]
        quarter = len(values) // 4
        if quarter < 1:
            raise ValueError("loaded steady window needs at least four samples")
        return float(np.linalg.norm(np.mean(values[-quarter:], axis=0) - np.mean(values[:quarter], axis=0)))

    @property
    def loaded_window_max_arm_qvel_rad_s(self) -> float:
        return float(np.max(np.abs(self.arm_qvel_rad_s[self.loaded_mask])))

    @property
    def loaded_window_sensitivity(self) -> Optional[float]:
        values = self.tip_world_m[self.loaded_mask]
        half = len(values) // 2
        signal = abs(self.delta_along_force_m)
        if half < 1 or signal <= 50 * self.loaded_window_noise_m:
            return None
        preload = self.preload_tip_world_m
        first = np.mean(values[:half], axis=0) - preload
        last = np.mean(values[-half:], axis=0) - preload
        axis = 0 if self.config.direction.endswith("X") else 1
        return abs(float(last[axis] - first[axis])) / signal

    @property
    def total_bend_rad(self):
        return _freeze(
            np.linalg.norm(
                self.composite_rotation_vectors_rad[:, :2], axis=1
            )
        )


@dataclass(frozen=True)
class AxisComplianceFit:
    common_tension_n: float
    axis: str
    force_axis_n: np.ndarray
    delta_axis_m: np.ndarray
    slope_compliance_m_n: float
    intercept_m: float
    r_squared: float
    residual_m: np.ndarray
    maximum_fit_residual_m: float
    central_compliance_m_n: np.ndarray
    central_relative_spread: float
    one_sided_relative_asymmetry: float
    maximum_cross_axis_residual_m: float
    maximum_cross_axis_relative: float
    maximum_window_sensitivity: float
    minimum_signal_to_noise: float
    stiffness_n_m: Optional[float]
    valid: bool


@dataclass(frozen=True)
class CommonModeEffect:
    common_tension_n: float
    compliance_m_n: float
    stiffness_n_m: Optional[float]
    stiffness_ratio: Optional[float]
    uncertainty_relative: float
    effect_floor_relative: float
    classification: str


def _phase(config, time_s):
    if time_s < config.tension_ramp_duration_s:
        return "tension_ramp"
    if time_s < config.preload_window_start_s:
        return "preload_settle"
    if time_s < config.force_start_s:
        return "preload_measure"
    if time_s < config.force_start_s + config.force_ramp_duration_s:
        return "force_ramp"
    if time_s < config.loaded_window_start_s:
        return "loaded_settle"
    return "loaded_measure"


def _command_and_force(config, control_tick):
    tension_scale = half_cosine_ramp_scale_at_tick(
        control_tick, config.tension_ramp_ticks
    )
    relative_force_tick = control_tick - config.force_start_tick
    force_scale = 0.0 if relative_force_tick <= 0 else half_cosine_ramp_scale_at_tick(
        relative_force_tick, config.force_ramp_ticks
    )
    return (
        (config.common_tension_n * tension_scale,) * len(CORNERS),
        force_scale * config.final_force_world_n,
    )


def _active_summary(config, accumulator, warning_latch):
    return ActiveSafetySummary(
        requested_steps=config.requested_steps,
        completed_steps=accumulator.completed_steps,
        warning_events=tuple(warning_latch.events),
        first_failure=None,
        maximum_contacts=accumulator.maximum_contacts,
        minimum_tendon_range_margin_m=accumulator.minimum_tendon_range_margin_m,
        maximum_abs_qvel_rad_s=accumulator.maximum_abs_qvel_rad_s,
        maximum_abs_qacc_rad_s2=accumulator.maximum_abs_qacc_rad_s2,
        maximum_force_command_error_n=accumulator.maximum_force_command_error_n,
        maximum_moment_residual=accumulator.maximum_moment_residual,
        maximum_generalized_actuator_power_residual_w=accumulator.maximum_generalized_actuator_power_residual_w,
        maximum_actuator_tendon_power_residual_w=accumulator.maximum_actuator_tendon_power_residual_w,
        maximum_command_tendon_power_residual_w=accumulator.maximum_command_tendon_power_residual_w,
        maximum_actuator_length_sign_residual_m=accumulator.maximum_actuator_length_sign_residual_m,
        maximum_actuator_velocity_sign_residual_m_s=accumulator.maximum_actuator_velocity_sign_residual_m_s,
        maximum_abs_joint_twist_rad=accumulator.maximum_abs_joint_twist_rad,
        maximum_joint_bend_rad=accumulator.maximum_joint_bend_rad,
        maximum_abs_composite_twist_rad=accumulator.maximum_abs_composite_twist_rad,
        maximum_composite_bend_rad=accumulator.maximum_composite_bend_rad,
    )


def run_common_mode_compliance(config: CommonModeComplianceRunConfig):
    """Run one static common-mode preload and tip-force trajectory."""
    if not isinstance(config, CommonModeComplianceRunConfig):
        raise TypeError("config must be CommonModeComplianceRunConfig")
    arm_spec = ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )
    drive = IdealTendonTensionSpec(config.max_tension_n)
    fixture = build_fixed_arm_fixture(
        arm_spec,
        "vertical",
        sim_freq=config.sim_freq,
        ctrl_freq=config.ctrl_freq,
        integrator=config.integrator,
        tendon_drive=drive,
    )
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    bindings = resolve_tendon_actuator_bindings(fixture)
    addresses = named_joint_addresses(model, manifest)
    arm_dofs = np.concatenate(
        [np.arange(dof, dof + 3) for _, _, dof in addresses]
    )
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_root_site_name
    )
    tip_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name
    )
    if root_id < 0 or tip_id < 0:
        raise AssertionError("manifest root/tip site did not resolve")
    tip_body_id = int(model.site_bodyid[tip_id])
    tip_body_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, tip_body_id
    )
    if tip_body_name != manifest.segment_body_names[-1]:
        raise AssertionError("tip site is not attached to the final segment")
    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    tendon_ranges = model.tendon_range[tendon_ids].copy()
    tendon_limited = model.tendon_limited[tendon_ids].copy()
    active_mask = np.full(len(CORNERS), config.common_tension_n > 0.0)
    final_command = (config.common_tension_n,) * len(CORNERS)

    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    warning_latch = WarningLatch()
    warning_latch.snapshot(data, stage="initial_forward", step_index=0)
    active_accumulator = _SafetyAccumulator()
    external_accumulator = _ExternalAccumulator()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    records = []

    def fail(stage, control_tick, physics_step, field, value):
        raise ComplianceRunSafetyError(
            ComplianceRunFailure(
                stage=stage,
                control_tick=int(control_tick),
                physics_step=int(physics_step),
                time_s=float(data.time),
                field=field,
                value=value,
            )
        )

    def prepare(control_tick, physics_step, *, record):
        expected_time = control_tick / config.ctrl_freq if record else data.time
        command, force = _command_and_force(config, control_tick)
        written, current_bindings = write_tension_command(
            fixture, command, drive, bindings=bindings
        )
        if current_bindings != bindings:
            raise AssertionError("actuator binding changed during run")
        external = apply_tip_force_and_check(
            fixture,
            tip_id,
            tip_body_id,
            force,
            external_accumulator,
            jacp=jacp,
            jacr=jacr,
            fail=lambda field, value: fail(
                "force_forward",
                control_tick,
                physics_step,
                field,
                value,
            ),
        )
        warning_latch.snapshot(
            data, stage="force_forward", step_index=physics_step
        )
        if warning_latch.triggered:
            fail(
                "force_forward",
                control_tick,
                physics_step,
                "mujoco_warning",
                warning_latch.events[-1],
            )
        if record and abs(float(data.time) - expected_time) > 1e-12:
            fail(
                "force_forward",
                control_tick,
                physics_step,
                "control_time_alignment",
                {"actual": float(data.time), "expected": expected_time},
            )
        synchronized = _check_synchronized_state(
            fixture,
            config,
            bindings,
            addresses,
            tip_id,
            tendon_ranges,
            tendon_limited,
            written,
            final_command,
            active_mask,
            active_accumulator,
            stage="force_forward",
            control_tick=control_tick,
            physics_step=physics_step,
        )
        external_accumulator.maximum_abs_actuator_power_w = max(
            external_accumulator.maximum_abs_actuator_power_w,
            abs(synchronized["generalized_power"]),
        )
        if not record:
            return
        records.append(
            {
                "phase": _phase(config, float(data.time)),
                "control_tick": control_tick,
                "physics_step": physics_step,
                "time": float(data.time),
                "command": np.asarray(written).copy(),
                "actuator_force": synchronized["actuator_force"],
                "force": external["force"],
                "point": external["point"],
                "qfrc_applied": np.asarray(data.qfrc_applied).copy(),
                "root": np.asarray(data.site_xpos[root_id]).copy(),
                "tip": synchronized["tip"],
                "tip_velocity": external["tip_velocity"],
                "qpos": np.asarray(data.qpos).copy(),
                "qvel": np.asarray(data.qvel).copy(),
                "qacc": np.asarray(data.qacc).copy(),
                "arm_qvel": np.asarray(data.qvel[arm_dofs]).copy(),
                "joints": synchronized["joints"],
                "composite": synchronized["composite"],
                "tendon_length": synchronized["tendon_length"],
                "tendon_velocity": synchronized["tendon_velocity"],
                "qfrc_actuator": np.asarray(data.qfrc_actuator).copy(),
                "generalized_power": synchronized["generalized_power"],
                "actuator_power": synchronized["actuator_power"],
                "tendon_power": synchronized["tendon_power"],
                "command_tendon_power": synchronized["command_tendon_power"],
                "external_generalized_power": external[
                    "external_generalized_power"
                ],
                "external_tip_power": external["external_tip_power"],
                "warning_counts": _warning_counts(data),
                "contacts": int(data.ncon),
                "margins": synchronized["margins"],
            }
        )

    try:
        for control_tick in range(config.total_ticks + 1):
            prepare(
                control_tick,
                active_accumulator.completed_steps,
                record=True,
            )
            if control_tick == config.total_ticks:
                break
            for substep in range(config.sample_stride):
                next_step = active_accumulator.completed_steps + 1
                mujoco.mj_step(model, data)
                warning_latch.snapshot(
                    data, stage="after_step", step_index=next_step
                )
                if warning_latch.triggered:
                    fail(
                        "after_step",
                        control_tick,
                        next_step,
                        "mujoco_warning",
                        warning_latch.events[-1],
                    )
                if not (
                    math.isfinite(float(data.time))
                    and np.all(np.isfinite(data.qpos))
                    and np.all(np.isfinite(data.qvel))
                ):
                    fail(
                        "after_step",
                        control_tick,
                        next_step,
                        "finite_raw_state",
                        False,
                    )
                active_accumulator.completed_steps = next_step
                if substep + 1 < config.sample_stride:
                    prepare(control_tick, next_step, record=False)
            expected = (control_tick + 1) / config.ctrl_freq
            if abs(float(data.time) - expected) > 1e-12:
                raise AssertionError(f"physics time mismatch: {data.time} != {expected}")
            data.time = expected
        if active_accumulator.completed_steps != config.requested_steps:
            raise AssertionError(
                f"completed {active_accumulator.completed_steps}/"
                f"{config.requested_steps} physics steps"
            )
    finally:
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0

    final_qfrc_applied_zero = bool(np.all(data.qfrc_applied == 0.0))
    final_xfrc_applied_zero = bool(np.all(data.xfrc_applied == 0.0))

    def values(name, dtype=None):
        return _freeze([item[name] for item in records], dtype=dtype)

    safety = ComplianceSafetySummary(
        actuation=_active_summary(config, active_accumulator, warning_latch),
        maximum_external_generalized_force_residual=(
            external_accumulator.maximum_generalized_force_residual
        ),
        maximum_external_power_residual_w=(
            external_accumulator.maximum_power_residual_w
        ),
        maximum_abs_external_power_w=(
            external_accumulator.maximum_abs_external_power_w
        ),
        maximum_abs_actuator_power_w=(
            external_accumulator.maximum_abs_actuator_power_w
        ),
        maximum_abs_applied_force_n=(
            external_accumulator.maximum_abs_applied_force_n
        ),
        final_qfrc_applied_zero=final_qfrc_applied_zero,
        final_xfrc_applied_zero=final_xfrc_applied_zero,
        first_failure=None,
    )
    return CommonModeComplianceTrace(
        config=config,
        tendon_names=tuple(manifest.tendon_names),
        actuator_names=tuple(manifest.tendon_actuator_names),
        root_site_name=manifest.arm_root_site_name,
        tip_site_name=manifest.arm_tip_site_name,
        tip_body_name=tip_body_name,
        root_site_id=int(root_id),
        tip_site_id=int(tip_id),
        tip_body_id=tip_body_id,
        tendon_ranges_m=_freeze(tendon_ranges),
        phase=values("phase"),
        control_tick=values("control_tick", int),
        physics_step=values("physics_step", int),
        time_s=values("time"),
        tension_command_n=values("command"),
        actuator_force_n=values("actuator_force"),
        tip_force_world_n=values("force"),
        application_point_world_m=values("point"),
        qfrc_applied=values("qfrc_applied"),
        root_world_m=values("root"),
        tip_world_m=values("tip"),
        tip_velocity_world_m_s=values("tip_velocity"),
        qpos=values("qpos"),
        qvel_rad_s=values("qvel"),
        qacc_rad_s2=values("qacc"),
        arm_qvel_rad_s=values("arm_qvel"),
        joint_rotation_vectors_rad=values("joints"),
        composite_rotation_vectors_rad=values("composite"),
        tendon_length_m=values("tendon_length"),
        tendon_velocity_m_s=values("tendon_velocity"),
        qfrc_actuator_nm=values("qfrc_actuator"),
        generalized_actuator_power_w=values("generalized_power"),
        actuator_space_power_w=values("actuator_power"),
        tendon_coordinate_power_w=values("tendon_power"),
        command_tendon_power_w=values("command_tendon_power"),
        external_generalized_power_w=values("external_generalized_power"),
        external_tip_power_w=values("external_tip_power"),
        warning_counts=values("warning_counts", int),
        contacts=values("contacts", int),
        tendon_range_margin_m=values("margins"),
        safety=safety,
    )


def fit_axis_compliance(traces, common_tension_n: float, axis: str):
    """Fit fixed-world-axis signed force/displacement data with free intercept."""
    if axis not in ("X", "Y"):
        raise ValueError("axis must be X or Y")
    axis_index = 0 if axis == "X" else 1
    cross_index = 1 - axis_index
    selected = [
        trace
        for trace in traces
        if trace.config.common_tension_n == common_tension_n
        and trace.config.direction.endswith(axis)
    ]
    if len(selected) != 2 * len(FORCE_MAGNITUDES_N):
        raise ValueError("axis fit requires all six signed force trajectories")
    selected.sort(key=lambda item: item.tip_force_world_n[-1, axis_index])
    force = np.asarray(
        [item.tip_force_world_n[-1, axis_index] for item in selected]
    )
    delta = np.asarray([item.delta_tip_world_m[axis_index] for item in selected])
    slope, intercept, r_squared, residual = signed_axis_ols(force, delta)
    central = []
    asymmetry = []
    for magnitude in FORCE_MAGNITUDES_N:
        positive = next(item for item in selected if item.config.direction == f"+{axis}" and item.config.force_magnitude_n == magnitude)
        negative = next(item for item in selected if item.config.direction == f"-{axis}" and item.config.force_magnitude_n == magnitude)
        central.append(
            (positive.delta_tip_world_m[axis_index] - negative.delta_tip_world_m[axis_index])
            / (2 * magnitude)
        )
        left = positive.one_sided_compliance_m_n
        right = negative.one_sided_compliance_m_n
        asymmetry.append(abs(left - right) / max(abs(left), abs(right)))
    central = np.asarray(central)
    central_spread = float(np.ptp(central) / abs(np.mean(central)))
    cross = np.asarray([abs(item.delta_tip_world_m[cross_index]) for item in selected])
    along = np.asarray([abs(item.delta_tip_world_m[axis_index]) for item in selected])
    sensitivities = [item.loaded_window_sensitivity for item in selected]
    noise_ratios = [
        abs(item.delta_along_force_m) / item.loaded_window_noise_m
        if item.loaded_window_noise_m > 0 else math.inf
        for item in selected
    ]
    steady_valid = all(
        item.loaded_window_drift_m
        <= max(1e-6, 0.005 * float(np.linalg.norm(item.delta_tip_world_m)))
        and item.loaded_window_max_arm_qvel_rad_s <= 1e-4
        and abs(item.delta_along_force_m) >= 50 * item.loaded_window_noise_m
        for item in selected
    )
    local_valid = all(
        np.linalg.norm(item.delta_tip_world_m[:2])
        <= 0.01 * arm_total_length_m()
        and np.max(item.total_bend_rad) <= 0.03
        and item.safety.actuation.maximum_joint_bend_rad <= 0.01
        for item in selected
    )
    symmetry_valid = (
        max(asymmetry) <= 0.001
        and np.all(cross <= np.maximum(1e-9, 0.001 * along))
    )
    linear_valid = (
        math.isfinite(float(slope))
        and slope > 0
        and abs(float(intercept)) <= 1e-7
        and np.max(np.abs(residual)) <= max(5e-8, 2e-4 * float(np.max(np.abs(delta))))
        and central_spread <= 0.0005
        and r_squared >= 0.999999
    )
    safety_valid = all(
        item.safety.actuation.completed_steps
        == item.safety.actuation.requested_steps
        and item.safety.actuation.warning_events == ()
        and item.safety.actuation.maximum_contacts == 0
        for item in selected
    )
    sensitivity_valid = all(value is not None for value in sensitivities)
    valid = bool(
        steady_valid
        and local_valid
        and symmetry_valid
        and linear_valid
        and safety_valid
        and sensitivity_valid
    )
    stiffness = float(1.0 / slope) if valid else None
    return AxisComplianceFit(
        common_tension_n=common_tension_n,
        axis=axis,
        force_axis_n=_freeze(force),
        delta_axis_m=_freeze(delta),
        slope_compliance_m_n=float(slope),
        intercept_m=float(intercept),
        r_squared=float(r_squared),
        residual_m=_freeze(residual),
        maximum_fit_residual_m=float(np.max(np.abs(residual))),
        central_compliance_m_n=_freeze(central),
        central_relative_spread=central_spread,
        one_sided_relative_asymmetry=float(max(asymmetry)),
        maximum_cross_axis_residual_m=float(np.max(cross)),
        maximum_cross_axis_relative=float(np.max(cross / along)),
        maximum_window_sensitivity=float(max(sensitivities)) if sensitivity_valid else math.inf,
        minimum_signal_to_noise=float(min(noise_ratios)),
        stiffness_n_m=stiffness,
        valid=valid,
    )


def arm_total_length_m() -> float:
    return ScaledStableSoftArmSpec().total_length


def signed_axis_ols(force_axis_n, delta_axis_m):
    """Fit signed fixed-axis displacement; never fold negative responses."""
    force = np.asarray(force_axis_n, dtype=float)
    delta = np.asarray(delta_axis_m, dtype=float)
    if force.ndim != 1 or delta.ndim != 1 or len(force) != len(delta):
        raise ValueError("signed OLS inputs must be equal one-dimensional arrays")
    if len(force) < 3 or not np.all(np.isfinite(force)) or not np.all(np.isfinite(delta)):
        raise ValueError("signed OLS inputs must contain at least three finite points")
    if np.ptp(force) <= 0:
        raise ValueError("signed OLS force values must span a nonzero interval")
    design = np.column_stack((force, np.ones(len(force))))
    slope, intercept = np.linalg.lstsq(design, delta, rcond=None)[0]
    residual = delta - (slope * force + intercept)
    denominator = float(np.sum((delta - np.mean(delta)) ** 2))
    r_squared = (
        1.0 - float(np.sum(residual ** 2)) / denominator
        if denominator > 0
        else math.nan
    )
    return float(slope), float(intercept), float(r_squared), _freeze(residual)


def classify_common_mode_effect(
    common_tension_n,
    x_fit,
    y_fit,
    baseline_x_fit,
    baseline_y_fit,
    convergence_relative_error,
):
    """Classify one level using only dimensionless uncertainty inputs."""
    fits = (x_fit, y_fit)
    baseline_fits = (baseline_x_fit, baseline_y_fit)
    if not all(item.valid and item.stiffness_n_m is not None for item in fits + baseline_fits):
        return CommonModeEffect(
            common_tension_n, math.nan, None, None, math.inf, math.inf, "possible instability"
        )
    compliance = float(np.mean([item.slope_compliance_m_n for item in fits]))
    stiffness = float(np.mean([item.stiffness_n_m for item in fits]))
    baseline_stiffness = float(np.mean([item.stiffness_n_m for item in baseline_fits]))
    ratio = stiffness / baseline_stiffness
    xy_variation = abs(x_fit.slope_compliance_m_n - y_fit.slope_compliance_m_n) / max(
        abs(x_fit.slope_compliance_m_n), abs(y_fit.slope_compliance_m_n)
    )
    convergence_error = _finite_float(
        "convergence_relative_error", convergence_relative_error
    )
    if convergence_error < 0:
        raise ValueError("convergence_relative_error must be non-negative")
    uncertainty = max(
        convergence_error,
        xy_variation,
        x_fit.one_sided_relative_asymmetry,
        y_fit.one_sided_relative_asymmetry,
        x_fit.central_relative_spread,
        y_fit.central_relative_spread,
        x_fit.maximum_window_sensitivity,
        y_fit.maximum_window_sensitivity,
    )
    floor = max(0.001, 3 * uncertainty)
    if ratio > 1 + floor:
        classification = "effective hardening"
    elif ratio < 1 - floor:
        classification = "effective softening"
    else:
        classification = "no resolved change"
    return CommonModeEffect(
        common_tension_n=common_tension_n,
        compliance_m_n=compliance,
        stiffness_n_m=stiffness,
        stiffness_ratio=ratio,
        uncertainty_relative=uncertainty,
        effect_floor_relative=floor,
        classification=classification,
    )
