"""Headless fixed-root acceleration-servo runner and metrics for Gate C2-B."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Tuple

import mujoco
import numpy as np

from multi_drone_mujoco.control.tendon_acceleration_controller import (
    TendonAccelerationControllerConfig,
    TendonContractionAccelerationController,
)
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)
from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_probe import (
    integrate_interval_power_w,
)
from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_probe_runner import (
    CTRL_FREQ_HZ,
    INTEGRATOR,
    SIM_FREQUENCIES_HZ,
    ProbeSafetySummary,
    _SafetyAccumulator,
    _check_synchronized_sample,
    _freeze,
    _grouped_physics_array,
    _records_array,
    _required_id,
    _state_sample,
    _time_tolerance,
    _validate_topology,
    resolve_probe_channel,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    JOINT_STIFFNESS,
    MAX_TENSION_N,
    TARGET_DAMPING_RATIO,
    resolve_tendon_actuator_bindings,
    write_tension_command,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    WarningLatch,
    build_fixed_arm_fixture,
    named_joint_addresses,
)


FORMAL_DURATION_S = 0.6
FORMAL_ACCELERATION_DURATION_S = 0.05
FORMAL_ACTIVE_CORNER = "xn_yp"


def _finite_float(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _aligned_ticks(name, duration_s, frequency_hz):
    raw = duration_s * frequency_hz
    ticks = int(round(raw))
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(raw))
    if ticks <= 0 or abs(raw - ticks) > tolerance:
        raise ValueError(f"{name} must align with the control tick grid")
    return ticks


@dataclass(frozen=True)
class FixedRootAccelerationRunConfig:
    """One deterministic accelerate-decelerate-hold closed-loop run."""

    sim_freq_hz: int
    controller: TendonAccelerationControllerConfig
    acceleration_m_s2: float
    active_tendon_corners: Tuple[str, ...] = (FORMAL_ACTIVE_CORNER,)
    applied_tendon_corners: Tuple[str, ...] = (FORMAL_ACTIVE_CORNER,)
    ctrl_freq_hz: int = CTRL_FREQ_HZ
    duration_s: float = FORMAL_DURATION_S
    acceleration_duration_s: float = FORMAL_ACCELERATION_DURATION_S

    def __post_init__(self):
        sim_freq = _integer("sim_freq_hz", self.sim_freq_hz)
        ctrl_freq = _integer("ctrl_freq_hz", self.ctrl_freq_hz)
        acceleration = _finite_float(
            "acceleration_m_s2", self.acceleration_m_s2
        )
        duration = _finite_float("duration_s", self.duration_s)
        pulse = _finite_float(
            "acceleration_duration_s", self.acceleration_duration_s
        )
        if sim_freq not in SIM_FREQUENCIES_HZ:
            raise ValueError(f"sim_freq_hz must be one of {SIM_FREQUENCIES_HZ}")
        if ctrl_freq != CTRL_FREQ_HZ:
            raise ValueError(f"ctrl_freq_hz must be exactly {CTRL_FREQ_HZ}")
        if sim_freq % ctrl_freq:
            raise ValueError("sim_freq_hz must be divisible by ctrl_freq_hz")
        if not isinstance(self.controller, TendonAccelerationControllerConfig):
            raise TypeError("controller must be TendonAccelerationControllerConfig")
        if self.controller.control_timestep_s != 1.0 / CTRL_FREQ_HZ:
            raise ValueError("controller timestep must be exactly 1/240 s")
        if self.controller.max_tension_n != MAX_TENSION_N:
            raise ValueError(f"controller max_tension_n must be {MAX_TENSION_N}")
        if acceleration <= 0.0:
            raise ValueError("acceleration_m_s2 must be positive")
        if duration <= 2.0 * pulse:
            raise ValueError("duration_s must include a hold after deceleration")
        active = tuple(self.active_tendon_corners)
        applied = tuple(self.applied_tendon_corners)
        for name, corners in (("active", active), ("applied", applied)):
            if not corners or any(not isinstance(item, str) or not item for item in corners):
                raise ValueError(f"{name}_tendon_corners must be non-empty names")
            if len(set(corners)) != len(corners):
                raise ValueError(f"{name}_tendon_corners must be unique")
        if not set(active).issubset(applied):
            raise ValueError("active tendon corners must be applied")
        object.__setattr__(self, "sim_freq_hz", sim_freq)
        object.__setattr__(self, "ctrl_freq_hz", ctrl_freq)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "acceleration_duration_s", pulse)
        object.__setattr__(self, "acceleration_m_s2", acceleration)
        object.__setattr__(self, "active_tendon_corners", active)
        object.__setattr__(self, "applied_tendon_corners", applied)
        _ = self.number_of_control_intervals
        _ = self.acceleration_ticks

    @property
    def number_of_control_intervals(self):
        return _aligned_ticks("duration_s", self.duration_s, self.ctrl_freq_hz)

    @property
    def acceleration_ticks(self):
        return _aligned_ticks(
            "acceleration_duration_s",
            self.acceleration_duration_s,
            self.ctrl_freq_hz,
        )

    @property
    def substeps_per_control_interval(self):
        return self.sim_freq_hz // self.ctrl_freq_hz

    @property
    def number_of_physics_steps(self):
        return self.number_of_control_intervals * self.substeps_per_control_interval


def build_acceleration_profile(config, active_indices):
    """Build +a, -a, hold on manifest-resolved channels."""
    active_indices = tuple(int(index) for index in active_indices)
    command = np.zeros((config.number_of_control_intervals, 4), dtype=float)
    stop = config.acceleration_ticks
    command[:stop, active_indices] = config.acceleration_m_s2
    command[stop : 2 * stop, active_indices] = -config.acceleration_m_s2
    command.setflags(write=False)
    return command


@dataclass(frozen=True)
class FixedRootAccelerationTrace:
    config: FixedRootAccelerationRunConfig
    bindings: tuple
    active_indices: Tuple[int, ...]
    applied_indices: Tuple[int, ...]
    time_s: np.ndarray
    tendon_length_m: np.ndarray
    tendon_velocity_m_s: np.ndarray
    tip_position_m: np.ndarray
    joint_rotation_vectors_rad: np.ndarray
    composite_rotation_vectors_rad: np.ndarray
    acceleration_command_m_s2: np.ndarray
    raw_tension_n: np.ndarray
    applied_tension_n: np.ndarray
    contraction_reference_m: np.ndarray
    contraction_velocity_reference_m_s: np.ndarray
    contraction_actual_m: np.ndarray
    contraction_velocity_actual_m_s: np.ndarray
    position_error_m: np.ndarray
    velocity_error_m_s: np.ndarray
    tension_clipped_low: np.ndarray
    tension_clipped_high: np.ndarray
    physics_command_power_w: np.ndarray
    physics_generalized_power_w: np.ndarray
    physics_contact_count: np.ndarray
    physics_warning_counts: np.ndarray
    controller_update_count: int
    number_of_physics_steps: int
    maximum_control_time_error_s: float
    safety: ProbeSafetySummary

    @property
    def command_work_j(self):
        return integrate_interval_power_w(
            self.physics_command_power_w,
            dt_sim_s=1.0 / self.config.sim_freq_hz,
        )


def _controller_record(controller, acceleration, raw, applied):
    state = controller.state
    return {
        "acceleration": np.asarray(acceleration).copy(),
        "raw": np.asarray(raw).copy(),
        "applied": np.asarray(applied).copy(),
        "s_ref": np.asarray(state.contraction_reference_m).copy(),
        "v_ref": np.asarray(state.contraction_velocity_reference_m_s).copy(),
        "s_actual": np.asarray(state.contraction_actual_m).copy(),
        "v_actual": np.asarray(state.contraction_velocity_actual_m_s).copy(),
        "position_error": np.asarray(state.position_error_m).copy(),
        "velocity_error": np.asarray(state.velocity_error_m_s).copy(),
        "clip_low": np.asarray(state.tension_was_clipped_low).copy(),
        "clip_high": np.asarray(state.tension_was_clipped_high).copy(),
    }


def run_fixed_root_acceleration_servo(config):
    """Run controller once per control tick and hold tension over substeps."""
    if not isinstance(config, FixedRootAccelerationRunConfig):
        raise TypeError("config must be FixedRootAccelerationRunConfig")
    arm_spec = ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )
    drive_spec = IdealTendonTensionSpec(max_tension=MAX_TENSION_N)
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
    _validate_topology(fixture, bindings, config)
    resolved = {
        corner: resolve_probe_channel(
            corner,
            manifest.tendon_names,
            manifest.tendon_actuator_names,
            bindings,
        )
        for corner in set(config.active_tendon_corners + config.applied_tendon_corners)
    }
    active_indices = tuple(
        resolved[corner].binding_index for corner in config.active_tendon_corners
    )
    applied_indices = tuple(
        resolved[corner].binding_index for corner in config.applied_tendon_corners
    )
    acceleration_profile = build_acceleration_profile(config, active_indices)
    addresses = named_joint_addresses(model, manifest)
    tip_id = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name)
    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    tendon_ranges = np.asarray(model.tendon_range[tendon_ids]).copy()
    tendon_limited = np.asarray(model.tendon_limited[tendon_ids]).copy()

    mujoco.mj_resetData(model, data)
    write_tension_command(fixture, (0.0,) * 4, drive_spec, bindings=bindings)
    mujoco.mj_forward(model, data)
    controller = TendonContractionAccelerationController(config.controller)
    controller.reset(data.ten_length[tendon_ids], data.ten_velocity[tendon_ids])
    warning_latch = WarningLatch()
    accumulator = _SafetyAccumulator()
    _check_synchronized_sample(
        fixture, bindings, (0.0,) * 4, applied_indices,
        tendon_ranges, tendon_limited, warning_latch, accumulator,
        stage="reset_forward", control_interval=0, physics_substep=0,
        physics_step=0,
    )
    state_records = [_state_sample(fixture, bindings, addresses, tip_id)]
    controller_records = []
    physics_records = []
    maximum_time_error = 0.0
    timestep = float(model.opt.timestep)

    for interval_index, acceleration in enumerate(acceleration_profile):
        expected_start = interval_index / config.ctrl_freq_hz
        if float(data.time) != expected_start:
            raise AssertionError("control interval did not start on exact grid")
        raw = controller.update(
            acceleration,
            data.ten_length[tendon_ids],
            data.ten_velocity[tendon_ids],
        )
        applied = np.zeros(4)
        applied[list(applied_indices)] = np.asarray(raw)[list(applied_indices)]
        command, current_bindings = write_tension_command(
            fixture, applied, drive_spec, bindings=bindings
        )
        if current_bindings != bindings:
            raise AssertionError("tendon bindings changed during run")
        mujoco.mj_forward(model, data)
        samples = [_check_synchronized_sample(
            fixture, bindings, command, applied_indices,
            tendon_ranges, tendon_limited, warning_latch, accumulator,
            stage="control_forward", control_interval=interval_index,
            physics_substep=0, physics_step=accumulator.completed_physics_steps,
        )]
        controller_records.append(
            _controller_record(controller, acceleration, raw, command)
        )
        for substep_index in range(1, config.substeps_per_control_interval + 1):
            next_step = accumulator.completed_physics_steps + 1
            mujoco.mj_step(model, data)
            if not (
                math.isfinite(float(data.time))
                and np.all(np.isfinite(data.qpos))
                and np.all(np.isfinite(data.qvel))
            ):
                raise AssertionError("non-finite raw state after physics step")
            mujoco.mj_forward(model, data)
            samples.append(_check_synchronized_sample(
                fixture, bindings, command, applied_indices,
                tendon_ranges, tendon_limited, warning_latch, accumulator,
                stage="physics_forward", control_interval=interval_index,
                physics_substep=substep_index, physics_step=next_step,
            ))
            accumulator.completed_physics_steps = next_step
        expected_end = (interval_index + 1) / config.ctrl_freq_hz
        time_error = abs(float(data.time) - expected_end)
        maximum_time_error = max(maximum_time_error, time_error)
        if time_error > _time_tolerance(
            expected_end, accumulator.completed_physics_steps, timestep
        ):
            raise AssertionError("control time drift exceeded tolerance")
        data.time = expected_end
        state_records.append(_state_sample(fixture, bindings, addresses, tip_id))
        physics_records.append(samples)

    if accumulator.completed_physics_steps != config.number_of_physics_steps:
        raise AssertionError("run ended before all physics steps")
    safety = ProbeSafetySummary(
        completed_physics_steps=accumulator.completed_physics_steps,
        warning_events=tuple(warning_latch.events),
        maximum_contacts=accumulator.maximum_contacts,
        minimum_tendon_range_margin_m=accumulator.minimum_tendon_range_margin_m,
        maximum_force_command_error_n=accumulator.maximum_force_command_error_n,
        maximum_moment_residual=accumulator.maximum_moment_residual,
        maximum_generalized_actuator_power_residual_w=accumulator.maximum_generalized_actuator_power_residual_w,
        maximum_actuator_tendon_power_residual_w=accumulator.maximum_actuator_tendon_power_residual_w,
        maximum_command_tendon_power_residual_w=accumulator.maximum_command_tendon_power_residual_w,
        maximum_actuator_length_sign_residual_m=accumulator.maximum_actuator_length_sign_residual_m,
        maximum_actuator_velocity_sign_residual_m_s=accumulator.maximum_actuator_velocity_sign_residual_m_s,
    )
    return FixedRootAccelerationTrace(
        config=config,
        bindings=bindings,
        active_indices=active_indices,
        applied_indices=applied_indices,
        time_s=_records_array(state_records, "time"),
        tendon_length_m=_records_array(state_records, "tendon_length"),
        tendon_velocity_m_s=_records_array(state_records, "tendon_velocity"),
        tip_position_m=_records_array(state_records, "tip"),
        joint_rotation_vectors_rad=_records_array(state_records, "joints"),
        composite_rotation_vectors_rad=_records_array(state_records, "composite"),
        acceleration_command_m_s2=_records_array(controller_records, "acceleration"),
        raw_tension_n=_records_array(controller_records, "raw"),
        applied_tension_n=_records_array(controller_records, "applied"),
        contraction_reference_m=_records_array(controller_records, "s_ref"),
        contraction_velocity_reference_m_s=_records_array(controller_records, "v_ref"),
        contraction_actual_m=_records_array(controller_records, "s_actual"),
        contraction_velocity_actual_m_s=_records_array(controller_records, "v_actual"),
        position_error_m=_records_array(controller_records, "position_error"),
        velocity_error_m_s=_records_array(controller_records, "velocity_error"),
        tension_clipped_low=_records_array(controller_records, "clip_low", dtype=bool),
        tension_clipped_high=_records_array(controller_records, "clip_high", dtype=bool),
        physics_command_power_w=_grouped_physics_array(physics_records, "command_power"),
        physics_generalized_power_w=_grouped_physics_array(physics_records, "generalized_power"),
        physics_contact_count=_grouped_physics_array(physics_records, "contacts", dtype=int),
        physics_warning_counts=_grouped_physics_array(physics_records, "warnings", dtype=int),
        controller_update_count=len(controller_records),
        number_of_physics_steps=accumulator.completed_physics_steps,
        maximum_control_time_error_s=maximum_time_error,
        safety=safety,
    )


@dataclass(frozen=True)
class AccelerationServoMetrics:
    target_contraction_m: float
    actual_final_contraction_m: float
    signed_peak_contraction_m: float
    overshoot_m: float
    peak_position_error_m: float
    peak_velocity_error_m_s: float
    steady_state_error_m: float
    settling_time_s: float
    peak_tension_n: float
    saturation_fraction: float
    longest_saturation_s: float
    saturation_recovered: bool
    inactive_force_peak_n: float
    inactive_contraction_peak_m: float
    cross_coupling_ratio: float
    tip_peak_displacement_m: float
    joint_peak_rotation_rad: float
    peak_abs_power_w: float
    work_j: float


def _longest_true_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def compute_acceleration_servo_metrics(trace, *, activity_resolution_m):
    if not isinstance(trace, FixedRootAccelerationTrace):
        raise TypeError("trace must be FixedRootAccelerationTrace")
    if len(trace.active_indices) != 1:
        raise ValueError("single-channel metrics require exactly one active tendon")
    resolution = _finite_float("activity_resolution_m", activity_resolution_m)
    if resolution < 0.0:
        raise ValueError("activity_resolution_m must be non-negative")
    index = trace.active_indices[0]
    contraction_state = trace.tendon_length_m[0, index] - trace.tendon_length_m[:, index]
    target = float(trace.contraction_reference_m[-1, index])
    actual_final = float(contraction_state[-1])
    signed_peak = float(np.max(contraction_state))
    overshoot = max(0.0, signed_peak - target)
    position_error = trace.position_error_m[:, index]
    velocity_error = trace.velocity_error_m_s[:, index]
    tail_ticks = max(12, int(round(0.1 * trace.config.ctrl_freq_hz)))
    steady = float(np.mean(position_error[-tail_ticks:]))
    band = max(0.05 * abs(target), 16.0 * resolution)
    state_error = target - contraction_state
    settled = np.flatnonzero(
        [np.all(np.abs(state_error[start:]) <= band) for start in range(state_error.size)]
    )
    settling_time = float(trace.time_s[settled[0]]) if settled.size else math.inf
    saturation = trace.tension_clipped_high[:, index]
    last_saturation = np.flatnonzero(saturation)
    recovered = bool(
        last_saturation.size == 0
        or (
            last_saturation[-1] < saturation.size - 1
            and trace.applied_tension_n[-1, index] < MAX_TENSION_N
        )
    )
    inactive = [item for item in range(4) if item not in trace.applied_indices]
    inactive_force = (
        float(np.max(np.abs(trace.applied_tension_n[:, inactive])))
        if inactive else 0.0
    )
    contraction_all = trace.tendon_length_m[0] - trace.tendon_length_m
    inactive_contraction = (
        float(np.max(np.abs(contraction_all[:, inactive])))
        if inactive else 0.0
    )
    denominator = max(abs(signed_peak), resolution)
    tip_displacement = trace.tip_position_m - trace.tip_position_m[0]
    return AccelerationServoMetrics(
        target_contraction_m=target,
        actual_final_contraction_m=actual_final,
        signed_peak_contraction_m=signed_peak,
        overshoot_m=overshoot,
        peak_position_error_m=float(np.max(np.abs(position_error))),
        peak_velocity_error_m_s=float(np.max(np.abs(velocity_error))),
        steady_state_error_m=steady,
        settling_time_s=settling_time,
        peak_tension_n=float(np.max(trace.applied_tension_n[:, index])),
        saturation_fraction=float(np.mean(saturation)),
        longest_saturation_s=_longest_true_run(saturation) / trace.config.ctrl_freq_hz,
        saturation_recovered=recovered,
        inactive_force_peak_n=inactive_force,
        inactive_contraction_peak_m=inactive_contraction,
        cross_coupling_ratio=inactive_contraction / denominator,
        tip_peak_displacement_m=float(np.max(np.linalg.norm(tip_displacement, axis=1))),
        joint_peak_rotation_rad=float(np.max(np.linalg.norm(trace.joint_rotation_vectors_rad, axis=2))),
        peak_abs_power_w=float(np.max(np.abs(trace.physics_command_power_w))),
        work_j=trace.command_work_j,
    )
