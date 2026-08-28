"""Fixed-root vector-tension calibration probes for Gate C2-C2-B."""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import ScaledStableSoftArmSpec
from multi_drone_mujoco.models.components.tendon_drive import IdealTendonTensionSpec
from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_probe_runner import (
    CTRL_FREQ_HZ,
    INTEGRATOR,
    SIM_FREQUENCIES_HZ,
    ProbeSafetySummary,
    _SafetyAccumulator,
    _check_synchronized_sample,
    _freeze,
    _records_array,
    _required_id,
    _state_sample,
    _time_tolerance,
    _validate_topology,
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


CORNERS = ("xp_yp", "xp_yn", "xn_yp", "xn_yn")
FORMAL_TENSIONS_N = (0.002, 0.005, 0.010)


@dataclass(frozen=True)
class VectorTensionProbeConfig:
    sim_freq_hz: int
    target_tension_n: tuple
    duration_s: float = 1.0
    ramp_duration_s: float = 0.05
    ctrl_freq_hz: int = CTRL_FREQ_HZ

    def __post_init__(self):
        if self.sim_freq_hz not in SIM_FREQUENCIES_HZ:
            raise ValueError(f"sim_freq_hz must be one of {SIM_FREQUENCIES_HZ}")
        if self.ctrl_freq_hz != CTRL_FREQ_HZ or self.sim_freq_hz % self.ctrl_freq_hz:
            raise ValueError("probe frequencies must preserve the 240 Hz control grid")
        target = tuple(float(value) for value in self.target_tension_n)
        if len(target) != 4 or not all(math.isfinite(value) for value in target):
            raise ValueError("target_tension_n must contain four finite values")
        if any(value < 0.0 or value > MAX_TENSION_N for value in target):
            raise ValueError(f"target tensions must lie in [0, {MAX_TENSION_N}]")
        for name in ("duration_s", "ramp_duration_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "target_tension_n", target)
        if self.ramp_duration_s > self.duration_s:
            raise ValueError("ramp_duration_s must not exceed duration_s")
        _ = self.number_of_control_intervals
        _ = self.ramp_ticks

    @property
    def number_of_control_intervals(self):
        return _aligned_ticks(self.duration_s, self.ctrl_freq_hz)

    @property
    def ramp_ticks(self):
        return _aligned_ticks(self.ramp_duration_s, self.ctrl_freq_hz)

    @property
    def substeps_per_control_interval(self):
        return self.sim_freq_hz // self.ctrl_freq_hz

    @property
    def number_of_physics_steps(self):
        return self.number_of_control_intervals * self.substeps_per_control_interval


def _aligned_ticks(duration, frequency):
    raw = duration * frequency
    ticks = int(round(raw))
    if ticks <= 0 or abs(raw - ticks) > 64 * np.finfo(float).eps * max(1.0, abs(raw)):
        raise ValueError("probe durations must align with the control grid")
    return ticks


def _ramp_scale(tick, ramp_ticks):
    if tick >= ramp_ticks:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * tick / ramp_ticks))


@dataclass(frozen=True)
class VectorTensionProbeTrace:
    config: VectorTensionProbeConfig
    tendon_names: tuple
    time_s: np.ndarray
    tendon_length_m: np.ndarray
    tendon_velocity_m_s: np.ndarray
    tip_position_m: np.ndarray
    joint_rotation_vectors_rad: np.ndarray
    composite_rotation_vectors_rad: np.ndarray
    commanded_tension_n: np.ndarray
    safety: ProbeSafetySummary

    @property
    def contraction_m(self):
        return _freeze(self.tendon_length_m[0] - self.tendon_length_m)


def run_vector_tension_probe(config):
    if not isinstance(config, VectorTensionProbeConfig):
        raise TypeError("config must be VectorTensionProbeConfig")
    arm_spec = ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )
    drive_spec = IdealTendonTensionSpec(max_tension=MAX_TENSION_N)
    fixture = build_fixed_arm_fixture(
        arm_spec, "vertical", sim_freq=config.sim_freq_hz,
        ctrl_freq=config.ctrl_freq_hz, integrator=INTEGRATOR,
        tendon_drive=drive_spec,
    )
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    bindings = tuple(resolve_tendon_actuator_bindings(fixture))
    _validate_topology(fixture, bindings, config)
    addresses = named_joint_addresses(model, manifest)
    tip_id = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name)
    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    tendon_ranges = np.asarray(model.tendon_range[tendon_ids]).copy()
    tendon_limited = np.asarray(model.tendon_limited[tendon_ids]).copy()
    active_indices = tuple(index for index, value in enumerate(config.target_tension_n) if value > 0.0)
    if not active_indices:
        active_indices = (0, 1, 2, 3)

    mujoco.mj_resetData(model, data)
    zero, current = write_tension_command(fixture, (0.0,) * 4, drive_spec, bindings=bindings)
    if current != bindings:
        raise AssertionError("bindings changed during reset")
    mujoco.mj_forward(model, data)
    warning_latch = WarningLatch()
    accumulator = _SafetyAccumulator()
    _check_synchronized_sample(
        fixture, bindings, zero, active_indices, tendon_ranges, tendon_limited,
        warning_latch, accumulator, stage="reset_forward", control_interval=0,
        physics_substep=0, physics_step=0,
    )
    states = [_state_sample(fixture, bindings, addresses, tip_id)]
    commands = []
    maximum_time_error = 0.0
    timestep = float(model.opt.timestep)
    target = np.asarray(config.target_tension_n)
    for interval in range(config.number_of_control_intervals):
        expected_start = interval / config.ctrl_freq_hz
        if float(data.time) != expected_start:
            raise AssertionError("control interval did not start on exact grid")
        requested = target * _ramp_scale(interval, config.ramp_ticks)
        command, current = write_tension_command(fixture, requested, drive_spec, bindings=bindings)
        if current != bindings:
            raise AssertionError("bindings changed during probe")
        commands.append(np.asarray(command).copy())
        mujoco.mj_forward(model, data)
        _check_synchronized_sample(
            fixture, bindings, command, active_indices, tendon_ranges, tendon_limited,
            warning_latch, accumulator, stage="control_forward", control_interval=interval,
            physics_substep=0, physics_step=accumulator.completed_physics_steps,
        )
        for substep in range(1, config.substeps_per_control_interval + 1):
            next_step = accumulator.completed_physics_steps + 1
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            _check_synchronized_sample(
                fixture, bindings, command, active_indices, tendon_ranges, tendon_limited,
                warning_latch, accumulator, stage="physics_forward", control_interval=interval,
                physics_substep=substep, physics_step=next_step,
            )
            accumulator.completed_physics_steps = next_step
        expected_end = (interval + 1) / config.ctrl_freq_hz
        error = abs(float(data.time) - expected_end)
        maximum_time_error = max(maximum_time_error, error)
        if error > _time_tolerance(expected_end, accumulator.completed_physics_steps, timestep):
            raise AssertionError("control time drift exceeded tolerance")
        data.time = expected_end
        states.append(_state_sample(fixture, bindings, addresses, tip_id))

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
    return VectorTensionProbeTrace(
        config=config,
        tendon_names=tuple(manifest.tendon_names),
        time_s=_records_array(states, "time"),
        tendon_length_m=_records_array(states, "tendon_length"),
        tendon_velocity_m_s=_records_array(states, "tendon_velocity"),
        tip_position_m=_records_array(states, "tip"),
        joint_rotation_vectors_rad=_records_array(states, "joints"),
        composite_rotation_vectors_rad=_records_array(states, "composite"),
        commanded_tension_n=_freeze(commands),
        safety=safety,
    )


def tail_mean_contraction(trace, start_s=0.8):
    mask = trace.time_s >= start_s
    return _freeze(np.mean(trace.contraction_m[mask], axis=0))


def response_matrix(zero, single_traces, tension_n):
    columns = []
    baseline = tail_mean_contraction(zero)
    for trace in single_traces:
        columns.append((tail_mean_contraction(trace) - baseline) / tension_n)
    return _freeze(np.column_stack(columns))


def retained_projector(matrix):
    u, singular_values, _ = np.linalg.svd(matrix)
    basis = u[:, :2]
    return singular_values, basis @ basis.T


def maximum_principal_angle_rad(first_projector, second_projector):
    first_values, first_vectors = np.linalg.eigh(first_projector)
    second_values, second_vectors = np.linalg.eigh(second_projector)
    first_basis = first_vectors[:, np.argsort(first_values)[-2:]]
    second_basis = second_vectors[:, np.argsort(second_values)[-2:]]
    cosines = np.linalg.svd(first_basis.T @ second_basis, compute_uv=False)
    return float(np.max(np.arccos(np.clip(cosines, -1.0, 1.0))))
