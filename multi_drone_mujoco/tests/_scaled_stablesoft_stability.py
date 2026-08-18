"""B3 timestep/integrator diagnostics for the scaled StableSoft arm.

This module is test-only.  It can run one isolated configuration as a child
process or orchestrate the approved matrix without writing result files.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Optional, Tuple

import mujoco
import numpy as np

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    WarningLatch,
    _named_object_id,
    _record_synchronized_state,
    build_fixed_arm_fixture,
    composite_rotation_vector,
    joint_rotation_vectors,
    named_joint_addresses,
)


SCHEMA_VERSION = "scaled_stablesoft_b3/v1"
CTRL_FREQ = 240
JOINT_STIFFNESS = 0.003086
TARGET_DAMPING_RATIO = 0.3
H1_DURATION = 2.0
H2_DURATION = 5.0
QVEL_HARD_LIMIT = 1000.0
QACC_EVIDENCE_LIMIT = 2.0e5
STABLE_QVEL_REFERENCE = 8.5363
VELOCITY_TREND_MULTIPLIER = 10.0
VELOCITY_TREND_TOTAL_GROWTH = 0.25
ENERGY_RELATIVE_TOLERANCE = 1.0e-4
ENERGY_HARD_MULTIPLIER = 5.0
ENERGY_JSON_PREFIX = "B3_RESULT_JSON="
MATRIX_CHILD_ENV = "B3_MATRIX_CHILD"

RUN_COMPLETED = "completed"
RUN_WARNING = "warning_reset"
RUN_RAW_NONFINITE = "raw_nonfinite"
RUN_SYNCHRONIZED_NONFINITE = "synchronized_nonfinite"
RUN_ENERGY_RUNAWAY = "energy_runaway"
RUN_FINITE_RUNAWAY = "finite_runaway"
RUN_CRASHED = "crashed"
RUN_TIMEOUT = "timeout"

ENERGY_CONSERVATIVE = "conservative_within_tolerance"
ENERGY_DISSIPATIVE = "dissipative_within_tolerance"
ENERGY_NUMERICAL_DISSIPATION = "numerical_dissipation"
ENERGY_NUMERICAL_INJECTION = "numerical_injection"
ENERGY_ACCURACY_FAILED = "energy_accuracy_failed"
ENERGY_UNAVAILABLE = "unavailable"

ANALYSIS_AVAILABLE = "available"
ANALYSIS_UNAVAILABLE = "analysis_unavailable"
CONVERGED = "converged"
NOT_CONVERGED = "not_converged"
INSUFFICIENT_REFERENCE = "insufficient_reference"
CONVERGENCE_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class StabilityRunConfig:
    case: str
    integrator: str
    sim_freq: int
    plane: str = "x"
    physical_duration: Optional[float] = None
    ctrl_freq: int = CTRL_FREQ

    def __post_init__(self):
        if self.case not in ("H1", "H2"):
            raise ValueError("case must be H1 or H2")
        if self.integrator not in ("RK4", "implicitfast"):
            raise ValueError("integrator must be RK4 or implicitfast")
        if self.plane not in ("x", "y"):
            raise ValueError("plane must be x or y")
        if self.sim_freq <= 0 or self.sim_freq % self.ctrl_freq != 0:
            raise ValueError("sim_freq must be divisible by ctrl_freq")
        if self.physical_duration is None:
            object.__setattr__(
                self,
                "physical_duration",
                H1_DURATION if self.case == "H1" else H2_DURATION,
            )
        if self.physical_duration <= 0:
            raise ValueError("physical_duration must be positive")

    @property
    def persistence_steps(self) -> int:
        return self.sim_freq // self.ctrl_freq

    @property
    def requested_steps(self) -> int:
        return int(round(self.physical_duration * self.sim_freq))


@dataclass(frozen=True)
class NumericalEnvironmentMetadata:
    schema_version: str
    python_version: str
    numpy_version: str
    mujoco_python_version: str
    mujoco_runtime_version: Optional[str]
    integrator_name: str
    integrator_enum: int
    timestep: float
    solver_name: str
    solver_enum: int
    iterations: int
    ls_iterations: int
    tolerance: float
    ls_tolerance: float
    enableflags: int
    disableflags: int


@dataclass(frozen=True)
class EnergyScale:
    delta_ug_scale: float
    elastic_90: float
    total: float
    tolerance: float
    hard_runaway: float
    single_step_hard_injection: float


@dataclass
class PersistenceTracker:
    current_steps: int = 0
    current_duration: float = 0.0
    longest_steps: int = 0
    longest_duration: float = 0.0

    def update(self, condition: bool, interval: float) -> None:
        if condition:
            self.current_steps += 1
            self.current_duration += interval
            self.longest_steps = max(self.longest_steps, self.current_steps)
            self.longest_duration = max(
                self.longest_duration,
                self.current_duration,
            )
        else:
            self.current_steps = 0
            self.current_duration = 0.0

    def reached(self, required_steps: int, required_duration: float) -> bool:
        duration_tolerance = 32 * np.finfo(float).eps * max(
            1.0,
            abs(required_duration),
        )
        return (
            self.current_steps >= required_steps
            and self.current_duration + duration_tolerance
            >= required_duration
        )


@dataclass
class StabilityResult:
    config: StabilityRunConfig
    metadata: NumericalEnvironmentMetadata
    energy_scale: EnergyScale
    run_status: str
    energy_status: str
    convergence_status: str
    analysis_status: str
    requested_steps: int
    completed_steps: int
    final_time: float
    persistence_steps: int
    persistence_duration: float
    first_failure: Optional[dict]
    warning_events: list
    metrics: dict
    energy: dict
    analysis: dict
    persistence: dict
    elapsed_seconds: float
    convergence_details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _json_ready(asdict(self))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _enum_name(enum_type, value: int) -> str:
    try:
        return enum_type(value).name
    except (TypeError, ValueError):
        return f"unknown_{value}"


def numerical_environment_metadata(model) -> NumericalEnvironmentMetadata:
    runtime_version = None
    if hasattr(mujoco, "mj_versionString"):
        runtime_version = str(mujoco.mj_versionString())
    return NumericalEnvironmentMetadata(
        schema_version=SCHEMA_VERSION,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        mujoco_python_version=str(mujoco.__version__),
        mujoco_runtime_version=runtime_version,
        integrator_name=_enum_name(
            mujoco.mjtIntegrator,
            int(model.opt.integrator),
        ),
        integrator_enum=int(model.opt.integrator),
        timestep=float(model.opt.timestep),
        solver_name=_enum_name(mujoco.mjtSolver, int(model.opt.solver)),
        solver_enum=int(model.opt.solver),
        iterations=int(model.opt.iterations),
        ls_iterations=int(model.opt.ls_iterations),
        tolerance=float(model.opt.tolerance),
        ls_tolerance=float(model.opt.ls_tolerance),
        enableflags=int(model.opt.enableflags),
        disableflags=int(model.opt.disableflags),
    )


def _arm_relative_gravity_potential(fixture) -> float:
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    root_id = _named_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        manifest.arm_root_site_name,
    )
    root_position = np.asarray(data.site_xpos[root_id], dtype=float)
    gravity = np.asarray(model.opt.gravity, dtype=float)
    result = 0.0
    for body_name in manifest.segment_body_names:
        body_id = _named_object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        relative_com = np.asarray(data.xipos[body_id], dtype=float) - root_position
        result -= float(model.body_mass[body_id]) * float(
            np.dot(gravity, relative_com)
        )
    return result


def calculate_energy_scale(
    fixture,
    spec: ScaledStableSoftArmSpec,
    config: StabilityRunConfig,
) -> EnergyScale:
    vertical_fixture = build_fixed_arm_fixture(
        spec,
        "vertical",
        sim_freq=config.sim_freq,
        ctrl_freq=config.ctrl_freq,
        integrator=config.integrator,
    )
    delta_ug = abs(
        _arm_relative_gravity_potential(vertical_fixture)
        - _arm_relative_gravity_potential(fixture)
    )
    addresses = named_joint_addresses(fixture.model, fixture.manifest)
    per_joint_90 = math.pi / (2 * len(addresses))
    elastic_90 = float(
        math.fsum(
            0.5
            * float(fixture.model.jnt_stiffness[joint_id])
            * per_joint_90 ** 2
            for joint_id, _, _ in addresses
        )
    )
    total = delta_ug + elastic_90
    tolerance = max(1.0e-9, ENERGY_RELATIVE_TOLERANCE * total)
    return EnergyScale(
        delta_ug_scale=float(delta_ug),
        elastic_90=elastic_90,
        total=float(total),
        tolerance=float(tolerance),
        hard_runaway=float(ENERGY_HARD_MULTIPLIER * total),
        single_step_hard_injection=float(total),
    )


def length_tolerance(*values: float) -> float:
    scale = max((abs(float(value)) for value in values), default=0.0)
    return max(1.0e-9, 1.0e-8 * scale)


def _all_finite(values) -> bool:
    return bool(np.all(np.isfinite(values)))


def _energy_increment_floor(*values: float) -> float:
    scale = max(1.0, *(abs(float(value)) for value in values))
    return 64 * np.finfo(float).eps * scale


def _warning_dict(event) -> dict:
    return {
        "stage": event.stage,
        "step_index": event.step_index,
        "time": event.time,
        "counts": list(event.counts),
        "raised_indices": list(event.raised_indices),
        "raised_names": [
            _enum_name(mujoco.mjtWarning, index)
            for index in event.raised_indices
        ],
    }


def _extrema_frequency(time_values, signal_values) -> Tuple[Optional[float], int, int]:
    sample_time = np.asarray(time_values, dtype=float)
    signal = np.asarray(signal_values, dtype=float)
    maxima = []
    minima = []
    for index in range(1, len(signal) - 1):
        if signal[index] >= signal[index - 1] and signal[index] > signal[index + 1]:
            maxima.append(index)
        if signal[index] <= signal[index - 1] and signal[index] < signal[index + 1]:
            minima.append(index)

    periods = []
    for indices in (maxima, minima):
        selected_times = sample_time[np.asarray(indices, dtype=int)]
        if len(selected_times) >= 2:
            periods.extend(
                float(value)
                for value in np.diff(selected_times)
                if value >= 0.05
            )
    if not periods:
        return None, len(maxima), len(minima)
    return float(1.0 / np.median(periods)), len(maxima), len(minima)


def _settling_time(time_values, bend_values, qvel_values) -> Optional[float]:
    sample_time = np.asarray(time_values, dtype=float)
    bend = np.asarray(bend_values, dtype=float)
    qvel = np.asarray(qvel_values, dtype=float)
    if len(sample_time) < 2:
        return None
    final_bend = float(bend[-1])
    angle_band = max(1.0e-3, 0.02 * abs(final_bend))
    qualified = (np.abs(bend - final_bend) <= angle_band) & (qvel <= 1.0e-3)
    for index in range(len(sample_time)):
        if sample_time[-1] - sample_time[index] < 0.25:
            break
        if np.all(qualified[index:]):
            return float(sample_time[index])
    return None


def _classify_energy(
    case: str,
    run_status: str,
    energy_metrics: dict,
    energy_scale: EnergyScale,
    persistent_upper_failed: bool,
    persistent_growth_failed: bool,
) -> str:
    if not energy_metrics.get("available", False):
        return ENERGY_UNAVAILABLE
    if run_status == RUN_ENERGY_RUNAWAY:
        return ENERGY_ACCURACY_FAILED

    maximum_absolute = energy_metrics["maximum_absolute_drift"]
    maximum_positive = energy_metrics["maximum_positive_drift"]
    minimum_delta = energy_metrics["minimum_drift"]
    final_delta = energy_metrics["final_drift"]
    tolerance = energy_scale.tolerance
    if case == "H1":
        if maximum_absolute <= tolerance:
            return ENERGY_CONSERVATIVE
        if maximum_positive > tolerance and maximum_positive >= abs(minimum_delta):
            return ENERGY_NUMERICAL_INJECTION
        if abs(minimum_delta) > tolerance:
            return ENERGY_NUMERICAL_DISSIPATION
        return ENERGY_ACCURACY_FAILED

    if (
        maximum_positive <= tolerance
        and final_delta <= tolerance
        and not persistent_upper_failed
        and not persistent_growth_failed
    ):
        return ENERGY_DISSIPATIVE
    if maximum_positive > tolerance or persistent_upper_failed or persistent_growth_failed:
        return ENERGY_NUMERICAL_INJECTION
    return ENERGY_ACCURACY_FAILED


def run_stability_case(config: StabilityRunConfig) -> StabilityResult:
    spec = ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=0.0 if config.case == "H1" else None,
        target_damping_ratio=(
            None if config.case == "H1" else TARGET_DAMPING_RATIO
        ),
    )
    fixture = build_fixed_arm_fixture(
        spec,
        config.plane,
        sim_freq=config.sim_freq,
        ctrl_freq=config.ctrl_freq,
        integrator=config.integrator,
    )
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    if fixture.sample_stride != config.persistence_steps:
        raise AssertionError("fixture sample_stride does not match persistence_steps")
    metadata = numerical_environment_metadata(model)
    energy_scale = calculate_energy_scale(fixture, spec, config)
    addresses = named_joint_addresses(model, manifest)
    arm_dof_addresses = np.concatenate(
        [
            np.arange(dof_address, dof_address + 3)
            for _, _, dof_address in addresses
        ]
    )
    tip_id = _named_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        manifest.arm_tip_site_name,
    )
    root_id = _named_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        manifest.arm_root_site_name,
    )
    tendon_ids = np.asarray(
        [
            _named_object_id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
            for name in manifest.tendon_names
        ],
        dtype=int,
    )
    tendon_ranges = np.asarray(model.tendon_range[tendon_ids], dtype=float)

    warning_latch = WarningLatch()
    warning_latch.snapshot(data, stage="initial_forward", step_index=0)
    initial_record = _record_synchronized_state(
        model,
        data,
        addresses,
        tip_id,
        root_id,
        arm_dof_addresses,
        tendon_ids,
    )
    initial_tip = np.asarray(initial_record["tip_relative"], dtype=float)
    initial_tendons = np.asarray(initial_record["tendon_lengths"], dtype=float)
    initial_energy = float(initial_record["total_energy"])
    initial_time = float(initial_record["time"])
    main_axis = 1 if config.plane == "x" else 0

    control_time = [initial_time]
    control_bend = [float(initial_record["composite"][main_axis])]
    control_qvel = [float(np.max(np.abs(initial_record["arm_qvel"])))]
    control_tip = [initial_tip.copy()]
    control_tendons = [initial_tendons.copy()]
    control_energy = [initial_energy]

    maximum_qvel = control_qvel[0]
    maximum_qacc = float(initial_record["max_abs_qacc"])
    maximum_joint_bend = 0.0
    maximum_joint_twist = 0.0
    maximum_total_bend = 0.0
    maximum_total_twist = 0.0
    maximum_tip_displacement = 0.0
    maximum_root_tip_length = float(np.linalg.norm(initial_tip))
    maximum_contacts = int(initial_record["contacts"])
    tendon_minimum = initial_tendons.copy()
    tendon_maximum = initial_tendons.copy()

    maximum_absolute_drift = 0.0
    maximum_absolute_drift_time = initial_time
    maximum_positive_drift = 0.0
    maximum_positive_drift_time = initial_time
    minimum_drift = 0.0
    minimum_drift_time = initial_time

    upper_tracker = PersistenceTracker()
    positive_tracker = PersistenceTracker()
    velocity_tracker = PersistenceTracker()
    positive_run_start_energy = initial_energy
    velocity_run_start = maximum_qvel
    persistent_upper_failed = False
    persistent_growth_failed = False

    previous_time = initial_time
    previous_energy = initial_energy
    previous_qvel = maximum_qvel
    last_synchronized_record = initial_record
    completed_steps = 0
    run_status = RUN_COMPLETED
    first_failure = None
    start_wall = time.perf_counter()

    for step_index in range(1, config.requested_steps + 1):
        mujoco.mj_step(model, data)
        warning_latch.snapshot(
            data,
            stage="after_step",
            step_index=step_index,
        )
        raw_flags = {
            "time_finite": math.isfinite(float(data.time)),
            "qpos_finite": _all_finite(data.qpos),
            "qvel_finite": _all_finite(data.qvel),
        }
        if not all(raw_flags.values()):
            run_status = RUN_RAW_NONFINITE
            first_failure = {
                "stage": "after_step",
                "step_index": step_index,
                "time": float(data.time),
                "reasons": [
                    name for name, passed in raw_flags.items() if not passed
                ],
            }
            break

        mujoco.mj_forward(model, data)
        warning_latch.snapshot(
            data,
            stage="after_forward",
            step_index=step_index,
        )
        synchronized_flags = {
            "qpos_finite": _all_finite(data.qpos),
            "qvel_finite": _all_finite(data.qvel),
            "qacc_finite": _all_finite(data.qacc),
            "site_finite": _all_finite(data.site_xpos),
            "tendon_finite": _all_finite(data.ten_length),
            "energy_finite": _all_finite(data.energy),
        }
        if not all(synchronized_flags.values()):
            run_status = RUN_SYNCHRONIZED_NONFINITE
            first_failure = {
                "stage": "after_forward",
                "step_index": step_index,
                "time": float(data.time),
                "reasons": [
                    name
                    for name, passed in synchronized_flags.items()
                    if not passed
                ],
            }
            break

        record = _record_synchronized_state(
            model,
            data,
            addresses,
            tip_id,
            root_id,
            arm_dof_addresses,
            tendon_ids,
        )
        last_synchronized_record = record
        completed_steps = step_index
        current_time = float(record["time"])
        interval = current_time - previous_time
        if interval <= 0 or not math.isfinite(interval):
            raise RuntimeError("synchronized physics_time is not increasing")

        joints = np.asarray(record["joints"], dtype=float)
        composite = np.asarray(record["composite"], dtype=float)
        tip_relative = np.asarray(record["tip_relative"], dtype=float)
        tendon_lengths = np.asarray(record["tendon_lengths"], dtype=float)
        current_energy = float(record["total_energy"])
        current_qvel = float(np.max(np.abs(record["arm_qvel"])))
        current_qacc = float(record["max_abs_qacc"])
        energy_delta = current_energy - initial_energy
        energy_increment = current_energy - previous_energy

        maximum_qvel = max(maximum_qvel, current_qvel)
        maximum_qacc = max(maximum_qacc, current_qacc)
        maximum_joint_bend = max(
            maximum_joint_bend,
            float(np.max(np.linalg.norm(joints[:, :2], axis=1))),
        )
        maximum_joint_twist = max(
            maximum_joint_twist,
            float(np.max(np.abs(joints[:, 2]))),
        )
        maximum_total_bend = max(
            maximum_total_bend,
            float(np.linalg.norm(composite[:2])),
        )
        maximum_total_twist = max(
            maximum_total_twist,
            abs(float(composite[2])),
        )
        maximum_tip_displacement = max(
            maximum_tip_displacement,
            float(np.linalg.norm(tip_relative - initial_tip)),
        )
        root_tip_length = float(np.linalg.norm(tip_relative))
        maximum_root_tip_length = max(maximum_root_tip_length, root_tip_length)
        maximum_contacts = max(maximum_contacts, int(record["contacts"]))
        tendon_minimum = np.minimum(tendon_minimum, tendon_lengths)
        tendon_maximum = np.maximum(tendon_maximum, tendon_lengths)

        absolute_drift = abs(energy_delta)
        if absolute_drift > maximum_absolute_drift:
            maximum_absolute_drift = absolute_drift
            maximum_absolute_drift_time = current_time
        if energy_delta > maximum_positive_drift:
            maximum_positive_drift = energy_delta
            maximum_positive_drift_time = current_time
        if energy_delta < minimum_drift:
            minimum_drift = energy_delta
            minimum_drift_time = current_time

        if step_index % fixture.sample_stride == 0:
            control_time.append(current_time)
            control_bend.append(float(composite[main_axis]))
            control_qvel.append(current_qvel)
            control_tip.append(tip_relative.copy())
            control_tendons.append(tendon_lengths.copy())
            control_energy.append(current_energy)

        if warning_latch.triggered:
            event = warning_latch.events[0]
            run_status = RUN_WARNING
            first_failure = {
                "stage": event.stage,
                "step_index": event.step_index,
                "time": event.time,
                "reasons": ["mujoco_warning"],
                "warning_indices": list(event.raised_indices),
            }
            break

        energy_hard_reasons = []
        if energy_delta > energy_scale.hard_runaway:
            energy_hard_reasons.append("total_energy_hard_limit")
        if energy_increment > energy_scale.single_step_hard_injection:
            energy_hard_reasons.append("single_step_energy_injection")

        geometry_tolerance = length_tolerance(
            spec.total_length,
            root_tip_length,
        )
        finite_reasons = []
        if current_qvel > QVEL_HARD_LIMIT:
            finite_reasons.append("qvel_hard_limit")
        if root_tip_length > spec.total_length + geometry_tolerance:
            finite_reasons.append("root_tip_length")
        if maximum_tip_displacement > 2 * spec.total_length + 2 * geometry_tolerance:
            finite_reasons.append("tip_displacement")
        for tendon_index, tendon_length in enumerate(tendon_lengths):
            lower, upper = tendon_ranges[tendon_index]
            tendon_tolerance = length_tolerance(
                lower,
                upper,
                tendon_length,
                spec.total_length,
            )
            if (
                tendon_length < lower - tendon_tolerance
                or tendon_length > upper + tendon_tolerance
            ):
                finite_reasons.append(
                    f"tendon_range:{manifest.tendon_names[tendon_index]}"
                )

        energy_increment_floor = _energy_increment_floor(
            current_energy,
            previous_energy,
            energy_scale.total,
        )
        upper_tracker.update(
            energy_delta > energy_scale.tolerance,
            interval,
        )
        positive_condition = energy_increment > energy_increment_floor
        if positive_condition and positive_tracker.current_steps == 0:
            positive_run_start_energy = previous_energy
        positive_tracker.update(positive_condition, interval)

        velocity_growth_tolerance = max(
            1.0e-6,
            1.0e-3 * max(previous_qvel, STABLE_QVEL_REFERENCE),
        )
        velocity_condition = (
            current_qvel - previous_qvel > velocity_growth_tolerance
        )
        if velocity_condition and velocity_tracker.current_steps == 0:
            velocity_run_start = previous_qvel
        velocity_tracker.update(velocity_condition, interval)

        persistence_duration = 1.0 / config.ctrl_freq
        if upper_tracker.reached(
            config.persistence_steps,
            persistence_duration,
        ):
            persistent_upper_failed = True
        if positive_tracker.reached(
            config.persistence_steps,
            persistence_duration,
        ):
            positive_run_increase = current_energy - positive_run_start_energy
            if positive_run_increase > energy_scale.tolerance:
                persistent_growth_failed = True
            if positive_run_increase > energy_scale.total:
                energy_hard_reasons.append("persistent_energy_growth")
        if velocity_tracker.reached(
            config.persistence_steps,
            persistence_duration,
        ):
            velocity_growth = current_qvel - velocity_run_start
            if (
                current_qvel
                > VELOCITY_TREND_MULTIPLIER * STABLE_QVEL_REFERENCE
                and velocity_growth
                > VELOCITY_TREND_TOTAL_GROWTH
                * max(velocity_run_start, STABLE_QVEL_REFERENCE)
            ):
                finite_reasons.append("persistent_velocity_growth")

        if energy_hard_reasons or finite_reasons:
            run_status = (
                RUN_ENERGY_RUNAWAY
                if energy_hard_reasons
                else RUN_FINITE_RUNAWAY
            )
            first_failure = {
                "stage": "after_forward",
                "step_index": step_index,
                "time": current_time,
                "reasons": energy_hard_reasons + finite_reasons,
            }
            break

        previous_time = current_time
        previous_energy = current_energy
        previous_qvel = current_qvel

    elapsed_seconds = time.perf_counter() - start_wall
    final_record = last_synchronized_record
    final_joints = np.asarray(final_record["joints"], dtype=float)
    final_composite = np.asarray(final_record["composite"], dtype=float)
    final_tip = np.asarray(final_record["tip_relative"], dtype=float)
    final_tendons = np.asarray(final_record["tendon_lengths"], dtype=float)
    final_energy = float(final_record["total_energy"])
    final_drift = final_energy - initial_energy

    energy_metrics = {
        "available": bool(math.isfinite(final_energy)),
        "initial": initial_energy,
        "maximum": initial_energy + maximum_positive_drift,
        "minimum": initial_energy + minimum_drift,
        "final": final_energy,
        "maximum_absolute_drift": maximum_absolute_drift,
        "maximum_absolute_drift_time": maximum_absolute_drift_time,
        "maximum_positive_drift": maximum_positive_drift,
        "maximum_positive_drift_time": maximum_positive_drift_time,
        "minimum_drift": minimum_drift,
        "minimum_drift_time": minimum_drift_time,
        "final_drift": final_drift,
        "normalized_maximum_absolute_drift": (
            maximum_absolute_drift / energy_scale.total
        ),
        "normalized_final_drift": final_drift / energy_scale.total,
    }
    energy_status = _classify_energy(
        config.case,
        run_status,
        energy_metrics,
        energy_scale,
        persistent_upper_failed,
        persistent_growth_failed,
    )

    analysis_status = ANALYSIS_UNAVAILABLE
    analysis = {
        "frequency": None,
        "settling_time": None,
        "maxima_count": 0,
        "minima_count": 0,
    }
    if run_status == RUN_COMPLETED and len(control_time) >= 3:
        frequency, maxima_count, minima_count = _extrema_frequency(
            control_time,
            control_bend,
        )
        settling = (
            _settling_time(control_time, control_bend, control_qvel)
            if config.case == "H2"
            else None
        )
        analysis = {
            "frequency": frequency,
            "settling_time": settling,
            "maxima_count": maxima_count,
            "minima_count": minima_count,
        }
        analysis_status = ANALYSIS_AVAILABLE

    metrics = {
        "maximum_abs_qvel": maximum_qvel,
        "maximum_abs_qacc": maximum_qacc,
        "qacc_evidence_limit_exceeded": maximum_qacc > QACC_EVIDENCE_LIMIT,
        "maximum_joint_bend": maximum_joint_bend,
        "maximum_joint_twist": maximum_joint_twist,
        "maximum_total_bend": maximum_total_bend,
        "maximum_total_twist": maximum_total_twist,
        "maximum_tip_displacement": maximum_tip_displacement,
        "maximum_root_tip_length": maximum_root_tip_length,
        "maximum_contacts": maximum_contacts,
        "initial_tip_relative": initial_tip,
        "final_tip_relative": final_tip,
        "final_total_bend": float(np.linalg.norm(final_composite[:2])),
        "final_signed_bend": float(final_composite[main_axis]),
        "final_joint_main_bends": final_joints[:, main_axis],
        "tendon_names": list(manifest.tendon_names),
        "tendon_ranges": tendon_ranges,
        "tendon_initial": initial_tendons,
        "tendon_minimum": tendon_minimum,
        "tendon_maximum": tendon_maximum,
        "tendon_final": final_tendons,
        "control_sample_count": len(control_time),
    }
    persistence = {
        "required_steps": config.persistence_steps,
        "required_duration": 1.0 / config.ctrl_freq,
        "energy_upper_longest_steps": upper_tracker.longest_steps,
        "energy_upper_longest_duration": upper_tracker.longest_duration,
        "positive_energy_longest_steps": positive_tracker.longest_steps,
        "positive_energy_longest_duration": positive_tracker.longest_duration,
        "velocity_growth_longest_steps": velocity_tracker.longest_steps,
        "velocity_growth_longest_duration": velocity_tracker.longest_duration,
    }
    return StabilityResult(
        config=config,
        metadata=metadata,
        energy_scale=energy_scale,
        run_status=run_status,
        energy_status=energy_status,
        convergence_status=CONVERGENCE_UNAVAILABLE,
        analysis_status=analysis_status,
        requested_steps=config.requested_steps,
        completed_steps=completed_steps,
        final_time=float(data.time),
        persistence_steps=config.persistence_steps,
        persistence_duration=1.0 / config.ctrl_freq,
        first_failure=first_failure,
        warning_events=[
            _warning_dict(event) for event in warning_latch.events
        ],
        metrics=_json_ready(metrics),
        energy=_json_ready(energy_metrics),
        analysis=_json_ready(analysis),
        persistence=_json_ready(persistence),
        elapsed_seconds=elapsed_seconds,
    )


def _relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-12)


def compare_to_reference(
    candidate: StabilityResult,
    reference: StabilityResult,
) -> Tuple[str, dict]:
    if (
        candidate.run_status != RUN_COMPLETED
        or reference.run_status != RUN_COMPLETED
        or candidate.analysis_status != ANALYSIS_AVAILABLE
        or reference.analysis_status != ANALYSIS_AVAILABLE
    ):
        return CONVERGENCE_UNAVAILABLE, {"reason": "run_or_analysis_unavailable"}
    tolerance = 0.01 if candidate.config.sim_freq >= 4800 else 0.02
    details = {}
    checks = []

    candidate_frequency = candidate.analysis.get("frequency")
    reference_frequency = reference.analysis.get("frequency")
    if candidate_frequency is not None and reference_frequency is not None:
        frequency_error = _relative_difference(
            candidate_frequency,
            reference_frequency,
        )
        details["frequency_relative_error"] = frequency_error
        checks.append(frequency_error <= tolerance)

    if candidate.config.case == "H1":
        checks.append(candidate.energy_status == ENERGY_CONSERVATIVE)
        details["energy_maximum_absolute_drift_ratio"] = (
            candidate.energy["maximum_absolute_drift"]
            / max(reference.energy["maximum_absolute_drift"], 1.0e-15)
        )
    else:
        bend_error = _relative_difference(
            candidate.metrics["final_total_bend"],
            reference.metrics["final_total_bend"],
        )
        details["final_bend_relative_error"] = bend_error
        checks.append(bend_error <= tolerance)

        candidate_tip = np.asarray(candidate.metrics["final_tip_relative"])
        reference_tip = np.asarray(reference.metrics["final_tip_relative"])
        tip_error = float(np.linalg.norm(candidate_tip - reference_tip))
        reference_displacement = float(
            np.linalg.norm(
                reference_tip
                - np.asarray(reference.metrics["initial_tip_relative"])
            )
        )
        tip_limit = max(
            1.0e-5 if tolerance == 0.01 else 2.0e-5,
            tolerance * reference_displacement,
        )
        details["tip_error"] = tip_error
        details["tip_limit"] = tip_limit
        checks.append(tip_error <= tip_limit)

        candidate_joints = np.asarray(
            candidate.metrics["final_joint_main_bends"]
        )
        reference_joints = np.asarray(
            reference.metrics["final_joint_main_bends"]
        )
        joint_match = bool(
            np.allclose(
                candidate_joints,
                reference_joints,
                rtol=tolerance,
                atol=1.0e-5,
            )
        )
        details["joint_distribution_match"] = joint_match
        checks.append(joint_match)

        candidate_settling = candidate.analysis.get("settling_time")
        reference_settling = reference.analysis.get("settling_time")
        if candidate_settling is not None and reference_settling is not None:
            settling_error = abs(candidate_settling - reference_settling)
            settling_limit = 0.02 if tolerance == 0.01 else 0.04
            details["settling_error"] = settling_error
            details["settling_limit"] = settling_limit
            checks.append(settling_error <= settling_limit)

        energy_error = abs(
            candidate.energy["final_drift"]
            - reference.energy["final_drift"]
        )
        energy_limit = tolerance * max(
            abs(reference.energy["final_drift"]),
            reference.energy_scale.tolerance,
        )
        details["final_energy_error"] = energy_error
        details["final_energy_limit"] = energy_limit
        checks.append(energy_error <= energy_limit)

        candidate_tendon_min = np.asarray(
            candidate.metrics["tendon_minimum"]
        )
        reference_tendon_min = np.asarray(
            reference.metrics["tendon_minimum"]
        )
        candidate_tendon_max = np.asarray(
            candidate.metrics["tendon_maximum"]
        )
        reference_tendon_max = np.asarray(
            reference.metrics["tendon_maximum"]
        )
        reference_initial = np.asarray(reference.metrics["tendon_initial"])
        reference_excursion = max(
            float(np.max(np.abs(reference_tendon_min - reference_initial))),
            float(np.max(np.abs(reference_tendon_max - reference_initial))),
        )
        tendon_limit = max(
            1.0e-8 if tolerance == 0.01 else 2.0e-8,
            tolerance * reference_excursion,
        )
        tendon_error = max(
            float(np.max(np.abs(candidate_tendon_min - reference_tendon_min))),
            float(np.max(np.abs(candidate_tendon_max - reference_tendon_max))),
        )
        details["tendon_extrema_error"] = tendon_error
        details["tendon_extrema_limit"] = tendon_limit
        checks.append(tendon_error <= tendon_limit)

    if not checks:
        return INSUFFICIENT_REFERENCE, details
    return (CONVERGED if all(checks) else NOT_CONVERGED), details


def approved_matrix_configs() -> list[StabilityRunConfig]:
    configs = [
        StabilityRunConfig("H1", "RK4", frequency, "x")
        for frequency in (240, 2400, 4800, 9600)
    ]
    configs.append(StabilityRunConfig("H1", "implicitfast", 9600, "x"))
    configs.extend(
        StabilityRunConfig("H2", integrator, frequency, "x")
        for integrator in ("RK4", "implicitfast")
        for frequency in (240, 2400, 4800, 9600)
    )
    configs.extend(
        StabilityRunConfig("H2", "implicitfast", frequency, "y")
        for frequency in (2400, 4800, 9600)
    )
    return configs


def _config_key(config: StabilityRunConfig) -> tuple:
    return (config.case, config.integrator, config.sim_freq, config.plane)


def apply_matrix_convergence(results: list[dict]) -> None:
    by_key = {
        (
            item["config"]["case"],
            item["config"]["integrator"],
            item["config"]["sim_freq"],
            item["config"]["plane"],
        ): item
        for item in results
    }
    for item in results:
        config = item["config"]
        if item["run_status"] != RUN_COMPLETED:
            item["convergence_status"] = CONVERGENCE_UNAVAILABLE
            continue
        if config["sim_freq"] == 9600:
            item["convergence_status"] = INSUFFICIENT_REFERENCE
            continue
        if config["sim_freq"] not in (2400, 4800):
            item["convergence_status"] = CONVERGENCE_UNAVAILABLE
            continue
        reference_dict = by_key.get(
            (
                config["case"],
                config["integrator"],
                9600,
                config["plane"],
            )
        )
        if reference_dict is None:
            item["convergence_status"] = INSUFFICIENT_REFERENCE
            continue
        candidate = stability_result_from_dict(item)
        reference = stability_result_from_dict(reference_dict)
        status, details = compare_to_reference(candidate, reference)
        item["convergence_status"] = status
        item["convergence_details"] = details


def stability_result_from_dict(value: dict) -> StabilityResult:
    return StabilityResult(
        config=StabilityRunConfig(**value["config"]),
        metadata=NumericalEnvironmentMetadata(**value["metadata"]),
        energy_scale=EnergyScale(**value["energy_scale"]),
        run_status=value["run_status"],
        energy_status=value["energy_status"],
        convergence_status=value["convergence_status"],
        analysis_status=value["analysis_status"],
        requested_steps=value["requested_steps"],
        completed_steps=value["completed_steps"],
        final_time=value["final_time"],
        persistence_steps=value["persistence_steps"],
        persistence_duration=value["persistence_duration"],
        first_failure=value.get("first_failure"),
        warning_events=value.get("warning_events", []),
        metrics=value.get("metrics", {}),
        energy=value.get("energy", {}),
        analysis=value.get("analysis", {}),
        persistence=value.get("persistence", {}),
        elapsed_seconds=value.get("elapsed_seconds", 0.0),
        convergence_details=value.get("convergence_details", {}),
    )


def _child_command(config: StabilityRunConfig) -> list[str]:
    return [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--case",
        config.case,
        "--integrator",
        config.integrator,
        "--sim-freq",
        str(config.sim_freq),
        "--plane",
        config.plane,
        "--duration",
        str(config.physical_duration),
    ]


def _run_child(config: StabilityRunConfig, timeout: float = 90.0) -> dict:
    environment = os.environ.copy()
    environment[MATRIX_CHILD_ENV] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        process = subprocess.run(
            _child_command(config),
            cwd="/tmp",
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(config),
            "run_status": RUN_TIMEOUT,
            "energy_status": ENERGY_UNAVAILABLE,
            "analysis_status": ANALYSIS_UNAVAILABLE,
            "convergence_status": CONVERGENCE_UNAVAILABLE,
            "infrastructure_error": {
                "timeout_seconds": timeout,
                "stderr": exc.stderr,
            },
        }

    payload_lines = [
        line[len(ENERGY_JSON_PREFIX):]
        for line in process.stdout.splitlines()
        if line.startswith(ENERGY_JSON_PREFIX)
    ]
    if process.returncode != 0 or len(payload_lines) != 1:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(config),
            "run_status": RUN_CRASHED,
            "energy_status": ENERGY_UNAVAILABLE,
            "analysis_status": ANALYSIS_UNAVAILABLE,
            "convergence_status": CONVERGENCE_UNAVAILABLE,
            "infrastructure_error": {
                "returncode": process.returncode,
                "stdout": process.stdout[-4000:],
                "stderr": process.stderr[-4000:],
            },
        }
    try:
        payload = json.loads(payload_lines[0])
    except json.JSONDecodeError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(config),
            "run_status": RUN_CRASHED,
            "energy_status": ENERGY_UNAVAILABLE,
            "analysis_status": ANALYSIS_UNAVAILABLE,
            "convergence_status": CONVERGENCE_UNAVAILABLE,
            "infrastructure_error": {
                "returncode": process.returncode,
                "protocol_error": repr(exc),
                "stdout": process.stdout[-4000:],
                "stderr": process.stderr[-4000:],
            },
        }
    if payload.get("schema_version") != SCHEMA_VERSION:
        payload["run_status"] = RUN_CRASHED
        payload["infrastructure_error"] = {"protocol_error": "schema mismatch"}
    return payload


def _render_matrix_table(results: list[dict]) -> None:
    header = (
        "case int          Hz plane run_status          energy_status                 "
        "conv                  steps       fail_time  wall_s"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        config = item["config"]
        failure = item.get("first_failure") or {}
        fail_time = failure.get("time")
        fail_text = "-" if fail_time is None else f"{fail_time:.6g}"
        completed = item.get("completed_steps", 0)
        requested = item.get("requested_steps", 0)
        print(
            f"{config['case']:<4} {config['integrator']:<12} "
            f"{config['sim_freq']:>5} {config['plane']:^5} "
            f"{item['run_status']:<19} {item['energy_status']:<29} "
            f"{item['convergence_status']:<21} "
            f"{completed:>6}/{requested:<6} {fail_text:>10} "
            f"{item.get('elapsed_seconds', 0.0):>7.3f}"
        )


def _matrix_acceptance_failures(results: list[dict]) -> list[str]:
    failures = []
    for item in results:
        if item["run_status"] in (RUN_CRASHED, RUN_TIMEOUT):
            failures.append(f"infrastructure:{item['config']}")
    required_stable = {
        ("H1", "RK4", frequency, "x"): ENERGY_CONSERVATIVE
        for frequency in (2400, 4800, 9600)
    }
    required_stable.update(
        {
            ("H2", "implicitfast", frequency, plane): ENERGY_DISSIPATIVE
            for plane in ("x", "y")
            for frequency in (2400, 4800, 9600)
        }
    )
    by_key = {
        (
            item["config"]["case"],
            item["config"]["integrator"],
            item["config"]["sim_freq"],
            item["config"]["plane"],
        ): item
        for item in results
    }
    for key, expected_energy in required_stable.items():
        item = by_key.get(key)
        if item is None:
            failures.append(f"missing_stable_candidate:{key}")
            continue
        if item["run_status"] != RUN_COMPLETED:
            failures.append(f"stable_candidate_run:{key}:{item['run_status']}")
        if item["energy_status"] != expected_energy:
            failures.append(
                f"stable_candidate_energy:{key}:{item['energy_status']}"
            )
        if key[2] in (2400, 4800) and item["convergence_status"] != CONVERGED:
            failures.append(
                f"stable_candidate_convergence:{key}:"
                f"{item['convergence_status']}"
            )
    return failures


def run_matrix(timeout: float = 90.0) -> Tuple[list[dict], list[str]]:
    results = []
    for config in approved_matrix_configs():
        result = _run_child(config, timeout=timeout)
        results.append(result)
    apply_matrix_convergence(results)
    for result in results:
        print("B3_MATRIX_RESULT=" + json.dumps(result, separators=(",", ":")))
    _render_matrix_table(results)
    failures = _matrix_acceptance_failures(results)
    if failures:
        print("B3_MATRIX_ACCEPTANCE_FAILURES=" + json.dumps(failures))
    else:
        print(
            "B3_RECOMMENDATION=fixed-root B3 supports 2400 Hz + "
            "implicitfast as the preferred candidate for the next free-base/"
            "full UAV-arm coupled validation stage"
        )
    return results, failures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--matrix", action="store_true")
    mode.add_argument("--single", action="store_true")
    parser.add_argument("--case", choices=("H1", "H2"))
    parser.add_argument("--integrator", choices=("RK4", "implicitfast"))
    parser.add_argument("--sim-freq", type=int)
    parser.add_argument("--plane", choices=("x", "y"), default="x")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.matrix:
        if os.environ.get(MATRIX_CHILD_ENV) == "1":
            print("matrix recursion is forbidden", file=sys.stderr)
            return 2
        _, failures = run_matrix(timeout=args.timeout)
        return 1 if failures else 0

    missing = [
        name
        for name, value in (
            ("--case", args.case),
            ("--integrator", args.integrator),
            ("--sim-freq", args.sim_freq),
        )
        if value is None
    ]
    if missing:
        print(f"single mode requires: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        result = run_stability_case(
            StabilityRunConfig(
                case=args.case,
                integrator=args.integrator,
                sim_freq=args.sim_freq,
                plane=args.plane,
                physical_duration=args.duration,
            )
        )
    except Exception as exc:
        print(f"B3 infrastructure failure: {exc!r}", file=sys.stderr)
        return 2
    payload = result.to_dict()
    payload["schema_version"] = SCHEMA_VERSION
    print(ENERGY_JSON_PREFIX + json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
