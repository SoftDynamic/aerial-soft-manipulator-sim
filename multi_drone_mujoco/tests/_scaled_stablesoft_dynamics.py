"""Test-only dynamics helpers for the scaled StableSoft arm."""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

import mujoco
import numpy as np

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ArmMjcfManifest,
    ScaledStableSoftArmSpec,
    build_scaled_stablesoft_arm_mjcf,
)


MOUNT_Z = -0.015
_SQRT_HALF = math.sqrt(0.5)
_FIXTURE_QUATERNIONS = {
    # The arm's nominal -Z centerline is rotated into the horizontal X/Y axis.
    "x": (_SQRT_HALF, 0.0, _SQRT_HALF, 0.0),
    "y": (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0),
    # B2 releases the nominal arm vertically downward under gravity.
    "vertical": (1.0, 0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class FixedArmFixture:
    model: mujoco.MjModel
    data: mujoco.MjData
    manifest: ArmMjcfManifest
    sample_stride: int


@dataclass(frozen=True)
class StaticCharacterization:
    plane: str
    timestep: float
    integrator: int
    model_counts: Tuple[int, ...]
    initial_tip_position: np.ndarray
    final_tip_position: np.ndarray
    tip_gravity_displacement: float
    total_bend: float
    total_twist: float
    joint_rotation_vectors: np.ndarray
    joint_main_bends: np.ndarray
    tip_window_drift: float
    max_arm_qvel: float
    joint_angle_window_change: float
    max_contacts: int
    warning_count: int
    all_finite: bool
    initial_energy: float
    maximum_energy: float
    final_energy: float
    root_to_tip_length: float
    tendon_lengths: np.ndarray


@dataclass(frozen=True)
class WarningEvent:
    stage: str
    step_index: int
    time: float
    counts: Tuple[int, ...]
    raised_indices: Tuple[int, ...]


@dataclass
class WarningLatch:
    """Keep warning evidence even if a later MuJoCo reset clears counters."""

    previous_counts: Tuple[int, ...] = ()
    events: list = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return bool(self.events)

    def snapshot_counts(
        self,
        counts,
        *,
        stage: str,
        step_index: int,
        time: float,
    ) -> None:
        normalized = tuple(int(value) for value in counts)
        previous = self.previous_counts or (0,) * len(normalized)
        raised = tuple(
            index
            for index, (current, prior) in enumerate(zip(normalized, previous))
            if current > prior
        )
        if raised:
            self.events.append(
                WarningEvent(
                    stage=stage,
                    step_index=step_index,
                    time=float(time),
                    counts=normalized,
                    raised_indices=raised,
                )
            )
        self.previous_counts = normalized

    def snapshot(self, data, *, stage: str, step_index: int) -> None:
        self.snapshot_counts(
            (warning.number for warning in data.warning),
            stage=stage,
            step_index=step_index,
            time=float(data.time),
        )


@dataclass(frozen=True)
class RawStateFailure:
    step_index: int
    time: float
    time_finite: bool
    qpos_finite: bool
    qvel_finite: bool


@dataclass(frozen=True)
class SynchronizedStateFailure:
    step_index: int
    time: float
    qpos_finite: bool
    qvel_finite: bool
    qacc_finite: bool
    site_finite: bool
    tendon_finite: bool
    energy_finite: bool


@dataclass(frozen=True)
class DecayTrace:
    bend_axis: str
    timestep: float
    integrator: int
    sample_stride: int
    requested_steps: int
    completed_steps: int
    physics_time: np.ndarray
    position_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    composite_rotation_vectors: np.ndarray
    joint_rotation_vectors: np.ndarray
    tip_relative_positions: np.ndarray
    arm_qvel: np.ndarray
    tendon_lengths: np.ndarray
    max_abs_qpos: np.ndarray
    max_abs_qvel: np.ndarray
    max_abs_qacc: np.ndarray
    max_abs_site_xpos: np.ndarray
    max_abs_tendon_length: np.ndarray
    contacts: np.ndarray
    tendon_names: Tuple[str, ...]
    tendon_ranges: np.ndarray
    warning_events: Tuple[WarningEvent, ...]
    raw_state_failure: Optional[RawStateFailure]
    synchronized_state_failure: Optional[SynchronizedStateFailure]

    @property
    def control_indices(self) -> np.ndarray:
        return np.arange(0, len(self.physics_time), self.sample_stride)

    @property
    def control_time(self) -> np.ndarray:
        return self.physics_time[self.control_indices]

    @property
    def control_composite_rotation_vectors(self) -> np.ndarray:
        return self.composite_rotation_vectors[self.control_indices]

    @property
    def control_joint_rotation_vectors(self) -> np.ndarray:
        return self.joint_rotation_vectors[self.control_indices]

    @property
    def control_tip_relative_positions(self) -> np.ndarray:
        return self.tip_relative_positions[self.control_indices]

    @property
    def control_arm_qvel(self) -> np.ndarray:
        return self.arm_qvel[self.control_indices]

    @property
    def control_tendon_lengths(self) -> np.ndarray:
        return self.tendon_lengths[self.control_indices]

    @property
    def target_axis_index(self) -> int:
        return 0 if self.bend_axis == "x" else 1

    @property
    def non_target_axis_index(self) -> int:
        return 1 if self.bend_axis == "x" else 0

    @property
    def control_signed_bend(self) -> np.ndarray:
        return self.control_composite_rotation_vectors[:, self.target_axis_index]

    @property
    def control_non_target_bend(self) -> np.ndarray:
        return self.control_composite_rotation_vectors[:, self.non_target_axis_index]

    @property
    def control_twist(self) -> np.ndarray:
        return self.control_composite_rotation_vectors[:, 2]

    @property
    def control_max_arm_qvel(self) -> np.ndarray:
        return np.max(np.abs(self.control_arm_qvel), axis=1)

    @property
    def all_synchronized_values_finite(self) -> bool:
        arrays = (
            self.physics_time,
            self.position_energy,
            self.kinetic_energy,
            self.total_energy,
            self.composite_rotation_vectors,
            self.joint_rotation_vectors,
            self.tip_relative_positions,
            self.arm_qvel,
            self.tendon_lengths,
            self.max_abs_qpos,
            self.max_abs_qvel,
            self.max_abs_qacc,
            self.max_abs_site_xpos,
            self.max_abs_tendon_length,
        )
        return all(np.all(np.isfinite(values)) for values in arrays)


@dataclass(frozen=True)
class DecayAnalysis:
    reliable_threshold: float
    peak_indices: np.ndarray
    peak_times: np.ndarray
    peak_amplitudes: np.ndarray
    zero_crossing_times: np.ndarray
    zero_crossing_directions: np.ndarray
    peak_frequency: Optional[float]
    zero_crossing_frequency: Optional[float]
    frequency: Optional[float]
    frequency_source: Optional[str]
    frequency_consistent: Optional[bool]
    log_decrements: np.ndarray
    median_log_decrement: Optional[float]
    decay_rate: Optional[float]
    apparent_damping_ratio: Optional[float]
    analysis_path: str
    envelope_valid: bool


@dataclass(frozen=True)
class EnergyAnalysis:
    tolerance: float
    maximum_after_release_delta: float
    final_delta: float
    longest_above_tolerance_run: int
    longest_positive_increment_run: int
    maximum_positive_run_increase: float
    hard_limit_failed: bool
    persistent_upper_bound_failed: bool
    sustained_growth_failed: bool
    final_bound_failed: bool

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.hard_limit_failed,
                self.persistent_upper_bound_failed,
                self.sustained_growth_failed,
                self.final_bound_failed,
            )
        )


@dataclass(frozen=True)
class DecayCharacterization:
    trace: DecayTrace
    decay: Optional[DecayAnalysis]
    settling_time: Optional[float]
    initial_spring_energy: float
    energy: Optional[EnergyAnalysis]


def _quaternion_text(quaternion) -> str:
    return " ".join(format(float(value), ".17g") for value in quaternion)


def build_fixed_arm_fixture(
    spec: ScaledStableSoftArmSpec,
    plane: str,
    *,
    sim_freq: int,
    ctrl_freq: int,
    integrator: str,
) -> FixedArmFixture:
    """Compile the production arm fragment under a welded test fixture."""
    if plane not in _FIXTURE_QUATERNIONS:
        raise ValueError("plane must be 'x', 'y', or 'vertical'")
    if sim_freq <= 0 or ctrl_freq <= 0 or sim_freq % ctrl_freq != 0:
        raise ValueError("sim_freq must be positive and divisible by ctrl_freq")
    if integrator not in ("RK4", "implicitfast"):
        raise ValueError("integrator must be RK4 or implicitfast")

    fragments = build_scaled_stablesoft_arm_mjcf("fixture", spec, MOUNT_Z)
    fixture_quaternion = _quaternion_text(_FIXTURE_QUATERNIONS[plane])
    xml = f'''<mujoco model="scaled_stablesoft_fixed_{plane}">
  <option gravity="0 0 -9.81" timestep="{1.0 / sim_freq}" integrator="{integrator}"/>
  <compiler inertiafromgeom="false" autolimits="true"/>
  <worldbody>
    <body name="fixture_mount" quat="{fixture_quaternion}">
{fragments.worldbody_xml}    </body>
  </worldbody>
  <tendon>
{fragments.tendon_xml}
  </tendon>
  <sensor>
{fragments.sensor_xml}
  </sensor>
</mujoco>
'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_forward(model, data)
    return FixedArmFixture(
        model=model,
        data=data,
        manifest=fragments.manifest,
        sample_stride=sim_freq // ctrl_freq,
    )


def named_joint_addresses(model, manifest):
    """Return joint IDs and qpos/qvel addresses without global index literals."""
    result = []
    for name in manifest.ball_joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise AssertionError(name)
        result.append(
            (
                joint_id,
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
        )
    return tuple(result)


def full_mass_matrix(model, data):
    """Expand MuJoCo's packed mass matrix across supported Python APIs."""
    result = np.zeros((model.nv, model.nv), dtype=float)
    try:
        mujoco.mj_fullM(model, data, result)
    except TypeError:
        packed = data.qM if hasattr(data, "qM") else data.M
        mujoco.mj_fullM(model, result, packed)
    return result


def _rotation_vector(quaternion):
    result = np.zeros(3, dtype=float)
    quaternion = np.asarray(quaternion, dtype=float).copy()
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    mujoco.mju_quat2Vel(result, quaternion, 1.0)
    return result


def joint_rotation_vectors(model, data, addresses):
    return np.asarray(
        [
            _rotation_vector(data.qpos[qpos_address:qpos_address + 4])
            for _, qpos_address, _ in addresses
        ]
    )


def composite_rotation_vector(model, data, addresses):
    total = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    product = np.empty(4, dtype=float)
    for _, qpos_address, _ in addresses:
        mujoco.mju_mulQuat(
            product,
            total,
            data.qpos[qpos_address:qpos_address + 4],
        )
        total = product.copy()
    return _rotation_vector(total)


def _warning_count(data):
    return sum(int(warning.number) for warning in data.warning)


def _sample_energy(model, data):
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    return float(np.sum(data.energy))


def run_static_characterization(
    plane: str,
    *,
    physical_duration: float = 5.0,
    steady_window: float = 0.5,
    sim_freq: int = 4800,
    ctrl_freq: int = 240,
    integrator: str = "implicitfast",
) -> StaticCharacterization:
    """Release a horizontal fixed-root arm and record its static response."""
    spec = ScaledStableSoftArmSpec(
        joint_stiffness=0.003086,
        joint_damping=None,
        target_damping_ratio=0.3,
    )
    fixture = build_fixed_arm_fixture(
        spec,
        plane,
        sim_freq=sim_freq,
        ctrl_freq=ctrl_freq,
        integrator=integrator,
    )
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    addresses = named_joint_addresses(model, manifest)
    arm_dof_addresses = np.concatenate(
        [np.arange(dof_address, dof_address + 3) for _, _, dof_address in addresses]
    )
    tip_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name
    )
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_root_site_name
    )
    initial_tip = np.asarray(data.site_xpos[tip_id], dtype=float).copy()

    tip_samples = [initial_tip]
    rotation_samples = [joint_rotation_vectors(model, data, addresses)]
    qvel_samples = [np.asarray(data.qvel[arm_dof_addresses]).copy()]
    energy_samples = [_sample_energy(model, data)]
    max_contacts = int(data.ncon)
    all_finite = True
    total_steps = int(round(physical_duration * sim_freq))

    for step_index in range(total_steps):
        mujoco.mj_step(model, data)
        all_finite = all_finite and all(
            np.all(np.isfinite(values))
            for values in (
                data.qpos,
                data.qvel,
                data.qacc,
                data.site_xpos,
                data.ten_length,
                data.energy,
            )
        )
        max_contacts = max(max_contacts, int(data.ncon))
        if _warning_count(data) or not all_finite:
            break
        if (step_index + 1) % fixture.sample_stride == 0:
            mujoco.mj_forward(model, data)
            tip_samples.append(np.asarray(data.site_xpos[tip_id]).copy())
            rotation_samples.append(
                joint_rotation_vectors(model, data, addresses)
            )
            qvel_samples.append(np.asarray(data.qvel[arm_dof_addresses]).copy())
            energy_samples.append(_sample_energy(model, data))
            all_finite = all_finite and all(
                np.all(np.isfinite(values))
                for values in (
                    data.site_xpos,
                    data.ten_length,
                    data.energy,
                )
            )

    mujoco.mj_forward(model, data)
    tips = np.asarray(tip_samples)
    rotations = np.asarray(rotation_samples)
    qvels = np.asarray(qvel_samples)
    energies = np.asarray(energy_samples)
    window_count = int(round(steady_window * ctrl_freq)) + 1
    window_tips = tips[-window_count:]
    window_rotations = rotations[-window_count:]
    window_qvels = qvels[-window_count:]
    final_rotations = joint_rotation_vectors(model, data, addresses)
    total_rotation = composite_rotation_vector(model, data, addresses)
    main_axis = 1 if plane == "x" else 0

    return StaticCharacterization(
        plane=plane,
        timestep=float(model.opt.timestep),
        integrator=int(model.opt.integrator),
        model_counts=(
            model.nbody,
            model.njnt,
            model.nq,
            model.nv,
            model.ngeom,
            model.nsite,
            model.ntendon,
            model.nu,
            model.nsensor,
        ),
        initial_tip_position=initial_tip,
        final_tip_position=np.asarray(data.site_xpos[tip_id]).copy(),
        tip_gravity_displacement=float(data.site_xpos[tip_id, 2] - initial_tip[2]),
        total_bend=float(np.linalg.norm(total_rotation[:2])),
        total_twist=float(abs(total_rotation[2])),
        joint_rotation_vectors=final_rotations,
        joint_main_bends=final_rotations[:, main_axis].copy(),
        tip_window_drift=float(
            np.max(np.linalg.norm(window_tips - window_tips[-1], axis=1))
        ),
        max_arm_qvel=float(np.max(np.abs(window_qvels))),
        joint_angle_window_change=float(
            np.max(
                np.linalg.norm(
                    window_rotations - window_rotations[-1], axis=2
                )
            )
        ),
        max_contacts=max_contacts,
        warning_count=_warning_count(data),
        all_finite=bool(all_finite),
        initial_energy=float(energies[0]),
        maximum_energy=float(np.max(energies)),
        final_energy=float(energies[-1]),
        root_to_tip_length=float(
            np.linalg.norm(data.site_xpos[tip_id] - data.site_xpos[root_id])
        ),
        tendon_lengths=np.asarray(data.ten_length).copy(),
    )


def initialize_constant_curvature(
    fixture: FixedArmFixture,
    bend_axis: str,
    *,
    total_bend: float = 0.2,
):
    """Set equal local ball-joint bends without changing model.qpos0."""
    if bend_axis not in ("x", "y"):
        raise ValueError("bend_axis must be 'x' or 'y'")
    if not math.isfinite(total_bend) or total_bend <= 0:
        raise ValueError("total_bend must be finite and positive")

    addresses = named_joint_addresses(fixture.model, fixture.manifest)
    angle = total_bend / len(addresses)
    axis_index = 0 if bend_axis == "x" else 1
    axis = np.zeros(3, dtype=float)
    axis[axis_index] = 1.0
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_axisAngle2Quat(quaternion, axis, angle)
    for _, qpos_address, dof_address in addresses:
        fixture.data.qpos[qpos_address:qpos_address + 4] = quaternion
        fixture.data.qvel[dof_address:dof_address + 3] = 0.0

    mujoco.mj_normalizeQuat(fixture.model, fixture.data.qpos)
    mujoco.mj_forward(fixture.model, fixture.data)
    return addresses


def _named_object_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise AssertionError(name)
    return int(object_id)


def _consecutive_true_run(values) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def analyze_energy_trace(
    total_energy,
    initial_spring_energy: float,
    *,
    consecutive_samples: int = 20,
) -> EnergyAnalysis:
    """Classify isolated energy noise separately from sustained injection."""
    energy = np.asarray(total_energy, dtype=float)
    if energy.ndim != 1 or len(energy) < 2:
        raise ValueError("total_energy must contain at least two samples")
    if not np.all(np.isfinite(energy)):
        raise ValueError("total_energy must be finite")
    if not math.isfinite(initial_spring_energy) or initial_spring_energy < 0:
        raise ValueError("initial_spring_energy must be finite and non-negative")
    if consecutive_samples <= 0:
        raise ValueError("consecutive_samples must be positive")

    tolerance = max(1e-9, 0.01 * initial_spring_energy)
    relative = energy - energy[0]
    above_tolerance = relative > tolerance
    longest_above = _consecutive_true_run(above_tolerance)

    increments = np.diff(energy)
    longest_positive = 0
    maximum_positive_increase = 0.0
    current_start = 0
    current_length = 0
    sustained_growth = False
    for increment_index, increment in enumerate(increments):
        if increment > 0:
            if current_length == 0:
                current_start = increment_index
            current_length += 1
            increase = energy[increment_index + 1] - energy[current_start]
            longest_positive = max(longest_positive, current_length)
            maximum_positive_increase = max(
                maximum_positive_increase,
                float(increase),
            )
            if current_length >= consecutive_samples and increase > tolerance:
                sustained_growth = True
        else:
            current_length = 0

    return EnergyAnalysis(
        tolerance=float(tolerance),
        maximum_after_release_delta=float(np.max(relative[1:])),
        final_delta=float(relative[-1]),
        longest_above_tolerance_run=longest_above,
        longest_positive_increment_run=longest_positive,
        maximum_positive_run_increase=maximum_positive_increase,
        hard_limit_failed=bool(np.any(relative[1:] > 5 * tolerance)),
        persistent_upper_bound_failed=longest_above >= consecutive_samples,
        sustained_growth_failed=sustained_growth,
        final_bound_failed=bool(relative[-1] > tolerance),
    )


def _same_direction_zero_frequency(times, directions):
    periods = []
    for direction in (-1, 1):
        selected = times[directions == direction]
        if len(selected) >= 2:
            periods.extend(np.diff(selected))
    if not periods:
        return None
    return float(1.0 / np.median(periods))


def analyze_decay_signal(
    time,
    signed_bend,
    *,
    initial_amplitude: float = 0.2,
    minimum_peak_distance: float = 0.05,
) -> DecayAnalysis:
    """Analyze a signed decay trace using NumPy-only peak statistics."""
    sample_time = np.asarray(time, dtype=float)
    signal = np.asarray(signed_bend, dtype=float)
    if sample_time.ndim != 1 or signal.ndim != 1:
        raise ValueError("time and signed_bend must be one-dimensional")
    if len(sample_time) != len(signal) or len(signal) < 3:
        raise ValueError("time and signed_bend must have equal usable lengths")
    if not np.all(np.isfinite(sample_time)) or not np.all(np.isfinite(signal)):
        raise ValueError("time and signed_bend must be finite")
    if not np.all(np.diff(sample_time) > 0):
        raise ValueError("time must be strictly increasing")
    if not math.isfinite(initial_amplitude) or initial_amplitude <= 0:
        raise ValueError("initial_amplitude must be finite and positive")
    if not math.isfinite(minimum_peak_distance) or minimum_peak_distance <= 0:
        raise ValueError("minimum_peak_distance must be finite and positive")

    signal_sign = 1.0 if signal[0] >= 0 else -1.0
    aligned = signal_sign * signal
    reliable_threshold = max(1e-4, 0.005 * initial_amplitude)
    peak_indices = []
    if aligned[0] >= reliable_threshold and aligned[0] >= aligned[1]:
        peak_indices.append(0)
    last_peak_time = sample_time[0] if peak_indices else -math.inf
    for index in range(1, len(aligned) - 1):
        if (
            aligned[index] >= reliable_threshold
            and aligned[index] >= aligned[index - 1]
            and aligned[index] > aligned[index + 1]
            and sample_time[index] - last_peak_time >= minimum_peak_distance
        ):
            peak_indices.append(index)
            last_peak_time = sample_time[index]

    peak_indices_array = np.asarray(peak_indices, dtype=int)
    peak_times = sample_time[peak_indices_array]
    peak_amplitudes = aligned[peak_indices_array]
    peak_frequency = None
    if len(peak_times) >= 2:
        peak_frequency = float(1.0 / np.median(np.diff(peak_times)))

    crossing_times = []
    crossing_directions = []
    nonzero_indices = np.flatnonzero(signal != 0.0)
    for left_index, right_index in zip(
        nonzero_indices[:-1],
        nonzero_indices[1:],
    ):
        left = signal[left_index]
        right = signal[right_index]
        if left * right >= 0.0:
            continue
        if right_index == left_index + 1:
            fraction = -left / (right - left)
            crossing_time = sample_time[left_index] + fraction * (
                sample_time[right_index] - sample_time[left_index]
            )
        else:
            zero_start_time = sample_time[left_index + 1]
            zero_end_time = sample_time[right_index - 1]
            crossing_time = 0.5 * (zero_start_time + zero_end_time)
        crossing_times.append(float(crossing_time))
        crossing_directions.append(1 if right > left else -1)
    crossing_times_array = np.asarray(crossing_times, dtype=float)
    crossing_directions_array = np.asarray(crossing_directions, dtype=int)
    zero_frequency = _same_direction_zero_frequency(
        crossing_times_array,
        crossing_directions_array,
    )

    reliable_peak_path = len(peak_amplitudes) >= 3
    log_decrements = np.empty(0, dtype=float)
    median_decrement = None
    decay_rate = None
    apparent_damping_ratio = None
    envelope_valid = False
    if reliable_peak_path:
        log_decrements = np.log(peak_amplitudes[:-1] / peak_amplitudes[1:])
        median_decrement = float(np.median(log_decrements))
        design = np.column_stack((peak_times, np.ones(len(peak_times))))
        slope, _ = np.linalg.lstsq(
            design,
            np.log(peak_amplitudes),
            rcond=None,
        )[0]
        decay_rate = float(-slope)
        apparent_damping_ratio = float(
            median_decrement
            / math.sqrt((2 * math.pi) ** 2 + median_decrement ** 2)
        )
        noise_tolerance = reliable_threshold
        envelope_valid = bool(
            peak_amplitudes[-1] < peak_amplitudes[0] - noise_tolerance
            and np.all(
                peak_amplitudes[1:]
                <= peak_amplitudes[:-1] + noise_tolerance
            )
            and median_decrement > 0
            and decay_rate > 0
        )

    if reliable_peak_path:
        frequency = peak_frequency
        frequency_source = "peaks"
        analysis_path = "peaks"
    else:
        frequency = zero_frequency
        frequency_source = "zero_crossings" if zero_frequency is not None else None
        analysis_path = "settling_fallback"
    frequency_consistent = None
    if peak_frequency is not None and zero_frequency is not None:
        frequency_consistent = bool(
            abs(peak_frequency - zero_frequency)
            / max(peak_frequency, zero_frequency)
            <= 0.02
        )

    return DecayAnalysis(
        reliable_threshold=reliable_threshold,
        peak_indices=peak_indices_array,
        peak_times=peak_times,
        peak_amplitudes=peak_amplitudes,
        zero_crossing_times=crossing_times_array,
        zero_crossing_directions=crossing_directions_array,
        peak_frequency=peak_frequency,
        zero_crossing_frequency=zero_frequency,
        frequency=frequency,
        frequency_source=frequency_source,
        frequency_consistent=frequency_consistent,
        log_decrements=log_decrements,
        median_log_decrement=median_decrement,
        decay_rate=decay_rate,
        apparent_damping_ratio=apparent_damping_ratio,
        analysis_path=analysis_path,
        envelope_valid=envelope_valid,
    )


def find_settling_time(
    time,
    signed_bend,
    max_arm_qvel,
    *,
    initial_amplitude: float = 0.2,
    minimum_hold: float = 0.25,
):
    """Find the first time after which angle and velocity remain settled."""
    sample_time = np.asarray(time, dtype=float)
    bend = np.asarray(signed_bend, dtype=float)
    velocity = np.asarray(max_arm_qvel, dtype=float)
    if not (len(sample_time) == len(bend) == len(velocity)):
        raise ValueError("settling inputs must have equal lengths")
    if len(sample_time) < 2 or not np.all(np.diff(sample_time) > 0):
        raise ValueError("settling time must be strictly increasing")
    if not all(np.all(np.isfinite(value)) for value in (sample_time, bend, velocity)):
        raise ValueError("settling inputs must be finite")

    angle_band = max(1e-3, 0.02 * initial_amplitude)
    qualified = (np.abs(bend) <= angle_band) & (velocity <= 1e-3)
    for index in range(len(sample_time)):
        if sample_time[-1] - sample_time[index] < minimum_hold:
            break
        if np.all(qualified[index:]):
            return float(sample_time[index])
    return None


def _record_synchronized_state(
    model,
    data,
    addresses,
    tip_id,
    root_id,
    arm_dof_addresses,
    tendon_ids,
):
    joints = joint_rotation_vectors(model, data, addresses)
    composite = composite_rotation_vector(model, data, addresses)
    return {
        "time": float(data.time),
        "position_energy": float(data.energy[0]),
        "kinetic_energy": float(data.energy[1]),
        "total_energy": float(np.sum(data.energy)),
        "composite": composite,
        "joints": joints,
        "tip_relative": np.asarray(
            data.site_xpos[tip_id] - data.site_xpos[root_id]
        ).copy(),
        "arm_qvel": np.asarray(data.qvel[arm_dof_addresses]).copy(),
        "tendon_lengths": np.asarray(data.ten_length[tendon_ids]).copy(),
        "max_abs_qpos": float(np.max(np.abs(data.qpos))),
        "max_abs_qvel": float(np.max(np.abs(data.qvel))),
        "max_abs_qacc": float(np.max(np.abs(data.qacc))),
        "max_abs_site_xpos": float(np.max(np.abs(data.site_xpos))),
        "max_abs_tendon_length": float(np.max(np.abs(data.ten_length))),
        "contacts": int(data.ncon),
    }


def _records_to_decay_trace(
    records,
    *,
    bend_axis,
    model,
    sample_stride,
    requested_steps,
    tendon_names,
    tendon_ranges,
    warning_latch,
    raw_state_failure,
    synchronized_state_failure,
):
    def values(name, dtype=float):
        return np.asarray([record[name] for record in records], dtype=dtype)

    return DecayTrace(
        bend_axis=bend_axis,
        timestep=float(model.opt.timestep),
        integrator=int(model.opt.integrator),
        sample_stride=sample_stride,
        requested_steps=requested_steps,
        completed_steps=len(records) - 1,
        physics_time=values("time"),
        position_energy=values("position_energy"),
        kinetic_energy=values("kinetic_energy"),
        total_energy=values("total_energy"),
        composite_rotation_vectors=values("composite"),
        joint_rotation_vectors=values("joints"),
        tip_relative_positions=values("tip_relative"),
        arm_qvel=values("arm_qvel"),
        tendon_lengths=values("tendon_lengths"),
        max_abs_qpos=values("max_abs_qpos"),
        max_abs_qvel=values("max_abs_qvel"),
        max_abs_qacc=values("max_abs_qacc"),
        max_abs_site_xpos=values("max_abs_site_xpos"),
        max_abs_tendon_length=values("max_abs_tendon_length"),
        contacts=values("contacts", dtype=int),
        tendon_names=tuple(tendon_names),
        tendon_ranges=np.asarray(tendon_ranges, dtype=float),
        warning_events=tuple(warning_latch.events),
        raw_state_failure=raw_state_failure,
        synchronized_state_failure=synchronized_state_failure,
    )


def run_decay_characterization(
    bend_axis: str,
    *,
    total_bend: float = 0.2,
    physical_duration: float = 5.0,
    sim_freq: int = 4800,
    ctrl_freq: int = 240,
    integrator: str = "implicitfast",
) -> DecayCharacterization:
    """Run one synchronized fixed-root vertical free-decay experiment."""
    spec = ScaledStableSoftArmSpec(
        joint_stiffness=0.003086,
        joint_damping=None,
        target_damping_ratio=0.3,
    )
    fixture = build_fixed_arm_fixture(
        spec,
        "vertical",
        sim_freq=sim_freq,
        ctrl_freq=ctrl_freq,
        integrator=integrator,
    )
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    addresses = initialize_constant_curvature(
        fixture,
        bend_axis,
        total_bend=total_bend,
    )
    arm_dof_addresses = np.concatenate(
        [np.arange(dof_address, dof_address + 3) for _, _, dof_address in addresses]
    )
    tip_id = _named_object_id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name
    )
    root_id = _named_object_id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_root_site_name
    )
    tendon_ids = np.asarray(
        [
            _named_object_id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
            for name in manifest.tendon_names
        ],
        dtype=int,
    )
    tendon_ranges = model.tendon_range[tendon_ids].copy()

    initial_joints = joint_rotation_vectors(model, data, addresses)
    initial_spring_energy = float(
        math.fsum(
            0.5
            * float(model.jnt_stiffness[joint_id])
            * float(np.dot(initial_joints[index], initial_joints[index]))
            for index, (joint_id, _, _) in enumerate(addresses)
        )
    )
    warning_latch = WarningLatch()
    warning_latch.snapshot(data, stage="initial_forward", step_index=0)
    records = [
        _record_synchronized_state(
            model,
            data,
            addresses,
            tip_id,
            root_id,
            arm_dof_addresses,
            tendon_ids,
        )
    ]
    raw_state_failure = None
    synchronized_state_failure = None
    total_steps = int(round(physical_duration * sim_freq))

    for step_index in range(total_steps if not warning_latch.triggered else 0):
        completed_step = step_index + 1
        mujoco.mj_step(model, data)
        warning_latch.snapshot(
            data,
            stage="after_step",
            step_index=completed_step,
        )
        time_finite = math.isfinite(float(data.time))
        qpos_finite = bool(np.all(np.isfinite(data.qpos)))
        qvel_finite = bool(np.all(np.isfinite(data.qvel)))
        if not (time_finite and qpos_finite and qvel_finite):
            raw_state_failure = RawStateFailure(
                step_index=completed_step,
                time=float(data.time),
                time_finite=time_finite,
                qpos_finite=qpos_finite,
                qvel_finite=qvel_finite,
            )
            break

        mujoco.mj_forward(model, data)
        warning_latch.snapshot(
            data,
            stage="after_forward",
            step_index=completed_step,
        )
        synchronized_record = _record_synchronized_state(
            model,
            data,
            addresses,
            tip_id,
            root_id,
            arm_dof_addresses,
            tendon_ids,
        )
        records.append(synchronized_record)
        synchronized_flags = {
            "qpos_finite": bool(np.all(np.isfinite(data.qpos))),
            "qvel_finite": bool(np.all(np.isfinite(data.qvel))),
            "qacc_finite": bool(np.all(np.isfinite(data.qacc))),
            "site_finite": bool(np.all(np.isfinite(data.site_xpos))),
            "tendon_finite": bool(np.all(np.isfinite(data.ten_length))),
            "energy_finite": bool(np.all(np.isfinite(data.energy))),
        }
        if not all(synchronized_flags.values()):
            synchronized_state_failure = SynchronizedStateFailure(
                step_index=completed_step,
                time=float(data.time),
                **synchronized_flags,
            )
        if warning_latch.triggered or synchronized_state_failure is not None:
            break

    trace = _records_to_decay_trace(
        records,
        bend_axis=bend_axis,
        model=model,
        sample_stride=fixture.sample_stride,
        requested_steps=total_steps,
        tendon_names=manifest.tendon_names,
        tendon_ranges=tendon_ranges,
        warning_latch=warning_latch,
        raw_state_failure=raw_state_failure,
        synchronized_state_failure=synchronized_state_failure,
    )
    decay = None
    settling_time = None
    run_failed = bool(
        trace.warning_events
        or trace.raw_state_failure is not None
        or trace.synchronized_state_failure is not None
    )
    if not run_failed and len(trace.control_time) >= 3:
        decay = analyze_decay_signal(
            trace.control_time,
            trace.control_signed_bend,
            initial_amplitude=total_bend,
        )
        settling_time = find_settling_time(
            trace.control_time,
            trace.control_signed_bend,
            trace.control_max_arm_qvel,
            initial_amplitude=total_bend,
        )
    energy = None
    if not run_failed and len(trace.total_energy) >= 2:
        energy = analyze_energy_trace(
            trace.total_energy,
            initial_spring_energy,
        )
    return DecayCharacterization(
        trace=trace,
        decay=decay,
        settling_time=settling_time,
        initial_spring_energy=initial_spring_energy,
        energy=energy,
    )
