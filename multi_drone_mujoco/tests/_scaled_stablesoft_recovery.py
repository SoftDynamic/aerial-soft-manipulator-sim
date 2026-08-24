"""Test-only common-mode disturbance-recovery helpers.

The matched-state experiment is a virtual-constraint release: it is not a
physical tip-force unload.  Strict modal metrics are computed from the full
physics-rate recovery trace; the control grid exists only for commands and
phase alignment.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Tuple, Union

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
    _SafetyAccumulator,
    _check_synchronized_state,
    _freeze,
    _warning_counts,
    half_cosine_ramp_scale_at_tick,
    resolve_tendon_actuator_bindings,
    write_tension_command,
)
from multi_drone_mujoco.tests._scaled_stablesoft_compliance import (
    _ExternalAccumulator,
    _active_summary,
    apply_tip_force_and_check,
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
EQUILIBRIUM_WINDOW_START_S = 1.0
MATCHED_RELEASE_TIME_S = 1.5
FORCE_START_TIME_S = 1.5
FORCE_RAMP_DURATION_S = 0.25
LOADED_WINDOW_START_S = 4.0
FORCE_RELEASE_TIME_S = 4.5
RECOVERY_DURATION_S = 3.0
MATCHED_TOTAL_BEND_RAD = 0.10
TIP_FORCE_N = 2e-4
MAX_RECOVERY_TIP_FORCE_N = 5e-4
COMMON_TENSION_LEVELS_N = (0.0, 0.002, 0.005, 0.01)
DIRECTIONS = ("+X", "-X", "+Y", "-Y")

_DIRECTION_VECTORS = {
    "+X": np.asarray((1.0, 0.0, 0.0)),
    "-X": np.asarray((-1.0, 0.0, 0.0)),
    "+Y": np.asarray((0.0, 1.0, 0.0)),
    "-Y": np.asarray((0.0, -1.0, 0.0)),
}
_MATCHED_ROTATION_AXES = {
    "+X": np.asarray((0.0, -1.0, 0.0)),
    "-X": np.asarray((0.0, 1.0, 0.0)),
    "+Y": np.asarray((1.0, 0.0, 0.0)),
    "-Y": np.asarray((-1.0, 0.0, 0.0)),
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
class _RecoveryGrid:
    common_tension_n: float
    direction: str
    sim_freq: int = SIM_FREQ
    ctrl_freq: int = CTRL_FREQ
    tension_ramp_duration_s: float = TENSION_RAMP_DURATION_S
    equilibrium_window_start_s: float = EQUILIBRIUM_WINDOW_START_S
    recovery_duration_s: float = RECOVERY_DURATION_S
    integrator: str = INTEGRATOR
    max_tension_n: float = MAX_TENSION_N

    def _normalize(self):
        for name in ("sim_freq", "ctrl_freq"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sim_freq % self.ctrl_freq:
            raise ValueError("sim_freq must be divisible by ctrl_freq")
        if self.integrator != INTEGRATOR:
            raise ValueError("recovery characterization requires implicitfast")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        for name in (
            "common_tension_n",
            "tension_ramp_duration_s",
            "equilibrium_window_start_s",
            "recovery_duration_s",
            "max_tension_n",
        ):
            object.__setattr__(self, name, _finite_float(name, getattr(self, name)))
        if not 0.0 <= self.common_tension_n <= self.max_tension_n:
            raise ValueError("common_tension_n must lie in [0, max_tension_n]")
        if self.max_tension_n <= 0.0 or self.max_tension_n > MAX_TENSION_N:
            raise ValueError("max_tension_n exceeds the C1 diagnostic limit")
        if self.tension_ramp_duration_s <= 0.0:
            raise ValueError("tension_ramp_duration_s must be positive")
        if self.equilibrium_window_start_s <= self.tension_ramp_duration_s:
            raise ValueError("equilibrium window must follow the tension ramp")
        if self.recovery_duration_s <= 0.0:
            raise ValueError("recovery_duration_s must be positive")
        for name in (
            "tension_ramp_duration_s",
            "equilibrium_window_start_s",
            "recovery_duration_s",
        ):
            _aligned_ticks(name, getattr(self, name), self.ctrl_freq)

    @property
    def sample_stride(self) -> int:
        return self.sim_freq // self.ctrl_freq

    @property
    def tension_ramp_ticks(self) -> int:
        return _aligned_ticks(
            "tension_ramp_duration_s",
            self.tension_ramp_duration_s,
            self.ctrl_freq,
        )

    @property
    def recovery_steps(self) -> int:
        return int(round(self.recovery_duration_s * self.sim_freq))

    @property
    def direction_vector(self):
        return _DIRECTION_VECTORS[self.direction].copy()

    @property
    def cross_direction_vector(self):
        direction = self.direction_vector
        return np.asarray((-direction[1], direction[0], 0.0))


@dataclass(frozen=True)
class MatchedStateDecayRunConfig(_RecoveryGrid):
    """One virtual-constraint matched-state release."""

    total_bend_rad: float = MATCHED_TOTAL_BEND_RAD
    release_time_s: float = MATCHED_RELEASE_TIME_S

    def __post_init__(self):
        self._normalize()
        for name in ("total_bend_rad", "release_time_s"):
            object.__setattr__(self, name, _finite_float(name, getattr(self, name)))
        if not 0.0 < self.total_bend_rad <= 0.20:
            raise ValueError("total_bend_rad must lie in (0, 0.20]")
        if self.release_time_s <= self.equilibrium_window_start_s:
            raise ValueError("release must follow the equilibrium window")
        _aligned_ticks("release_time_s", self.release_time_s, self.ctrl_freq)

    @property
    def release_tick(self) -> int:
        return _aligned_ticks("release_time_s", self.release_time_s, self.ctrl_freq)

    @property
    def total_steps(self) -> int:
        return self.release_tick * self.sample_stride + self.recovery_steps

    @property
    def requested_steps(self) -> int:
        return self.total_steps


@dataclass(frozen=True)
class ForceReleaseRecoveryRunConfig(_RecoveryGrid):
    """One ramped-and-held tip force followed by an instantaneous release."""

    force_magnitude_n: float = TIP_FORCE_N
    force_start_time_s: float = FORCE_START_TIME_S
    force_ramp_duration_s: float = FORCE_RAMP_DURATION_S
    loaded_window_start_s: float = LOADED_WINDOW_START_S
    release_time_s: float = FORCE_RELEASE_TIME_S
    max_tip_force_n: float = MAX_RECOVERY_TIP_FORCE_N

    def __post_init__(self):
        self._normalize()
        for name in (
            "force_magnitude_n",
            "force_start_time_s",
            "force_ramp_duration_s",
            "loaded_window_start_s",
            "release_time_s",
            "max_tip_force_n",
        ):
            object.__setattr__(self, name, _finite_float(name, getattr(self, name)))
        if not 0.0 < self.force_magnitude_n <= self.max_tip_force_n:
            raise ValueError("force_magnitude_n must lie in (0, max_tip_force_n]")
        if not 0.0 < self.max_tip_force_n <= MAX_RECOVERY_TIP_FORCE_N:
            raise ValueError("max_tip_force_n exceeds the A4b diagnostic limit")
        force_end = self.force_start_time_s + self.force_ramp_duration_s
        if not (
            self.equilibrium_window_start_s < self.force_start_time_s
            and self.force_ramp_duration_s > 0.0
            and force_end < self.loaded_window_start_s < self.release_time_s
        ):
            raise ValueError("force and loaded windows are not ordered")
        for name in (
            "force_start_time_s",
            "force_ramp_duration_s",
            "loaded_window_start_s",
            "release_time_s",
        ):
            _aligned_ticks(name, getattr(self, name), self.ctrl_freq)

    @property
    def force_start_tick(self) -> int:
        return _aligned_ticks(
            "force_start_time_s", self.force_start_time_s, self.ctrl_freq
        )

    @property
    def force_ramp_ticks(self) -> int:
        return _aligned_ticks(
            "force_ramp_duration_s", self.force_ramp_duration_s, self.ctrl_freq
        )

    @property
    def release_tick(self) -> int:
        return _aligned_ticks("release_time_s", self.release_time_s, self.ctrl_freq)

    @property
    def total_steps(self) -> int:
        return self.release_tick * self.sample_stride + self.recovery_steps

    @property
    def requested_steps(self) -> int:
        return self.total_steps


RecoveryRunConfig = Union[
    MatchedStateDecayRunConfig,
    ForceReleaseRecoveryRunConfig,
]


@dataclass(frozen=True)
class DampedSinusoidFit:
    offset: float
    cosine_coefficient: float
    sine_coefficient: float
    frequency_hz: float
    decay_rate_s: float
    normalized_rms_residual: float
    frequency_search_bounds_hz: Tuple[float, float]
    alpha_search_bounds_s: Tuple[float, float]
    levels: int
    frequency_grid_size: int
    alpha_grid_size: int
    final_frequency_resolution_hz: float


@dataclass(frozen=True)
class RecoveryAnalysis:
    signal_floor: float
    positive_peak_times_s: np.ndarray
    positive_peak_amplitudes: np.ndarray
    negative_peak_times_s: np.ndarray
    negative_peak_amplitudes: np.ndarray
    zero_crossing_times_s: np.ndarray
    zero_crossing_directions: np.ndarray
    peak_frequency_hz: Optional[float]
    post_transient_peak_frequency_hz: Optional[float]
    zero_crossing_frequency_hz: Optional[float]
    fit: Optional[DampedSinusoidFit]
    frequency_method_spread_relative: Optional[float]
    positive_log_decrements: np.ndarray
    negative_log_decrements: np.ndarray
    median_log_decrement: Optional[float]
    decay_rate_s: Optional[float]
    release_inclusive_decay_rate_s: Optional[float]
    damping_ratio: Optional[float]
    natural_frequency_hz: Optional[float]
    envelope_log_rms_residual: Optional[float]
    settling_time_s: Optional[float]
    first_complete_peak_time_s: Optional[float]
    first_complete_peak_amplitude: Optional[float]
    maximum_reverse_peak: float
    reverse_peak_ratio: Optional[float]
    high_frequency_energy_fraction: float
    secondary_frequency_hz: Optional[float]
    secondary_energy_ratio: float
    within_trace_mode_shape_residual: Optional[float]
    single_mode_adequate: bool
    unresolved_reason: Optional[str]


@dataclass(frozen=True)
class RecoverySafetySummary:
    active: object
    maximum_external_generalized_force_residual: float
    maximum_external_power_residual_w: float
    maximum_abs_external_power_w: float
    maximum_abs_actuator_power_w: float
    maximum_abs_applied_force_n: float
    maximum_post_release_force_n: float
    maximum_post_release_qfrc_applied: float
    maximum_quaternion_norm_error: float


@dataclass(frozen=True)
class CommonModeRecoveryTrace:
    config: RecoveryRunConfig
    experiment_kind: str
    tendon_names: Tuple[str, ...]
    actuator_names: Tuple[str, ...]
    tendon_ranges_m: np.ndarray
    equilibrium_tip_world_m: np.ndarray
    release_control_tick: int
    release_physics_step: int
    release_time_s: float
    control_tick: np.ndarray
    control_time_s: np.ndarray
    control_phase: np.ndarray
    control_tension_command_n: np.ndarray
    control_tip_force_world_n: np.ndarray
    control_tip_world_m: np.ndarray
    physics_time_s: np.ndarray
    recovery_time_s: np.ndarray
    tension_command_n: np.ndarray
    actuator_force_n: np.ndarray
    tip_force_world_n: np.ndarray
    qfrc_applied: np.ndarray
    tip_world_m: np.ndarray
    tip_velocity_world_m_s: np.ndarray
    qpos: np.ndarray
    arm_qvel_rad_s: np.ndarray
    arm_qacc_rad_s2: np.ndarray
    joint_rotation_vectors_rad: np.ndarray
    composite_rotation_vectors_rad: np.ndarray
    tendon_length_m: np.ndarray
    tendon_velocity_m_s: np.ndarray
    actuator_power_w: np.ndarray
    external_power_w: np.ndarray
    warning_counts: np.ndarray
    contacts: np.ndarray
    tendon_range_margin_m: np.ndarray
    safety: RecoverySafetySummary
    analysis: RecoveryAnalysis

    @property
    def recovery_signal_m(self):
        delta = self.tip_world_m - self.equilibrium_tip_world_m
        return _freeze(delta @ self.config.direction_vector)

    @property
    def cross_axis_signal_m(self):
        delta = self.tip_world_m - self.equilibrium_tip_world_m
        return _freeze(delta @ self.config.cross_direction_vector)

    @property
    def z_signal_m(self):
        return _freeze(self.tip_world_m[:, 2] - self.equilibrium_tip_world_m[2])

    @property
    def maximum_arm_qvel_rad_s(self):
        return _freeze(np.max(np.abs(self.arm_qvel_rad_s), axis=1))

    @property
    def release_qpos(self):
        return self.qpos[0]

    @property
    def release_qvel(self):
        return self.arm_qvel_rad_s[0]

    @property
    def release_qacc(self):
        return self.arm_qacc_rad_s2[0]


@dataclass(frozen=True)
class FrequencyEffect:
    common_tension_n: float
    frequency_hz: Optional[float]
    frequency_ratio: Optional[float]
    uncertainty_relative: float
    effect_floor_relative: float
    classification: str


def _quadratic_peak(time, values, index):
    denominator = values[index - 1] - 2 * values[index] + values[index + 1]
    if not math.isfinite(float(denominator)) or denominator >= 0.0:
        return None
    offset = 0.5 * (values[index - 1] - values[index + 1]) / denominator
    if not -1.0 <= offset <= 1.0:
        return None
    dt = time[index + 1] - time[index]
    peak_time = time[index] + offset * dt
    amplitude = values[index] - 0.25 * (
        values[index - 1] - values[index + 1]
    ) * offset
    if not math.isfinite(float(amplitude)):
        return None
    return float(peak_time), float(amplitude)


def _polarity_peaks(time, aligned_signal, threshold, polarity):
    values = polarity * aligned_signal
    peaks = []
    if values[0] >= threshold and values[0] >= values[1]:
        peaks.append((float(time[0]), float(values[0])))
    for index in range(1, len(values) - 1):
        if not (
            values[index] >= threshold
            and values[index] >= values[index - 1]
            and values[index] > values[index + 1]
        ):
            continue
        peak = _quadratic_peak(time, values, index)
        if peak is not None and peak[1] >= threshold:
            peaks.append(peak)
    if not peaks:
        return np.empty(0), np.empty(0)
    return (
        np.asarray([item[0] for item in peaks]),
        np.asarray([item[1] for item in peaks]),
    )


def _zero_crossings(time, signal):
    crossing_times = []
    crossing_directions = []
    nonzero = np.flatnonzero(signal != 0.0)
    for left_index, right_index in zip(nonzero[:-1], nonzero[1:]):
        left = signal[left_index]
        right = signal[right_index]
        if left * right >= 0.0:
            continue
        if right_index == left_index + 1:
            fraction = -left / (right - left)
            crossing_time = time[left_index] + fraction * (
                time[right_index] - time[left_index]
            )
        else:
            crossing_time = 0.5 * (
                time[left_index + 1] + time[right_index - 1]
            )
        crossing_times.append(float(crossing_time))
        crossing_directions.append(1 if right > left else -1)
    return np.asarray(crossing_times), np.asarray(crossing_directions, dtype=int)


def _period_frequency(groups):
    periods = []
    for values in groups:
        if len(values) >= 2:
            periods.extend(np.diff(values))
    if not periods:
        return None
    return float(1.0 / np.median(periods))


def fit_damped_sinusoid(time, signal, seed_frequency_hz, seed_alpha_s):
    """Fit offset plus one damped sinusoid using deterministic NumPy grids."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time) < 8 or len(time) != len(signal):
        return None
    if not (
        np.all(np.isfinite(time))
        and np.all(np.isfinite(signal))
        and math.isfinite(seed_frequency_hz)
        and seed_frequency_hz > 0.0
    ):
        return None
    seed_alpha_s = max(0.0, float(seed_alpha_s))
    initial_f_bounds = (
        max(0.5, 0.75 * seed_frequency_hz),
        min(20.0, 1.25 * seed_frequency_hz),
    )
    initial_a_bounds = (0.0, max(8.0, 3.0 * seed_alpha_s))
    f_bounds = initial_f_bounds
    a_bounds = initial_a_bounds
    f_size = 11
    a_size = 9
    levels = 6
    best = None
    best_sse = math.inf
    final_f_step = math.inf
    signal_energy = float(signal @ signal)

    for _ in range(levels):
        frequencies = np.linspace(*f_bounds, f_size)
        alphas = np.linspace(*a_bounds, a_size)
        final_f_step = float(frequencies[1] - frequencies[0])
        level_best = None
        level_sse = math.inf
        # Ascending grids plus strict improvement give deterministic
        # low-alpha, low-frequency tie breaking.
        for alpha in alphas:
            envelope = np.exp(-alpha * time)
            for frequency in frequencies:
                phase = 2 * math.pi * frequency * time
                design = np.column_stack(
                    (np.ones(len(time)), envelope * np.cos(phase), envelope * np.sin(phase))
                )
                coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
                residual = signal - design @ coefficients
                sse = float(residual @ residual)
                if math.isfinite(sse) and sse < level_sse:
                    level_sse = sse
                    level_best = (float(frequency), float(alpha), coefficients, residual)
        if level_best is None:
            return None
        best = level_best
        best_sse = level_sse
        frequency, alpha = best[:2]
        f_bounds = (
            max(0.5, frequency - final_f_step),
            min(20.0, frequency + final_f_step),
        )
        a_step = float(alphas[1] - alphas[0])
        a_bounds = (max(0.0, alpha - a_step), alpha + a_step)

    if best is None or not math.isfinite(best_sse):
        return None
    frequency, alpha, coefficients, residual = best
    amplitude_scale = max(
        float(np.ptp(signal)),
        math.sqrt(max(signal_energy, 0.0) / len(signal)),
        np.finfo(float).tiny,
    )
    normalized_rms = float(
        math.sqrt(np.mean(residual ** 2)) / amplitude_scale
    )
    return DampedSinusoidFit(
        offset=float(coefficients[0]),
        cosine_coefficient=float(coefficients[1]),
        sine_coefficient=float(coefficients[2]),
        frequency_hz=frequency,
        decay_rate_s=alpha,
        normalized_rms_residual=normalized_rms,
        frequency_search_bounds_hz=initial_f_bounds,
        alpha_search_bounds_s=initial_a_bounds,
        levels=levels,
        frequency_grid_size=f_size,
        alpha_grid_size=a_size,
        final_frequency_resolution_hz=final_f_step,
    )


def _mode_shape_residual(time, peak_times, joint_target_axis):
    if joint_target_axis is None or len(peak_times) < 3:
        return None
    joints = np.asarray(joint_target_axis, dtype=float)
    shapes = []
    for peak_time in peak_times[1:]:
        index = int(np.searchsorted(time, peak_time))
        index = min(max(index, 1), len(time) - 1)
        left = index - 1
        fraction = (peak_time - time[left]) / (time[index] - time[left])
        values = (1.0 - fraction) * joints[left] + fraction * joints[index]
        norm = float(np.linalg.norm(values))
        if norm > 0.0:
            shapes.append(values / norm)
    if len(shapes) < 2:
        return None
    reference = shapes[0]
    return float(max(np.linalg.norm(item - reference) for item in shapes[1:]))


def find_recovery_settling_time(
    time,
    signal,
    projected_tip_velocity,
    maximum_arm_qvel,
    *,
    minimum_hold_s=0.25,
    cross_axis_signal=None,
):
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    tip_velocity = np.asarray(projected_tip_velocity, dtype=float)
    arm_qvel = np.asarray(maximum_arm_qvel, dtype=float)
    if not (
        len(time) == len(signal) == len(tip_velocity) == len(arm_qvel)
        and len(time) >= 2
        and np.all(np.diff(time) > 0.0)
    ):
        raise ValueError("settling inputs must share a strictly increasing grid")
    angle_band = max(1e-6, 0.02 * abs(float(signal[0])))
    tip_velocity_band = max(1e-5, 0.02 * float(np.max(np.abs(tip_velocity))))
    arm_qvel_band = max(1e-3, 0.02 * float(np.max(np.abs(arm_qvel))))
    qualified = (
        (np.abs(signal) <= angle_band)
        & (np.abs(tip_velocity) <= tip_velocity_band)
        & (arm_qvel <= arm_qvel_band)
    )
    if cross_axis_signal is not None:
        cross = np.asarray(cross_axis_signal, dtype=float)
        if len(cross) != len(time) or not np.all(np.isfinite(cross)):
            raise ValueError("cross_axis_signal must match the settling grid")
        cross_band = max(1e-8, 0.001 * float(np.max(np.abs(signal))))
        qualified &= np.abs(cross) <= cross_band
    for index in range(len(time)):
        end_time = time[index] + minimum_hold_s
        end = int(np.searchsorted(time, end_time, side="left"))
        if end >= len(time):
            break
        if np.all(qualified[index:end + 1]):
            return float(time[index])
    return None


def analyze_recovery_signal(
    time,
    signal,
    projected_tip_velocity=None,
    maximum_arm_qvel=None,
    joint_target_axis=None,
    *,
    equilibrium_noise=0.0,
    cross_axis_signal=None,
):
    """Analyze one physics-rate recovery signal without SciPy."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if (
        time.ndim != 1
        or signal.ndim != 1
        or len(time) != len(signal)
        or len(time) < 8
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(signal))
        or not np.all(np.diff(time) > 0.0)
    ):
        raise ValueError("time and signal must be finite physics-rate vectors")
    equilibrium_noise = _finite_float("equilibrium_noise", equilibrium_noise)
    if equilibrium_noise < 0.0:
        raise ValueError("equilibrium_noise must be non-negative")
    initial_sign = 1.0 if signal[0] >= 0.0 else -1.0
    aligned = initial_sign * signal
    floor = max(1e-7, 10 * equilibrium_noise, 0.001 * abs(float(signal[0])))
    positive_times, positive_amplitudes = _polarity_peaks(
        time, aligned, floor, 1.0
    )
    negative_times, negative_amplitudes = _polarity_peaks(
        time, aligned, floor, -1.0
    )
    crossing_times, crossing_directions = _zero_crossings(time, aligned)
    peak_frequency = _period_frequency((positive_times, negative_times))
    post_transient_peak_frequency = _period_frequency(
        (positive_times[1:], negative_times)
    )
    zero_frequency = _period_frequency(
        tuple(
            crossing_times[crossing_directions == direction]
            for direction in (-1, 1)
        )
    )
    positive_decrements = (
        np.log(positive_amplitudes[1:-1] / positive_amplitudes[2:])
        if len(positive_amplitudes) >= 3
        else np.empty(0)
    )
    negative_decrements = (
        np.log(negative_amplitudes[:-1] / negative_amplitudes[1:])
        if len(negative_amplitudes) >= 2
        else np.empty(0)
    )
    all_decrements = np.concatenate((positive_decrements, negative_decrements))
    median_decrement = (
        float(np.median(all_decrements)) if len(all_decrements) else None
    )

    decay_rate = None
    release_inclusive_decay_rate = None
    envelope_residual = None
    envelope_times = np.concatenate((positive_times[1:], negative_times))
    envelope_amplitudes = np.concatenate(
        (positive_amplitudes[1:], negative_amplitudes)
    )
    if len(envelope_times) >= 6:
        order = np.argsort(envelope_times)
        envelope_times = envelope_times[order]
        envelope_amplitudes = envelope_amplitudes[order]
        design = np.column_stack((envelope_times, np.ones(len(envelope_times))))
        coefficients = np.linalg.lstsq(
            design, np.log(envelope_amplitudes), rcond=None
        )[0]
        fitted = design @ coefficients
        decay_rate = float(-coefficients[0])
        envelope_residual = float(
            np.sqrt(np.mean((np.log(envelope_amplitudes) - fitted) ** 2))
        )
        inclusive_times = np.concatenate((positive_times, negative_times))
        inclusive_amplitudes = np.concatenate(
            (positive_amplitudes, negative_amplitudes)
        )
        inclusive_order = np.argsort(inclusive_times)
        inclusive_design = np.column_stack(
            (
                inclusive_times[inclusive_order],
                np.ones(len(inclusive_times)),
            )
        )
        release_inclusive_decay_rate = float(
            -np.linalg.lstsq(
                inclusive_design,
                np.log(inclusive_amplitudes[inclusive_order]),
                rcond=None,
            )[0][0]
        )

    fit = None
    if peak_frequency is not None and decay_rate is not None:
        first_complete = positive_times[1] if len(positive_times) >= 2 else time[0]
        last_reliable = max(
            positive_times[-1] if len(positive_times) else time[-1],
            negative_times[-1] if len(negative_times) else time[-1],
        )
        fit_mask = (time >= first_complete) & (time <= last_reliable)
        fit = fit_damped_sinusoid(
            time[fit_mask] - time[fit_mask][0],
            signal[fit_mask],
            peak_frequency,
            decay_rate,
        )

    methods = [
        value
        for value in (
            peak_frequency,
            zero_frequency,
            None if fit is None else fit.frequency_hz,
        )
        if value is not None and math.isfinite(value)
    ]
    method_spread = None
    if len(methods) >= 2:
        method_spread = float((max(methods) - min(methods)) / max(methods))

    damping_ratio = None
    natural_frequency = None
    if decay_rate is not None and peak_frequency is not None and decay_rate > 0.0:
        omega_d = 2 * math.pi * peak_frequency
        omega_n = math.sqrt(omega_d ** 2 + decay_rate ** 2)
        damping_ratio = float(decay_rate / omega_n)
        natural_frequency = float(omega_n / (2 * math.pi))

    settling_time = None
    if projected_tip_velocity is not None and maximum_arm_qvel is not None:
        settling_time = find_recovery_settling_time(
            time,
            signal,
            projected_tip_velocity,
            maximum_arm_qvel,
            cross_axis_signal=cross_axis_signal,
        )

    first_complete_time = (
        float(positive_times[1]) if len(positive_times) >= 2 else None
    )
    first_complete_amplitude = (
        float(positive_amplitudes[1]) if len(positive_amplitudes) >= 2 else None
    )
    reverse_peak = float(np.max(-aligned))
    reverse_ratio = (
        reverse_peak / abs(float(signal[0])) if signal[0] != 0.0 else None
    )

    # A Hann window limits leakage from the strongly decaying primary mode.
    # The secondary diagnostic excludes a wide main-mode neighborhood; it is
    # not used as a frequency estimator or adequacy gate by itself.
    centered = (signal - np.mean(signal)) * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(signal), time[1] - time[0])
    non_dc_energy = float(np.sum(spectrum[1:]))
    high_fraction = (
        float(np.sum(spectrum[frequencies >= 20.0]) / non_dc_energy)
        if non_dc_energy > 0.0
        else 0.0
    )
    secondary_frequency = None
    secondary_ratio = 0.0
    if non_dc_energy > 0.0 and peak_frequency is not None:
        secondary_mask = (
            frequencies >= peak_frequency + 2.0
        )
        if np.any(secondary_mask):
            indices = np.flatnonzero(secondary_mask)
            secondary_index = indices[np.argmax(spectrum[secondary_mask])]
            primary_index = int(np.argmin(np.abs(frequencies - peak_frequency)))
            primary_energy = max(float(spectrum[primary_index]), np.finfo(float).tiny)
            secondary_frequency = float(frequencies[secondary_index])
            secondary_ratio = float(spectrum[secondary_index] / primary_energy)

    mode_shape_residual = _mode_shape_residual(
        time, positive_times, joint_target_axis
    )
    reasons = []
    if len(positive_times) < 7 or len(negative_times) < 6:
        reasons.append("insufficient reliable peaks")
    if peak_frequency is None or zero_frequency is None or fit is None:
        reasons.append("frequency estimator unavailable")
    if method_spread is None or method_spread > 5e-4:
        reasons.append("frequency estimators disagree")
    if fit is None or fit.normalized_rms_residual > 0.02:
        reasons.append("damped-sinusoid residual too large")
    if decay_rate is None or decay_rate <= 0.0:
        reasons.append("non-decaying envelope")
    if envelope_residual is None or envelope_residual > 0.01:
        reasons.append("envelope residual too large")
    if mode_shape_residual is not None and mode_shape_residual > 0.01:
        reasons.append("joint mode shape changed")

    return RecoveryAnalysis(
        signal_floor=float(floor),
        positive_peak_times_s=_freeze(positive_times),
        positive_peak_amplitudes=_freeze(positive_amplitudes),
        negative_peak_times_s=_freeze(negative_times),
        negative_peak_amplitudes=_freeze(negative_amplitudes),
        zero_crossing_times_s=_freeze(crossing_times),
        zero_crossing_directions=_freeze(crossing_directions, dtype=int),
        peak_frequency_hz=peak_frequency,
        post_transient_peak_frequency_hz=post_transient_peak_frequency,
        zero_crossing_frequency_hz=zero_frequency,
        fit=fit,
        frequency_method_spread_relative=method_spread,
        positive_log_decrements=_freeze(positive_decrements),
        negative_log_decrements=_freeze(negative_decrements),
        median_log_decrement=median_decrement,
        decay_rate_s=decay_rate,
        release_inclusive_decay_rate_s=release_inclusive_decay_rate,
        damping_ratio=damping_ratio,
        natural_frequency_hz=natural_frequency,
        envelope_log_rms_residual=envelope_residual,
        settling_time_s=settling_time,
        first_complete_peak_time_s=first_complete_time,
        first_complete_peak_amplitude=first_complete_amplitude,
        maximum_reverse_peak=reverse_peak,
        reverse_peak_ratio=reverse_ratio,
        high_frequency_energy_fraction=high_fraction,
        secondary_frequency_hz=secondary_frequency,
        secondary_energy_ratio=secondary_ratio,
        within_trace_mode_shape_residual=mode_shape_residual,
        single_mode_adequate=not reasons,
        unresolved_reason="; ".join(reasons) if reasons else None,
    )


def _set_matched_qpos(fixture, addresses, direction, total_bend):
    angle = total_bend / len(addresses)
    quaternion = np.empty(4)
    mujoco.mju_axisAngle2Quat(
        quaternion,
        _MATCHED_ROTATION_AXES[direction],
        angle,
    )
    for _, qpos_address, dof_address in addresses:
        fixture.data.qpos[qpos_address:qpos_address + 4] = quaternion
        fixture.data.qvel[dof_address:dof_address + 3] = 0.0
    mujoco.mj_normalizeQuat(fixture.model, fixture.data.qpos)


def _phase(config, control_tick):
    time_s = control_tick / config.ctrl_freq
    if time_s < config.tension_ramp_duration_s:
        return "tension_ramp"
    if time_s < config.equilibrium_window_start_s:
        return "preload_settle"
    if isinstance(config, MatchedStateDecayRunConfig):
        return "equilibrium_measure" if control_tick < config.release_tick else "recovery"
    if time_s < config.force_start_time_s:
        return "equilibrium_measure"
    if time_s < config.force_start_time_s + config.force_ramp_duration_s:
        return "force_ramp"
    if time_s < config.loaded_window_start_s:
        return "loaded_settle"
    if control_tick < config.release_tick:
        return "loaded_measure"
    return "recovery"


def _force_at_tick(config, control_tick):
    if not isinstance(config, ForceReleaseRecoveryRunConfig):
        return np.zeros(3)
    if control_tick >= config.release_tick:
        return np.zeros(3)
    relative = control_tick - config.force_start_tick
    scale = (
        0.0
        if relative <= 0
        else half_cosine_ramp_scale_at_tick(relative, config.force_ramp_ticks)
    )
    return scale * config.force_magnitude_n * config.direction_vector


def _run_recovery(config):
    if not isinstance(config, (MatchedStateDecayRunConfig, ForceReleaseRecoveryRunConfig)):
        raise TypeError("config must be a recovery run config")
    spec = ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )
    drive = IdealTendonTensionSpec(config.max_tension_n)
    fixture = build_fixed_arm_fixture(
        spec,
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
    tip_body_id = int(model.site_bodyid[tip_id])
    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    output_addresses = np.asarray(
        [item.output_address for item in bindings], dtype=int
    )
    tendon_ranges = model.tendon_range[tendon_ids].copy()
    tendon_limited = model.tendon_limited[tendon_ids].copy()
    active_mask = np.full(len(CORNERS), config.common_tension_n > 0.0)
    final_command = (config.common_tension_n,) * len(CORNERS)
    warning_latch = WarningLatch()
    active_accumulator = _SafetyAccumulator()
    external_accumulator = _ExternalAccumulator()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    control_records = []
    physics_records = []
    equilibrium_samples = []
    maximum_post_force = 0.0
    maximum_post_qfrc = 0.0
    maximum_quaternion_error = 0.0

    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    warning_latch.snapshot(data, stage="initial_forward", step_index=0)

    def failure(field, value):
        raise AssertionError(
            f"{field} at time={data.time}, step={active_accumulator.completed_steps}: {value}"
        )

    def command_at_tick(control_tick):
        scale = half_cosine_ramp_scale_at_tick(
            control_tick, config.tension_ramp_ticks
        )
        return (config.common_tension_n * scale,) * len(CORNERS)

    def synchronize(control_tick, force, *, record_control):
        command = command_at_tick(control_tick)
        written, current = write_tension_command(
            fixture, command, drive, bindings=bindings
        )
        if current != bindings:
            raise AssertionError("actuator binding changed")
        external = apply_tip_force_and_check(
            fixture,
            tip_id,
            tip_body_id,
            force,
            external_accumulator,
            jacp=jacp,
            jacr=jacr,
            fail=failure,
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
            stage="synchronized",
            control_tick=control_tick,
            physics_step=active_accumulator.completed_steps,
        )
        external_accumulator.maximum_abs_actuator_power_w = max(
            external_accumulator.maximum_abs_actuator_power_w,
            abs(synchronized["generalized_power"]),
        )
        if record_control:
            control_records.append(
                {
                    "tick": control_tick,
                    "time": float(data.time),
                    "phase": _phase(config, control_tick),
                    "command": np.asarray(written).copy(),
                    "force": external["force"],
                    "tip": synchronized["tip"],
                }
            )
        return written, synchronized, external

    # Pre-release control-grid phases.  External force is refreshed at every
    # physics substep, matching the A4a timing contract.
    for control_tick in range(config.release_tick + 1):
        expected_time = control_tick / config.ctrl_freq
        data.time = expected_time
        force = _force_at_tick(config, control_tick)
        if control_tick == config.release_tick:
            break
        _, synchronized, _ = synchronize(
            control_tick, force, record_control=True
        )
        time_s = float(data.time)
        if config.equilibrium_window_start_s <= time_s <= (
            config.force_start_time_s
            if isinstance(config, ForceReleaseRecoveryRunConfig)
            else config.release_time_s
        ):
            equilibrium_samples.append(synchronized["tip"].copy())
        for substep in range(config.sample_stride):
            mujoco.mj_step(model, data)
            active_accumulator.completed_steps += 1
            warning_latch.snapshot(
                data,
                stage="pre_release_step",
                step_index=active_accumulator.completed_steps,
            )
            if warning_latch.triggered:
                failure("mujoco_warning", warning_latch.events[-1])
            if substep + 1 < config.sample_stride:
                synchronize(control_tick, force, record_control=False)
        data.time = (control_tick + 1) / config.ctrl_freq

    if not equilibrium_samples:
        raise AssertionError("equilibrium window produced no samples")
    equilibrium_tip = np.mean(equilibrium_samples, axis=0)

    # Exact aligned release.  This sample is recorded before the first step.
    data.time = config.release_time_s
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    write_tension_command(fixture, final_command, drive, bindings=bindings)
    if isinstance(config, MatchedStateDecayRunConfig):
        _set_matched_qpos(
            fixture,
            addresses,
            config.direction,
            config.total_bend_rad,
        )
        data.qvel[arm_dofs] = 0.0
        # Exactly one forward synchronizes the reset state before its sample.
        mujoco.mj_forward(model, data)
        external = {
            "force": np.zeros(3),
            "point": np.asarray(data.site_xpos[tip_id]).copy(),
            "tip_velocity": jacp @ data.qvel,
            "external_generalized_power": 0.0,
            "external_tip_power": 0.0,
        }
    else:
        external = apply_tip_force_and_check(
            fixture,
            tip_id,
            tip_body_id,
            np.zeros(3),
            external_accumulator,
            jacp=jacp,
            jacr=jacr,
            fail=failure,
        )
    synchronized = _check_synchronized_state(
        fixture,
        config,
        bindings,
        addresses,
        tip_id,
        tendon_ranges,
        tendon_limited,
        final_command,
        final_command,
        active_mask,
        active_accumulator,
        stage="release_sample",
        control_tick=config.release_tick,
        physics_step=active_accumulator.completed_steps,
    )
    control_records.append(
        {
            "tick": config.release_tick,
            "time": float(data.time),
            "phase": "recovery",
            "command": np.asarray(final_command),
            "force": np.zeros(3),
            "tip": synchronized["tip"],
        }
    )

    def record_recovery(recovery_step, synchronized_values, external_values):
        nonlocal maximum_post_force, maximum_post_qfrc, maximum_quaternion_error
        joints = synchronized_values["joints"]
        quaternion_error = max(
            abs(np.linalg.norm(data.qpos[qpos:qpos + 4]) - 1.0)
            for _, qpos, _ in addresses
        )
        maximum_quaternion_error = max(maximum_quaternion_error, quaternion_error)
        maximum_post_force = max(
            maximum_post_force,
            float(np.max(np.abs(external_values["force"]))),
        )
        maximum_post_qfrc = max(
            maximum_post_qfrc,
            float(np.max(np.abs(data.qfrc_applied))),
        )
        physics_records.append(
            {
                "time": float(data.time),
                "recovery_time": recovery_step / config.sim_freq,
                "command": np.asarray(final_command).copy(),
                "actuator_force": synchronized_values["actuator_force"],
                "force": external_values["force"],
                "qfrc_applied": np.asarray(data.qfrc_applied).copy(),
                "tip": synchronized_values["tip"],
                "tip_velocity": external_values["tip_velocity"],
                "qpos": np.asarray(data.qpos).copy(),
                "qvel": np.asarray(data.qvel[arm_dofs]).copy(),
                "qacc": np.asarray(data.qacc[arm_dofs]).copy(),
                "joints": joints,
                "composite": synchronized_values["composite"],
                "tendon_length": synchronized_values["tendon_length"],
                "tendon_velocity": synchronized_values["tendon_velocity"],
                "actuator_power": synchronized_values["generalized_power"],
                "external_power": external_values["external_generalized_power"],
                "warnings": _warning_counts(data),
                "contacts": int(data.ncon),
                "margins": synchronized_values["margins"],
            }
        )

    record_recovery(0, synchronized, external)
    for recovery_step in range(1, config.recovery_steps + 1):
        # External arrays remain exactly zero; one and only one step advances
        # each physics sample after the virtual or physical release.
        mujoco.mj_step(model, data)
        active_accumulator.completed_steps += 1
        data.time = config.release_time_s + recovery_step / config.sim_freq
        warning_latch.snapshot(
            data,
            stage="recovery_step",
            step_index=active_accumulator.completed_steps,
        )
        if warning_latch.triggered:
            failure("mujoco_warning", warning_latch.events[-1])
        mujoco.mj_forward(model, data)
        synchronized = _check_synchronized_state(
            fixture,
            config,
            bindings,
            addresses,
            tip_id,
            tendon_ranges,
            tendon_limited,
            final_command,
            final_command,
            active_mask,
            active_accumulator,
            stage="recovery_forward",
            control_tick=config.release_tick
            + recovery_step // config.sample_stride,
            physics_step=active_accumulator.completed_steps,
        )
        mujoco.mj_jacSite(model, data, jacp, jacr, tip_id)
        external = {
            "force": np.zeros(3),
            "tip_velocity": (jacp @ data.qvel).copy(),
            "external_generalized_power": float(data.qfrc_applied @ data.qvel),
        }
        record_recovery(recovery_step, synchronized, external)
        if recovery_step % config.sample_stride == 0:
            control_records.append(
                {
                    "tick": config.release_tick
                    + recovery_step // config.sample_stride,
                    "time": float(data.time),
                    "phase": "recovery",
                    "command": np.asarray(final_command).copy(),
                    "force": np.zeros(3),
                    "tip": synchronized["tip"],
                }
            )

    if active_accumulator.completed_steps != config.total_steps:
        raise AssertionError(
            f"completed {active_accumulator.completed_steps}/{config.total_steps} steps"
        )

    def control_values(name, dtype=None):
        return _freeze([item[name] for item in control_records], dtype=dtype)

    def physics_values(name, dtype=None):
        return _freeze([item[name] for item in physics_records], dtype=dtype)

    tip_values = physics_values("tip")
    recovery_signal = (
        tip_values - equilibrium_tip
    ) @ config.direction_vector
    cross_axis_signal = (
        tip_values - equilibrium_tip
    ) @ config.cross_direction_vector
    tip_velocity = physics_values("tip_velocity") @ config.direction_vector
    maximum_arm_qvel = np.max(np.abs(physics_values("qvel")), axis=1)
    joints = physics_values("joints")
    rotation_axis = int(np.argmax(np.abs(_MATCHED_ROTATION_AXES[config.direction])))
    joint_target = joints[:, :, rotation_axis]
    equilibrium_array = np.asarray(equilibrium_samples)
    equilibrium_noise = float(
        np.max(np.linalg.norm(equilibrium_array - equilibrium_tip, axis=1))
    )
    analysis = analyze_recovery_signal(
        physics_values("recovery_time"),
        recovery_signal,
        tip_velocity,
        maximum_arm_qvel,
        joint_target,
        equilibrium_noise=equilibrium_noise,
        cross_axis_signal=cross_axis_signal,
    )
    safety = RecoverySafetySummary(
        active=_active_summary(config, active_accumulator, warning_latch),
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
        maximum_post_release_force_n=maximum_post_force,
        maximum_post_release_qfrc_applied=maximum_post_qfrc,
        maximum_quaternion_norm_error=maximum_quaternion_error,
    )
    return CommonModeRecoveryTrace(
        config=config,
        experiment_kind=(
            "virtual_constraint_matched_state"
            if isinstance(config, MatchedStateDecayRunConfig)
            else "instantaneous_tip_force_release"
        ),
        tendon_names=tuple(manifest.tendon_names),
        actuator_names=tuple(manifest.tendon_actuator_names),
        tendon_ranges_m=_freeze(tendon_ranges),
        equilibrium_tip_world_m=_freeze(equilibrium_tip),
        release_control_tick=config.release_tick,
        release_physics_step=config.release_tick * config.sample_stride,
        release_time_s=config.release_time_s,
        control_tick=control_values("tick", int),
        control_time_s=control_values("time"),
        control_phase=control_values("phase"),
        control_tension_command_n=control_values("command"),
        control_tip_force_world_n=control_values("force"),
        control_tip_world_m=control_values("tip"),
        physics_time_s=physics_values("time"),
        recovery_time_s=physics_values("recovery_time"),
        tension_command_n=physics_values("command"),
        actuator_force_n=physics_values("actuator_force"),
        tip_force_world_n=physics_values("force"),
        qfrc_applied=physics_values("qfrc_applied"),
        tip_world_m=tip_values,
        tip_velocity_world_m_s=physics_values("tip_velocity"),
        qpos=physics_values("qpos"),
        arm_qvel_rad_s=physics_values("qvel"),
        arm_qacc_rad_s2=physics_values("qacc"),
        joint_rotation_vectors_rad=joints,
        composite_rotation_vectors_rad=physics_values("composite"),
        tendon_length_m=physics_values("tendon_length"),
        tendon_velocity_m_s=physics_values("tendon_velocity"),
        actuator_power_w=physics_values("actuator_power"),
        external_power_w=physics_values("external_power"),
        warning_counts=physics_values("warnings", int),
        contacts=physics_values("contacts", int),
        tendon_range_margin_m=physics_values("margins"),
        safety=safety,
        analysis=analysis,
    )


def run_matched_state_decay(config: MatchedStateDecayRunConfig):
    return _run_recovery(config)


def run_force_release_recovery(config: ForceReleaseRecoveryRunConfig):
    return _run_recovery(config)


def classify_frequency_effect(
    common_tension_n,
    frequency_hz,
    baseline_frequency_hz,
    uncertainty_components,
    *,
    practical_floor=5e-4,
    eligible=True,
):
    components = np.asarray(tuple(uncertainty_components), dtype=float)
    if (
        not eligible
        or frequency_hz is None
        or baseline_frequency_hz is None
        or not np.all(np.isfinite(components))
        or np.any(components < 0.0)
    ):
        return FrequencyEffect(
            float(common_tension_n),
            frequency_hz,
            None,
            math.inf,
            math.inf,
            "unresolved",
        )
    uncertainty = float(np.max(components)) if len(components) else 0.0
    floor = max(_finite_float("practical_floor", practical_floor), 3 * uncertainty)
    ratio = float(frequency_hz / baseline_frequency_hz)
    if ratio > 1.0 + floor:
        classification = "resolved increase"
    elif ratio < 1.0 - floor:
        classification = "resolved decrease"
    else:
        classification = "no resolved change"
    return FrequencyEffect(
        common_tension_n=float(common_tension_n),
        frequency_hz=float(frequency_hz),
        frequency_ratio=ratio,
        uncertainty_relative=uncertainty,
        effect_floor_relative=floor,
        classification=classification,
    )
