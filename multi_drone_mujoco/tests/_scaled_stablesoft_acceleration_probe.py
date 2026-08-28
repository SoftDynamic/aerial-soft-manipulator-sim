"""Pure analysis helpers for the Gate C2-B1 fixed-tension probe."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Optional

import numpy as np

from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    half_cosine_ramp_scale_at_tick,
)


_NUM_TENDONS = 4


def _integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _finite_float(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _aligned_tick_count(name: str, duration_s: float, frequency_hz: int) -> int:
    raw_ticks = duration_s * frequency_hz
    ticks = int(round(raw_ticks))
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(raw_ticks))
    if abs(raw_ticks - ticks) > tolerance:
        raise ValueError(f"{name} must align with the control tick grid")
    if ticks <= 0:
        raise ValueError(f"{name} must contain at least one control tick")
    return ticks


def _readonly_float_array(name: str, values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _channel_index(value, *, num_tendons: int = _NUM_TENDONS) -> int:
    index = _integer("active_channel_index", value)
    if not 0 <= index < num_tendons:
        raise ValueError(
            f"active_channel_index must be in [0, {num_tendons - 1}]"
        )
    return index


@dataclass(frozen=True)
class FixedTensionProfileConfig:
    """Immutable control-grid configuration for a single-tendon profile."""

    ctrl_freq_hz: int
    duration_s: float
    ramp_duration_s: float
    target_tension_n: float
    max_tension_n: float
    num_tendons: int
    active_channel_index: int

    def __post_init__(self) -> None:
        ctrl_freq = _integer("ctrl_freq_hz", self.ctrl_freq_hz)
        num_tendons = _integer("num_tendons", self.num_tendons)
        active_index = _integer(
            "active_channel_index", self.active_channel_index
        )
        duration = _finite_float("duration_s", self.duration_s)
        ramp_duration = _finite_float(
            "ramp_duration_s", self.ramp_duration_s
        )
        target = _finite_float("target_tension_n", self.target_tension_n)
        maximum = _finite_float("max_tension_n", self.max_tension_n)

        if ctrl_freq <= 0:
            raise ValueError("ctrl_freq_hz must be positive")
        if duration <= 0.0:
            raise ValueError("duration_s must be positive")
        if ramp_duration <= 0.0:
            raise ValueError("ramp_duration_s must be positive")
        if ramp_duration > duration:
            raise ValueError("ramp_duration_s must not exceed duration_s")
        if num_tendons != _NUM_TENDONS:
            raise ValueError("num_tendons must be exactly 4")
        if not 0 <= active_index < num_tendons:
            raise ValueError("active_channel_index must be in [0, 3]")
        if maximum <= 0.0:
            raise ValueError("max_tension_n must be positive")
        if not 0.0 <= target <= maximum:
            raise ValueError(
                "target_tension_n must lie in [0, max_tension_n]"
            )

        object.__setattr__(self, "ctrl_freq_hz", ctrl_freq)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "ramp_duration_s", ramp_duration)
        object.__setattr__(self, "target_tension_n", target)
        object.__setattr__(self, "max_tension_n", maximum)
        object.__setattr__(self, "num_tendons", num_tendons)
        object.__setattr__(self, "active_channel_index", active_index)
        _ = self.number_of_control_intervals
        _ = self.ramp_ticks

    @property
    def number_of_control_intervals(self) -> int:
        return _aligned_tick_count(
            "duration_s", self.duration_s, self.ctrl_freq_hz
        )

    @property
    def ramp_ticks(self) -> int:
        return _aligned_tick_count(
            "ramp_duration_s", self.ramp_duration_s, self.ctrl_freq_hz
        )


def build_fixed_tension_command_profile(
    config: FixedTensionProfileConfig,
) -> np.ndarray:
    """Return read-only ``(K, 4)`` commands for intervals ``[t_k,t_k+1)``."""
    if not isinstance(config, FixedTensionProfileConfig):
        raise TypeError("config must be a FixedTensionProfileConfig")
    command = np.zeros(
        (config.number_of_control_intervals, config.num_tendons),
        dtype=float,
    )
    if config.target_tension_n > 0.0:
        command[:, config.active_channel_index] = [
            config.target_tension_n
            * half_cosine_ramp_scale_at_tick(tick, config.ramp_ticks)
            for tick in range(config.number_of_control_intervals)
        ]
    command.setflags(write=False)
    return command


@dataclass(frozen=True)
class AlignedProbeTrace:
    """MuJoCo-independent state-grid data already aligned at control rate."""

    time_s: np.ndarray
    tendon_length_m: np.ndarray
    tendon_velocity_m_s: np.ndarray
    tip_position_m: np.ndarray

    def __post_init__(self) -> None:
        time = _readonly_float_array("time_s", self.time_s)
        tendon_length = _readonly_float_array(
            "tendon_length_m", self.tendon_length_m
        )
        tendon_velocity = _readonly_float_array(
            "tendon_velocity_m_s", self.tendon_velocity_m_s
        )
        tip_position = _readonly_float_array(
            "tip_position_m", self.tip_position_m
        )
        if time.ndim != 1:
            raise ValueError("time_s must be one-dimensional")
        if time.size < 2:
            raise ValueError("time_s must contain at least two state samples")
        if not np.all(np.diff(time) > 0.0):
            raise ValueError("time_s must be strictly increasing")
        expected_tendon_shape = (time.size, _NUM_TENDONS)
        if tendon_length.shape != expected_tendon_shape:
            raise ValueError(
                "tendon_length_m must have shape (state_samples, 4)"
            )
        if tendon_velocity.shape != expected_tendon_shape:
            raise ValueError(
                "tendon_velocity_m_s must have shape (state_samples, 4)"
            )
        if tip_position.shape != (time.size, 3):
            raise ValueError("tip_position_m must have shape (state_samples, 3)")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "tendon_length_m", tendon_length)
        object.__setattr__(self, "tendon_velocity_m_s", tendon_velocity)
        object.__setattr__(self, "tip_position_m", tip_position)

    @property
    def contraction_m(self) -> np.ndarray:
        return _readonly_float_array(
            "contraction_m", self.tendon_length_m[0] - self.tendon_length_m
        )

    @property
    def contraction_velocity_m_s(self) -> np.ndarray:
        return _readonly_float_array(
            "contraction_velocity_m_s", -self.tendon_velocity_m_s
        )

    @property
    def tip_displacement_m(self) -> np.ndarray:
        return _readonly_float_array(
            "tip_displacement_m", self.tip_position_m - self.tip_position_m[0]
        )


@dataclass(frozen=True)
class BaselineCorrectedResponse:
    """Read-only zero/nonzero response on one exact common state grid."""

    time_s: np.ndarray
    zero_contraction_m: np.ndarray
    tension_contraction_m: np.ndarray
    delta_contraction_m: np.ndarray
    zero_contraction_velocity_m_s: np.ndarray
    tension_contraction_velocity_m_s: np.ndarray
    delta_contraction_velocity_m_s: np.ndarray
    tension_tip_displacement_m: np.ndarray
    delta_tip_displacement_m: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _readonly_float_array(name, getattr(self, name))
            )


def baseline_correct_probe_traces(
    zero_trace: AlignedProbeTrace,
    tension_trace: AlignedProbeTrace,
) -> BaselineCorrectedResponse:
    """Subtract aligned zero-run motion without interpolation or flattening."""
    if not isinstance(zero_trace, AlignedProbeTrace) or not isinstance(
        tension_trace, AlignedProbeTrace
    ):
        raise TypeError("zero_trace and tension_trace must be AlignedProbeTrace")
    if not np.array_equal(zero_trace.time_s, tension_trace.time_s):
        raise ValueError("zero and tension time grids must be exactly equal")
    if zero_trace.tendon_length_m.shape != tension_trace.tendon_length_m.shape:
        raise ValueError("zero and tension tendon shapes must be equal")
    zero_contraction = zero_trace.contraction_m
    tension_contraction = tension_trace.contraction_m
    zero_velocity = zero_trace.contraction_velocity_m_s
    tension_velocity = tension_trace.contraction_velocity_m_s
    return BaselineCorrectedResponse(
        time_s=zero_trace.time_s,
        zero_contraction_m=zero_contraction,
        tension_contraction_m=tension_contraction,
        delta_contraction_m=tension_contraction - zero_contraction,
        zero_contraction_velocity_m_s=zero_velocity,
        tension_contraction_velocity_m_s=tension_velocity,
        delta_contraction_velocity_m_s=tension_velocity - zero_velocity,
        tension_tip_displacement_m=tension_trace.tip_displacement_m,
        delta_tip_displacement_m=(
            tension_trace.tip_displacement_m - zero_trace.tip_displacement_m
        ),
    )


def _tail_mask(
    time_s: np.ndarray,
    tail_start_s,
    tail_end_s,
    minimum_tail_samples,
) -> np.ndarray:
    start = _finite_float("tail_start_s", tail_start_s)
    end = _finite_float("tail_end_s", tail_end_s)
    minimum = _integer("minimum_tail_samples", minimum_tail_samples)
    if minimum <= 0:
        raise ValueError("minimum_tail_samples must be positive")
    if start > end:
        raise ValueError("tail_start_s must not exceed tail_end_s")
    if start < time_s[0] or end > time_s[-1]:
        raise ValueError("tail window must lie within the trace time range")
    mask = (time_s >= start) & (time_s <= end)
    if int(np.count_nonzero(mask)) < minimum:
        raise ValueError(
            f"tail window must contain at least {minimum} state samples"
        )
    return mask


@dataclass(frozen=True)
class ProbeMetrics:
    peak_contraction_m: float
    peak_absolute_contraction_m: float
    tail_mean_contraction_m: float
    tail_std_contraction_m: float
    peak_contraction_velocity_m_s: float
    minimum_contraction_velocity_m_s: float
    maximum_contraction_velocity_m_s: float
    baseline_corrected_peak_contraction_m: float
    baseline_corrected_tail_contraction_m: float
    tip_peak_displacement_m: float
    baseline_corrected_tip_peak_displacement_m: float
    inactive_peak_contraction_m: np.ndarray
    cross_coupling_ratio: Optional[float]

    def __post_init__(self) -> None:
        inactive = _readonly_float_array(
            "inactive_peak_contraction_m", self.inactive_peak_contraction_m
        )
        if inactive.shape != (3,):
            raise ValueError("inactive_peak_contraction_m must have shape (3,)")
        object.__setattr__(self, "inactive_peak_contraction_m", inactive)


def compute_probe_metrics(
    response: BaselineCorrectedResponse,
    *,
    active_channel_index: int,
    tail_start_s: float,
    tail_end_s: float,
    activity_resolution_m: float,
    minimum_tail_samples: int = 2,
) -> ProbeMetrics:
    """Compute signed active metrics and absolute geometric cross-coupling."""
    if not isinstance(response, BaselineCorrectedResponse):
        raise TypeError("response must be a BaselineCorrectedResponse")
    active = _channel_index(active_channel_index)
    resolution = _finite_float("activity_resolution_m", activity_resolution_m)
    if resolution < 0.0:
        raise ValueError("activity_resolution_m must be non-negative")
    tail = _tail_mask(
        response.time_s,
        tail_start_s,
        tail_end_s,
        minimum_tail_samples,
    )
    contraction = response.tension_contraction_m[:, active]
    velocity = response.tension_contraction_velocity_m_s[:, active]
    corrected = response.delta_contraction_m[:, active]
    corrected_peak = float(np.max(corrected))
    inactive_indices = [index for index in range(_NUM_TENDONS) if index != active]
    inactive_peak = np.max(
        np.abs(response.delta_contraction_m[:, inactive_indices]), axis=0
    )
    ratio = None
    if corrected_peak > resolution:
        ratio = float(np.max(inactive_peak) / corrected_peak)
    tip_norm = np.linalg.norm(response.tension_tip_displacement_m, axis=1)
    corrected_tip_norm = np.linalg.norm(response.delta_tip_displacement_m, axis=1)
    return ProbeMetrics(
        peak_contraction_m=float(np.max(contraction)),
        peak_absolute_contraction_m=float(np.max(np.abs(contraction))),
        tail_mean_contraction_m=float(np.mean(contraction[tail])),
        tail_std_contraction_m=float(np.std(contraction[tail])),
        peak_contraction_velocity_m_s=float(np.max(velocity)),
        minimum_contraction_velocity_m_s=float(np.min(velocity)),
        maximum_contraction_velocity_m_s=float(np.max(velocity)),
        baseline_corrected_peak_contraction_m=corrected_peak,
        baseline_corrected_tail_contraction_m=float(
            np.mean(corrected[tail])
        ),
        tip_peak_displacement_m=float(np.max(tip_norm)),
        baseline_corrected_tip_peak_displacement_m=float(
            np.max(corrected_tip_norm)
        ),
        inactive_peak_contraction_m=inactive_peak,
        cross_coupling_ratio=ratio,
    )


@dataclass(frozen=True)
class ProbeCondition:
    """Explicit provenance for a synthetic or future measured probe."""

    target_tension_n: float
    sim_freq_hz: int
    ctrl_freq_hz: int

    def __post_init__(self) -> None:
        target = _finite_float("target_tension_n", self.target_tension_n)
        sim_freq = _integer("sim_freq_hz", self.sim_freq_hz)
        ctrl_freq = _integer("ctrl_freq_hz", self.ctrl_freq_hz)
        if target <= 0.0:
            raise ValueError("target_tension_n must be positive for S_probe")
        if sim_freq <= 0 or ctrl_freq <= 0:
            raise ValueError("probe frequencies must be positive")
        object.__setattr__(self, "target_tension_n", target)
        object.__setattr__(self, "sim_freq_hz", sim_freq)
        object.__setattr__(self, "ctrl_freq_hz", ctrl_freq)


@dataclass(frozen=True)
class SProbeMeasurement:
    condition: ProbeCondition
    active_channel_index: int
    tail_start_s: float
    tail_end_s: float
    value_m: float


def measure_s_probe(
    response: BaselineCorrectedResponse,
    condition: ProbeCondition,
    *,
    active_channel_index: int,
    tail_start_s: float,
    tail_end_s: float,
    minimum_tail_samples: int = 2,
) -> SProbeMeasurement:
    """Return signed tail contraction for an explicitly identified condition.

    Gate C2-B1 defines ``S_probe`` by selecting the 0.002 N, 2400 Hz
    measurement; those experiment values remain caller-owned metadata.
    """
    if not isinstance(response, BaselineCorrectedResponse):
        raise TypeError("response must be a BaselineCorrectedResponse")
    if not isinstance(condition, ProbeCondition):
        raise TypeError("condition must be a ProbeCondition")
    active = _channel_index(active_channel_index)
    start = _finite_float("tail_start_s", tail_start_s)
    end = _finite_float("tail_end_s", tail_end_s)
    tail = _tail_mask(response.time_s, start, end, minimum_tail_samples)
    return SProbeMeasurement(
        condition=condition,
        active_channel_index=active,
        tail_start_s=start,
        tail_end_s=end,
        value_m=float(np.mean(response.delta_contraction_m[tail, active])),
    )


@dataclass(frozen=True)
class ZeroRunActivityFloor:
    n_s_m: float
    n_v_m_s: float
    rms_contraction_m: float
    rms_contraction_velocity_m_s: float


def compute_zero_run_activity_floor(
    zero_trace: AlignedProbeTrace,
) -> ZeroRunActivityFloor:
    """Measure zero-run activity, including legitimate passive transients."""
    if not isinstance(zero_trace, AlignedProbeTrace):
        raise TypeError("zero_trace must be an AlignedProbeTrace")
    contraction = zero_trace.contraction_m
    velocity = zero_trace.contraction_velocity_m_s
    return ZeroRunActivityFloor(
        n_s_m=float(np.max(np.abs(contraction))),
        n_v_m_s=float(np.max(np.abs(velocity))),
        rms_contraction_m=float(np.sqrt(np.mean(np.square(contraction)))),
        rms_contraction_velocity_m_s=float(
            np.sqrt(np.mean(np.square(velocity)))
        ),
    )


def machine_resolution_floor_m(
    initial_tendon_length_m,
    *,
    fallback_length_scale_m=None,
) -> float:
    """Return ``64*eps*length_scale`` with an explicit zero-scale fallback."""
    initial = _readonly_float_array(
        "initial_tendon_length_m", initial_tendon_length_m
    )
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("initial_tendon_length_m must be a non-empty vector")
    scale = float(np.max(np.abs(initial)))
    if scale == 0.0:
        if fallback_length_scale_m is None:
            raise ValueError(
                "fallback_length_scale_m is required for a zero length scale"
            )
        scale = _finite_float(
            "fallback_length_scale_m", fallback_length_scale_m
        )
        if scale <= 0.0:
            raise ValueError("fallback_length_scale_m must be positive")
    return float(64 * np.finfo(float).eps * scale)


@dataclass(frozen=True)
class AlignedResponseDifference:
    contraction_difference_m: np.ndarray
    velocity_difference_m_s: np.ndarray
    d_s_m: float
    d_v_m_s: float

    def __post_init__(self) -> None:
        contraction = _readonly_float_array(
            "contraction_difference_m", self.contraction_difference_m
        )
        velocity = _readonly_float_array(
            "velocity_difference_m_s", self.velocity_difference_m_s
        )
        if contraction.shape != (_NUM_TENDONS,):
            raise ValueError("contraction_difference_m must have shape (4,)")
        if velocity.shape != (_NUM_TENDONS,):
            raise ValueError("velocity_difference_m_s must have shape (4,)")
        object.__setattr__(self, "contraction_difference_m", contraction)
        object.__setattr__(self, "velocity_difference_m_s", velocity)


def compare_aligned_responses(
    first: BaselineCorrectedResponse,
    second: BaselineCorrectedResponse,
) -> AlignedResponseDifference:
    """Compute d_s/d_v from two already aligned corrected responses."""
    if not isinstance(first, BaselineCorrectedResponse) or not isinstance(
        second, BaselineCorrectedResponse
    ):
        raise TypeError("responses must be BaselineCorrectedResponse")
    if not np.array_equal(first.time_s, second.time_s):
        raise ValueError("response time grids must be exactly equal")
    if first.delta_contraction_m.shape != second.delta_contraction_m.shape:
        raise ValueError("response tendon shapes must be equal")
    contraction = np.max(
        np.abs(first.delta_contraction_m - second.delta_contraction_m),
        axis=0,
    )
    velocity = np.max(
        np.abs(
            first.delta_contraction_velocity_m_s
            - second.delta_contraction_velocity_m_s
        ),
        axis=0,
    )
    return AlignedResponseDifference(
        contraction_difference_m=contraction,
        velocity_difference_m_s=velocity,
        d_s_m=float(np.max(contraction)),
        d_v_m_s=float(np.max(velocity)),
    )


def integrate_interval_power_w(
    interval_power_w,
    *,
    dt_sim_s: float,
) -> float:
    """Integrate power per interval without bridging command boundaries."""
    power = _readonly_float_array("interval_power_w", interval_power_w)
    dt_sim = _finite_float("dt_sim_s", dt_sim_s)
    if dt_sim <= 0.0:
        raise ValueError("dt_sim_s must be positive")
    if power.ndim != 2:
        raise ValueError("interval_power_w must be a two-dimensional array")
    if power.shape[0] == 0:
        raise ValueError("interval_power_w must contain at least one interval")
    if power.shape[1] < 2:
        raise ValueError(
            "each interval must contain start and end power samples"
        )
    interval_work = dt_sim * (
        0.5 * power[:, 0]
        + np.sum(power[:, 1:-1], axis=1)
        + 0.5 * power[:, -1]
    )
    total = float(np.sum(interval_work))
    if not math.isfinite(total):
        raise ValueError("integrated work must be finite")
    return total
