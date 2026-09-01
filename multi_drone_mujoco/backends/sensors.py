"""Pure ENU/FLU to NED/FRD transforms and ideal sensor generation."""

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np

from multi_drone_mujoco.backends.contracts import (
    SimulatorClock,
    finite_tuple,
    nonnegative_int,
)


ENU_TO_NED = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
)
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])
EARTH_RADIUS_M = 6_378_137.0


def _vector3(name, values) -> np.ndarray:
    return np.asarray(finite_tuple(name, values, 3), dtype=float)


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def enu_to_ned(vector) -> Tuple[float, float, float]:
    """Convert one polar vector from MuJoCo ENU to PX4 NED."""
    return tuple(ENU_TO_NED @ _vector3("vector", vector))


def flu_to_frd(vector) -> Tuple[float, float, float]:
    """Convert one body vector from FLU to FRD."""
    return tuple(FLU_TO_FRD @ _vector3("vector", vector))


def quaternion_wxyz_to_matrix(quaternion) -> np.ndarray:
    """Convert a scalar-first active quaternion to a rotation matrix."""
    q = np.asarray(finite_tuple("quaternion", quaternion, 4), dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("quaternion must have unit norm")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion_wxyz(matrix) -> Tuple[float, float, float, float]:
    """Convert a proper active rotation matrix to scalar-first quaternion."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must be a finite 3x3 array")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-9):
        raise ValueError("matrix must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-9):
        raise ValueError("matrix must be a proper rotation")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
             (matrix[0, 2] - matrix[2, 0]) / s,
             (matrix[1, 0] - matrix[0, 1]) / s]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            q = np.array([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                          (matrix[0, 1] + matrix[1, 0]) / s,
                          (matrix[0, 2] + matrix[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            q = np.array([(matrix[0, 2] - matrix[2, 0]) / s,
                          (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s,
                          (matrix[1, 2] + matrix[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            q = np.array([(matrix[1, 0] - matrix[0, 1]) / s,
                          (matrix[0, 2] + matrix[2, 0]) / s,
                          (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return tuple(q)


def quaternion_enu_flu_to_ned_frd(quaternion) -> Tuple[float, float, float, float]:
    """Convert body-to-world attitude from ENU/FLU to NED/FRD."""
    rotation_enu_flu = quaternion_wxyz_to_matrix(quaternion)
    return matrix_to_quaternion_wxyz(
        ENU_TO_NED @ rotation_enu_flu @ FLU_TO_FRD
    )


@dataclass(frozen=True)
class MujocoTruthFrame:
    """Noise-free MuJoCo truth before any simulated sensor effects."""

    position_enu_m: Tuple[float, float, float]
    velocity_enu_m_s: Tuple[float, float, float]
    quaternion_enu_flu_wxyz: Tuple[float, float, float, float]
    angular_velocity_flu_rad_s: Tuple[float, float, float]
    linear_acceleration_enu_m_s2: Tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_enu_m", finite_tuple("position_enu_m", self.position_enu_m, 3))
        object.__setattr__(self, "velocity_enu_m_s", finite_tuple("velocity_enu_m_s", self.velocity_enu_m_s, 3))
        quaternion_wxyz_to_matrix(self.quaternion_enu_flu_wxyz)
        object.__setattr__(self, "quaternion_enu_flu_wxyz", finite_tuple("quaternion_enu_flu_wxyz", self.quaternion_enu_flu_wxyz, 4))
        object.__setattr__(self, "angular_velocity_flu_rad_s", finite_tuple("angular_velocity_flu_rad_s", self.angular_velocity_flu_rad_s, 3))
        object.__setattr__(self, "linear_acceleration_enu_m_s2", finite_tuple("linear_acceleration_enu_m_s2", self.linear_acceleration_enu_m_s2, 3))


@dataclass(frozen=True)
class SensorFrame:
    """Ideal PX4-facing sensor values at one simulation instant."""

    timestamp_us: int
    sequence: int
    position_ned_m: Tuple[float, float, float]
    velocity_ned_m_s: Tuple[float, float, float]
    quaternion_ned_frd_wxyz: Tuple[float, float, float, float]
    angular_velocity_frd_rad_s: Tuple[float, float, float]
    specific_force_frd_m_s2: Tuple[float, float, float]
    linear_acceleration_ned_m_s2: Tuple[float, float, float]
    gps_latitude_deg: float
    gps_longitude_deg: float
    gps_altitude_m: float
    barometric_altitude_m: float
    magnetic_field_frd_gauss: Tuple[float, float, float]
    imu_updated: bool
    gps_updated: bool
    barometer_updated: bool
    magnetometer_updated: bool

    def __post_init__(self) -> None:
        nonnegative_int("timestamp_us", self.timestamp_us)
        nonnegative_int("sequence", self.sequence)
        for name in (
            "position_ned_m",
            "velocity_ned_m_s",
            "angular_velocity_frd_rad_s",
            "specific_force_frd_m_s2",
            "linear_acceleration_ned_m_s2",
            "magnetic_field_frd_gauss",
        ):
            object.__setattr__(self, name, finite_tuple(name, getattr(self, name), 3))
        quaternion_wxyz_to_matrix(self.quaternion_ned_frd_wxyz)
        object.__setattr__(
            self,
            "quaternion_ned_frd_wxyz",
            finite_tuple(
                "quaternion_ned_frd_wxyz", self.quaternion_ned_frd_wxyz, 4
            ),
        )
        for name in (
            "gps_latitude_deg",
            "gps_longitude_deg",
            "gps_altitude_m",
            "barometric_altitude_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not -90.0 <= self.gps_latitude_deg <= 90.0:
            raise ValueError("gps_latitude_deg must be in [-90, 90]")
        if not -180.0 <= self.gps_longitude_deg <= 180.0:
            raise ValueError("gps_longitude_deg must be in [-180, 180]")
        for name in (
            "imu_updated",
            "gps_updated",
            "barometer_updated",
            "magnetometer_updated",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")


@dataclass(frozen=True)
class IdealSensorConfig:
    """Ideal sensor origin, fields, gravity, and independent update rates."""

    origin_latitude_deg: float
    origin_longitude_deg: float
    origin_altitude_m: float
    gravity_m_s2: float
    magnetic_field_ned_gauss: Tuple[float, float, float]
    imu_rate_hz: float
    gps_rate_hz: float
    barometer_rate_hz: float
    magnetometer_rate_hz: float

    def __post_init__(self) -> None:
        for name in ("origin_latitude_deg", "origin_longitude_deg", "origin_altitude_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not -90.0 <= self.origin_latitude_deg <= 90.0:
            raise ValueError("origin_latitude_deg must be in [-90, 90]")
        if not -180.0 <= self.origin_longitude_deg <= 180.0:
            raise ValueError("origin_longitude_deg must be in [-180, 180]")
        object.__setattr__(self, "gravity_m_s2", _positive_finite("gravity_m_s2", self.gravity_m_s2))
        object.__setattr__(self, "magnetic_field_ned_gauss", finite_tuple("magnetic_field_ned_gauss", self.magnetic_field_ned_gauss, 3))
        for name in ("imu_rate_hz", "gps_rate_hz", "barometer_rate_hz", "magnetometer_rate_hz"):
            object.__setattr__(self, name, _positive_finite(name, getattr(self, name)))


class IdealSensorAdapter:
    """Convert MuJoCo truth to ideal PX4 sensors without noise or latency."""

    def __init__(self, config: IdealSensorConfig):
        self.config = config
        self._last_timestamp_us = -1
        self._last_updates: Dict[str, int] = {}

    def reset(self) -> None:
        self._last_timestamp_us = -1
        self._last_updates.clear()

    def _due(self, name: str, rate_hz: float, timestamp_us: int) -> bool:
        previous = self._last_updates.get(name)
        period_us = 1_000_000.0 / rate_hz
        due = previous is None or timestamp_us - previous + 1e-9 >= period_us
        if due:
            self._last_updates[name] = timestamp_us
        return due

    def sample(self, truth: MujocoTruthFrame, clock: SimulatorClock) -> SensorFrame:
        if clock.timestamp_us <= self._last_timestamp_us:
            raise ValueError("sensor timestamps must increase monotonically")
        self._last_timestamp_us = clock.timestamp_us
        position_enu = _vector3("position_enu_m", truth.position_enu_m)
        position_ned = ENU_TO_NED @ position_enu
        velocity_ned = ENU_TO_NED @ _vector3("velocity_enu_m_s", truth.velocity_enu_m_s)
        rotation_enu_flu = quaternion_wxyz_to_matrix(truth.quaternion_enu_flu_wxyz)
        rotation_ned_frd = ENU_TO_NED @ rotation_enu_flu @ FLU_TO_FRD
        acceleration_enu = _vector3("linear_acceleration_enu_m_s2", truth.linear_acceleration_enu_m_s2)
        acceleration_ned = ENU_TO_NED @ acceleration_enu
        gravity_enu = np.array([0.0, 0.0, -self.config.gravity_m_s2])
        specific_force_flu = rotation_enu_flu.T @ (acceleration_enu - gravity_enu)
        specific_force_frd = FLU_TO_FRD @ specific_force_flu
        angular_velocity_frd = FLU_TO_FRD @ _vector3("angular_velocity_flu_rad_s", truth.angular_velocity_flu_rad_s)
        magnetic_field_frd = rotation_ned_frd.T @ np.asarray(self.config.magnetic_field_ned_gauss)

        latitude_rad = math.radians(self.config.origin_latitude_deg)
        latitude = self.config.origin_latitude_deg + math.degrees(position_enu[1] / EARTH_RADIUS_M)
        longitude = self.config.origin_longitude_deg + math.degrees(
            position_enu[0] / (EARTH_RADIUS_M * max(math.cos(latitude_rad), 1e-12))
        )
        altitude = self.config.origin_altitude_m + position_enu[2]
        return SensorFrame(
            timestamp_us=clock.timestamp_us,
            sequence=clock.sequence,
            position_ned_m=tuple(position_ned),
            velocity_ned_m_s=tuple(velocity_ned),
            quaternion_ned_frd_wxyz=matrix_to_quaternion_wxyz(rotation_ned_frd),
            angular_velocity_frd_rad_s=tuple(angular_velocity_frd),
            specific_force_frd_m_s2=tuple(specific_force_frd),
            linear_acceleration_ned_m_s2=tuple(acceleration_ned),
            gps_latitude_deg=latitude,
            gps_longitude_deg=longitude,
            gps_altitude_m=altitude,
            barometric_altitude_m=float(position_enu[2]),
            magnetic_field_frd_gauss=tuple(magnetic_field_frd),
            imu_updated=self._due("imu", self.config.imu_rate_hz, clock.timestamp_us),
            gps_updated=self._due("gps", self.config.gps_rate_hz, clock.timestamp_us),
            barometer_updated=self._due("barometer", self.config.barometer_rate_hz, clock.timestamp_us),
            magnetometer_updated=self._due("magnetometer", self.config.magnetometer_rate_hz, clock.timestamp_us),
        )


class MujocoTruthSource:
    """Extract one drone's truth from an aviary without coupling it to PX4.

    `BaseAviary.ang_v` is interpreted as an ENU-world angular velocity and is
    rotated into the FLU body frame required by :class:`MujocoTruthFrame`.
    Linear acceleration is the finite difference of world velocity over the
    explicit simulation interval.
    """

    def __init__(self, drone_index: int = 0):
        if isinstance(drone_index, bool) or not isinstance(drone_index, int) or drone_index < 0:
            raise ValueError("drone_index must be a non-negative integer")
        self.drone_index = drone_index
        self._previous_velocity = None

    def reset(self, aviary) -> None:
        self._previous_velocity = self._read_vector(aviary.vel, "vel").copy()

    def sample(self, aviary, dt_s: float) -> MujocoTruthFrame:
        dt_s = float(dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        position = self._read_vector(aviary.pos, "pos")
        velocity = self._read_vector(aviary.vel, "vel")
        quaternion = np.asarray(aviary.quat[self.drone_index], dtype=float)
        rotation_enu_flu = quaternion_wxyz_to_matrix(quaternion)
        angular_velocity_enu = self._read_vector(aviary.ang_v, "ang_v")
        angular_velocity_flu = rotation_enu_flu.T @ angular_velocity_enu
        if self._previous_velocity is None:
            acceleration = np.zeros(3, dtype=float)
        else:
            acceleration = (velocity - self._previous_velocity) / dt_s
        self._previous_velocity = velocity.copy()
        return MujocoTruthFrame(
            tuple(position),
            tuple(velocity),
            tuple(quaternion),
            tuple(angular_velocity_flu),
            tuple(acceleration),
        )

    def _read_vector(self, values, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3 or self.drone_index >= array.shape[0]:
            raise ValueError(f"aviary {name} must have shape (num_drones, 3)")
        vector = array[self.drone_index]
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"aviary {name} values must be finite")
        return vector
