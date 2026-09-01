"""Gate P3 analytic frame and ideal-sensor tests."""

import math

import numpy as np
import pytest

from multi_drone_mujoco.backends.contracts import SimulatorClock
from multi_drone_mujoco.backends.sensors import (
    IdealSensorAdapter,
    IdealSensorConfig,
    MujocoTruthFrame,
    MujocoTruthSource,
    enu_to_ned,
    flu_to_frd,
    quaternion_wxyz_to_matrix,
    quaternion_enu_flu_to_ned_frd,
)


def config():
    return IdealSensorConfig(
        origin_latitude_deg=30.0,
        origin_longitude_deg=120.0,
        origin_altitude_m=100.0,
        gravity_m_s2=9.81,
        magnetic_field_ned_gauss=(0.2, 0.0, 0.4),
        imu_rate_hz=1000,
        gps_rate_hz=10,
        barometer_rate_hz=50,
        magnetometer_rate_hz=100,
    )


def truth(quaternion=(1, 0, 0, 0), acceleration=(0, 0, 0)):
    return MujocoTruthFrame(
        position_enu_m=(10, 20, 3),
        velocity_enu_m_s=(1, 2, 3),
        quaternion_enu_flu_wxyz=quaternion,
        angular_velocity_flu_rad_s=(1, 2, 3),
        linear_acceleration_enu_m_s2=acceleration,
    )


def test_unit_axes_enu_ned_and_flu_frd():
    assert enu_to_ned((1, 0, 0)) == (0, 1, 0)
    assert enu_to_ned((0, 1, 0)) == (1, 0, 0)
    assert enu_to_ned((0, 0, 1)) == (0, 0, -1)
    assert flu_to_frd((1, 2, 3)) == (1, -2, -3)


def test_level_stationary_and_free_fall_specific_force():
    adapter = IdealSensorAdapter(config())
    stationary = adapter.sample(truth(), SimulatorClock(0, 0))
    assert stationary.position_ned_m == (20, 10, -3)
    assert stationary.velocity_ned_m_s == (2, 1, -3)
    assert np.allclose(stationary.specific_force_frd_m_s2, (0, 0, -9.81))
    assert stationary.linear_acceleration_ned_m_s2 == (0, 0, 0)
    assert stationary.angular_velocity_frd_rad_s == (1, -2, -3)
    adapter.reset()
    free_fall = adapter.sample(truth(acceleration=(0, 0, -9.81)), SimulatorClock(0, 0))
    assert np.allclose(free_fall.specific_force_frd_m_s2, (0, 0, 0))


def test_ninety_degree_yaw_quaternion_rotation_direction():
    yaw_90 = (math.sqrt(0.5), 0, 0, math.sqrt(0.5))
    converted = quaternion_enu_flu_to_ned_frd(yaw_90)
    rotation = quaternion_wxyz_to_matrix(converted)
    # Body-forward points ENU north after +90 deg yaw, hence NED north.
    assert np.allclose(rotation @ np.array([1, 0, 0]), (1, 0, 0), atol=1e-12)


def test_gps_baro_magnetic_field_and_independent_rates():
    adapter = IdealSensorAdapter(config())
    first = adapter.sample(truth(), SimulatorClock(0, 0))
    assert first.gps_latitude_deg > 30.0
    assert first.gps_longitude_deg > 120.0
    assert first.gps_altitude_m == 103.0
    assert first.barometric_altitude_m == 3.0
    assert first.imu_updated and first.gps_updated and first.barometer_updated
    second = adapter.sample(truth(), SimulatorClock(1_000, 1))
    assert second.imu_updated
    assert not second.gps_updated
    assert not second.barometer_updated
    with pytest.raises(ValueError, match="increase"):
        adapter.sample(truth(), SimulatorClock(1_000, 2))
    adapter.reset()
    assert adapter.sample(truth(), SimulatorClock(0, 0)).gps_updated


def test_invalid_quaternion_and_nonfinite_truth_are_rejected():
    with pytest.raises(ValueError, match="unit norm"):
        truth(quaternion=(2, 0, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        MujocoTruthFrame((0, 0, 0), (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, np.nan, 0))


def test_mujoco_truth_source_rotates_world_rate_and_differences_velocity():
    class Aviary:
        pos = np.array([[1.0, 2.0, 3.0]])
        vel = np.array([[1.0, 0.0, 0.0]])
        quat = np.array([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
        ang_v = np.array([[0.0, 1.0, 0.0]])

    source = MujocoTruthSource()
    source.reset(Aviary)
    Aviary.vel = np.array([[1.1, 0.0, 0.0]])
    result = source.sample(Aviary, 0.1)
    assert np.allclose(result.linear_acceleration_enu_m_s2, (1, 0, 0))
    # World +Y is body +X at +90 degree yaw.
    assert np.allclose(result.angular_velocity_flu_rad_s, (1, 0, 0), atol=1e-12)
