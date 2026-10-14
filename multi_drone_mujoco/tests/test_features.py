"""Tests for the retained wind model and BaseAviary integration."""

import numpy as np
import pytest

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import Physics
from multi_drone_mujoco.wrappers.wind import WindField, WindConfig, WindModel



class TestWindModel:
    def test_constant_wind(self):
        config = WindConfig(
            model=WindModel.CONSTANT,
            constant_wind=np.array([2.0, 0, 0]),
        )
        wind = WindField(config)
        wind.reset(seed=0)
        force = wind.get_force(dt=1/240, position=np.array([0, 0, 1.0]),
                               velocity=np.zeros(3))
        assert force[0] > 0  # positive x force from headwind
        assert abs(force[1]) < 1e-10
        assert abs(force[2]) < 1e-10

    def test_dryden_turbulence(self):
        config = WindConfig(model=WindModel.DRYDEN, turbulence_intensity=2.0)
        wind = WindField(config)
        wind.reset(seed=42)
        forces = []
        for _ in range(100):
            f = wind.get_force(1/240, np.array([0, 0, 1.0]), np.zeros(3))
            forces.append(f)
        forces = np.array(forces)
        # Should produce non-zero, varying forces
        assert np.std(forces) > 0
        # Should be bounded (not exploding)
        assert np.max(np.abs(forces)) < 1.0

    def test_gust(self):
        config = WindConfig(model=WindModel.GUST, gust_probability=1.0, gust_intensity=0.01)
        wind = WindField(config)
        wind.reset(seed=0)
        force = wind.get_force(1/240, np.zeros(3), np.zeros(3))
        # With prob=1, first call should trigger a gust
        assert np.linalg.norm(force) > 0

    def test_wind_in_env(self):
        """Test wind integration in BaseAviary."""
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240)
        wind_config = WindConfig(
            model=WindModel.CONSTANT,
            constant_wind=np.array([5.0, 0, 0]),
            drag_coefficient=0.01,
        )
        env.set_wind(wind_config)
        env.reset()
        # Step with hover RPM — wind should push drone in +x
        hover = np.full(4, env.HOVER_RPM)
        for _ in range(240):
            env.step(hover)
        # Drone should have moved in x direction due to wind
        assert env.pos[0, 0] > 0.001
        env.close()

    def test_sinusoidal(self):
        config = WindConfig(model=WindModel.SINUSOIDAL, sinusoidal_amplitude=0.01)
        wind = WindField(config)
        wind.reset()
        f1 = wind.get_force(1/240, np.zeros(3), np.zeros(3))
        for _ in range(120):
            wind.get_force(1/240, np.zeros(3), np.zeros(3))
        f2 = wind.get_force(1/240, np.zeros(3), np.zeros(3))
        # Should produce different forces at different times
        assert not np.allclose(f1, f2)
