"""Tests for all environment types."""

import numpy as np
import pytest
import gymnasium as gym

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import Physics, ActionType


class TestBaseAviary:
    """Tests for BaseAviary."""

    def test_instantiation(self):
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240)
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_multi_drone(self):
        env = BaseAviary(num_drones=3, ctrl_freq=240, sim_freq=240)
        obs, _ = env.reset()
        assert env.pos.shape == (3, 3)
        env.close()

    def test_hover_rpm_stability(self):
        """Hover RPM should maintain exact position."""
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240, physics=Physics.MJC)
        env.reset()
        z0 = env.pos[0, 2]
        hover_rpm = np.full(4, env.HOVER_RPM)
        for _ in range(240):
            env.step(hover_rpm)
        assert abs(env.pos[0, 2] - z0) < 1e-4
        assert abs(env.vel[0, 2]) < 1e-4
        env.close()

    def test_physics_modes(self):
        """All physics modes should instantiate and step."""
        for phys in [Physics.MJC, Physics.DYN, Physics.MJC_GND, Physics.MJC_DRAG, Physics.MJC_DW, Physics.MJC_GND_DRAG_DW]:
            env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240, physics=phys)
            env.reset()
            env.step(np.full(4, env.HOVER_RPM))
            env.close()

    def test_action_space_shape(self):
        env = BaseAviary(num_drones=2, ctrl_freq=240, sim_freq=240)
        env.reset()
        assert env.action_space.shape == (8,)
        env.close()
