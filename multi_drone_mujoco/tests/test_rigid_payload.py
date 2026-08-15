"""Tests for the single-CF2X 3 g rigid payload benchmark."""

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.envs.base_aviary import BaseAviary, CF2X_3G_PAYLOAD_PARAMS
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


def _make_benchmark_env(enable_3g_payload=False, **overrides):
    kwargs = {
        "drone_model": DroneModel.CF2X,
        "num_drones": 1,
        "initial_xyzs": np.array([[0.0, 0.0, 1.0]]),
        "physics": Physics.MJC,
        "sim_freq": 240,
        "ctrl_freq": 240,
        "gui": False,
        "record": False,
        "obstacles": False,
        "vision_attributes": False,
        "act_type": ActionType.RPM,
        "render_mode": None,
        "enable_3g_payload": enable_3g_payload,
    }
    kwargs.update(overrides)
    return BaseAviary(**kwargs)


def _body_subtree_ids(model, root_name):
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    assert root_id >= 0

    subtree_ids = []
    for body_id in range(model.nbody):
        ancestor_id = body_id
        while True:
            if ancestor_id == root_id:
                subtree_ids.append(body_id)
                break
            parent_id = int(model.body_parentid[ancestor_id])
            if parent_id == ancestor_id:
                break
            ancestor_id = parent_id
    return root_id, subtree_ids


def _body_subtree_mass(model, root_name):
    _, subtree_ids = _body_subtree_ids(model, root_name)
    return float(np.sum(model.body_mass[subtree_ids]))


def _run_open_loop(env, rpm, steps=240):
    observation, _ = env.reset(seed=0)
    z_initial = float(env.pos[0, 2])
    joint_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint"
    )
    z_dof_address = int(env.model.jnt_dofadr[joint_id]) + 2
    rpm_action = np.full(4, rpm, dtype=float)

    first_qacc_z = None
    delta_z_02 = None
    finite = True
    for step in range(steps):
        observation, _, _, _, _ = env.step(rpm_action)
        if step == 0:
            first_qacc_z = float(env.data.qacc[z_dof_address])
        if step == 47:
            delta_z_02 = float(env.pos[0, 2] - z_initial)

        finite = finite and all(
            np.all(np.isfinite(values))
            for values in (
                observation,
                env.data.qpos,
                env.data.qvel,
                env.data.qacc,
                rpm_action,
                env.last_clipped_action,
            )
        )

    return {
        "first_qacc_z": first_qacc_z,
        "delta_z_02": delta_z_02,
        "delta_z_1": float(env.pos[0, 2] - z_initial),
        "final_vz": float(env.vel[0, 2]),
        "finite": finite,
    }


def test_payload_model_structure_and_named_subtree_mass():
    unloaded = _make_benchmark_env()
    loaded = _make_benchmark_env(enable_3g_payload=True)
    try:
        assert _body_subtree_mass(unloaded.model, "drone0") == pytest.approx(
            0.027, rel=0.0, abs=1e-12
        )
        assert mujoco.mj_name2id(
            unloaded.model, mujoco.mjtObj.mjOBJ_BODY, "drone0_payload"
        ) == -1

        drone_id, loaded_subtree_ids = _body_subtree_ids(loaded.model, "drone0")
        payload_id = mujoco.mj_name2id(
            loaded.model, mujoco.mjtObj.mjOBJ_BODY, "drone0_payload"
        )
        payload_geom_id = mujoco.mj_name2id(
            loaded.model, mujoco.mjtObj.mjOBJ_GEOM, "drone0_payload_geom"
        )

        assert payload_id >= 0
        assert payload_geom_id >= 0
        assert payload_id in loaded_subtree_ids
        assert int(loaded.model.body_parentid[payload_id]) == drone_id
        assert int(loaded.model.body_jntnum[payload_id]) == 0
        assert float(np.sum(loaded.model.body_mass[loaded_subtree_ids])) == pytest.approx(
            0.030, rel=0.0, abs=1e-12
        )

        payload_mass = CF2X_3G_PAYLOAD_PARAMS["mass"]
        radius = CF2X_3G_PAYLOAD_PARAMS["radius"]
        half_height = CF2X_3G_PAYLOAD_PARAMS["half_height"]
        full_height = 2 * half_height
        expected_ixx = payload_mass * (3 * radius ** 2 + full_height ** 2) / 12
        expected_izz = 0.5 * payload_mass * radius ** 2

        assert loaded.model.body_mass[payload_id] == pytest.approx(
            payload_mass, rel=0.0, abs=1e-12
        )
        np.testing.assert_allclose(
            loaded.model.body_pos[payload_id],
            CF2X_3G_PAYLOAD_PARAMS["position"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            loaded.model.body_inertia[payload_id],
            [expected_ixx, expected_ixx, expected_izz],
            rtol=0.0,
            atol=1e-12,
        )
        assert int(loaded.model.geom_bodyid[payload_geom_id]) == payload_id
        assert loaded.model.geom_type[payload_geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        np.testing.assert_allclose(
            loaded.model.geom_size[payload_geom_id, :2],
            [radius, half_height],
            rtol=0.0,
            atol=1e-12,
        )
        assert int(loaded.model.geom_contype[payload_geom_id]) == 1
        assert int(loaded.model.geom_conaffinity[payload_geom_id]) == 1
    finally:
        loaded.close()
        unloaded.close()


def test_payload_changes_only_mass_weight_and_hover_rpm():
    unloaded = _make_benchmark_env()
    loaded = _make_benchmark_env(enable_3g_payload=True)
    try:
        assert unloaded.M == pytest.approx(0.027, rel=0.0, abs=1e-12)
        assert loaded.M == pytest.approx(0.030, rel=0.0, abs=1e-12)
        assert loaded.PAYLOAD_MASS == pytest.approx(0.003, rel=0.0, abs=1e-12)
        assert loaded.GRAVITY == pytest.approx(loaded.G * loaded.M, rel=0.0, abs=1e-12)
        assert loaded.HOVER_RPM == pytest.approx(
            np.sqrt(loaded.M * loaded.G / (4 * loaded.KF)), rel=1e-12
        )
        assert loaded.HOVER_RPM / unloaded.HOVER_RPM == pytest.approx(
            np.sqrt(0.030 / 0.027), rel=1e-12
        )

        for attribute in (
            "THRUST2WEIGHT_RATIO",
            "MAX_RPM",
            "MAX_THRUST",
            "MAX_XY_TORQUE",
            "MAX_Z_TORQUE",
            "KF",
            "KM",
            "L",
        ):
            assert getattr(loaded, attribute) == getattr(unloaded, attribute)
        np.testing.assert_array_equal(loaded.J, unloaded.J)

        assert unloaded.MAX_THRUST / unloaded.GRAVITY == pytest.approx(
            2.25, rel=0.0, abs=1e-12
        )
        assert loaded.MAX_THRUST / loaded.GRAVITY == pytest.approx(
            2.025, rel=0.0, abs=1e-12
        )
    finally:
        loaded.close()
        unloaded.close()


@pytest.mark.parametrize("drone_model", [DroneModel.CF2P, DroneModel.RACE])
def test_payload_rejects_unsupported_drone_models(drone_model):
    with pytest.raises(ValueError, match="only supported for DroneModel.CF2X"):
        _make_benchmark_env(enable_3g_payload=True, drone_model=drone_model)


def test_payload_rejects_multiple_drones():
    with pytest.raises(ValueError, match="requires num_drones == 1"):
        _make_benchmark_env(enable_3g_payload=True, num_drones=2)


@pytest.mark.parametrize("physics", [mode for mode in Physics if mode != Physics.MJC])
def test_payload_rejects_unsupported_physics_modes(physics):
    with pytest.raises(ValueError, match="only supported with Physics.MJC"):
        _make_benchmark_env(enable_3g_payload=True, physics=physics)


@pytest.mark.parametrize("act_type", [mode for mode in ActionType if mode != ActionType.RPM])
def test_payload_rejects_unsupported_action_modes(act_type):
    with pytest.raises(ValueError, match="only supported with ActionType.RPM"):
        _make_benchmark_env(enable_3g_payload=True, act_type=act_type)


def test_three_open_loop_payload_benchmark_conditions():
    unloaded = _make_benchmark_env()
    loaded_old_rpm = _make_benchmark_env(enable_3g_payload=True)
    loaded_new_rpm = _make_benchmark_env(enable_3g_payload=True)
    try:
        unloaded_result = _run_open_loop(unloaded, unloaded.HOVER_RPM)
        loaded_old_result = _run_open_loop(loaded_old_rpm, unloaded.HOVER_RPM)
        loaded_new_result = _run_open_loop(loaded_new_rpm, loaded_new_rpm.HOVER_RPM)

        assert unloaded_result["finite"]
        assert abs(unloaded_result["delta_z_1"]) < 1e-4
        assert abs(unloaded_result["final_vz"]) < 1e-4

        expected_loaded_acceleration = loaded_old_rpm.G * (
            unloaded.M / loaded_old_rpm.M - 1
        )
        assert loaded_old_result["finite"]
        assert loaded_old_result["first_qacc_z"] == pytest.approx(
            expected_loaded_acceleration, rel=0.0, abs=2e-3
        )
        assert loaded_old_result["delta_z_02"] < -0.015

        assert loaded_new_result["finite"]
        assert abs(loaded_new_result["delta_z_1"]) < 1e-4
        assert abs(loaded_new_result["final_vz"]) < 1e-4
    finally:
        loaded_new_rpm.close()
        loaded_old_rpm.close()
        unloaded.close()
