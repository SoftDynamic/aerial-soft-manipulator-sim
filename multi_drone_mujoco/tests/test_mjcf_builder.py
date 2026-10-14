"""Focused tests for aviary MJCF construction."""

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.mjcf_builder import build_aviary_mjcf
from multi_drone_mujoco.models.parameters import DRONE_PARAMS
from multi_drone_mujoco.utils.enums import DroneModel


@pytest.mark.parametrize("drone_model", list(DroneModel))
def test_builder_compiles_supported_drone_models(drone_model):
    timestep = 1 / 480
    model = mujoco.MjModel.from_xml_string(build_aviary_mjcf(
        num_drones=1,
        drone_model=drone_model,
        init_xyzs=np.array([[0.0, 0.0, 1.0]]),
        init_rpys=np.zeros((1, 3)),
        timestep=timestep,
    ).xml)

    drone_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "drone0"
    )
    params = DRONE_PARAMS[drone_model]
    assert model.body_mass[drone_id] == pytest.approx(params["mass"])
    np.testing.assert_allclose(
        model.body_inertia[drone_id],
        [params["ixx"], params["iyy"], params["izz"]],
    )
    assert model.opt.timestep == pytest.approx(timestep)


def test_builder_preserves_multi_drone_scene_features_and_initial_poses():
    init_xyzs = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]])
    init_rpys = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]])
    model = mujoco.MjModel.from_xml_string(build_aviary_mjcf(
        num_drones=2,
        drone_model=DroneModel.CF2X,
        init_xyzs=init_xyzs,
        init_rpys=init_rpys,
        obstacles=True,
        vision=True,
    ).xml)

    expected_quats = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
    ])
    for drone_id in range(2):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{drone_id}_joint"
        )
        qpos_address = model.jnt_qposadr[joint_id]
        np.testing.assert_allclose(
            model.qpos0[qpos_address:qpos_address + 3], init_xyzs[drone_id]
        )
        np.testing.assert_allclose(
            model.qpos0[qpos_address + 3:qpos_address + 7],
            expected_quats[drone_id],
        )
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, f"drone{drone_id}_cam"
        ) >= 0

    for body_name in ("obstacle_box", "obstacle_sphere", "obstacle_cylinder"):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        ) >= 0


def test_builder_compiles_3g_payload_with_expected_parent():
    model = mujoco.MjModel.from_xml_string(build_aviary_mjcf(
        num_drones=1,
        drone_model=DroneModel.CF2X,
        init_xyzs=np.array([[0.0, 0.0, 1.0]]),
        init_rpys=np.zeros((1, 3)),
        enable_3g_payload=True,
    ).xml)

    drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
    payload_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "drone0_payload"
    )
    assert payload_id >= 0
    assert model.body_parentid[payload_id] == drone_id
    assert model.body_jntnum[payload_id] == 0


def test_save_generated_xml_writes_the_compiled_model(tmp_path):
    env = BaseAviary()
    output_path = tmp_path / "nested" / "aviary.xml"
    try:
        assert list(tmp_path.iterdir()) == []

        saved_path = env.save_generated_xml(output_path)

        assert saved_path == output_path
        assert saved_path.read_text(encoding="utf-8") == env._generated_xml
        reloaded_model = mujoco.MjModel.from_xml_path(str(saved_path))
        assert reloaded_model.nq == env.model.nq
    finally:
        env.close()
