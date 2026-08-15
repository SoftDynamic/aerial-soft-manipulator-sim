"""Tests for the single-link passive arm physics benchmark."""

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.components.passive_arm import PassiveArmSpec
from multi_drone_mujoco.models.mjcf_builder import build_aviary_xml
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


ARM_LENGTH = 0.08
ARM_RADIUS = 0.003
ARM_MASS = 0.003
ARM_DAMPING = 1e-5


def _arm_spec(initial_angle=0.0):
    return PassiveArmSpec(
        length=ARM_LENGTH,
        radius=ARM_RADIUS,
        mass=ARM_MASS,
        initial_angle=initial_angle,
        joint_damping=ARM_DAMPING,
    )


def _make_env(initial_angle=0.0, **overrides):
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
        "passive_arm": _arm_spec(initial_angle),
    }
    kwargs.update(overrides)
    return BaseAviary(**kwargs)


def _model_ids(model):
    drone_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "drone0"
    )
    arm_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "drone0_passive_arm"
    )
    arm_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_passive_arm_joint"
    )
    arm_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "drone0_passive_arm_geom"
    )
    return drone_body_id, arm_body_id, arm_joint_id, arm_geom_id


def _pitch_from_quaternion(quaternion):
    w, x, y, z = quaternion
    return np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))


def _assert_dynamic_values_are_finite(env, rpm):
    motor_forces = np.asarray(rpm) ** 2 * env.KF
    for values in (
        env.data.qpos,
        env.data.qvel,
        env.data.qacc,
        rpm,
        motor_forces,
        env.data.xfrc_applied,
        env.data.qfrc_applied,
    ):
        assert np.all(np.isfinite(values))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("length", 0.0, "length must be positive"),
        ("radius", -0.01, "radius must be positive"),
        ("mass", np.inf, "parameters must be finite"),
        ("joint_damping", -1.0, "damping must be non-negative"),
    ],
)
def test_passive_arm_spec_rejects_invalid_parameters(field, value, match):
    kwargs = {
        "length": ARM_LENGTH,
        "radius": ARM_RADIUS,
        "mass": ARM_MASS,
        "initial_angle": 0.0,
        "joint_damping": ARM_DAMPING,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        PassiveArmSpec(**kwargs)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"drone_model": DroneModel.CF2P},
            "only supported for DroneModel.CF2X",
        ),
        (
            {"drone_model": DroneModel.RACE},
            "only supported for DroneModel.CF2X",
        ),
        (
            {
                "num_drones": 2,
                "init_xyzs": np.zeros((2, 3)),
                "init_rpys": np.zeros((2, 3)),
            },
            "requires num_drones == 1",
        ),
        (
            {"enable_3g_payload": True},
            "cannot be combined with the 3 g rigid payload",
        ),
    ],
)
def test_builder_rejects_unsupported_passive_arm_combinations(overrides, match):
    kwargs = {
        "num_drones": 1,
        "drone_model": DroneModel.CF2X,
        "init_xyzs": np.zeros((1, 3)),
        "init_rpys": np.zeros((1, 3)),
        "passive_arm": _arm_spec(),
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        build_aviary_xml(**kwargs)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"physics": Physics.DYN}, "only supported with Physics.MJC"),
        ({"act_type": ActionType.VEL}, "only supported with ActionType.RPM"),
        (
            {"enable_3g_payload": True},
            "cannot be combined with the 3 g rigid payload",
        ),
    ],
)
def test_environment_rejects_unsupported_passive_arm_combinations(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_env(**overrides)


def test_passive_arm_model_mass_inertia_hardware_and_reset():
    initial_angle = 0.35
    unloaded = BaseAviary(initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    armed = _make_env(initial_angle)
    try:
        model = armed.model
        drone_id, arm_id, joint_id, geom_id = _model_ids(model)
        joint_qpos_address = int(model.jnt_qposadr[joint_id])
        joint_dof_address = int(model.jnt_dofadr[joint_id])

        assert model.nq == 8
        assert model.nv == 7
        assert model.nbody == 3
        assert model.njnt == 2
        assert model.ngeom == 10
        assert model.nu == 0
        assert int(model.body_parentid[arm_id]) == drone_id
        assert int(model.jnt_bodyid[joint_id]) == arm_id
        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        np.testing.assert_allclose(
            model.jnt_axis[joint_id], [0.0, 1.0, 0.0], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            model.body_pos[arm_id], [0.0, 0.0, -0.015], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            model.body_ipos[arm_id],
            [0.0, 0.0, -ARM_LENGTH / 2],
            rtol=0.0,
            atol=1e-12,
        )

        transverse_inertia = ARM_MASS * (
            3 * ARM_RADIUS ** 2 + ARM_LENGTH ** 2
        ) / 12
        axial_inertia = 0.5 * ARM_MASS * ARM_RADIUS ** 2
        assert model.body_mass[arm_id] == pytest.approx(
            ARM_MASS, rel=0.0, abs=1e-12
        )
        np.testing.assert_allclose(
            model.body_inertia[arm_id],
            [transverse_inertia, transverse_inertia, axial_inertia],
            rtol=0.0,
            atol=1e-12,
        )
        assert int(model.geom_bodyid[geom_id]) == arm_id
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        np.testing.assert_allclose(
            model.geom_size[geom_id, :2],
            [ARM_RADIUS, ARM_LENGTH / 2],
            rtol=0.0,
            atol=1e-12,
        )
        assert model.dof_damping[joint_dof_address] == pytest.approx(
            ARM_DAMPING, rel=0.0, abs=1e-12
        )
        assert model.jnt_stiffness[joint_id] == 0.0
        assert model.dof_armature[joint_dof_address] == 0.0
        assert model.dof_frictionloss[joint_dof_address] == 0.0
        assert model.body_subtreemass[drone_id] == pytest.approx(
            armed.M, rel=0.0, abs=1e-12
        )

        assert armed.ARM_MASS == pytest.approx(ARM_MASS, rel=0.0, abs=1e-12)
        assert armed.M == pytest.approx(0.030, rel=0.0, abs=1e-12)
        assert armed.HOVER_RPM == pytest.approx(
            np.sqrt(armed.M * armed.G / (4 * armed.KF)), rel=1e-12
        )
        assert armed.HOVER_RPM < armed.MAX_RPM
        for attribute in (
            "MAX_RPM",
            "MAX_THRUST",
            "MAX_XY_TORQUE",
            "MAX_Z_TORQUE",
            "KF",
            "KM",
            "L",
        ):
            assert getattr(armed, attribute) == getattr(unloaded, attribute)
        np.testing.assert_array_equal(armed.J, unloaded.J)
        np.testing.assert_array_equal(armed.J_INV, unloaded.J_INV)

        assert model.qpos0[joint_qpos_address] == pytest.approx(
            initial_angle, rel=0.0, abs=1e-12
        )
        armed.data.qpos[joint_qpos_address] = -0.2
        armed.data.qvel[joint_dof_address] = 1.0
        armed.reset()
        assert armed.data.qpos[joint_qpos_address] == pytest.approx(
            initial_angle, rel=0.0, abs=1e-12
        )
        assert armed.data.qvel[joint_dof_address] == 0.0

        expected_quaternion = [
            np.cos(initial_angle / 2),
            0.0,
            np.sin(initial_angle / 2),
            0.0,
        ]
        np.testing.assert_allclose(
            armed.data.xquat[arm_id], expected_quaternion, rtol=0.0, atol=1e-12
        )
        expected_center = np.array([
            -np.sin(initial_angle) * ARM_LENGTH / 2,
            0.0,
            1.0 - 0.015 - np.cos(initial_angle) * ARM_LENGTH / 2,
        ])
        np.testing.assert_allclose(
            armed.data.xipos[arm_id], expected_center, rtol=0.0, atol=1e-12
        )
        assert armed.data.ncon == 0
    finally:
        armed.close()
        unloaded.close()


def test_passive_arm_default_height_and_explicit_position():
    default_position = _make_env(initial_xyzs=None)
    explicit_position = _make_env(
        initial_xyzs=np.array([[0.0, 0.0, 1.0]])
    )
    try:
        _, _, _, default_geom_id = _model_ids(default_position.model)
        arm_tip_z = (
            default_position.data.geom_xpos[default_geom_id, 2]
            - default_position.model.geom_size[default_geom_id, 1]
        )
        assert default_position.INIT_XYZS[0, 2] == pytest.approx(
            0.115 + ARM_LENGTH, rel=0.0, abs=1e-12
        )
        assert arm_tip_z == pytest.approx(0.1, rel=0.0, abs=1e-12)
        assert default_position.data.ncon == 0

        np.testing.assert_allclose(
            explicit_position.INIT_XYZS[0],
            [0.0, 0.0, 1.0],
            rtol=0.0,
            atol=1e-12,
        )
        explicit_position.reset()
        explicit_drone_joint_id = mujoco.mj_name2id(
            explicit_position.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "drone0_joint",
        )
        explicit_qpos_address = int(
            explicit_position.model.jnt_qposadr[explicit_drone_joint_id]
        )
        np.testing.assert_allclose(
            explicit_position.data.qpos[
                explicit_qpos_address:explicit_qpos_address + 3
            ],
            [0.0, 0.0, 1.0],
            rtol=0.0,
            atol=1e-12,
        )
    finally:
        explicit_position.close()
        default_position.close()


def test_passive_arm_static_equilibrium_with_compensated_hover_rpm():
    env = _make_env(initial_angle=0.0)
    try:
        env.reset()
        mujoco.mj_forward(env.model, env.data)
        drone_id, _, joint_id, _ = _model_ids(env.model)
        drone_joint_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint"
        )
        drone_qpos_address = int(env.model.jnt_qposadr[drone_joint_id])
        joint_qpos_address = int(env.model.jnt_qposadr[joint_id])
        initial_position = env.data.qpos[
            drone_qpos_address:drone_qpos_address + 3
        ].copy()
        rpm = np.full(4, env.HOVER_RPM)
        np.testing.assert_allclose(env._preprocessAction(rpm), rpm.reshape(1, 4))
        assert env.data.ncon == 0

        for _ in range(240):
            env.step(rpm)
            assert env.data.ncon == 0
            _assert_dynamic_values_are_finite(env, rpm)

        np.testing.assert_allclose(
            env.data.qpos[drone_qpos_address:drone_qpos_address + 3],
            initial_position,
            rtol=0.0,
            atol=1e-4,
        )
        np.testing.assert_allclose(env.data.qvel, 0.0, rtol=0.0, atol=1e-4)
        assert abs(_pitch_from_quaternion(env.data.xquat[drone_id])) < 1e-4
        assert abs(env.data.qpos[joint_qpos_address]) < 1e-4
    finally:
        env.close()


def test_passive_arm_release_disturbs_drone_without_contact():
    initial_angle = 0.35
    env = _make_env(initial_angle=initial_angle)
    try:
        env.reset()
        mujoco.mj_forward(env.model, env.data)
        drone_id, _, joint_id, _ = _model_ids(env.model)
        drone_joint_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint"
        )
        drone_qpos_address = int(env.model.jnt_qposadr[drone_joint_id])
        joint_qpos_address = int(env.model.jnt_qposadr[joint_id])
        initial_x = float(env.data.qpos[drone_qpos_address])
        rpm = np.full(4, env.HOVER_RPM)
        max_joint_delta = 0.0
        max_pitch = 0.0
        max_x_delta = 0.0
        assert env.data.ncon == 0

        for _ in range(120):
            env.step(rpm)
            assert env.data.ncon == 0
            _assert_dynamic_values_are_finite(env, rpm)
            max_joint_delta = max(
                max_joint_delta,
                abs(float(env.data.qpos[joint_qpos_address]) - initial_angle),
            )
            max_pitch = max(
                max_pitch,
                abs(float(_pitch_from_quaternion(env.data.xquat[drone_id]))),
            )
            max_x_delta = max(
                max_x_delta,
                abs(float(env.data.qpos[drone_qpos_address]) - initial_x),
            )

        assert max_joint_delta > 0.20
        assert max_pitch > 0.03
        assert max_x_delta > 5e-4
    finally:
        env.close()


def test_passive_arm_zero_gravity_reaction_is_opposite_and_contact_free():
    env = _make_env(initial_angle=0.0)
    try:
        env.reset()
        env.model.opt.gravity[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        drone_id, _, arm_joint_id, _ = _model_ids(env.model)
        drone_joint_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint"
        )
        drone_qpos_address = int(env.model.jnt_qposadr[drone_joint_id])
        drone_dof_address = int(env.model.jnt_dofadr[drone_joint_id])
        arm_qpos_address = int(env.model.jnt_qposadr[arm_joint_id])
        arm_dof_address = int(env.model.jnt_dofadr[arm_joint_id])
        initial_hinge_angle = float(env.data.qpos[arm_qpos_address])
        initial_pitch = float(_pitch_from_quaternion(env.data.xquat[drone_id]))
        rpm = np.zeros(4)
        assert env.data.ncon == 0

        for step in range(120):
            env.data.qfrc_applied[:] = 0.0
            env.data.xfrc_applied[:] = 0.0
            if step < 12:
                env.data.qfrc_applied[arm_dof_address] = 2e-5

            _assert_dynamic_values_are_finite(env, rpm)
            mujoco.mj_step(env.model, env.data)
            assert env.data.ncon == 0
            _assert_dynamic_values_are_finite(env, rpm)
            assert np.all(np.isfinite(
                env.data.qpos[drone_qpos_address:drone_qpos_address + 7]
            ))
            assert np.all(np.isfinite(
                env.data.qvel[drone_dof_address:drone_dof_address + 6]
            ))
            assert np.all(np.isfinite(env.data.xquat[drone_id]))

        hinge_delta = (
            float(env.data.qpos[arm_qpos_address]) - initial_hinge_angle
        )
        pitch_delta = (
            float(_pitch_from_quaternion(env.data.xquat[drone_id]))
            - initial_pitch
        )
        assert abs(hinge_delta) > 0.05
        assert abs(pitch_delta) > 0.01
        assert hinge_delta * pitch_delta < 0.0
    finally:
        env.close()
