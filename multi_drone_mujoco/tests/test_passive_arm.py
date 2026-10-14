"""Tests for the multi-segment two-direction passive flexible arm."""

import math

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.components.passive_arm import PassiveArmSpec
from multi_drone_mujoco.models.mjcf_builder import build_aviary_mjcf
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


ARM_LENGTH = 0.08
ARM_RADIUS = 0.003
ARM_MASS = 0.003
NUM_SEGMENTS = 4
INITIAL_BEND_X = 0.24
INITIAL_BEND_Y = -0.16
BEND_STIFFNESS_X = 2e-4
BEND_STIFFNESS_Y = 3e-4
BEND_DAMPING_X = 2.5e-7
BEND_DAMPING_Y = 3.5e-7
MAX_BEND_X = 0.70
MAX_BEND_Y = 0.60
MOUNT_Z = -0.015


def _arm_spec(**overrides):
    kwargs = {
        "total_length": ARM_LENGTH,
        "radius": ARM_RADIUS,
        "total_mass": ARM_MASS,
        "num_segments": NUM_SEGMENTS,
        "initial_bend_x": 0.0,
        "initial_bend_y": 0.0,
        "bend_stiffness_x": BEND_STIFFNESS_X,
        "bend_stiffness_y": BEND_STIFFNESS_Y,
        "bend_damping_x": BEND_DAMPING_X,
        "bend_damping_y": BEND_DAMPING_Y,
        "max_bend_x": MAX_BEND_X,
        "max_bend_y": MAX_BEND_Y,
    }
    kwargs.update(overrides)
    return PassiveArmSpec(**kwargs)


def _make_env(**overrides):
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
        "passive_arm": _arm_spec(),
    }
    kwargs.update(overrides)
    return BaseAviary(**kwargs)


def _segment_name(segment_index):
    return f"drone0_passive_arm_segment_{segment_index:02d}"


def _joint_name(segment_index, axis):
    return f"{_segment_name(segment_index)}_bend_{axis}_joint"


def _name_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, name
    return object_id


def _body_id(model, name):
    return _name_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _joint_id(model, name):
    return _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _geom_id(model, name):
    return _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _site_id(model, name):
    return _name_id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def _joint_ids(model, axis, num_segments=NUM_SEGMENTS):
    return [
        _joint_id(model, _joint_name(segment_index, axis))
        for segment_index in range(num_segments)
    ]


def _qpos_addresses(model, axis, num_segments=NUM_SEGMENTS):
    return np.array([
        int(model.jnt_qposadr[joint_id])
        for joint_id in _joint_ids(model, axis, num_segments)
    ])


def _dof_addresses(model, axis, num_segments=NUM_SEGMENTS):
    return np.array([
        int(model.jnt_dofadr[joint_id])
        for joint_id in _joint_ids(model, axis, num_segments)
    ])


def _quaternion_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def _quaternion_to_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _reference_segment_quaternion(initial_bend_x, initial_bend_y, num_segments):
    bend_x = initial_bend_x / num_segments
    bend_y = initial_bend_y / num_segments
    bend_x_quaternion = np.array([
        math.cos(bend_x / 2), 0.0, -math.sin(bend_x / 2), 0.0
    ])
    bend_y_quaternion = np.array([
        math.cos(bend_y / 2), math.sin(bend_y / 2), 0.0, 0.0
    ])
    return _quaternion_multiply(bend_y_quaternion, bend_x_quaternion)


def _rpy_from_quaternion(quaternion):
    w, x, y, z = quaternion
    return np.array([
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ])


def _assert_dynamic_values_are_finite(env, rpm):
    for values in (
        env.data.qpos,
        env.data.qvel,
        env.data.qacc,
        rpm,
        np.asarray(rpm) ** 2 * env.KF,
        env.data.xfrc_applied,
        env.data.qfrc_applied,
    ):
        assert np.all(np.isfinite(values))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("total_length", 0.0, "total_length must be positive"),
        ("radius", -0.01, "radius must be positive"),
        ("total_mass", np.inf, "parameters must be finite"),
        ("num_segments", 0, "num_segments must be at least 1"),
        ("num_segments", 1.5, "num_segments must be an integer"),
        ("num_segments", True, "num_segments must be an integer"),
        ("bend_stiffness_x", -1.0, "stiffness must be non-negative"),
        ("bend_stiffness_y", -1.0, "stiffness must be non-negative"),
        ("bend_damping_x", -1.0, "damping must be non-negative"),
        ("bend_damping_y", -1.0, "damping must be non-negative"),
        ("max_bend_x", 0.0, "max bend must be positive"),
        ("max_bend_y", -0.1, "max bend must be positive"),
        ("initial_bend_x", MAX_BEND_X + 0.01, "within max_bend_x"),
        ("initial_bend_y", -(MAX_BEND_Y + 0.01), "within max_bend_y"),
    ],
)
def test_passive_arm_spec_rejects_invalid_parameters(field, value, match):
    kwargs = _arm_spec().__dict__.copy()
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        PassiveArmSpec(**kwargs)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"drone_model": DroneModel.CF2P}, "only supported for DroneModel.CF2X"),
        ({"drone_model": DroneModel.RACE}, "only supported for DroneModel.CF2X"),
        (
            {
                "num_drones": 2,
                "init_xyzs": np.zeros((2, 3)),
                "init_rpys": np.zeros((2, 3)),
            },
            "requires num_drones == 1",
        ),
        ({"enable_3g_payload": True}, "cannot be combined with the 3 g rigid payload"),
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
        build_aviary_mjcf(**kwargs).xml


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"physics": Physics.DYN}, "only supported with Physics.MJC"),
        ({"act_type": ActionType.VEL}, "only supported with ActionType.RPM"),
        ({"enable_3g_payload": True}, "cannot be combined with the 3 g rigid payload"),
    ],
)
def test_environment_rejects_unsupported_passive_arm_combinations(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_env(**overrides)


def test_overweight_passive_arm_is_allowed_and_rpm_clipping_is_unchanged():
    heavy_spec = _arm_spec(total_mass=0.04)
    env = _make_env(passive_arm=heavy_spec)
    try:
        assert env.HOVER_RPM >= env.MAX_RPM
        clipped = env._preprocessAction(np.full(4, env.HOVER_RPM))
        np.testing.assert_allclose(clipped, env.MAX_RPM)
    finally:
        env.close()


@pytest.mark.parametrize("num_segments", [1, 4])
def test_passive_arm_structure_and_discrete_parameter_mapping(num_segments):
    spec = _arm_spec(
        num_segments=num_segments,
        initial_bend_x=INITIAL_BEND_X,
        initial_bend_y=INITIAL_BEND_Y,
    )
    model = mujoco.MjModel.from_xml_string(build_aviary_mjcf(
        num_drones=1,
        drone_model=DroneModel.CF2X,
        init_xyzs=np.array([[0.0, 0.0, 1.0]]),
        init_rpys=np.zeros((1, 3)),
        passive_arm=spec,
    ).xml)

    assert model.nq == 7 + 2 * num_segments
    assert model.nv == 6 + 2 * num_segments
    assert model.nbody == 2 + num_segments
    assert model.njnt == 1 + 2 * num_segments
    assert model.ngeom == 9 + num_segments
    assert model.nsite == 6
    assert model.nu == 0

    drone_id = _body_id(model, "drone0")
    segment_length = ARM_LENGTH / num_segments
    segment_mass = ARM_MASS / num_segments
    expected_transverse_inertia = segment_mass * (
        3 * ARM_RADIUS ** 2 + segment_length ** 2
    ) / 12
    expected_axial_inertia = 0.5 * segment_mass * ARM_RADIUS ** 2

    for segment_index in range(num_segments):
        body_name = _segment_name(segment_index)
        body_id = _body_id(model, body_name)
        expected_parent = (
            drone_id
            if segment_index == 0
            else _body_id(model, _segment_name(segment_index - 1))
        )
        assert int(model.body_parentid[body_id]) == expected_parent
        np.testing.assert_allclose(
            model.body_pos[body_id],
            [0.0, 0.0, MOUNT_Z if segment_index == 0 else -segment_length],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model.body_ipos[body_id],
            [0.0, 0.0, -segment_length / 2],
            rtol=0.0,
            atol=1e-12,
        )
        assert model.body_mass[body_id] == pytest.approx(segment_mass, abs=1e-12)
        np.testing.assert_allclose(
            model.body_inertia[body_id],
            [expected_transverse_inertia, expected_transverse_inertia,
             expected_axial_inertia],
            rtol=0.0,
            atol=1e-12,
        )

        x_joint_id = _joint_id(model, _joint_name(segment_index, "x"))
        y_joint_id = _joint_id(model, _joint_name(segment_index, "y"))
        assert int(model.jnt_bodyid[x_joint_id]) == body_id
        assert int(model.jnt_bodyid[y_joint_id]) == body_id
        assert y_joint_id == x_joint_id + 1
        np.testing.assert_allclose(model.jnt_axis[x_joint_id], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(model.jnt_axis[y_joint_id], [1.0, 0.0, 0.0])

        for axis, joint_id, initial_bend, stiffness, damping, max_bend in (
            ("x", x_joint_id, INITIAL_BEND_X, BEND_STIFFNESS_X,
             BEND_DAMPING_X, MAX_BEND_X),
            ("y", y_joint_id, INITIAL_BEND_Y, BEND_STIFFNESS_Y,
             BEND_DAMPING_Y, MAX_BEND_Y),
        ):
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
            assert model.qpos0[qpos_address] == pytest.approx(
                initial_bend / num_segments, abs=1e-12
            ), axis
            assert model.qpos_spring[qpos_address] == pytest.approx(0.0, abs=1e-12)
            assert model.jnt_stiffness[joint_id] == pytest.approx(
                num_segments * stiffness, abs=1e-12
            )
            assert model.dof_damping[dof_address] == pytest.approx(
                num_segments * damping, abs=1e-12
            )
            np.testing.assert_allclose(
                model.jnt_range[joint_id],
                [-max_bend / num_segments, max_bend / num_segments],
                rtol=0.0,
                atol=1e-12,
            )
            assert model.jnt_limited[joint_id]

        geom_id = _geom_id(model, f"{body_name}_geom")
        assert int(model.geom_bodyid[geom_id]) == body_id
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        np.testing.assert_allclose(
            model.geom_size[geom_id, :2],
            [ARM_RADIUS, segment_length / 2],
            rtol=0.0,
            atol=1e-12,
        )
        assert int(model.geom_contype[geom_id]) == 0
        assert int(model.geom_conaffinity[geom_id]) == 0

    tip_site_id = _site_id(model, "drone0_passive_arm_tip_site")
    assert int(model.site_bodyid[tip_site_id]) == _body_id(
        model, _segment_name(num_segments - 1)
    )
    np.testing.assert_allclose(
        model.site_pos[tip_site_id], [0.0, 0.0, -segment_length], atol=1e-12
    )


def test_passive_arm_mass_hardware_preload_geometry_and_reset():
    spec = _arm_spec(
        initial_bend_x=INITIAL_BEND_X,
        initial_bend_y=INITIAL_BEND_Y,
    )
    unloaded = BaseAviary(initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    armed = _make_env(passive_arm=spec)
    try:
        model = armed.model
        data = armed.data
        drone_id = _body_id(model, "drone0")
        x_qpos = _qpos_addresses(model, "x")
        y_qpos = _qpos_addresses(model, "y")
        x_dofs = _dof_addresses(model, "x")
        y_dofs = _dof_addresses(model, "y")

        assert armed.ARM_MASS == pytest.approx(ARM_MASS, abs=1e-12)
        assert armed.M == pytest.approx(0.030, abs=1e-12)
        assert model.body_subtreemass[drone_id] == pytest.approx(
            armed.M, abs=1e-12
        )
        assert sum(
            model.body_mass[_body_id(model, _segment_name(segment_index))]
            for segment_index in range(NUM_SEGMENTS)
        ) == pytest.approx(ARM_MASS, abs=1e-12)
        assert armed.HOVER_RPM == pytest.approx(
            np.sqrt(armed.M * armed.G / (4 * armed.KF)), rel=1e-12
        )
        assert armed.HOVER_RPM < armed.MAX_RPM
        for attribute in (
            "MAX_RPM", "MAX_THRUST", "MAX_XY_TORQUE", "MAX_Z_TORQUE",
            "KF", "KM", "L",
        ):
            assert getattr(armed, attribute) == getattr(unloaded, attribute)
        np.testing.assert_array_equal(armed.J, unloaded.J)
        np.testing.assert_array_equal(armed.J_INV, unloaded.J_INV)

        armed.reset()
        np.testing.assert_allclose(
            data.qpos[x_qpos], INITIAL_BEND_X / NUM_SEGMENTS, atol=1e-12
        )
        np.testing.assert_allclose(
            data.qpos[y_qpos], INITIAL_BEND_Y / NUM_SEGMENTS, atol=1e-12
        )

        segment_quaternion = _reference_segment_quaternion(
            INITIAL_BEND_X, INITIAL_BEND_Y, NUM_SEGMENTS
        )
        segment_rotation = _quaternion_to_matrix(segment_quaternion)
        cumulative_rotation = np.eye(3)
        expected_origin = np.array([0.0, 0.0, 1.0 + MOUNT_Z])
        segment_length = ARM_LENGTH / NUM_SEGMENTS
        for segment_index in range(NUM_SEGMENTS):
            body_id = _body_id(model, _segment_name(segment_index))
            cumulative_rotation = cumulative_rotation @ segment_rotation
            np.testing.assert_allclose(data.xpos[body_id], expected_origin, atol=1e-12)
            np.testing.assert_allclose(
                data.xmat[body_id].reshape(3, 3), cumulative_rotation, atol=1e-12
            )
            expected_origin = expected_origin + cumulative_rotation @ np.array([
                0.0, 0.0, -segment_length
            ])
        tip_site_id = _site_id(model, "drone0_passive_arm_tip_site")
        np.testing.assert_allclose(data.site_xpos[tip_site_id], expected_origin, atol=1e-12)

        data.qpos[x_qpos] = 0.0
        data.qpos[y_qpos] = 0.0
        mujoco.mj_forward(model, data)
        for segment_index in range(NUM_SEGMENTS):
            body_id = _body_id(model, _segment_name(segment_index))
            np.testing.assert_allclose(
                data.xmat[body_id].reshape(3, 3), np.eye(3), atol=1e-12
            )
            np.testing.assert_allclose(
                data.xpos[body_id],
                [0.0, 0.0, 1.0 + MOUNT_Z - segment_index * segment_length],
                atol=1e-12,
            )
        np.testing.assert_allclose(
            data.site_xpos[tip_site_id],
            [0.0, 0.0, 1.0 + MOUNT_Z - ARM_LENGTH],
            atol=1e-12,
        )

        data.qpos[x_qpos] = -0.1
        data.qpos[y_qpos] = 0.1
        data.qvel[x_dofs] = 1.0
        data.qvel[y_dofs] = -1.0
        armed.reset()
        np.testing.assert_allclose(
            data.qpos[x_qpos], INITIAL_BEND_X / NUM_SEGMENTS, atol=1e-12
        )
        np.testing.assert_allclose(
            data.qpos[y_qpos], INITIAL_BEND_Y / NUM_SEGMENTS, atol=1e-12
        )
        np.testing.assert_allclose(data.qvel[x_dofs], 0.0, atol=1e-12)
        np.testing.assert_allclose(data.qvel[y_dofs], 0.0, atol=1e-12)
        assert data.ncon == 0
    finally:
        armed.close()
        unloaded.close()


def test_passive_arm_default_height_is_conservative_and_explicit_position_is_preserved():
    bent_spec = _arm_spec(
        initial_bend_x=INITIAL_BEND_X,
        initial_bend_y=INITIAL_BEND_Y,
    )
    default_position = _make_env(initial_xyzs=None, passive_arm=bent_spec)
    explicit_position = _make_env(
        initial_xyzs=np.array([[0.0, 0.0, 0.5]]),
        passive_arm=bent_spec,
    )
    try:
        assert default_position.INIT_XYZS[0, 2] == pytest.approx(
            0.115 + ARM_LENGTH, abs=1e-12
        )
        np.testing.assert_allclose(
            explicit_position.INIT_XYZS[0], [0.0, 0.0, 0.5], atol=1e-12
        )
        explicit_position.reset()
        drone_joint_id = _joint_id(explicit_position.model, "drone0_joint")
        qpos_address = int(explicit_position.model.jnt_qposadr[drone_joint_id])
        np.testing.assert_allclose(
            explicit_position.data.qpos[qpos_address:qpos_address + 3],
            [0.0, 0.0, 0.5],
            atol=1e-12,
        )
    finally:
        default_position.close()
        explicit_position.close()


def test_passive_arm_none_preserves_baseline_model():
    model = mujoco.MjModel.from_xml_string(build_aviary_mjcf(
        num_drones=1,
        drone_model=DroneModel.CF2X,
        init_xyzs=np.array([[0.0, 0.0, 1.0]]),
        init_rpys=np.zeros((1, 3)),
        passive_arm=None,
    ).xml)
    assert model.nq == 7
    assert model.nv == 6
    assert model.nbody == 2
    assert model.njnt == 1
    assert model.ngeom == 9
    assert model.nsite == 5
    assert model.nu == 0
    floor_id = _geom_id(model, "floor")
    assert int(model.geom_contype[floor_id]) == 1
    assert int(model.geom_conaffinity[floor_id]) == 1
    assert mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, _segment_name(0)
    ) == -1


def test_passive_arm_static_equilibrium_with_compensated_hover_rpm():
    env = _make_env()
    try:
        env.reset()
        mujoco.mj_forward(env.model, env.data)
        drone_id = _body_id(env.model, "drone0")
        drone_joint_id = _joint_id(env.model, "drone0_joint")
        drone_qpos = int(env.model.jnt_qposadr[drone_joint_id])
        x_qpos = _qpos_addresses(env.model, "x")
        y_qpos = _qpos_addresses(env.model, "y")
        initial_position = env.data.qpos[drone_qpos:drone_qpos + 3].copy()
        rpm = np.full(4, env.HOVER_RPM)
        max_contacts = env.data.ncon

        for _ in range(240):
            env.data.qfrc_applied[:] = 0.0
            env.step(rpm)
            max_contacts = max(max_contacts, env.data.ncon)
            _assert_dynamic_values_are_finite(env, rpm)

        np.testing.assert_allclose(
            env.data.qpos[drone_qpos:drone_qpos + 3],
            initial_position,
            rtol=0.0,
            atol=1e-6,
        )
        assert np.max(np.abs(env.data.qvel)) < 1e-6
        assert np.max(np.abs(env.data.qpos[x_qpos])) < 1e-6
        assert np.max(np.abs(env.data.qpos[y_qpos])) < 1e-6
        assert np.max(np.abs(_rpy_from_quaternion(env.data.xquat[drone_id])[:2])) < 1e-6
        assert max_contacts == 0
    finally:
        env.close()


@pytest.mark.parametrize(
    "axis,target_rpy_index",
    [("x", 1), ("y", 0)],
)
def test_passive_arm_single_axis_pulse_reacts_on_drone(
    axis, target_rpy_index
):
    env = _make_env()
    try:
        env.reset()
        mujoco.mj_forward(env.model, env.data)
        drone_id = _body_id(env.model, "drone0")
        qpos_addresses = _qpos_addresses(env.model, axis)
        base_dof = int(_dof_addresses(env.model, axis)[0])
        rpm = np.full(4, env.HOVER_RPM)
        max_total_bend = 0.0
        max_target_attitude = 0.0
        max_cross_attitude = 0.0
        max_yaw = 0.0
        max_contacts = env.data.ncon

        for step in range(240):
            env.data.qfrc_applied[:] = 0.0
            if step < 12:
                env.data.qfrc_applied[base_dof] = 5e-5
            env.step(rpm)
            rpy = _rpy_from_quaternion(env.data.xquat[drone_id])
            total_bend = float(np.sum(env.data.qpos[qpos_addresses]))
            target_attitude = float(rpy[target_rpy_index])
            if abs(target_attitude) > max_target_attitude:
                max_target_attitude = abs(target_attitude)
            max_total_bend = max(max_total_bend, abs(total_bend))
            max_cross_attitude = max(
                max_cross_attitude, abs(float(rpy[1 - target_rpy_index]))
            )
            max_yaw = max(max_yaw, abs(float(rpy[2])))
            max_contacts = max(max_contacts, env.data.ncon)
            _assert_dynamic_values_are_finite(env, rpm)

        assert max_total_bend > 0.05
        assert max_target_attitude > 0.04
        assert max_cross_attitude < 1e-5
        assert max_yaw < 1e-5
        assert max_contacts == 0
    finally:
        env.close()


def test_passive_arm_positive_stiffness_moves_preload_toward_straight():
    env = _make_env(passive_arm=_arm_spec(initial_bend_x=0.20))
    try:
        env.reset()
        env.model.opt.gravity[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        qpos_addresses = _qpos_addresses(env.model, "x")
        initial_total_bend = float(np.sum(env.data.qpos[qpos_addresses]))
        minimum_absolute_bend = abs(initial_total_bend)

        for _ in range(60):
            env.data.qfrc_applied[:] = 0.0
            env.data.xfrc_applied[:] = 0.0
            mujoco.mj_step(env.model, env.data)
            mujoco.mj_forward(env.model, env.data)
            minimum_absolute_bend = min(
                minimum_absolute_bend,
                abs(float(np.sum(env.data.qpos[qpos_addresses]))),
            )
            _assert_dynamic_values_are_finite(env, np.zeros(4))

        assert initial_total_bend == pytest.approx(0.20, abs=1e-12)
        assert minimum_absolute_bend < 0.15
        assert env.data.ncon == 0
    finally:
        env.close()


def test_passive_arm_positive_damping_reduces_late_oscillation_amplitude():
    env = _make_env(passive_arm=_arm_spec(initial_bend_x=0.30))
    try:
        env.reset()
        env.model.opt.gravity[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        qpos_addresses = _qpos_addresses(env.model, "x")
        amplitudes = []

        for _ in range(480):
            env.data.qfrc_applied[:] = 0.0
            env.data.xfrc_applied[:] = 0.0
            mujoco.mj_step(env.model, env.data)
            mujoco.mj_forward(env.model, env.data)
            amplitudes.append(abs(float(np.sum(env.data.qpos[qpos_addresses]))))
            _assert_dynamic_values_are_finite(env, np.zeros(4))

        early_peak = max(amplitudes[:120])
        late_peak = max(amplitudes[-120:])
        assert early_peak > 0.20
        assert late_peak < 0.9 * early_peak
        assert env.data.ncon == 0
    finally:
        env.close()
