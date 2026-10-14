"""C1-B1 tests for production ideal-tension UAV-arm coupling."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import multi_drone_mujoco.envs.base_aviary as base_aviary_module
from multi_drone_mujoco.control.ideal_tendon_tension import (
    IdealTendonActuatorBinding,
    write_ideal_tendon_tension_command,
)
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)
from multi_drone_mujoco.models.mjcf_builder import (
    build_aviary_mjcf,
)
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


SIM_FREQ = 2400
CTRL_FREQ = 240
MAX_TENSION_N = 0.01


def _arm_spec():
    return ScaledStableSoftArmSpec(
        joint_stiffness=0.003086,
        joint_damping=None,
        target_damping_ratio=0.3,
    )


def _drive_spec():
    return IdealTendonTensionSpec(max_tension=MAX_TENSION_N)


def _builder_kwargs():
    return {
        "num_drones": 1,
        "drone_model": DroneModel.CF2X,
        "init_xyzs": np.array([[0.0, 0.0, 1.0]]),
        "init_rpys": np.zeros((1, 3)),
        "timestep": 1 / SIM_FREQ,
        "integrator": "implicitfast",
        "scaled_stablesoft_arm": _arm_spec(),
    }


def _make_env(*, active: bool, sim_freq: int = SIM_FREQ):
    return BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.MJC,
        sim_freq=sim_freq,
        ctrl_freq=CTRL_FREQ,
        gui=False,
        record=False,
        obstacles=False,
        vision_attributes=False,
        act_type=ActionType.RPM,
        render_mode=None,
        scaled_stablesoft_arm=_arm_spec(),
        scaled_stablesoft_tendon_drive=(
            _drive_spec() if active else None
        ),
        integrator="implicitfast",
    )


def _object_names(model, object_type, count):
    return tuple(
        mujoco.mj_id2name(model, object_type, index)
        for index in range(count)
    )


def _quaternion_geodesic(q_a, q_b):
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    q_a /= np.linalg.norm(q_a)
    q_b /= np.linalg.norm(q_b)
    dot = float(np.clip(abs(np.dot(q_a, q_b)), 0.0, 1.0))
    return 2.0 * np.arccos(dot)


def test_passive_builder_interface_and_xml_are_unchanged():
    kwargs = _builder_kwargs()
    build = build_aviary_mjcf(**kwargs)

    assert build_aviary_mjcf(
        **kwargs,
        scaled_stablesoft_tendon_drive=None,
    ).xml == build.xml
    assert build.scaled_stablesoft_arm_manifest is not None
    assert not build.scaled_stablesoft_arm_manifest.tendon_actuator_names

    model = mujoco.MjModel.from_xml_string(build.xml)
    assert model.ntendon == 4
    assert model.nu == 0
    assert model.na == 0


def test_drive_without_scaled_arm_is_rejected():
    kwargs = _builder_kwargs()
    kwargs.pop("scaled_stablesoft_arm")
    with pytest.raises(ValueError, match="requires a scaled StableSoft arm"):
        build_aviary_mjcf(
            **kwargs,
            scaled_stablesoft_tendon_drive=_drive_spec(),
        )
    with pytest.raises(ValueError, match="requires a scaled StableSoft arm"):
        BaseAviary(scaled_stablesoft_tendon_drive=_drive_spec())


def test_base_uses_manifest_from_its_single_production_build(monkeypatch):
    real_builder = base_aviary_module.build_aviary_mjcf
    builds = []

    def recording_builder(*args, **kwargs):
        result = real_builder(*args, **kwargs)
        builds.append(result)
        return result

    monkeypatch.setattr(
        base_aviary_module,
        "build_aviary_mjcf",
        recording_builder,
    )
    env = _make_env(active=True)
    try:
        assert len(builds) == 1
        assert env.SCALED_STABLESOFT_ARM_MANIFEST is (
            builds[0].scaled_stablesoft_arm_manifest
        )
    finally:
        env.close()


def test_active_model_manifest_transmissions_and_runtime_bindings():
    env = _make_env(active=True)
    try:
        model = env.model
        manifest = env.SCALED_STABLESOFT_ARM_MANIFEST
        bindings = env.SCALED_STABLESOFT_TENDON_BINDINGS

        assert model.ntendon == 4
        assert model.nu == 4
        assert model.na == 0
        assert len(manifest.tendon_names) == 4
        assert len(manifest.tendon_actuator_names) == 4
        assert len(bindings) == 4

        for tendon_name, actuator_name, binding in zip(
            manifest.tendon_names,
            manifest.tendon_actuator_names,
            bindings,
        ):
            tendon_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_TENDON,
                tendon_name,
            )
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            assert tendon_id >= 0
            assert actuator_id >= 0
            assert int(model.actuator_trnid[actuator_id, 0]) == tendon_id
            assert int(model.actuator_ctrlnum[actuator_id]) == 1
            assert binding.tendon_id == tendon_id
            assert binding.actuator_id == actuator_id
            assert binding.control_address == int(
                model.actuator_ctrladr[actuator_id]
            )
    finally:
        env.close()


def test_runtime_writer_uses_control_addresses_not_actuator_ids():
    addresses = (3, 1, 0, 2)
    bindings = tuple(
        IdealTendonActuatorBinding(
            tendon_name=f"tendon_{index}",
            actuator_name=f"actuator_{index}",
            tendon_id=index,
            actuator_id=index,
            control_address=address,
            output_address=index,
        )
        for index, address in enumerate(addresses)
    )
    data = SimpleNamespace(ctrl=np.zeros(4))
    command = (0.001, 0.002, 0.003, 0.004)

    values = write_ideal_tendon_tension_command(
        data,
        bindings,
        command,
        _drive_spec(),
    )

    assert values == command
    np.testing.assert_allclose(
        data.ctrl,
        np.array([0.003, 0.002, 0.004, 0.001]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "command",
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-0.001, 0.0, 0.0, 0.0),
        (0.011, 0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 0.0),
        (float("inf"), 0.0, 0.0, 0.0),
        (True, 0.0, 0.0, 0.0),
        True,
    ],
)
def test_public_setter_rejects_invalid_commands(command):
    env = _make_env(active=True)
    try:
        with pytest.raises(ValueError):
            env.set_scaled_stablesoft_tension_command(command)
    finally:
        env.close()


def test_public_setter_writes_valid_command_and_rpm_action_stays_separate():
    passive = _make_env(active=False, sim_freq=CTRL_FREQ)
    active = _make_env(active=True, sim_freq=CTRL_FREQ)
    try:
        np.testing.assert_array_equal(active.action_space.low, passive.action_space.low)
        np.testing.assert_array_equal(active.action_space.high, passive.action_space.high)
        assert active.action_space.shape == (4,)

        command = (0.001, 0.002, 0.003, 0.004)
        assert active.set_scaled_stablesoft_tension_command(command) == command
        for value, binding in zip(
            command,
            active.SCALED_STABLESOFT_TENDON_BINDINGS,
        ):
            assert active.data.ctrl[binding.control_address] == value

        rotor_rpm = np.full(4, 0.5 * active.HOVER_RPM)
        active.step(rotor_rpm)
        np.testing.assert_allclose(
            active.last_clipped_action[0],
            rotor_rpm,
            rtol=0.0,
            atol=0.0,
        )
        for value, binding in zip(
            command,
            active.SCALED_STABLESOFT_TENDON_BINDINGS,
        ):
            assert active.data.ctrl[binding.control_address] == value
    finally:
        passive.close()
        active.close()


def test_reset_zeros_cached_command_and_actuator_controls():
    env = _make_env(active=True)
    try:
        env.set_scaled_stablesoft_tension_command(
            (0.001, 0.002, 0.003, 0.004)
        )
        env.reset()

        assert env._scaled_stablesoft_tension_command_n == (0.0,) * 4
        for binding in env.SCALED_STABLESOFT_TENDON_BINDINGS:
            assert env.data.ctrl[binding.control_address] == 0.0
    finally:
        env.close()


def test_active_and_passive_physical_model_semantics_match():
    passive = _make_env(active=False)
    active = _make_env(active=True)
    try:
        p_model = passive.model
        a_model = active.model
        count_fields = (
            "nbody",
            "njnt",
            "nq",
            "nv",
            "ngeom",
            "nsite",
            "ntendon",
            "nsensor",
            "nmesh",
        )
        for field in count_fields:
            assert getattr(a_model, field) == getattr(p_model, field)

        name_specs = (
            (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
            (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
            (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
            (mujoco.mjtObj.mjOBJ_TENDON, "ntendon"),
            (mujoco.mjtObj.mjOBJ_SENSOR, "nsensor"),
        )
        for object_type, count_field in name_specs:
            count = getattr(a_model, count_field)
            assert _object_names(a_model, object_type, count) == (
                _object_names(p_model, object_type, count)
            )

        physical_arrays = (
            "body_mass",
            "body_inertia",
            "jnt_type",
            "jnt_bodyid",
            "jnt_pos",
            "jnt_axis",
            "jnt_range",
            "jnt_stiffness",
            "site_bodyid",
            "site_type",
            "site_pos",
            "site_quat",
            "site_size",
            "tendon_range",
            "tendon_adr",
            "tendon_num",
            "tendon_stiffness",
            "tendon_damping",
            "tendon_frictionloss",
            "wrap_type",
            "wrap_objid",
            "wrap_prm",
            "sensor_type",
            "sensor_datatype",
            "sensor_needstage",
            "sensor_objtype",
            "sensor_objid",
            "sensor_dim",
            "geom_type",
            "geom_dataid",
            "geom_bodyid",
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_group",
            "geom_contype",
            "geom_conaffinity",
            "mesh_vertadr",
            "mesh_vertnum",
            "mesh_faceadr",
            "mesh_facenum",
            "mesh_scale",
            "mesh_vert",
            "mesh_face",
        )
        for field in physical_arrays:
            np.testing.assert_array_equal(
                getattr(a_model, field),
                getattr(p_model, field),
                err_msg=field,
            )
    finally:
        passive.close()
        active.close()


def test_zero_tension_active_and_passive_hover_are_equivalent_for_one_second():
    passive = _make_env(active=False)
    active = _make_env(active=True)
    try:
        passive.reset()
        active.reset()
        active.set_scaled_stablesoft_tension_command((0.0,) * 4)

        p_manifest = passive.SCALED_STABLESOFT_ARM_MANIFEST
        a_manifest = active.SCALED_STABLESOFT_ARM_MANIFEST
        p_drone_joint = mujoco.mj_name2id(
            passive.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "drone0_joint",
        )
        a_drone_joint = mujoco.mj_name2id(
            active.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "drone0_joint",
        )
        p_qpos = int(passive.model.jnt_qposadr[p_drone_joint])
        a_qpos = int(active.model.jnt_qposadr[a_drone_joint])
        p_tip = mujoco.mj_name2id(
            passive.model,
            mujoco.mjtObj.mjOBJ_SITE,
            p_manifest.arm_tip_site_name,
        )
        a_tip = mujoco.mj_name2id(
            active.model,
            mujoco.mjtObj.mjOBJ_SITE,
            a_manifest.arm_tip_site_name,
        )
        rotor_rpm = np.full(4, passive.HOVER_RPM)

        max_position_error = 0.0
        max_orientation_error = 0.0
        max_velocity_error = 0.0
        max_tip_error = 0.0
        max_tendon_error = 0.0
        max_energy_error = 0.0
        for _ in range(CTRL_FREQ):
            passive.step(rotor_rpm)
            active.step(rotor_rpm)
            mujoco.mj_energyPos(passive.model, passive.data)
            mujoco.mj_energyVel(passive.model, passive.data)
            mujoco.mj_energyPos(active.model, active.data)
            mujoco.mj_energyVel(active.model, active.data)

            max_position_error = max(
                max_position_error,
                float(np.linalg.norm(
                    passive.data.qpos[p_qpos:p_qpos + 3]
                    - active.data.qpos[a_qpos:a_qpos + 3]
                )),
            )
            max_orientation_error = max(
                max_orientation_error,
                _quaternion_geodesic(
                    passive.data.qpos[p_qpos + 3:p_qpos + 7],
                    active.data.qpos[a_qpos + 3:a_qpos + 7],
                ),
            )
            max_velocity_error = max(
                max_velocity_error,
                float(np.max(np.abs(passive.data.qvel - active.data.qvel))),
            )
            max_tip_error = max(
                max_tip_error,
                float(np.linalg.norm(
                    passive.data.site_xpos[p_tip]
                    - active.data.site_xpos[a_tip]
                )),
            )
            max_tendon_error = max(
                max_tendon_error,
                float(np.max(np.abs(
                    passive.data.ten_length - active.data.ten_length
                ))),
            )
            max_energy_error = max(
                max_energy_error,
                float(np.max(np.abs(passive.data.energy - active.data.energy))),
            )
            np.testing.assert_allclose(
                active.data.actuator_force,
                0.0,
                rtol=0.0,
                atol=1.0e-15,
            )
            np.testing.assert_allclose(
                active.data.qfrc_actuator,
                0.0,
                rtol=0.0,
                atol=1.0e-15,
            )

        assert max_position_error <= 1.0e-10
        assert max_orientation_error <= 1.0e-10
        assert max_velocity_error <= 1.0e-9
        assert max_tip_error <= 1.0e-10
        assert max_tendon_error <= 1.0e-10
        assert max_energy_error <= 1.0e-10
    finally:
        passive.close()
        active.close()
