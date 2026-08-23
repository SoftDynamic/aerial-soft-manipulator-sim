"""Tests for the scaled StableSoft passive topology and tendon routing."""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.components.passive_arm import PassiveArmSpec
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
    build_scaled_stablesoft_arm_mjcf,
    solid_cylinder_diagonal_inertia,
)
from multi_drone_mujoco.models.mjcf_builder import build_aviary_xml
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


TOTAL_LENGTH = 0.08
TOTAL_MASS = 0.003
BODY_RADIUS = 0.003
MOUNT_Z = -0.015
CORNERS = {
    "xp_yp": (1.0, 1.0),
    "xp_yn": (1.0, -1.0),
    "xn_yp": (-1.0, 1.0),
    "xn_yn": (-1.0, -1.0),
}


def _spec(**overrides):
    kwargs = {
        "num_segments": 9,
        "total_length": TOTAL_LENGTH,
        "total_mass": TOTAL_MASS,
        "body_radius": BODY_RADIUS,
        "tendon_range": None,
        "tendon_frictionloss": 0.0,
        "joint_stiffness": 0.0,
        "joint_damping": 0.0,
        "target_damping_ratio": None,
    }
    kwargs.update(overrides)
    return ScaledStableSoftArmSpec(**kwargs)


def _legacy_passive_arm_spec():
    return PassiveArmSpec(
        total_length=0.08,
        radius=0.003,
        total_mass=0.003,
        num_segments=4,
        initial_bend_x=0.0,
        initial_bend_y=0.0,
        bend_stiffness_x=2e-4,
        bend_stiffness_y=2e-4,
        bend_damping_x=2.5e-7,
        bend_damping_y=2.5e-7,
        max_bend_x=0.7,
        max_bend_y=0.7,
    )


def _build_xml(spec=None, **overrides):
    kwargs = {
        "num_drones": 1,
        "drone_model": DroneModel.CF2X,
        "init_xyzs": np.array([[0.0, 0.0, 1.0]]),
        "init_rpys": np.zeros((1, 3)),
        "scaled_stablesoft_arm": _spec() if spec is None else spec,
    }
    kwargs.update(overrides)
    return build_aviary_xml(**kwargs)


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
        "scaled_stablesoft_arm": _spec(),
    }
    kwargs.update(overrides)
    return BaseAviary(**kwargs)


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


def _tendon_id(model, name):
    return _name_id(model, mujoco.mjtObj.mjOBJ_TENDON, name)


def _without_mount_visuals(xml, manifest):
    root = ET.fromstring(xml)
    drone_body = root.find(".//body[@name='drone0']")
    assert drone_body is not None
    adapter = drone_body.find(
        f"geom[@name='{manifest.arm_mount_adapter_geom_name}']"
    )
    mount_site = drone_body.find(
        f"site[@name='{manifest.arm_mount_site_name}']"
    )
    root_body = drone_body.find(
        f"body[@name='{manifest.segment_body_names[0]}']"
    )
    assert adapter is not None
    assert mount_site is not None
    assert root_body is not None
    root_site = root_body.find(
        f"site[@name='{manifest.arm_root_site_name}']"
    )
    assert root_site is not None
    drone_body.remove(adapter)
    drone_body.remove(mount_site)
    root_body.remove(root_site)
    return ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("num_segments", 8, "must equal 9"),
        ("num_segments", 10, "must equal 9"),
        ("num_segments", 9.0, "must be an integer"),
        ("num_segments", True, "must be an integer"),
        ("total_length", 0.0, "total_length must be positive"),
        ("total_length", -0.1, "total_length must be positive"),
        ("total_length", True, "must be a real number"),
        ("total_length", np.nan, "must be finite"),
        ("total_mass", 0.0, "total_mass must be positive"),
        ("total_mass", False, "must be a real number"),
        ("total_mass", np.inf, "must be finite"),
        ("body_radius", -0.001, "body_radius must be positive"),
        ("body_radius", True, "must be a real number"),
        ("body_radius", np.nan, "must be finite"),
        ("tendon_frictionloss", -0.1, "must be non-negative"),
        ("tendon_frictionloss", False, "must be a real number"),
        ("tendon_frictionloss", np.inf, "must be finite"),
    ],
)
def test_scaled_stablesoft_spec_rejects_invalid_parameters(
    field, value, match
):
    kwargs = _spec().__dict__.copy()
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        ScaledStableSoftArmSpec(**kwargs)


@pytest.mark.parametrize(
    "tendon_range,match",
    [
        ((0.0,), "must contain two values"),
        ((False, 0.2), "must be a real number"),
        ((0.0, True), "must be a real number"),
        ((0.0, np.inf), "must be finite"),
        ((-0.01, 0.2), "0 <= lower < upper"),
        ((0.2, 0.1), "0 <= lower < upper"),
        ((0.1, 0.2), "initial tendon length"),
        ((0.0, 0.01), "initial tendon length"),
    ],
)
def test_scaled_stablesoft_spec_rejects_invalid_tendon_range(
    tendon_range, match
):
    with pytest.raises(ValueError, match=match):
        _spec(tendon_range=tendon_range)


def test_scaled_stablesoft_spec_normalizes_numeric_inputs_to_floats():
    spec = _spec(
        total_length=np.float32(TOTAL_LENGTH),
        total_mass=np.float64(TOTAL_MASS),
        body_radius=np.float32(BODY_RADIUS),
        tendon_frictionloss=np.int64(0),
        tendon_range=[np.int64(0), np.float32(0.2)],
    )

    assert type(spec.total_length) is float
    assert type(spec.total_mass) is float
    assert type(spec.body_radius) is float
    assert type(spec.tendon_frictionloss) is float
    assert isinstance(spec.tendon_range, tuple)
    assert all(type(endpoint) is float for endpoint in spec.tendon_range)


def test_scaled_stablesoft_passive_parameters_preserve_gate_a_defaults():
    spec = ScaledStableSoftArmSpec()

    assert spec.joint_stiffness == 0.0
    assert spec.joint_damping == 0.0
    assert spec.target_damping_ratio is None
    assert spec.resolved_joint_stiffness == (0.0,) * 8
    assert spec.resolved_joint_damping == (0.0,) * 8


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("joint_stiffness", True, "joint_stiffness.*not bool"),
        ("joint_stiffness", -1.0, "joint_stiffness must be non-negative"),
        ("joint_stiffness", np.nan, "joint_stiffness must be finite"),
        ("joint_stiffness", np.inf, "joint_stiffness must be finite"),
        ("joint_stiffness", 1 + 2j, "joint_stiffness must be"),
        ("joint_stiffness", [0.0] * 7, "sequence must have length 8"),
        ("joint_stiffness", [0.0] * 7 + [-1.0], "values must be non-negative"),
        ("joint_damping", False, "joint_damping.*not bool"),
        ("joint_damping", -1.0, "joint_damping must be non-negative"),
        ("joint_damping", np.nan, "joint_damping must be finite"),
        ("joint_damping", np.inf, "joint_damping must be finite"),
        ("joint_damping", 1 + 2j, "joint_damping must be a real number"),
        ("joint_damping", [0.0] * 9, "sequence must have length 8"),
        ("joint_damping", [0.0] * 7 + [np.inf], r"joint_damping\[7\].*finite"),
        ("target_damping_ratio", True, "target_damping_ratio.*not bool"),
        ("target_damping_ratio", -0.1, "target_damping_ratio must be non-negative"),
        ("target_damping_ratio", np.nan, "target_damping_ratio must be finite"),
        ("target_damping_ratio", np.inf, "target_damping_ratio must be finite"),
        (
            "target_damping_ratio",
            1 + 2j,
            "target_damping_ratio must be a real number",
        ),
    ],
)
def test_scaled_stablesoft_spec_rejects_invalid_passive_parameters(
    field, value, match
):
    kwargs = _spec().__dict__.copy()
    kwargs[field] = value
    if field == "target_damping_ratio":
        kwargs["joint_damping"] = None
    with pytest.raises(ValueError, match=match):
        ScaledStableSoftArmSpec(**kwargs)


def test_scaled_stablesoft_spec_rejects_ambiguous_damping_modes():
    with pytest.raises(ValueError, match="cannot both be set"):
        _spec(joint_damping=0.0, target_damping_ratio=0.3)


def test_scaled_stablesoft_spec_resolves_unspecified_damping_to_zero():
    spec = _spec(joint_damping=None, target_damping_ratio=None)

    assert spec.joint_damping is None
    assert spec.resolved_joint_damping == (0.0,) * 8


def test_scaled_stablesoft_spec_copies_and_normalizes_joint_sequences():
    stiffness = [0.001 * (index + 1) for index in range(8)]
    damping = np.arange(8, dtype=float) * 1e-6
    spec = _spec(
        joint_stiffness=stiffness,
        joint_damping=damping,
    )
    expected_stiffness = tuple(stiffness)
    expected_damping = tuple(float(value) for value in damping)

    stiffness[:] = [0.0] * 8
    damping[:] = 1.0

    assert spec.joint_stiffness == expected_stiffness
    assert spec.joint_damping == expected_damping
    assert spec.resolved_joint_stiffness == expected_stiffness
    assert spec.resolved_joint_damping == expected_damping
    assert all(type(value) is float for value in spec.joint_stiffness)
    assert all(type(value) is float for value in spec.joint_damping)


def test_scaled_stablesoft_derives_b1_damping_from_downstream_inertia():
    stiffness = [0.003086] * 8
    spec = _spec(
        joint_stiffness=stiffness,
        joint_damping=None,
        target_damping_ratio=0.3,
    )
    expected_damping = tuple(
        2 * 0.3 * math.sqrt(k * inertia)
        for k, inertia in zip(stiffness, spec.bending_effective_inertias)
    )

    assert spec.resolved_joint_damping == pytest.approx(expected_damping)
    assert spec.resolved_joint_damping == pytest.approx(
        (
            7.071313274573578e-05,
            5.788962718019289e-05,
            4.595333125392906e-05,
            3.497608602784423e-05,
            2.5050809424837078e-05,
            1.6304549262910235e-05,
            8.927077439170014e-06,
            3.2536908197641157e-06,
        ),
        rel=1e-12,
    )


def test_scaled_stablesoft_rejects_derived_damping_underflow_and_overflow():
    with pytest.raises(ValueError, match="damping product.*underflowed"):
        _spec(
            joint_stiffness=np.nextafter(0.0, 1.0),
            joint_damping=None,
            target_damping_ratio=0.3,
        )
    with pytest.raises(ValueError, match="derived joint damping.*finite"):
        _spec(
            joint_stiffness=1.0,
            joint_damping=None,
            target_damping_ratio=np.finfo(float).max,
        )


def test_scaled_stablesoft_spec_copies_mutable_tendon_range_input():
    source_range = [0.0, 0.2]
    spec = _spec(tendon_range=source_range)
    expected_range = spec.tendon_range
    expected_effective_range = spec.effective_tendon_range

    source_range[:] = [0.1, 0.11]

    assert spec.tendon_range == expected_range == (0.0, 0.2)
    assert spec.effective_tendon_range == expected_effective_range == (0.0, 0.2)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"total_length": np.finfo(float).max},
            "derived quantities overflowed",
        ),
        (
            {"body_radius": np.finfo(float).max},
            "derived quantities overflowed",
        ),
        (
            {"total_mass": np.finfo(float).max, "body_radius": 10.0},
            "solid-cylinder inertia must be finite",
        ),
    ],
)
def test_scaled_stablesoft_spec_rejects_derived_overflow(overrides, match):
    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"total_length": np.nextafter(0.0, 1.0)},
            "derived segment_pitch must be positive",
        ),
        (
            {"total_mass": np.nextafter(0.0, 1.0)},
            "derived segment_mass must be positive",
        ),
        (
            {"body_radius": np.nextafter(0.0, 1.0)},
            "solid-cylinder inertia must be positive",
        ),
    ],
)
def test_scaled_stablesoft_spec_rejects_derived_underflow(overrides, match):
    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


def test_scaled_stablesoft_spec_rejects_degenerate_inertia_triangle():
    with pytest.raises(ValueError, match="strict triangle inequalities"):
        _spec(total_length=1e-8, body_radius=1.0)


def test_scaled_stablesoft_spec_derives_scaled_geometry_and_legacy_range():
    spec = _spec()
    scale = TOTAL_LENGTH / (9 * 0.096)

    assert spec.num_ball_joints == 8
    assert spec.segment_pitch == pytest.approx(TOTAL_LENGTH / 9)
    assert spec.segment_mass == pytest.approx(TOTAL_MASS / 9)
    assert spec.tendon_site_half_width == pytest.approx(0.035 * scale)
    assert spec.proximal_site_offset == pytest.approx(0.008 * scale)
    assert spec.distal_site_offset == pytest.approx(0.088 * scale)
    assert spec.initial_tendon_length == pytest.approx(0.848 * scale)
    assert spec.effective_tendon_range == pytest.approx((0.0, 2.0 * scale))
    assert spec.tendon_width == pytest.approx(
        max(0.0004 * scale, 0.05 * BODY_RADIUS)
    )
    assert spec.routing_site_size == pytest.approx(
        max(0.001 * scale, 0.08 * BODY_RADIUS)
    )
    assert spec.collision_enabled is False
    assert spec.tip_payload_mass == pytest.approx(0.0)

    doubled = _spec(total_length=2 * TOTAL_LENGTH)
    assert doubled.effective_tendon_range[1] == pytest.approx(4.0 * scale)
    assert doubled.initial_tendon_length == pytest.approx(
        2 * spec.initial_tendon_length
    )

    wider = _spec(body_radius=0.01)
    assert wider.tendon_width == pytest.approx(0.05 * wider.body_radius)
    assert wider.routing_site_size == pytest.approx(0.08 * wider.body_radius)


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
        (
            {"passive_arm": _legacy_passive_arm_spec()},
            "cannot be combined with the passive arm",
        ),
    ],
)
def test_builder_rejects_unsupported_scaled_stablesoft_combinations(
    overrides, match
):
    with pytest.raises(ValueError, match=match):
        _build_xml(**overrides)


@pytest.mark.parametrize(
    "integrator,match",
    [
        (None, "must be a string"),
        (False, "must be a string"),
        ("", "must not be empty"),
        ("Euler", "must be one of"),
        ("rk4", "must be one of"),
    ],
)
def test_builder_rejects_invalid_integrator(integrator, match):
    with pytest.raises(ValueError, match=match):
        _build_xml(integrator=integrator)


@pytest.mark.parametrize(
    "integrator,expected",
    [
        ("RK4", mujoco.mjtIntegrator.mjINT_RK4),
        ("implicitfast", mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
    ],
)
def test_builder_compiles_explicit_integrator(integrator, expected):
    xml = _build_xml(integrator=integrator)
    model = mujoco.MjModel.from_xml_string(xml)

    assert f'integrator="{integrator}"' in xml
    assert model.opt.integrator == expected


def test_builder_default_integrator_remains_rk4():
    model = mujoco.MjModel.from_xml_string(_build_xml())

    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4


def test_environment_default_integrator_remains_rk4():
    env = _make_env()
    try:
        assert env.INTEGRATOR == "RK4"
        assert env.model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        assert env.model.opt.timestep == pytest.approx(1.0 / 240)
    finally:
        env.close()


def test_environment_passes_explicit_integrator_and_timestep_to_builder():
    env = _make_env(
        sim_freq=2400,
        ctrl_freq=240,
        integrator="implicitfast",
    )
    try:
        assert env.INTEGRATOR == "implicitfast"
        assert (
            env.model.opt.integrator
            == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        )
        assert env.model.opt.timestep == pytest.approx(1.0 / 2400)
        assert env.model.nu == 0
        assert env.model.ntendon == 4
    finally:
        env.close()


def test_environment_leaves_integrator_validation_to_builder():
    with pytest.raises(ValueError, match="integrator must be one of"):
        _make_env(integrator="Euler")


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"drone_model": DroneModel.CF2P},
            "only supported for DroneModel.CF2X",
        ),
        (
            {"num_drones": 2},
            "requires num_drones == 1",
        ),
        ({"physics": Physics.DYN}, "only supported with Physics.MJC"),
        (
            {"act_type": ActionType.VEL},
            "only supported with ActionType.RPM",
        ),
        (
            {"enable_3g_payload": True},
            "cannot be combined with the 3 g rigid payload",
        ),
        (
            {"passive_arm": _legacy_passive_arm_spec()},
            "cannot be combined with the passive arm",
        ),
    ],
)
def test_environment_rejects_unsupported_scaled_stablesoft_combinations(
    overrides, match
):
    with pytest.raises(ValueError, match=match):
        _make_env(**overrides)


def test_scaled_stablesoft_fragments_have_stable_manifest_and_empty_actuator():
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", _spec(), MOUNT_Z)
    manifest = fragments.manifest

    assert len(manifest.segment_body_names) == 9
    assert len(manifest.ball_joint_names) == 8
    assert len(manifest.routing_site_names) == 72
    assert len(manifest.tendon_names) == 4
    assert manifest.tendon_actuator_names == ()
    assert len(manifest.tendon_sensor_names) == 4
    assert len(manifest.tendon_site_routes) == 4
    assert all(len(route) == 18 for route in manifest.tendon_site_routes)
    assert len(set(manifest.routing_site_names)) == 72
    assert len(set(manifest.tendon_names)) == 4
    assert manifest.segment_body_names[0].endswith("segment_00")
    assert manifest.segment_body_names[-1].endswith("segment_08")
    assert manifest.ball_joint_names[0].endswith("segment_01_ball_joint")
    assert manifest.arm_mount_adapter_geom_name == (
        "drone0_arm_mount_adapter_geom"
    )
    assert manifest.arm_mount_site_name == "drone0_arm_mount_site"
    assert manifest.arm_root_site_name == (
        "drone0_scaled_stablesoft_arm_root_site"
    )
    assert manifest.arm_tip_site_name == (
        "drone0_scaled_stablesoft_arm_tip_site"
    )
    assert fragments.actuator_xml == ""
    assert "<muscle" not in fragments.worldbody_xml + fragments.tendon_xml
    assert "<motor" not in fragments.worldbody_xml + fragments.tendon_xml


def test_scaled_stablesoft_xml_compiles_with_expected_counts():
    xml = _build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    xml_root = ET.fromstring(xml)
    actuator = xml_root.find("actuator")

    assert model.nbody == 11
    assert model.njnt == 9
    assert model.nq == 39
    assert model.nv == 30
    assert model.ngeom == 19
    assert model.nsite == 80
    assert model.ntendon == 4
    assert model.nu == 0
    assert model.nsensor == 10
    assert actuator is None or len(actuator) == 0
    assert "<actuator>" not in xml
    assert "<muscle" not in xml
    assert "<motor" not in xml
    assert "mocap=" not in xml
    assert 'name="J0"' not in xml


def test_scaled_stablesoft_named_body_joint_geometry_and_tip():
    spec = _spec()
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z)
    model = mujoco.MjModel.from_xml_string(_build_xml(spec))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    drone_id = _body_id(model, "drone0")

    xml_root = ET.fromstring(_build_xml(spec))
    adapter_element = xml_root.find(
        ".//body[@name='drone0']/geom"
        f"[@name='{fragments.manifest.arm_mount_adapter_geom_name}']"
    )
    assert adapter_element is not None
    assert adapter_element.attrib["fromto"] == "0 0 0 0 0 -0.014999999999999999"
    assert float(adapter_element.attrib["size"]) == pytest.approx(
        spec.body_radius
    )
    assert adapter_element.attrib["contype"] == "0"
    assert adapter_element.attrib["conaffinity"] == "0"

    adapter_id = _geom_id(
        model, fragments.manifest.arm_mount_adapter_geom_name
    )
    assert int(model.geom_bodyid[adapter_id]) == drone_id
    assert model.geom_type[adapter_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert int(model.geom_contype[adapter_id]) == 0
    assert int(model.geom_conaffinity[adapter_id]) == 0

    mount_site_id = _site_id(model, fragments.manifest.arm_mount_site_name)
    root_site_id = _site_id(model, fragments.manifest.arm_root_site_name)
    assert int(model.site_bodyid[mount_site_id]) == drone_id
    np.testing.assert_allclose(
        model.site_pos[mount_site_id], [0.0, 0.0, MOUNT_Z], atol=1e-12
    )

    for segment_index, body_name in enumerate(
        fragments.manifest.segment_body_names
    ):
        body_id = _body_id(model, body_name)
        parent_id = (
            drone_id
            if segment_index == 0
            else _body_id(
                model, fragments.manifest.segment_body_names[segment_index - 1]
            )
        )
        assert int(model.body_parentid[body_id]) == parent_id
        expected_z = MOUNT_Z if segment_index == 0 else -spec.segment_pitch
        np.testing.assert_allclose(
            model.body_pos[body_id], [0.0, 0.0, expected_z], atol=1e-12
        )
        np.testing.assert_allclose(
            model.body_ipos[body_id],
            [0.0, 0.0, -spec.segment_pitch / 2],
            atol=1e-12,
        )
        assert int(model.body_jntnum[body_id]) == (0 if segment_index == 0 else 1)

        if segment_index > 0:
            joint_id = _joint_id(model, f"{body_name}_ball_joint")
            assert int(model.jnt_bodyid[joint_id]) == body_id
            assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL
            assert model.jnt_stiffness[joint_id] == pytest.approx(0.0)
            dof_address = int(model.jnt_dofadr[joint_id])
            np.testing.assert_allclose(
                model.dof_damping[dof_address:dof_address + 3], 0.0
            )
            np.testing.assert_allclose(
                model.dof_frictionloss[dof_address:dof_address + 3], 0.0
            )

        geom_id = _geom_id(model, f"{body_name}_geom")
        assert int(model.geom_bodyid[geom_id]) == body_id
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        np.testing.assert_allclose(
            model.geom_pos[geom_id],
            [0.0, 0.0, -spec.segment_pitch / 2],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model.geom_size[geom_id, :2],
            [spec.body_radius, spec.segment_pitch / 2],
            atol=1e-12,
        )

    tip_id = _site_id(model, fragments.manifest.arm_tip_site_name)
    final_body_id = _body_id(model, fragments.manifest.segment_body_names[-1])
    assert int(model.site_bodyid[tip_id]) == final_body_id
    np.testing.assert_allclose(
        model.site_pos[tip_id], [0.0, 0.0, -spec.segment_pitch], atol=1e-12
    )
    root_body_id = _body_id(model, fragments.manifest.segment_body_names[0])
    assert int(model.site_bodyid[root_site_id]) == root_body_id
    np.testing.assert_allclose(
        model.site_pos[root_site_id], [0.0, 0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        data.site_xpos[mount_site_id],
        data.site_xpos[root_site_id],
        rtol=0.0,
        atol=1e-12,
    )
    assert np.linalg.norm(
        data.site_xpos[tip_id] - data.xpos[root_body_id]
    ) == pytest.approx(spec.total_length, abs=1e-12)
    for segment_index, body_name in enumerate(
        fragments.manifest.segment_body_names
    ):
        body_id = _body_id(model, body_name)
        np.testing.assert_allclose(
            data.xpos[body_id],
            [0.0, 0.0, 1.0 + MOUNT_Z - segment_index * spec.segment_pitch],
            atol=1e-12,
        )


def test_scaled_stablesoft_compiles_per_joint_stiffness_and_damping():
    stiffness = tuple(0.001 * (index + 1) for index in range(8))
    damping = tuple(1e-6 * (index + 1) for index in range(8))
    spec = _spec(joint_stiffness=stiffness, joint_damping=damping)
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z)
    model = mujoco.MjModel.from_xml_string(_build_xml(spec))

    for index, joint_name in enumerate(fragments.manifest.ball_joint_names):
        joint_id = _joint_id(model, joint_name)
        dof_address = int(model.jnt_dofadr[joint_id])
        assert model.jnt_stiffness[joint_id] == pytest.approx(stiffness[index])
        np.testing.assert_allclose(
            model.dof_damping[dof_address:dof_address + 3],
            damping[index],
            rtol=1e-12,
            atol=0.0,
        )


def test_scaled_stablesoft_mass_and_cylinder_inertia():
    spec = _spec()
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z)
    model = mujoco.MjModel.from_xml_string(_build_xml(spec))
    expected_inertia = solid_cylinder_diagonal_inertia(
        spec.segment_mass, spec.body_radius, spec.segment_pitch
    )

    masses = []
    for body_name in fragments.manifest.segment_body_names:
        body_id = _body_id(model, body_name)
        masses.append(float(model.body_mass[body_id]))
        inertia = np.asarray(model.body_inertia[body_id])
        assert model.body_mass[body_id] == pytest.approx(spec.segment_mass)
        assert inertia.tolist() == pytest.approx(expected_inertia, abs=1e-15)
        assert np.all(np.isfinite(inertia))
        assert np.all(inertia > 0)
        for axis in range(3):
            other_axes = [index for index in range(3) if index != axis]
            assert inertia[axis] < sum(inertia[index] for index in other_axes)
        assert not np.allclose(
            inertia, [0.0003, 0.0003, 0.0004], rtol=0.0, atol=1e-12
        )

    assert math.isclose(
        math.fsum(masses), spec.total_mass, rel_tol=0.0, abs_tol=1e-15
    )
    drone_id = _body_id(model, "drone0")
    assert model.body_subtreemass[drone_id] == pytest.approx(0.030, abs=1e-12)


def test_mount_visuals_preserve_mass_inertia_com_length_and_tendons():
    spec = _spec()
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z)
    with_visuals = mujoco.MjModel.from_xml_string(_build_xml(spec))
    without_visuals = mujoco.MjModel.from_xml_string(
        _without_mount_visuals(_build_xml(spec), fragments.manifest)
    )
    with_data = mujoco.MjData(with_visuals)
    without_data = mujoco.MjData(without_visuals)
    mujoco.mj_forward(with_visuals, with_data)
    mujoco.mj_forward(without_visuals, without_data)

    np.testing.assert_allclose(
        with_visuals.body_mass,
        without_visuals.body_mass,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        with_visuals.body_subtreemass,
        without_visuals.body_subtreemass,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        with_visuals.body_inertia,
        without_visuals.body_inertia,
        rtol=1e-12,
        atol=1e-15,
    )

    with_drone_id = _body_id(with_visuals, "drone0")
    without_drone_id = _body_id(without_visuals, "drone0")
    np.testing.assert_allclose(
        with_data.subtree_com[with_drone_id],
        without_data.subtree_com[without_drone_id],
        rtol=0.0,
        atol=1e-12,
    )

    with_root_id = _site_id(with_visuals, fragments.manifest.arm_root_site_name)
    with_tip_id = _site_id(with_visuals, fragments.manifest.arm_tip_site_name)
    without_root_body_id = _body_id(
        without_visuals, fragments.manifest.segment_body_names[0]
    )
    without_tip_id = _site_id(
        without_visuals, fragments.manifest.arm_tip_site_name
    )
    assert np.linalg.norm(
        with_data.site_xpos[with_tip_id] - with_data.site_xpos[with_root_id]
    ) == pytest.approx(spec.total_length, abs=1e-12)
    assert np.linalg.norm(
        without_data.site_xpos[without_tip_id]
        - without_data.xpos[without_root_body_id]
    ) == pytest.approx(spec.total_length, abs=1e-12)

    np.testing.assert_allclose(
        with_data.ten_length,
        without_data.ten_length,
        rtol=1e-10,
        atol=1e-12,
    )


def test_scaled_stablesoft_routing_sites_match_scaled_contract():
    spec = _spec()
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z)
    model = mujoco.MjModel.from_xml_string(_build_xml(spec))
    manifest = fragments.manifest

    assert len(set(manifest.routing_site_names)) == 72
    for body_name in manifest.segment_body_names:
        body_id = _body_id(model, body_name)
        body_site_names = []
        for corner_name, (x_sign, y_sign) in CORNERS.items():
            for position_name, offset in (
                ("proximal", spec.proximal_site_offset),
                ("distal", spec.distal_site_offset),
            ):
                site_name = f"{body_name}_{corner_name}_{position_name}_site"
                site_id = _site_id(model, site_name)
                body_site_names.append(site_name)
                assert int(model.site_bodyid[site_id]) == body_id
                np.testing.assert_allclose(
                    model.site_pos[site_id],
                    [
                        x_sign * spec.tendon_site_half_width,
                        y_sign * spec.tendon_site_half_width,
                        -offset,
                    ],
                    atol=1e-12,
                )
        assert len(body_site_names) == 8
        assert set(body_site_names).issubset(manifest.routing_site_names)
    assert manifest.arm_tip_site_name not in manifest.routing_site_names


def test_scaled_stablesoft_spatial_tendons_route_all_segments_in_order():
    spec = _spec()
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z)
    xml = _build_xml(spec)
    root = ET.fromstring(xml)
    model = mujoco.MjModel.from_xml_string(xml)

    assert len(set(fragments.manifest.tendon_site_routes)) == 4
    tendon_element = root.find("tendon")
    assert tendon_element is not None
    for tendon_name, expected_route in zip(
        fragments.manifest.tendon_names,
        fragments.manifest.tendon_site_routes,
    ):
        spatial = tendon_element.find(f"spatial[@name='{tendon_name}']")
        assert spatial is not None
        assert tuple(
            site_element.attrib["site"] for site_element in spatial.findall("site")
        ) == expected_route

        tendon_id = _tendon_id(model, tendon_name)
        wrap_address = int(model.tendon_adr[tendon_id])
        wrap_count = int(model.tendon_num[tendon_id])
        assert wrap_count == 18
        compiled_route = []
        for wrap_index in range(wrap_address, wrap_address + wrap_count):
            assert model.wrap_type[wrap_index] == mujoco.mjtWrap.mjWRAP_SITE
            compiled_route.append(
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    int(model.wrap_objid[wrap_index]),
                )
            )
        assert tuple(compiled_route) == expected_route


def test_scaled_stablesoft_initial_tendon_lengths_and_limits():
    spec = _spec()
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z)
    model = mujoco.MjModel.from_xml_string(_build_xml(spec))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scale = spec.total_length / (9 * 0.096)

    assert spec.initial_tendon_length == pytest.approx(0.848 * scale)
    for tendon_name in fragments.manifest.tendon_names:
        tendon_id = _tendon_id(model, tendon_name)
        length = float(data.ten_length[tendon_id])
        lower, upper = model.tendon_range[tendon_id]
        assert math.isfinite(length)
        assert length == pytest.approx(spec.initial_tendon_length, abs=1e-12)
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(2.0 * scale)
        assert lower <= length <= upper
        assert bool(model.tendon_limited[tendon_id])


def test_scaled_stablesoft_arm_geoms_are_noncolliding():
    spec = _spec()
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z)
    model = mujoco.MjModel.from_xml_string(_build_xml(spec))

    for body_name in fragments.manifest.segment_body_names:
        geom_id = _geom_id(model, f"{body_name}_geom")
        assert int(model.geom_contype[geom_id]) == 0
        assert int(model.geom_conaffinity[geom_id]) == 0
    floor_id = _geom_id(model, "floor")
    drone_geom_id = _geom_id(model, "drone0_collision")
    assert int(model.geom_contype[floor_id]) == 1
    assert int(model.geom_conaffinity[floor_id]) == 1
    assert int(model.geom_contype[drone_geom_id]) == 1
    assert int(model.geom_conaffinity[drone_geom_id]) == 1


def test_scaled_stablesoft_tendon_position_sensors_bind_by_name():
    spec = _spec()
    fragments = build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z)
    model = mujoco.MjModel.from_xml_string(_build_xml(spec))

    for sensor_name, tendon_name in zip(
        fragments.manifest.tendon_sensor_names,
        fragments.manifest.tendon_names,
    ):
        sensor_id = _name_id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        tendon_id = _tendon_id(model, tendon_name)
        assert model.sensor_type[sensor_id] == mujoco.mjtSensor.mjSENS_TENDONPOS
        assert model.sensor_objtype[sensor_id] == mujoco.mjtObj.mjOBJ_TENDON
        assert int(model.sensor_objid[sensor_id]) == tendon_id


def test_scaled_stablesoft_environment_mass_and_default_height():
    unloaded = BaseAviary(initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    armed = _make_env()
    default_position = _make_env(initial_xyzs=None)
    explicit_position = _make_env(
        initial_xyzs=np.array([[0.0, 0.0, 0.5]])
    )
    try:
        assert armed.ARM_MASS == pytest.approx(TOTAL_MASS)
        assert armed.M == pytest.approx(0.030)
        assert armed.GRAVITY == pytest.approx(armed.G * armed.M)
        assert armed.HOVER_RPM == pytest.approx(
            np.sqrt(armed.M * armed.G / (4 * armed.KF))
        )
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
        assert default_position.INIT_XYZS[0, 2] == pytest.approx(
            0.115 + TOTAL_LENGTH
        )
        np.testing.assert_allclose(
            explicit_position.INIT_XYZS[0], [0.0, 0.0, 0.5]
        )
    finally:
        explicit_position.close()
        default_position.close()
        armed.close()
        unloaded.close()


def test_scaled_stablesoft_none_preserves_baseline_model():
    xml = build_aviary_xml(
        num_drones=1,
        drone_model=DroneModel.CF2X,
        init_xyzs=np.array([[0.0, 0.0, 1.0]]),
        init_rpys=np.zeros((1, 3)),
        scaled_stablesoft_arm=None,
    )
    model = mujoco.MjModel.from_xml_string(xml)

    assert model.nq == 7
    assert model.nv == 6
    assert model.nbody == 2
    assert model.njnt == 1
    assert model.ngeom == 9
    assert model.nsite == 5
    assert model.ntendon == 0
    assert model.nu == 0
    assert "scaled_stablesoft_arm" not in xml
    assert "<tendon>" not in xml
    assert "<actuator>" not in xml
