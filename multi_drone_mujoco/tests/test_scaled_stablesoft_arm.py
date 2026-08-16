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
    assert len(manifest.tendon_sensor_names) == 4
    assert len(manifest.tendon_site_routes) == 4
    assert all(len(route) == 18 for route in manifest.tendon_site_routes)
    assert len(set(manifest.routing_site_names)) == 72
    assert len(set(manifest.tendon_names)) == 4
    assert manifest.segment_body_names[0].endswith("segment_00")
    assert manifest.segment_body_names[-1].endswith("segment_08")
    assert manifest.ball_joint_names[0].endswith("segment_01_ball_joint")
    assert manifest.arm_tip_site_name == (
        "drone0_scaled_stablesoft_arm_tip_site"
    )
    assert fragments.actuator_xml == ""
    assert "<muscle" not in fragments.worldbody_xml + fragments.tendon_xml
    assert "<motor" not in fragments.worldbody_xml + fragments.tendon_xml


def test_scaled_stablesoft_xml_compiles_with_expected_counts():
    xml = _build_xml()
    model = mujoco.MjModel.from_xml_string(xml)

    assert model.nbody == 11
    assert model.njnt == 9
    assert model.nq == 39
    assert model.nv == 30
    assert model.ngeom == 18
    assert model.nsite == 78
    assert model.ntendon == 4
    assert model.nu == 0
    assert model.nsensor == 10
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
