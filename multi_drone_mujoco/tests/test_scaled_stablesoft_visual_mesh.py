"""Visual-mesh structure and dynamics-equivalence acceptance tests."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ORIGINAL_STABLESOFT_SEGMENT_PITCH_M,
    STABLESOFT_VISUAL_MESH_NAME,
    STABLESOFT_VISUAL_MESH_PATH,
    ScaledStableSoftArmSpec,
    ScaledStableSoftVisualSpec,
    build_scaled_stablesoft_arm_mjcf,
)
from multi_drone_mujoco.models.mjcf_builder import build_aviary_xml
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    ActiveTendonRunConfig,
    TendonCommandRunConfig,
    run_single_tendon_characterization,
    run_tendon_command_characterization,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    build_fixed_arm_fixture,
    initialize_constant_curvature,
    joint_rotation_vectors,
    named_joint_addresses,
)
from multi_drone_mujoco.utils.enums import DroneModel


EXPECTED_MESH_SHA256 = (
    "e0b891b694589ea7378a08c174f1e736ce01905968fec2bd7547663604fa4db8"
)
MESH_VISUAL = ScaledStableSoftVisualSpec(mode="mesh")
CYLINDER_VISUAL = ScaledStableSoftVisualSpec(mode="cylinder")
ARM_SPEC = ScaledStableSoftArmSpec(
    joint_stiffness=0.003086,
    joint_damping=None,
    target_damping_ratio=0.3,
)
DRIVE_SPEC = IdealTendonTensionSpec(max_tension=0.01)
MOUNT_Z = -0.015
SIM_FREQ = 2400
CTRL_FREQ = 240
INTEGRATOR = "implicitfast"


def _fixture(visual, *, active=False):
    return build_fixed_arm_fixture(
        ARM_SPEC,
        "vertical",
        sim_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        integrator=INTEGRATOR,
        tendon_drive=DRIVE_SPEC if active else None,
        visual=visual,
    )


def _named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, name
    return int(object_id)


def _object_names(model, object_type, count):
    return tuple(
        mujoco.mj_id2name(model, object_type, object_id)
        for object_id in range(count)
    )


def _segment_geom_ids(fixture):
    return np.asarray(
        [
            _named_id(
                fixture.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{body_name}_geom",
            )
            for body_name in fixture.manifest.segment_body_names
        ],
        dtype=int,
    )


def _mesh_local_vertices(model, geom_id):
    mesh_id = int(model.geom_dataid[geom_id])
    vertex_address = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vertex_address:vertex_address + vertex_count],
        dtype=float,
    )
    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    return vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]


def _aviary_xml(visual=None):
    kwargs = {}
    if visual is not None:
        kwargs["scaled_stablesoft_visual"] = visual
    return build_aviary_xml(
        num_drones=1,
        drone_model=DroneModel.CF2X,
        init_xyzs=np.asarray([[0.0, 0.0, 1.0]]),
        init_rpys=np.zeros((1, 3)),
        scaled_stablesoft_arm=ARM_SPEC,
        integrator=INTEGRATOR,
        timestep=1.0 / SIM_FREQ,
        **kwargs,
    )


def _maximum_errors(left, right, fields):
    return {
        field: float(
            np.max(
                np.abs(
                    np.asarray(getattr(left, field))
                    - np.asarray(getattr(right, field))
                )
            )
        )
        for field in fields
    }


def test_visual_spec_defaults_to_mesh_and_validates_mode():
    assert ScaledStableSoftVisualSpec().mode == "mesh"
    assert CYLINDER_VISUAL.mode == "cylinder"
    for invalid in ("", "both", "Mesh", None, 1):
        with pytest.raises(ValueError, match="mesh.*cylinder"):
            ScaledStableSoftVisualSpec(mode=invalid)


def test_authorized_mesh_is_packaged_unchanged_and_source_is_portable():
    package_root = Path(__file__).resolve().parents[1]
    assert STABLESOFT_VISUAL_MESH_PATH.is_file()
    STABLESOFT_VISUAL_MESH_PATH.relative_to(package_root)
    digest = hashlib.sha256(STABLESOFT_VISUAL_MESH_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_MESH_SHA256

    forbidden_machine_path = "/" + "home/ustc/"
    production_sources = (
        package_root / "models" / "components" / "scaled_stablesoft_arm.py",
        package_root / "models" / "mjcf_builder.py",
        package_root / "envs" / "base_aviary.py",
    )
    for source in production_sources:
        assert forbidden_machine_path not in source.read_text(encoding="utf-8")


def test_default_aviary_assembles_one_mesh_asset_for_nine_segment_geoms():
    xml = _aviary_xml()
    root = ET.fromstring(xml)
    mesh_assets = root.findall(
        f"asset/mesh[@name='{STABLESOFT_VISUAL_MESH_NAME}']"
    )
    mesh_geoms = root.findall(
        f".//geom[@mesh='{STABLESOFT_VISUAL_MESH_NAME}']"
    )

    assert len(mesh_assets) == 1
    assert len(mesh_geoms) == ARM_SPEC.num_segments
    expected_scale = (
        ARM_SPEC.segment_pitch / ORIGINAL_STABLESOFT_SEGMENT_PITCH_M
    )
    assert tuple(float(value) for value in mesh_assets[0].attrib["scale"].split()) == (
        expected_scale,
    ) * 3
    for geom in mesh_geoms:
        assert geom.attrib["type"] == "mesh"
        assert geom.attrib["group"] == "2"
        assert geom.attrib["contype"] == "0"
        assert geom.attrib["conaffinity"] == "0"
        assert geom.attrib["pos"] == "0 0 0"
        assert geom.attrib["quat"] == "0 1 0 0"

    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nmesh == 40


def test_mesh_and_cylinder_modes_compile_with_expected_visual_geometry():
    mesh = _fixture(MESH_VISUAL)
    cylinder = _fixture(CYLINDER_VISUAL)
    mesh_geom_ids = _segment_geom_ids(mesh)
    cylinder_geom_ids = _segment_geom_ids(cylinder)

    assert mesh.model.nmesh == 1
    assert cylinder.model.nmesh == 0
    assert mesh.model.ngeom == cylinder.model.ngeom == 10
    np.testing.assert_array_equal(
        mesh.model.geom_type[mesh_geom_ids],
        mujoco.mjtGeom.mjGEOM_MESH,
    )
    np.testing.assert_array_equal(
        cylinder.model.geom_type[cylinder_geom_ids],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    )
    np.testing.assert_array_equal(mesh.model.geom_group[mesh_geom_ids], 2)
    np.testing.assert_array_equal(mesh.model.geom_contype[mesh_geom_ids], 0)
    np.testing.assert_array_equal(mesh.model.geom_conaffinity[mesh_geom_ids], 0)
    assert len(set(mesh.model.geom_dataid[mesh_geom_ids].tolist())) == 1


def test_compiled_mesh_pose_matches_scaled_negative_z_segment_contract():
    fixture = _fixture(MESH_VISUAL)
    geom_ids = _segment_geom_ids(fixture)
    vertices = _mesh_local_vertices(fixture.model, int(geom_ids[0]))
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    extent = np.ptp(vertices, axis=0)

    np.testing.assert_allclose(
        bounds_min,
        [-0.00370370362, -0.00370370362, -0.00885104002],
        rtol=0.0,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        bounds_max,
        [0.00370370362, 0.00370370362, -0.0000378489144],
        rtol=0.0,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        extent,
        [0.00740740724, 0.00740740724, 0.00881319110],
        rtol=0.0,
        atol=3e-10,
    )
    assert bounds_max[2] < 0.0
    assert bounds_min[2] > -ARM_SPEC.segment_pitch
    visible_gap = ARM_SPEC.segment_pitch - extent[2]
    assert visible_gap == pytest.approx(0.0000756978, abs=2e-10)
    for geom_id in geom_ids[1:]:
        np.testing.assert_allclose(
            _mesh_local_vertices(fixture.model, int(geom_id)),
            vertices,
            rtol=0.0,
            atol=0.0,
        )


def test_mesh_and_cylinder_models_are_structurally_and_inertially_equivalent():
    mesh = _fixture(MESH_VISUAL, active=True)
    cylinder = _fixture(CYLINDER_VISUAL, active=True)
    left, right = mesh.model, cylinder.model

    counts = lambda model: (
        model.nbody,
        model.njnt,
        model.nq,
        model.nv,
        model.ngeom,
        model.nsite,
        model.ntendon,
        model.nu,
        model.na,
    )
    assert counts(left) == counts(right) == (11, 8, 32, 24, 10, 75, 4, 4, 0)
    assert left.nmesh == 1
    assert right.nmesh == 0

    exact_arrays = (
        "body_parentid",
        "body_jntnum",
        "body_jntadr",
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "site_bodyid",
        "tendon_adr",
        "tendon_num",
        "tendon_limited",
        "actuator_ctrladr",
        "actuator_ctrlnum",
        "actuator_outadr",
        "actuator_outnum",
        "actuator_trntype",
        "actuator_trnid",
    )
    for name in exact_arrays:
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))

    float_arrays = (
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "jnt_pos",
        "jnt_stiffness",
        "dof_damping",
        "site_pos",
        "site_quat",
        "tendon_range",
        "qpos0",
        "qpos_spring",
        "actuator_ctrlrange",
        "actuator_gear",
    )
    for name in float_arrays:
        np.testing.assert_allclose(
            getattr(left, name),
            getattr(right, name),
            rtol=0.0,
            atol=0.0,
            err_msg=name,
        )

    for object_type, count_name in (
        (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
        (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
        (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
        (mujoco.mjtObj.mjOBJ_TENDON, "ntendon"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
    ):
        count = getattr(left, count_name)
        assert _object_names(left, object_type, count) == _object_names(
            right, object_type, count
        )

    np.testing.assert_allclose(
        mesh.data.ten_length,
        cylinder.data.ten_length,
        rtol=0.0,
        atol=0.0,
    )


def _zero_input_trace(visual):
    fixture = _fixture(visual)
    addresses = initialize_constant_curvature(
        fixture,
        "x",
        total_bend=0.2,
    )
    tip_id = _named_id(
        fixture.model,
        mujoco.mjtObj.mjOBJ_SITE,
        fixture.manifest.arm_tip_site_name,
    )
    records = []
    for step in range(241):
        records.append(
            (
                fixture.data.qpos.copy(),
                fixture.data.qvel.copy(),
                fixture.data.site_xpos[tip_id].copy(),
                joint_rotation_vectors(
                    fixture.model,
                    fixture.data,
                    addresses,
                ),
                fixture.data.ten_length.copy(),
                fixture.data.ten_velocity.copy(),
                np.asarray(
                    [warning.number for warning in fixture.data.warning]
                ),
                int(fixture.data.ncon),
            )
        )
        if step < 240:
            mujoco.mj_step(fixture.model, fixture.data)
            mujoco.mj_forward(fixture.model, fixture.data)
    return tuple(np.asarray([record[index] for record in records]) for index in range(8))


def test_zero_input_mesh_and_cylinder_trajectories_are_float_exact():
    mesh = _zero_input_trace(MESH_VISUAL)
    cylinder = _zero_input_trace(CYLINDER_VISUAL)
    for left, right in zip(mesh, cylinder):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "runner,config",
    (
        (
            run_single_tendon_characterization,
            ActiveTendonRunConfig("xp_yp", 0.01, sim_freq=SIM_FREQ),
        ),
        (
            run_tendon_command_characterization,
            TendonCommandRunConfig(
                (0.01, 0.01, 0.0, 0.0),
                sim_freq=SIM_FREQ,
            ),
        ),
    ),
    ids=("single-xp-yp", "multi-positive-x"),
)
def test_active_mesh_and_cylinder_trajectories_are_float_exact(runner, config):
    mesh = runner(config, visual=MESH_VISUAL)
    cylinder = runner(config, visual=CYLINDER_VISUAL)
    fields = (
        "qpos",
        "qvel_rad_s",
        "tip_world_m",
        "tip_displacement_m",
        "joint_rotation_vectors_rad",
        "composite_rotation_vectors_rad",
        "tendon_length_m",
        "tendon_velocity_m_s",
        "actuator_force_n",
        "qfrc_actuator_nm",
        "generalized_actuator_power_w",
        "actuator_space_power_w",
        "tendon_coordinate_power_w",
        "command_tendon_power_w",
        "warning_counts",
        "contacts",
    )
    errors = _maximum_errors(mesh, cylinder, fields)
    assert all(error == 0.0 for error in errors.values()), errors
    assert mesh.safety == cylinder.safety
