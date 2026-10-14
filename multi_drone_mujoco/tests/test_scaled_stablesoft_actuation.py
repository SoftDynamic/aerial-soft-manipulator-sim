"""C1-A1 ideal tendon-tension topology and sign tests."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
    build_scaled_stablesoft_arm_mjcf,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
    build_ideal_tendon_tension_actuator_xml,
    validate_ideal_tendon_tension_command,
)
from multi_drone_mujoco.control.ideal_tendon_tension import (
    resolve_ideal_tendon_actuator_bindings,
    write_ideal_tendon_tension_command,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    MOUNT_Z,
    build_fixed_arm_fixture,
    initialize_constant_curvature,
    joint_rotation_vectors,
    named_joint_addresses,
)


SIM_FREQ = 2400
CTRL_FREQ = 240
INTEGRATOR = "implicitfast"
MAX_TENSION = 0.01  # C1-A1 diagnostic limit, not a hardware rating.
JOINT_STIFFNESS = 0.003086
TARGET_DAMPING_RATIO = 0.3
TOTAL_BEND = 0.2
ZERO_INPUT_DURATION = 2.0
CORNERS = ("xp_yp", "xp_yn", "xn_yp", "xn_yn")


def _arm_spec():
    return ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )


def _drive_spec():
    return IdealTendonTensionSpec(max_tension=MAX_TENSION)


def _fixture(*, active: bool):
    return build_fixed_arm_fixture(
        _arm_spec(),
        "vertical",
        sim_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        integrator=INTEGRATOR,
        tendon_drive=_drive_spec() if active else None,
    )


def _name_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, name
    return int(object_id)


def _actuator_bindings(fixture):
    return resolve_ideal_tendon_actuator_bindings(fixture.model, fixture.manifest)


def _write_tension_command(fixture, command):
    bindings = _actuator_bindings(fixture)
    values = write_ideal_tendon_tension_command(fixture.data, bindings, command, _drive_spec())
    return values, bindings


def _warning_count(data):
    return sum(int(warning.number) for warning in data.warning)


def _sample_state(fixture):
    model, data = fixture.model, fixture.data
    manifest = fixture.manifest
    addresses = named_joint_addresses(model, manifest)
    tip_id = _name_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        manifest.arm_tip_site_name,
    )
    tendon_ids = np.asarray(
        [
            _name_id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
            for name in manifest.tendon_names
        ],
        dtype=int,
    )
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    return {
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "tip": np.asarray(data.site_xpos[tip_id]).copy(),
        "joints": joint_rotation_vectors(model, data, addresses),
        "tendons": np.asarray(data.ten_length[tendon_ids]).copy(),
        "energy": float(np.sum(data.energy)),
    }


def _run_zero_input_pair():
    passive = _fixture(active=False)
    active = _fixture(active=True)
    initialize_constant_curvature(
        passive,
        "x",
        total_bend=TOTAL_BEND,
    )
    initialize_constant_curvature(
        active,
        "x",
        total_bend=TOTAL_BEND,
    )
    _write_tension_command(active, (0.0, 0.0, 0.0, 0.0))

    passive_samples = []
    active_samples = []
    total_steps = int(round(ZERO_INPUT_DURATION * SIM_FREQ))
    for step_index in range(total_steps + 1):
        if step_index % passive.sample_stride == 0:
            passive_samples.append(_sample_state(passive))
            active_samples.append(_sample_state(active))
        if step_index == total_steps:
            break

        mujoco.mj_step(passive.model, passive.data)
        mujoco.mj_step(active.model, active.data)
        mujoco.mj_forward(passive.model, passive.data)
        mujoco.mj_forward(active.model, active.data)

        for fixture in (passive, active):
            data = fixture.data
            assert _warning_count(data) == 0
            assert data.ncon == 0
            assert all(
                np.all(np.isfinite(values))
                for values in (
                    data.qpos,
                    data.qvel,
                    data.qacc,
                    data.site_xpos,
                    data.ten_length,
                    data.energy,
                )
            )
            np.testing.assert_array_less(
                fixture.model.tendon_range[:, 0] - 1e-9,
                data.ten_length,
            )
            np.testing.assert_array_less(
                data.ten_length,
                fixture.model.tendon_range[:, 1] + 1e-9,
            )
        np.testing.assert_allclose(active.data.actuator_force, 0.0)
        np.testing.assert_allclose(active.data.qfrc_actuator, 0.0)

    def stacked(records, field):
        return np.asarray([record[field] for record in records])

    return {
        field: (
            stacked(passive_samples, field),
            stacked(active_samples, field),
        )
        for field in passive_samples[0]
    }


@pytest.mark.parametrize("value", [0.0, -0.01, np.nan, np.inf, -np.inf])
def test_ideal_tendon_tension_spec_rejects_invalid_maximum(value):
    with pytest.raises(ValueError):
        IdealTendonTensionSpec(max_tension=value)


def test_ideal_tendon_tension_spec_normalizes_maximum_to_float():
    spec = IdealTendonTensionSpec(max_tension=np.float32(MAX_TENSION))

    assert type(spec.max_tension) is float
    assert spec.max_tension == pytest.approx(MAX_TENSION)


def test_ideal_tendon_tension_command_returns_immutable_float_copy():
    source = [0, np.float32(0.002), 0.004, MAX_TENSION]
    command = validate_ideal_tendon_tension_command(source, _drive_spec())
    source[:] = [MAX_TENSION] * 4

    assert command == pytest.approx((0.0, 0.002, 0.004, MAX_TENSION))
    assert isinstance(command, tuple)
    assert all(type(value) is float for value in command)


@pytest.mark.parametrize(
    "command",
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, np.nan, 0.0, 0.0),
        (0.0, np.inf, 0.0, 0.0),
        (0.0, -1e-6, 0.0, 0.0),
        (0.0, 0.0, 0.0, MAX_TENSION + 1e-6),
    ],
)
def test_ideal_tendon_tension_command_rejects_invalid_values(command):
    with pytest.raises(ValueError):
        validate_ideal_tendon_tension_command(command, _drive_spec())


def test_ideal_tendon_tension_xml_is_minimal_and_ordered():
    tendon_names = tuple(
        f"fixture_scaled_stablesoft_arm_{corner}_tendon"
        for corner in ("xp_yp", "xp_yn", "xn_yp", "xn_yn")
    )
    xml, actuator_names = build_ideal_tendon_tension_actuator_xml(
        tendon_names,
        _drive_spec(),
    )
    root = ET.fromstring(f"<actuator>{xml}</actuator>")

    assert actuator_names == tuple(
        f"t{index}_tension_N"
        for index in range(len(tendon_names))
    )
    assert [element.tag for element in root] == ["motor"] * 4
    for element, tendon_name, actuator_name in zip(
        root,
        tendon_names,
        actuator_names,
    ):
        assert element.attrib == {
            "name": actuator_name,
            "tendon": tendon_name,
            "gear": "-1",
            "ctrllimited": "true",
            "ctrlrange": "0 0.01",
        }


def test_passive_fixture_preserves_empty_actuation_contract():
    fragments = build_scaled_stablesoft_arm_mjcf(
        "fixture",
        _arm_spec(),
        MOUNT_Z,
    )
    fixture = _fixture(active=False)
    model = fixture.model

    assert fragments.actuator_xml == ""
    assert fragments.manifest.tendon_actuator_names == ()
    assert model.ntendon == 4
    assert model.nu == 0
    assert model.na == 0
    assert model.nout == 0
    if hasattr(model, "nactuator"):
        assert model.nactuator == 0
    assert (
        model.nbody,
        model.njnt,
        model.nq,
        model.nv,
        model.ngeom,
        model.nsite,
        model.ntendon,
        model.nu,
        model.nsensor,
    ) == (11, 8, 32, 24, 10, 75, 4, 0, 4)


def test_active_fixture_compiles_named_one_to_one_tendon_transmissions():
    fragments = build_scaled_stablesoft_arm_mjcf(
        "fixture",
        _arm_spec(),
        MOUNT_Z,
        tendon_drive=_drive_spec(),
    )
    fixture = _fixture(active=True)
    model = fixture.model
    manifest = fixture.manifest
    bindings = _actuator_bindings(fixture)

    assert model.ntendon == 4
    assert model.nu == 4
    assert model.na == 0
    assert model.nout == 4
    if hasattr(model, "nactuator"):
        assert model.nactuator == 4
    assert len(manifest.tendon_actuator_names) == 4
    assert fragments.manifest.tendon_actuator_names == (
        manifest.tendon_actuator_names
    )
    for index, (corner, tendon_name, actuator_name) in enumerate(zip(
        CORNERS,
        manifest.tendon_names,
        manifest.tendon_actuator_names,
    )):
        assert tendon_name.endswith(f"_{corner}_tendon")
        assert actuator_name == f"t{index}_tension_N"
    assert tuple(
        binding.actuator_id for binding in bindings
    ) == tuple(sorted(binding.actuator_id for binding in bindings))
    assert tuple(
        binding.control_address for binding in bindings
    ) == tuple(sorted(binding.control_address for binding in bindings))

    for binding in bindings:
        actuator_id = binding.actuator_id
        assert (
            model.actuator_trntype[actuator_id]
            == mujoco.mjtTrn.mjTRN_TENDON
        )
        assert int(model.actuator_trnid[actuator_id, 0]) == binding.tendon_id
        assert int(model.actuator_trnid[actuator_id, 1]) == -1
        np.testing.assert_allclose(
            model.actuator_gear[actuator_id],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            rtol=0.0,
            atol=0.0,
        )
        assert bool(model.actuator_ctrllimited[actuator_id])
        np.testing.assert_allclose(
            model.actuator_ctrlrange[actuator_id],
            [0.0, MAX_TENSION],
            rtol=0.0,
            atol=1e-15,
        )
        assert (
            model.actuator_dyntype[actuator_id]
            == mujoco.mjtDyn.mjDYN_NONE
        )
        assert (
            model.actuator_gaintype[actuator_id]
            == mujoco.mjtGain.mjGAIN_FIXED
        )
        assert (
            model.actuator_biastype[actuator_id]
            == mujoco.mjtBias.mjBIAS_NONE
        )
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(1.0)
        np.testing.assert_allclose(
            model.actuator_biasprm[actuator_id],
            0.0,
            rtol=0.0,
            atol=0.0,
        )


def test_valid_tension_command_is_direct_actuator_force():
    fixture = _fixture(active=True)
    command, bindings = _write_tension_command(
        fixture,
        (0.002, 0.004, 0.006, 0.008),
    )
    mujoco.mj_forward(fixture.model, fixture.data)
    force = tuple(
        float(fixture.data.actuator_force[binding.output_address])
        for binding in bindings
    )

    np.testing.assert_allclose(force, command, rtol=0.0, atol=1e-15)


def test_zero_tension_active_dynamics_match_passive_fixture():
    traces = _run_zero_input_pair()
    tolerances = {
        "qpos": (1e-11, 1e-12),
        "qvel": (1e-10, 1e-11),
        "tip": (1e-10, 1e-12),
        "joints": (1e-10, 1e-11),
        "tendons": (1e-10, 1e-12),
        "energy": (1e-10, 1e-12),
    }
    for field, (passive, active) in traces.items():
        rtol, atol = tolerances[field]
        np.testing.assert_allclose(
            active,
            passive,
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"{field} max_abs_error="
                f"{np.max(np.abs(active - passive))}"
            ),
        )


def test_tendon_transmission_sign_moment_and_power_identities():
    fixture = _fixture(active=True)
    model, data = fixture.model, fixture.data
    addresses = initialize_constant_curvature(
        fixture,
        "x",
        total_bend=TOTAL_BEND,
    )
    arm_dofs = np.concatenate(
        [
            np.arange(dof_address, dof_address + 3)
            for _, _, dof_address in addresses
        ]
    )
    data.qvel[:] = 0.0
    data.qvel[arm_dofs] = np.linspace(-0.012, 0.015, len(arm_dofs))
    command, bindings = _write_tension_command(
        fixture,
        (0.002, 0.004, 0.006, 0.008),
    )
    mujoco.mj_forward(model, data)

    tendon_ids = np.asarray(
        [binding.tendon_id for binding in bindings],
        dtype=int,
    )
    output_addresses = np.asarray(
        [binding.output_address for binding in bindings],
        dtype=int,
    )
    actuator_length = np.asarray(data.actuator_length)[output_addresses]
    actuator_velocity = np.asarray(data.actuator_velocity)[output_addresses]
    actuator_force = np.asarray(data.actuator_force)[output_addresses]
    tendon_length = np.asarray(data.ten_length)[tendon_ids]
    tendon_velocity = np.asarray(data.ten_velocity)[tendon_ids]
    actuator_moment = np.asarray(data.actuator_moment).reshape(
        model.nout,
        model.nv,
    )
    expected_qfrc = (
        actuator_moment[output_addresses].T @ actuator_force
    )

    np.testing.assert_allclose(
        actuator_length,
        -tendon_length,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actuator_velocity,
        -tendon_velocity,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actuator_force,
        command,
        rtol=0.0,
        atol=1e-15,
    )

    moment_residual = float(
        np.max(np.abs(data.qfrc_actuator - expected_qfrc))
    )
    moment_scale = max(
        1.0,
        float(np.max(np.abs(data.qfrc_actuator))),
        float(np.max(np.abs(expected_qfrc))),
    )
    assert moment_residual <= 1e-12 + 1e-10 * moment_scale

    generalized_power = float(data.qfrc_actuator @ data.qvel)
    actuator_power = float(actuator_force @ actuator_velocity)
    tendon_power = float(-np.asarray(command) @ tendon_velocity)
    assert max(
        abs(generalized_power),
        abs(actuator_power),
        abs(tendon_power),
    ) > 1e-12
    power_scale = max(
        1e-12,
        abs(generalized_power),
        abs(actuator_power),
        abs(tendon_power),
    )
    power_tolerance = 1e-12 + 1e-9 * power_scale
    assert abs(generalized_power - actuator_power) <= power_tolerance
    assert abs(actuator_power - tendon_power) <= power_tolerance
