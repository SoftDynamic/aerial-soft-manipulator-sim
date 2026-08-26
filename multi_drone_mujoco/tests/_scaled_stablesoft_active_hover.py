"""C1-B3 active tendon coupling under gravity and open-loop hover thrust.

The formal model is derived from the production active aviary XML in memory.
Only the floor is removed; production gravity and all active topology remain.
This module never writes XML, traces, logs, images, or viewer state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Optional
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from multi_drone_mujoco.control.ideal_tendon_tension import (
    resolve_ideal_tendon_actuator_bindings,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    half_cosine_ramp_scale_at_tick,
    sample_ideal_tendon_transmission,
)
from multi_drone_mujoco.tests._scaled_stablesoft_active_coupled import (
    MAX_TENSION_N,
    QACC_HARD_LIMIT,
    QVEL_HARD_LIMIT,
    TOTAL_ARM_LENGTH_M,
    TOTAL_SYSTEM_MASS_KG,
    _active_counts,
    _make_active_production_env,
    _object_names,
    _resolve_entities,
    _verify_bindings,
)
from multi_drone_mujoco.tests._scaled_stablesoft_coupled import (
    _component_momentum,
)


CTRL_FREQ = 240
DURATION_S = 3.0
RAMP_TICKS = 60
RELEASE_TICK = 180
TOTAL_CONTROL_TICKS = 720
FORMAL_TENSION_N = 0.002
FORCED_END_S = 0.75
EARLY_END_S = 1.5

CORNERS = ("xp_yp", "xp_yn", "xn_yp", "xn_yn")
CASE_CORNERS = MappingProxyType(
    {
        "baseline": (),
        "single_xp_yp": ("xp_yp",),
        "plus_x_pair": ("xp_yp", "xp_yn"),
        "common_mode": CORNERS,
    }
)
FORMAL_CASES = tuple(CASE_CORNERS)


def early_linear_impulse_limit(peak_impulse_ns):
    return max(6e-4, 0.02 * float(peak_impulse_ns))


def early_angular_impulse_limit(peak_delta_h_nms, peak_impulse_nms):
    return max(
        1.5e-6,
        0.25 * max(float(peak_delta_h_nms), float(peak_impulse_nms)),
    )


def forced_linear_impulse_limit(peak_impulse_ns):
    """Frozen after three deterministic probes at both formal frequencies."""
    return max(2.5e-6, 0.001 * float(peak_impulse_ns))


def forced_energy_ledger_limit(peak_combined_work_j):
    return max(5e-7, 0.005 * float(peak_combined_work_j))


def early_energy_ledger_limit(peak_combined_work_j):
    return max(5e-6, 0.04 * float(peak_combined_work_j))


def full_duration_energy_guard(peak_combined_work_j):
    return max(1e-3, 0.40 * float(peak_combined_work_j))


def ledger_convergence_limit(reference_peak_combined_work_j):
    return max(2e-4, 0.002 * float(reference_peak_combined_work_j))


@dataclass(frozen=True)
class C1B3RunConfig:
    sim_freq: int
    case: str
    tension_n: float = FORMAL_TENSION_N
    ctrl_freq: int = CTRL_FREQ
    duration_s: float = DURATION_S
    integrator: str = "implicitfast"

    def __post_init__(self):
        if self.sim_freq not in (2400, 4800):
            raise ValueError("C1-B3 sim_freq must be 2400 or 4800")
        if self.case not in CASE_CORNERS:
            raise ValueError(f"unknown C1-B3 case: {self.case}")
        if self.ctrl_freq != CTRL_FREQ:
            raise ValueError("C1-B3 ctrl_freq must be 240")
        if self.duration_s != DURATION_S:
            raise ValueError("C1-B3 duration_s must be 3.0")
        if self.integrator != "implicitfast":
            raise ValueError("C1-B3 integrator must be implicitfast")
        if not math.isfinite(self.tension_n):
            raise ValueError("C1-B3 tension must be finite")
        if self.tension_n < 0.0 or self.tension_n > MAX_TENSION_N:
            raise ValueError("C1-B3 tension must be in [0, 0.01] N")
        if self.case == "baseline" and self.tension_n != FORMAL_TENSION_N:
            raise ValueError("baseline uses the formal tension metadata")

    @property
    def sample_stride(self):
        return self.sim_freq // self.ctrl_freq

    @property
    def requested_steps(self):
        return int(round(self.duration_s * self.sim_freq))


@dataclass(frozen=True)
class C1B3Failure:
    case: str
    sim_freq: int
    stage: str
    physics_step: int
    control_tick: int
    substep: int
    time_s: float
    field: str
    value: object


@dataclass(frozen=True)
class C1B3TopologyProof:
    production_counts: tuple
    hover_counts: tuple
    production_gravity: tuple
    hover_gravity: tuple
    production_floor_count: int
    hover_floor_count: int
    named_topology_match: bool
    physical_arrays_match: bool
    actuator_arrays_match: bool
    timestep_match: bool
    integrator_match: bool
    system_mass_kg: float
    tendon_names: tuple
    actuator_names: tuple


@dataclass(frozen=True)
class C1B3InitialHoverProof:
    environment_mass_kg: float
    model_system_mass_kg: float
    arm_mass_kg: float
    hover_rpm: float
    maximum_rpm: float
    rotor_rpm: tuple
    rotor_force_world_n: tuple
    rotor_torque_world_nm: tuple
    force_balance_world_n: tuple
    initial_quaternion: tuple
    initial_qvel_max: float
    initial_root_to_tip_body_m: tuple
    initial_contacts: int
    initial_warning_counts: tuple


@dataclass(frozen=True)
class C1B3SafetySummary:
    requested_steps: int
    completed_steps: int
    first_failure: Optional[C1B3Failure]
    maximum_contacts: int
    minimum_tendon_margin_m: float
    maximum_abs_qvel_native: float
    maximum_abs_qacc_native: float
    maximum_root_to_tip_length_m: float
    maximum_force_command_error_n: float
    maximum_actuator_moment_residual: float
    maximum_tendon_power_identity_residual_w: float
    maximum_rotor_power_residual_w: float
    maximum_p_partition_residual_ns: float
    maximum_h_partition_residual_nms: float
    maximum_gravity_torque_residual_nm: float


@dataclass(frozen=True)
class C1B3Trace:
    config: C1B3RunConfig
    status: str
    topology: C1B3TopologyProof
    initial_hover: C1B3InitialHoverProof
    bindings: tuple
    tendon_ranges_m: np.ndarray
    physics: MappingProxyType
    control: MappingProxyType
    relative: MappingProxyType
    safety: C1B3SafetySummary


class _SteppingFailure(Exception):
    def __init__(self, failure):
        self.failure = failure
        super().__init__(str(failure))


def _remove_floor_preserving_gravity(xml):
    root = ET.fromstring(xml)
    option = root.find("option")
    if option is None:
        raise AssertionError("production XML has no option element")
    gravity_before = option.get("gravity")
    floors = root.findall(".//geom[@name='floor']")
    if len(floors) != 1:
        raise AssertionError(f"expected one floor geom, found {len(floors)}")
    floor = floors[0]
    parents = [parent for parent in root.iter() if floor in list(parent)]
    if len(parents) != 1:
        raise AssertionError("floor geom parent is ambiguous")
    parents[0].remove(floor)
    if option.get("gravity") != gravity_before:
        raise AssertionError("floor removal changed gravity text")
    return ET.tostring(root, encoding="unicode")


def _without_floor(names):
    return tuple(name for name in names if name != "floor")


def _verify_hover_transformation(production, hover, manifest):
    production_counts = _active_counts(production)
    hover_counts = _active_counts(hover)
    expected = list(production_counts)
    expected[4] -= 1
    if tuple(expected) != hover_counts:
        raise AssertionError(
            f"hover count mismatch: {production_counts} -> {hover_counts}"
        )
    production_geoms = _object_names(
        production, mujoco.mjtObj.mjOBJ_GEOM, production.ngeom
    )
    hover_geoms = _object_names(hover, mujoco.mjtObj.mjOBJ_GEOM, hover.ngeom)
    named_match = _without_floor(production_geoms) == hover_geoms
    for object_type, left_count, right_count in (
        (mujoco.mjtObj.mjOBJ_BODY, production.nbody, hover.nbody),
        (mujoco.mjtObj.mjOBJ_JOINT, production.njnt, hover.njnt),
        (mujoco.mjtObj.mjOBJ_SITE, production.nsite, hover.nsite),
        (mujoco.mjtObj.mjOBJ_TENDON, production.ntendon, hover.ntendon),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, production.nu, hover.nu),
        (mujoco.mjtObj.mjOBJ_SENSOR, production.nsensor, hover.nsensor),
    ):
        named_match = named_match and _object_names(
            production, object_type, left_count
        ) == _object_names(hover, object_type, right_count)
    if not named_match:
        raise AssertionError("floor removal changed named active topology")

    physical_fields = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "jnt_type",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "tendon_range",
        "tendon_adr",
        "tendon_num",
        "wrap_type",
        "wrap_objid",
        "site_pos",
        "site_quat",
    )
    physical_match = all(
        np.array_equal(getattr(production, field), getattr(hover, field))
        for field in physical_fields
    )
    floor_id = mujoco.mj_name2id(
        production, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    geom_fields = (
        "geom_type",
        "geom_bodyid",
        "geom_contype",
        "geom_conaffinity",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_group",
        "geom_rgba",
    )
    physical_match = physical_match and all(
        np.array_equal(
            np.delete(getattr(production, field), floor_id, axis=0),
            getattr(hover, field),
        )
        for field in geom_fields
    )
    if not physical_match:
        raise AssertionError("floor removal changed mass/inertia/routing arrays")
    actuator_fields = (
        "actuator_trntype",
        "actuator_trnid",
        "actuator_gear",
        "actuator_ctrllimited",
        "actuator_ctrlrange",
        "actuator_forcelimited",
    )
    actuator_match = all(
        np.array_equal(getattr(production, field), getattr(hover, field))
        for field in actuator_fields
    )
    if not actuator_match:
        raise AssertionError("floor removal changed actuator arrays")
    if not np.array_equal(production.opt.gravity, hover.opt.gravity):
        raise AssertionError("floor removal changed production gravity")
    if not np.array_equal(hover.opt.gravity, [0.0, 0.0, -9.81]):
        raise AssertionError(f"unexpected hover gravity: {hover.opt.gravity}")
    if production.opt.timestep != hover.opt.timestep:
        raise AssertionError("floor removal changed timestep")
    if production.opt.integrator != hover.opt.integrator:
        raise AssertionError("floor removal changed integrator")
    system_mass = float(
        hover.body_subtreemass[
            mujoco.mj_name2id(hover, mujoco.mjtObj.mjOBJ_BODY, "drone0")
        ]
    )
    if not math.isclose(
        system_mass, TOTAL_SYSTEM_MASS_KG, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(f"unexpected hover system mass: {system_mass}")
    return C1B3TopologyProof(
        production_counts=production_counts,
        hover_counts=hover_counts,
        production_gravity=tuple(float(x) for x in production.opt.gravity),
        hover_gravity=tuple(float(x) for x in hover.opt.gravity),
        production_floor_count=int("floor" in production_geoms),
        hover_floor_count=int("floor" in hover_geoms),
        named_topology_match=named_match,
        physical_arrays_match=physical_match,
        actuator_arrays_match=actuator_match,
        timestep_match=bool(production.opt.timestep == hover.opt.timestep),
        integrator_match=bool(production.opt.integrator == hover.opt.integrator),
        system_mass_kg=system_mass,
        tendon_names=tuple(manifest.tendon_names),
        actuator_names=tuple(manifest.tendon_actuator_names),
    )


def _binding_by_corner(bindings):
    result = {}
    for binding in bindings:
        matches = [
            corner
            for corner in CORNERS
            if binding.tendon_name.endswith(f"_{corner}_tendon")
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"cannot resolve tendon corner: {binding.tendon_name}"
            )
        result[matches[0]] = binding
    if tuple(result) != CORNERS:
        raise AssertionError(f"unexpected manifest corner order: {tuple(result)}")
    return MappingProxyType(result)


def _command_at_tick(config, bindings_by_corner, tick):
    if tick >= RELEASE_TICK:
        scale = 0.0
    else:
        scale = half_cosine_ramp_scale_at_tick(tick, RAMP_TICKS)
    active = set(CASE_CORNERS[config.case])
    by_actuator = {
        binding.actuator_id: config.tension_n * scale
        if corner in active
        else 0.0
        for corner, binding in bindings_by_corner.items()
    }
    return tuple(
        float(by_actuator[binding.actuator_id])
        for binding in bindings_by_corner.values()
    )


def _warning_counts(data):
    return tuple(int(warning.number) for warning in data.warning)


def _failure(config, data, *, stage, step, tick, substep, field, value):
    return C1B3Failure(
        case=config.case,
        sim_freq=config.sim_freq,
        stage=stage,
        physics_step=step,
        control_tick=tick,
        substep=substep,
        time_s=float(data.time),
        field=field,
        value=value,
    )


def _require(
    condition,
    config,
    data,
    *,
    stage,
    step,
    tick,
    substep,
    field,
    value,
    stepping,
):
    if condition:
        return
    item = _failure(
        config,
        data,
        stage=stage,
        step=step,
        tick=tick,
        substep=substep,
        field=field,
        value=value,
    )
    if stepping:
        raise _SteppingFailure(item)
    raise AssertionError(str(item))


def _normalized_quaternion(quaternion):
    result = np.asarray(quaternion, dtype=float).copy()
    result /= np.linalg.norm(result)
    if result[0] < 0.0:
        result *= -1.0
    return result


def _relative_quaternion(current, reference):
    inverse = np.empty(4)
    result = np.empty(4)
    mujoco.mju_negQuat(inverse, _normalized_quaternion(reference))
    mujoco.mju_mulQuat(result, _normalized_quaternion(current), inverse)
    return _normalized_quaternion(result)


def _rotation_vector(quaternion):
    result = np.empty(3)
    mujoco.mju_quat2Vel(result, _normalized_quaternion(quaternion), 1.0)
    return result


def _joint_quaternions(data, addresses):
    return np.asarray(
        [
            _normalized_quaternion(data.qpos[qpos:qpos + 4])
            for _, qpos, _ in addresses
        ]
    )


def _composite_quaternion(joints):
    total = np.array([1.0, 0.0, 0.0, 0.0])
    product = np.empty(4)
    for quaternion in joints:
        mujoco.mju_mulQuat(product, total, quaternion)
        total = _normalized_quaternion(product)
    return total


def _rotor_mechanics(model, data, entities):
    force = np.asarray(data.xfrc_applied[entities.drone_id, :3]).copy()
    torque = np.asarray(data.xfrc_applied[entities.drone_id, 3:]).copy()
    point = np.asarray(data.xipos[entities.drone_id]).copy()
    qfrc = np.zeros(model.nv)
    mujoco.mj_applyFT(
        model, data, force, torque, point, entities.drone_id, qfrc
    )
    jacp = np.empty((3, model.nv))
    jacr = np.empty((3, model.nv))
    mujoco.mj_jacBodyCom(
        model, data, jacp, jacr, entities.drone_id
    )
    velocity = jacp @ data.qvel
    omega = jacr @ data.qvel
    generalized_power = float(qfrc @ data.qvel)
    cartesian_power = float(force @ velocity + torque @ omega)
    return {
        "force": force,
        "torque": torque,
        "wrench": np.concatenate((force, torque)),
        "point": point,
        "qfrc": qfrc,
        "velocity": velocity,
        "omega": omega,
        "generalized_power": generalized_power,
        "cartesian_power": cartesian_power,
    }


def _system_mechanics(model, data, entities, rotor):
    mujoco.mj_subtreeVel(model, data)
    com = np.asarray(data.subtree_com[entities.drone_id]).copy()
    momentum = (
        entities.system_mass
        * np.asarray(data.subtree_linvel[entities.drone_id]).copy()
    )
    angular = np.asarray(data.subtree_angmom[entities.drone_id]).copy()
    gravity = np.asarray(model.opt.gravity)
    net_force = rotor["force"] + entities.system_mass * gravity
    rotor_torque_com = rotor["torque"] + np.cross(
        rotor["point"] - com, rotor["force"]
    )
    gravity_torque = np.zeros(3)
    for body_id in entities.uav_body_ids + entities.arm_body_ids:
        mass = float(model.body_mass[body_id])
        gravity_torque += np.cross(
            np.asarray(data.xipos[body_id]) - com,
            mass * gravity,
        )
    return {
        "com": com,
        "momentum": momentum,
        "angular_momentum": angular,
        "net_force": net_force,
        "rotor_torque_com": rotor_torque_com,
        "gravity_torque": gravity_torque,
        "external_torque": rotor_torque_com + gravity_torque,
    }


def _synchronized(
    model,
    data,
    config,
    bindings,
    entities,
    command,
    tendon_ranges,
    accumulator,
    *,
    stage,
    step,
    tick,
    substep,
    stepping,
):
    transmission = sample_ideal_tendon_transmission(
        model, data, bindings, command
    )
    rotor = _rotor_mechanics(model, data, entities)
    system = _system_mechanics(model, data, entities, rotor)
    warnings = _warning_counts(data)
    arrays = (
        data.qpos,
        data.qvel,
        data.qacc,
        data.site_xpos,
        data.ten_length,
        data.energy,
        rotor["qfrc"],
        system["momentum"],
        system["angular_momentum"],
    )
    context = dict(
        stage=stage,
        step=step,
        tick=tick,
        substep=substep,
        stepping=stepping,
    )
    _require(
        math.isfinite(float(data.time)) and all(np.all(np.isfinite(x)) for x in arrays),
        config,
        data,
        field="finite_synchronized_state",
        value=False,
        **context,
    )
    _require(
        not any(warnings), config, data, field="mujoco_warning", value=warnings, **context
    )
    _require(
        int(data.ncon) == 0,
        config,
        data,
        field="contacts",
        value=int(data.ncon),
        **context,
    )
    _require(
        np.array_equal(data.qfrc_applied, np.zeros(model.nv)),
        config,
        data,
        field="native_qfrc_applied",
        value=float(np.max(np.abs(data.qfrc_applied))),
        **context,
    )
    margins = np.minimum(
        transmission.tendon_length - tendon_ranges[:, 0],
        tendon_ranges[:, 1] - transmission.tendon_length,
    )
    qvel_max = float(np.max(np.abs(data.qvel)))
    qacc_max = float(np.max(np.abs(data.qacc)))
    root = np.asarray(data.site_xpos[entities.root_site_id])
    tip = np.asarray(data.site_xpos[entities.tip_site_id])
    root_tip_length = float(np.linalg.norm(tip - root))
    force_error = float(
        np.max(np.abs(transmission.actuator_force - np.asarray(command)))
    )
    moment_residual = float(
        np.max(np.abs(data.qfrc_actuator - transmission.expected_qfrc))
    )
    tendon_power_residual = max(
        abs(transmission.generalized_power - transmission.actuator_power),
        abs(transmission.actuator_power - transmission.tendon_power),
        abs(transmission.tendon_power - transmission.command_tendon_power),
    )
    rotor_power_residual = abs(
        rotor["generalized_power"] - rotor["cartesian_power"]
    )
    accumulator["maximum_contacts"] = max(
        accumulator["maximum_contacts"], int(data.ncon)
    )
    accumulator["minimum_tendon_margin_m"] = min(
        accumulator["minimum_tendon_margin_m"], float(np.min(margins))
    )
    accumulator["maximum_abs_qvel_native"] = max(
        accumulator["maximum_abs_qvel_native"], qvel_max
    )
    accumulator["maximum_abs_qacc_native"] = max(
        accumulator["maximum_abs_qacc_native"], qacc_max
    )
    accumulator["maximum_root_to_tip_length_m"] = max(
        accumulator["maximum_root_to_tip_length_m"], root_tip_length
    )
    accumulator["maximum_force_command_error_n"] = max(
        accumulator["maximum_force_command_error_n"], force_error
    )
    accumulator["maximum_actuator_moment_residual"] = max(
        accumulator["maximum_actuator_moment_residual"], moment_residual
    )
    accumulator["maximum_tendon_power_identity_residual_w"] = max(
        accumulator["maximum_tendon_power_identity_residual_w"],
        tendon_power_residual,
    )
    accumulator["maximum_rotor_power_residual_w"] = max(
        accumulator["maximum_rotor_power_residual_w"], rotor_power_residual
    )
    accumulator["maximum_gravity_torque_residual_nm"] = max(
        accumulator["maximum_gravity_torque_residual_nm"],
        float(np.linalg.norm(system["gravity_torque"])),
    )
    for condition, field, value in (
        (
            float(np.min(margins)) >= -1e-9,
            "tendon_range_margin_m",
            float(np.min(margins)),
        ),
        (qvel_max < QVEL_HARD_LIMIT, "qvel_hard_limit", qvel_max),
        (qacc_max < QACC_HARD_LIMIT, "qacc_hard_limit", qacc_max),
        (
            root_tip_length <= TOTAL_ARM_LENGTH_M + 1e-9,
            "root_tip_length_m",
            root_tip_length,
        ),
        (force_error <= 1e-15, "force_command_error_n", force_error),
        (
            moment_residual <= 1e-12,
            "actuator_moment_residual",
            moment_residual,
        ),
        (
            tendon_power_residual <= 1e-12,
            "tendon_power_identity_w",
            tendon_power_residual,
        ),
        (
            rotor_power_residual
            <= 1e-12
            + 1e-9
            * max(
                abs(rotor["generalized_power"]),
                abs(rotor["cartesian_power"]),
            ),
            "rotor_power_identity_w",
            rotor_power_residual,
        ),
    ):
        _require(condition, config, data, field=field, value=value, **context)
    damping_power = float(-data.qfrc_damper @ data.qvel)
    _require(
        damping_power >= -1e-15,
        config,
        data,
        field="damping_power_w",
        value=damping_power,
        **context,
    )
    return {
        "transmission": transmission,
        "rotor": rotor,
        "system": system,
        "warnings": warnings,
        "margins": margins,
        "qvel_max": qvel_max,
        "qacc_max": qacc_max,
        "damping_power": damping_power,
    }


def _control_sample(
    model,
    data,
    config,
    entities,
    command,
    rpm,
    synchronized,
    cumulative,
    tick,
    step,
):
    qpos = entities.drone_qpos_address
    dof = entities.drone_dof_address
    rotation = np.asarray(data.xmat[entities.drone_id]).reshape(3, 3).copy()
    quaternion = _normalized_quaternion(data.qpos[qpos + 3:qpos + 7])
    omega_body = np.asarray(data.qvel[dof + 3:dof + 6]).copy()
    omega_world = synchronized["rotor"]["omega"]
    if not np.allclose(
        omega_world, rotation @ omega_body, rtol=1e-10, atol=1e-12
    ):
        raise _SteppingFailure(
            C1B3Failure(
                case=config.case,
                sim_freq=config.sim_freq,
                stage="control_sample",
                physics_step=step,
                control_tick=tick,
                substep=0,
                time_s=float(data.time),
                field="angular_velocity_frame_consistency",
                value=float(np.max(np.abs(omega_world - rotation @ omega_body))),
            )
        )
    root = np.asarray(data.site_xpos[entities.root_site_id]).copy()
    tip = np.asarray(data.site_xpos[entities.tip_site_id]).copy()
    joint_quaternions = _joint_quaternions(data, entities.joint_addresses)
    composite_quaternion = _composite_quaternion(joint_quaternions)
    system_com = synchronized["system"]["com"]
    p_uav, h_uav = _component_momentum(
        model, data, entities.uav_body_ids, system_com
    )
    p_arm, h_arm = _component_momentum(
        model, data, entities.arm_body_ids, system_com
    )
    accumulator_p = float(
        np.linalg.norm(p_uav + p_arm - synchronized["system"]["momentum"])
    )
    accumulator_h = float(
        np.linalg.norm(
            h_uav + h_arm - synchronized["system"]["angular_momentum"]
        )
    )
    transmission = synchronized["transmission"]
    return {
        "time": float(data.time),
        "tick": int(tick),
        "physics_step": int(step),
        "command": np.asarray(command).copy(),
        "actuator_force": transmission.actuator_force.copy(),
        "tendon_length": transmission.tendon_length.copy(),
        "tendon_velocity": transmission.tendon_velocity.copy(),
        "uav_position_world_m": np.asarray(data.qpos[qpos:qpos + 3]).copy(),
        "uav_linear_velocity_world_m_s": np.asarray(data.qvel[dof:dof + 3]).copy(),
        "uav_quaternion": quaternion,
        "uav_rotation_matrix": rotation,
        "uav_angular_velocity_body_rad_s": omega_body,
        "uav_angular_velocity_world_rad_s": omega_world.copy(),
        "uav_inertial_com_world_m": np.asarray(
            data.xipos[entities.drone_id]
        ).copy(),
        "system_com_world_m": system_com.copy(),
        "root_world_m": root,
        "tip_world_m": tip,
        "root_to_tip_world_m": tip - root,
        "root_to_tip_uav_body_m": rotation.T @ (tip - root),
        "tip_minus_uav_world_m": tip - np.asarray(data.xpos[entities.drone_id]),
        "joint_quaternion": joint_quaternions,
        "joint_rotation_vector_rad": np.asarray(
            [_rotation_vector(q) for q in joint_quaternions]
        ),
        "composite_quaternion": composite_quaternion,
        "composite_rotation_vector_rad": _rotation_vector(composite_quaternion),
        "arm_qvel_rad_s": np.concatenate(
            [
                data.qvel[dof_:dof_ + 3]
                for _, _, dof_ in entities.joint_addresses
            ]
        ).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "qacc": np.asarray(data.qacc).copy(),
        "system_linear_momentum_ns": synchronized["system"]["momentum"].copy(),
        "system_angular_momentum_com_nms": synchronized["system"][
            "angular_momentum"
        ].copy(),
        "p_uav_ns": p_uav,
        "p_arm_ns": p_arm,
        "h_uav_com_nms": h_uav,
        "h_arm_com_nms": h_arm,
        "p_partition_residual_ns": accumulator_p,
        "h_partition_residual_nms": accumulator_h,
        "rotor_rpm": np.asarray(rpm).copy(),
        "rotor_force_world_n": synchronized["rotor"]["force"].copy(),
        "rotor_torque_world_nm": synchronized["rotor"]["torque"].copy(),
        "rotor_xfrc_applied": synchronized["rotor"]["wrench"].copy(),
        "rotor_generalized_force": synchronized["rotor"]["qfrc"].copy(),
        "mechanical_energy_j": float(np.sum(data.energy)),
        "tendon_work_j": float(cumulative["tendon_work"]),
        "rotor_work_j": float(cumulative["rotor_work"]),
        "damping_dissipation_j": float(cumulative["damping"]),
        "external_linear_impulse_ns": cumulative["linear_impulse"].copy(),
        "external_angular_impulse_nms": cumulative["angular_impulse"].copy(),
        "linear_impulse_residual_ns": cumulative["linear_residual"].copy(),
        "angular_impulse_residual_nms": cumulative["angular_residual"].copy(),
        "ledger_residual_j": float(cumulative["ledger_residual"]),
    }


def _physics_sample(data, command, synchronized, cumulative, tick, step):
    transmission = synchronized["transmission"]
    return {
        "time": float(data.time),
        "step": int(step),
        "control_tick": int(tick),
        "command": np.asarray(command).copy(),
        "actuator_force": transmission.actuator_force.copy(),
        "tendon_length": transmission.tendon_length.copy(),
        "tendon_velocity": transmission.tendon_velocity.copy(),
        "actuator_power_w": float(transmission.actuator_power),
        "tendon_power_w": float(transmission.tendon_power),
        "rotor_generalized_power_w": float(synchronized["rotor"]["generalized_power"]),
        "rotor_cartesian_power_w": float(synchronized["rotor"]["cartesian_power"]),
        "damping_power_w": float(synchronized["damping_power"]),
        "mechanical_energy_j": float(np.sum(data.energy)),
        "tendon_work_j": float(cumulative["tendon_work"]),
        "rotor_work_j": float(cumulative["rotor_work"]),
        "damping_dissipation_j": float(cumulative["damping"]),
        "ledger_residual_j": float(cumulative["ledger_residual"]),
        "system_linear_momentum_ns": synchronized["system"]["momentum"].copy(),
        "system_angular_momentum_com_nms": synchronized["system"]["angular_momentum"].copy(),
        "net_external_force_world_n": synchronized["system"]["net_force"].copy(),
        "external_torque_system_com_nm": synchronized["system"]["external_torque"].copy(),
        "gravity_torque_system_com_nm": synchronized["system"]["gravity_torque"].copy(),
        "external_linear_impulse_ns": cumulative["linear_impulse"].copy(),
        "external_angular_impulse_nms": cumulative["angular_impulse"].copy(),
        "linear_impulse_residual_ns": cumulative["linear_residual"].copy(),
        "angular_impulse_residual_nms": cumulative["angular_residual"].copy(),
        "rotor_force_world_n": synchronized["rotor"]["force"].copy(),
        "rotor_torque_world_nm": synchronized["rotor"]["torque"].copy(),
        "rotor_xfrc_applied": synchronized["rotor"]["wrench"].copy(),
        "rotor_generalized_force_norm": float(np.linalg.norm(synchronized["rotor"]["qfrc"])),
        "warnings": np.asarray(synchronized["warnings"], dtype=int),
        "contacts": int(data.ncon),
        "minimum_tendon_margin_m": float(np.min(synchronized["margins"])),
        "maximum_abs_qvel": float(synchronized["qvel_max"]),
        "maximum_abs_qacc": float(synchronized["qacc_max"]),
    }


def _freeze_records(records):
    if not records:
        return MappingProxyType({})
    frozen = {}
    for name in records[0]:
        values = np.asarray([record[name] for record in records])
        values.setflags(write=False)
        frozen[name] = values
    return MappingProxyType(frozen)


def _freeze_array(values):
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


def _relative_records(control, baseline):
    if len(control["time"]) != len(baseline["time"]):
        size = min(len(control["time"]), len(baseline["time"]))
    else:
        size = len(control["time"])
    relative_rotation = np.einsum(
        "nij,nkj->nik",
        control["uav_rotation_matrix"][:size],
        baseline["uav_rotation_matrix"][:size],
    )
    relative_quaternion = np.asarray(
        [
            _relative_quaternion(current, reference)
            for current, reference in zip(
                control["uav_quaternion"][:size],
                baseline["uav_quaternion"][:size],
            )
        ]
    )
    relative_rotation_vector = np.asarray(
        [_rotation_vector(q) for q in relative_quaternion]
    )
    joint_relative_quaternion = np.asarray(
        [
            [
                _relative_quaternion(current, reference)
                for current, reference in zip(current_set, reference_set)
            ]
            for current_set, reference_set in zip(
                control["joint_quaternion"][:size],
                baseline["joint_quaternion"][:size],
            )
        ]
    )
    joint_relative_rotation_vector = np.asarray(
        [
            [_rotation_vector(q) for q in sample]
            for sample in joint_relative_quaternion
        ]
    )
    composite_relative_quaternion = np.asarray(
        [
            _relative_quaternion(current, reference)
            for current, reference in zip(
                control["composite_quaternion"][:size],
                baseline["composite_quaternion"][:size],
            )
        ]
    )
    result = {
        "time": control["time"][:size].copy(),
        "uav_position_world_m": (
            control["uav_position_world_m"][:size]
            - baseline["uav_position_world_m"][:size]
        ),
        "uav_linear_velocity_world_m_s": (
            control["uav_linear_velocity_world_m_s"][:size]
            - baseline["uav_linear_velocity_world_m_s"][:size]
        ),
        "uav_rotation_matrix": relative_rotation,
        "uav_quaternion": relative_quaternion,
        "uav_rotation_vector_rad": relative_rotation_vector,
        "uav_quaternion_geodesic_rad": np.linalg.norm(
            relative_rotation_vector, axis=1
        ),
        "root_to_tip_uav_body_m": (
            control["root_to_tip_uav_body_m"][:size]
            - baseline["root_to_tip_uav_body_m"][:size]
        ),
        "tip_minus_uav_world_m": (
            control["tip_minus_uav_world_m"][:size]
            - baseline["tip_minus_uav_world_m"][:size]
        ),
        "joint_quaternion": joint_relative_quaternion,
        "joint_rotation_vector_rad": joint_relative_rotation_vector,
        "composite_quaternion": composite_relative_quaternion,
        "composite_rotation_vector_rad": np.asarray(
            [_rotation_vector(q) for q in composite_relative_quaternion]
        ),
        "tendon_excursion_m": (
            control["tendon_length"][:size]
            - control["tendon_length"][0]
            - baseline["tendon_length"][:size]
            + baseline["tendon_length"][0]
        ),
        "arm_qvel_rad_s": (
            control["arm_qvel_rad_s"][:size]
            - baseline["arm_qvel_rad_s"][:size]
        ),
    }
    for values in result.values():
        values.setflags(write=False)
    return MappingProxyType(result)


def _safety_accumulator():
    return {
        "maximum_contacts": 0,
        "minimum_tendon_margin_m": math.inf,
        "maximum_abs_qvel_native": 0.0,
        "maximum_abs_qacc_native": 0.0,
        "maximum_root_to_tip_length_m": 0.0,
        "maximum_force_command_error_n": 0.0,
        "maximum_actuator_moment_residual": 0.0,
        "maximum_tendon_power_identity_residual_w": 0.0,
        "maximum_rotor_power_residual_w": 0.0,
        "maximum_p_partition_residual_ns": 0.0,
        "maximum_h_partition_residual_nms": 0.0,
        "maximum_gravity_torque_residual_nm": 0.0,
    }


def run_c1_b3(config, baseline=None):
    """Run one C1-B3 trajectory and optionally attach baseline-relative data."""
    if not isinstance(config, C1B3RunConfig):
        raise TypeError("config must be C1B3RunConfig")
    if baseline is not None:
        if baseline.config.sim_freq != config.sim_freq:
            raise ValueError("baseline frequency must match active trace")
        if baseline.config.case != "baseline":
            raise ValueError("relative reference must be a baseline trace")
    env = None
    try:
        env = _make_active_production_env(config)
        production_model = env.model
        manifest = env.SCALED_STABLESOFT_ARM_MANIFEST
        if manifest is None:
            raise AssertionError("production active build returned no manifest")
        hover_xml = _remove_floor_preserving_gravity(env._generated_xml)
        model = mujoco.MjModel.from_xml_string(hover_xml)
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        data = mujoco.MjData(model)
        topology = _verify_hover_transformation(
            production_model, model, manifest
        )
        env.model, env.data = model, data
        bindings = resolve_ideal_tendon_actuator_bindings(model, manifest)
        _verify_bindings(model, manifest, bindings)
        env.SCALED_STABLESOFT_TENDON_BINDINGS = bindings
        bindings_by_corner = _binding_by_corner(bindings)
        entities = _resolve_entities(model, manifest, bindings)
        if not math.isclose(env.M, entities.system_mass, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"environment/model mass mismatch: {env.M} != {entities.system_mass}"
            )
        tendon_ranges = np.asarray(model.tendon_range)[
            np.asarray(entities.tendon_ids, dtype=int)
        ].copy()

        env.reset()
        data.qvel[:] = 0.0
        command0 = _command_at_tick(config, bindings_by_corner, 0)
        env.set_scaled_stablesoft_tension_command(command0)
        rpm = np.full(4, env.HOVER_RPM)
        env._physics(rpm, 0)
        mujoco.mj_forward(model, data)
        mujoco.mj_energyPos(model, data)
        mujoco.mj_energyVel(model, data)

        accumulator = _safety_accumulator()
        initial_sync = _synchronized(
            model,
            data,
            config,
            bindings,
            entities,
            command0,
            tendon_ranges,
            accumulator,
            stage="initial_forward",
            step=0,
            tick=0,
            substep=0,
            stepping=False,
        )
        initial_root_tip_body = (
            np.asarray(data.xmat[entities.drone_id]).reshape(3, 3).T
            @ (
                data.site_xpos[entities.tip_site_id]
                - data.site_xpos[entities.root_site_id]
            )
        )
        force_balance = (
            initial_sync["rotor"]["force"]
            + entities.system_mass * np.asarray(model.opt.gravity)
        )
        initial_hover = C1B3InitialHoverProof(
            environment_mass_kg=float(env.M),
            model_system_mass_kg=entities.system_mass,
            arm_mass_kg=float(env.ARM_MASS),
            hover_rpm=float(env.HOVER_RPM),
            maximum_rpm=float(env.MAX_RPM),
            rotor_rpm=tuple(float(x) for x in rpm),
            rotor_force_world_n=tuple(
                float(x) for x in initial_sync["rotor"]["force"]
            ),
            rotor_torque_world_nm=tuple(
                float(x) for x in initial_sync["rotor"]["torque"]
            ),
            force_balance_world_n=tuple(float(x) for x in force_balance),
            initial_quaternion=tuple(
                float(x)
                for x in data.qpos[
                    entities.drone_qpos_address + 3:
                    entities.drone_qpos_address + 7
                ]
            ),
            initial_qvel_max=float(np.max(np.abs(data.qvel))),
            initial_root_to_tip_body_m=tuple(float(x) for x in initial_root_tip_body),
            initial_contacts=int(data.ncon),
            initial_warning_counts=_warning_counts(data),
        )
        if np.linalg.norm(force_balance) > 1e-14:
            raise AssertionError(f"initial hover force imbalance: {force_balance}")
        if np.linalg.norm(initial_sync["rotor"]["torque"]) > 1e-15:
            raise AssertionError("initial equal-RPM rotor torque is nonzero")

        initial_energy = float(np.sum(data.energy))
        initial_p = initial_sync["system"]["momentum"].copy()
        initial_h = initial_sync["system"]["angular_momentum"].copy()
        cumulative = {
            "tendon_work": 0.0,
            "rotor_work": 0.0,
            "damping": 0.0,
            "linear_impulse": np.zeros(3),
            "angular_impulse": np.zeros(3),
            "linear_residual": np.zeros(3),
            "angular_residual": np.zeros(3),
            "ledger_residual": 0.0,
        }
        physics_records = [
            _physics_sample(data, command0, initial_sync, cumulative, 0, 0)
        ]
        control_records = []
        failure = None
        completed_steps = 0

        try:
            for tick in range(TOTAL_CONTROL_TICKS + 1):
                command = _command_at_tick(config, bindings_by_corner, tick)
                written = env.set_scaled_stablesoft_tension_command(command)
                if written != command:
                    raise AssertionError(
                        f"production setter changed command: {written} != {command}"
                    )
                env._physics(rpm, 0)
                mujoco.mj_forward(model, data)
                mujoco.mj_energyPos(model, data)
                mujoco.mj_energyVel(model, data)
                synchronized = _synchronized(
                    model,
                    data,
                    config,
                    bindings,
                    entities,
                    command,
                    tendon_ranges,
                    accumulator,
                    stage="control_forward",
                    step=completed_steps,
                    tick=tick,
                    substep=0,
                    stepping=completed_steps > 0,
                )
                expected_time = tick / CTRL_FREQ
                _require(
                    abs(float(data.time) - expected_time) <= 1e-12,
                    config,
                    data,
                    stage="control_forward",
                    step=completed_steps,
                    tick=tick,
                    substep=0,
                    field="control_time_alignment_s",
                    value={"actual": float(data.time), "expected": expected_time},
                    stepping=completed_steps > 0,
                )
                cumulative["linear_residual"] = (
                    synchronized["system"]["momentum"]
                    - initial_p
                    - cumulative["linear_impulse"]
                )
                cumulative["angular_residual"] = (
                    synchronized["system"]["angular_momentum"]
                    - initial_h
                    - cumulative["angular_impulse"]
                )
                cumulative["ledger_residual"] = (
                    float(np.sum(data.energy))
                    - initial_energy
                    - (
                        cumulative["tendon_work"]
                        + cumulative["rotor_work"]
                        - cumulative["damping"]
                    )
                )
                sample = _control_sample(
                    model,
                    data,
                    config,
                    entities,
                    command,
                    rpm,
                    synchronized,
                    cumulative,
                    tick,
                    completed_steps,
                )
                accumulator["maximum_p_partition_residual_ns"] = max(
                    accumulator["maximum_p_partition_residual_ns"],
                    sample["p_partition_residual_ns"],
                )
                accumulator["maximum_h_partition_residual_nms"] = max(
                    accumulator["maximum_h_partition_residual_nms"],
                    sample["h_partition_residual_nms"],
                )
                control_records.append(sample)
                if tick == TOTAL_CONTROL_TICKS:
                    break

                previous = synchronized
                for substep in range(1, config.sample_stride + 1):
                    next_step = completed_steps + 1
                    env._physics(rpm, 0)
                    mujoco.mj_step(model, data)
                    raw_ok = (
                        math.isfinite(float(data.time))
                        and np.all(np.isfinite(data.qpos))
                        and np.all(np.isfinite(data.qvel))
                    )
                    _require(
                        raw_ok,
                        config,
                        data,
                        stage="after_step",
                        step=next_step,
                        tick=tick,
                        substep=substep,
                        field="raw_state_finite",
                        value=raw_ok,
                        stepping=True,
                    )
                    mujoco.mj_forward(model, data)
                    mujoco.mj_energyPos(model, data)
                    mujoco.mj_energyVel(model, data)
                    current = _synchronized(
                        model,
                        data,
                        config,
                        bindings,
                        entities,
                        command,
                        tendon_ranges,
                        accumulator,
                        stage="after_forward",
                        step=next_step,
                        tick=tick,
                        substep=substep,
                        stepping=True,
                    )
                    dt = 1.0 / config.sim_freq
                    cumulative["tendon_work"] += 0.5 * (
                        previous["transmission"].actuator_power
                        + current["transmission"].actuator_power
                    ) * dt
                    cumulative["rotor_work"] += 0.5 * (
                        previous["rotor"]["generalized_power"]
                        + current["rotor"]["generalized_power"]
                    ) * dt
                    cumulative["damping"] += 0.5 * (
                        previous["damping_power"]
                        + current["damping_power"]
                    ) * dt
                    cumulative["linear_impulse"] += 0.5 * (
                        previous["system"]["net_force"]
                        + current["system"]["net_force"]
                    ) * dt
                    cumulative["angular_impulse"] += 0.5 * (
                        previous["system"]["external_torque"]
                        + current["system"]["external_torque"]
                    ) * dt
                    cumulative["linear_residual"] = (
                        current["system"]["momentum"]
                        - initial_p
                        - cumulative["linear_impulse"]
                    )
                    cumulative["angular_residual"] = (
                        current["system"]["angular_momentum"]
                        - initial_h
                        - cumulative["angular_impulse"]
                    )
                    cumulative["ledger_residual"] = (
                        float(np.sum(data.energy))
                        - initial_energy
                        - (
                            cumulative["tendon_work"]
                            + cumulative["rotor_work"]
                            - cumulative["damping"]
                        )
                    )
                    completed_steps = next_step
                    physics_records.append(
                        _physics_sample(
                            data,
                            command,
                            current,
                            cumulative,
                            tick,
                            completed_steps,
                        )
                    )
                    previous = current
        except _SteppingFailure as exc:
            failure = exc.failure

        physics = _freeze_records(physics_records)
        control = _freeze_records(control_records)
        tendon_ranges = _freeze_array(tendon_ranges)
        safety = C1B3SafetySummary(
            requested_steps=config.requested_steps,
            completed_steps=completed_steps,
            first_failure=failure,
            **accumulator,
        )
        trace = C1B3Trace(
            config=config,
            status="completed" if failure is None else "failed",
            topology=topology,
            initial_hover=initial_hover,
            bindings=tuple(bindings),
            tendon_ranges_m=tendon_ranges,
            physics=physics,
            control=control,
            relative=MappingProxyType({}),
            safety=safety,
        )
        reference_control = control if baseline is None else baseline.control
        return replace(
            trace,
            relative=_relative_records(control, reference_control),
        )
    finally:
        if env is not None:
            env.close()


def run_manual_tension_diagnostic(sim_freq, tension_n):
    """Explicit non-CI entry point for 0.005/0.010 N single-tendon probes."""
    if tension_n not in (0.005, 0.01):
        raise ValueError("manual diagnostic tension must be 0.005 or 0.01 N")
    baseline = run_c1_b3(C1B3RunConfig(sim_freq, "baseline"))
    return run_c1_b3(
        C1B3RunConfig(sim_freq, "single_xp_yp", tension_n=tension_n),
        baseline=baseline,
    )


def _window(trace, end_s):
    return trace.physics["time"] <= end_s + 1e-12


def impulse_metrics(trace):
    result = {}
    for name, end in (
        ("forced", FORCED_END_S),
        ("early", EARLY_END_S),
        ("full", DURATION_S),
    ):
        mask = _window(trace, end)
        p_residual = np.linalg.norm(
            trace.physics["linear_impulse_residual_ns"][mask], axis=1
        )
        h_residual = np.linalg.norm(
            trace.physics["angular_impulse_residual_nms"][mask], axis=1
        )
        linear_impulse = np.linalg.norm(
            trace.physics["external_linear_impulse_ns"][mask], axis=1
        )
        angular_impulse = np.linalg.norm(
            trace.physics["external_angular_impulse_nms"][mask], axis=1
        )
        delta_h = np.linalg.norm(
            trace.physics["system_angular_momentum_com_nms"][mask]
            - trace.physics["system_angular_momentum_com_nms"][0],
            axis=1,
        )
        result[name] = MappingProxyType(
            {
                "maximum_linear_residual_ns": float(np.max(p_residual)),
                "maximum_angular_residual_nms": float(np.max(h_residual)),
                "peak_linear_impulse_ns": float(np.max(linear_impulse)),
                "peak_angular_impulse_nms": float(np.max(angular_impulse)),
                "peak_delta_h_nms": float(np.max(delta_h)),
            }
        )
    return MappingProxyType(result)


def energy_window_metrics(trace):
    result = {}
    for name, end in (
        ("forced", FORCED_END_S),
        ("early", EARLY_END_S),
        ("full", DURATION_S),
    ):
        mask = _window(trace, end)
        combined = (
            trace.physics["tendon_work_j"][mask]
            + trace.physics["rotor_work_j"][mask]
        )
        result[name] = MappingProxyType(
            {
                "maximum_abs_ledger_residual_j": float(
                    np.max(np.abs(trace.physics["ledger_residual_j"][mask]))
                ),
                "peak_abs_combined_work_j": float(np.max(np.abs(combined))),
            }
        )
    return MappingProxyType(result)


def response_metrics(trace):
    relative = trace.relative
    release = RELEASE_TICK
    early = int(round(EARLY_END_S * CTRL_FREQ))
    tip_norm = np.linalg.norm(relative["root_to_tip_uav_body_m"], axis=1)
    bend_norm = np.linalg.norm(
        relative["composite_rotation_vector_rad"], axis=1
    )
    return MappingProxyType(
        {
            "release_position_m": float(
                np.linalg.norm(relative["uav_position_world_m"][release])
            ),
            "release_attitude_rad": float(
                relative["uav_quaternion_geodesic_rad"][release]
            ),
            "release_tip_m": float(tip_norm[release]),
            "release_bend_rad": float(bend_norm[release]),
            "release_tendon_excursion_m": float(
                np.max(np.abs(relative["tendon_excursion_m"][release]))
            ),
            "early_tip_m": float(tip_norm[early]),
            "early_bend_rad": float(bend_norm[early]),
            "early_tendon_excursion_m": float(
                np.max(np.abs(relative["tendon_excursion_m"][early]))
            ),
            "final_tip_m": float(tip_norm[-1]),
            "final_bend_rad": float(bend_norm[-1]),
            "final_tendon_excursion_m": float(
                np.max(np.abs(relative["tendon_excursion_m"][-1]))
            ),
            "maximum_tip_m": float(np.max(tip_norm)),
            "maximum_bend_rad": float(np.max(bend_norm)),
            "maximum_tendon_excursion_m": float(
                np.max(np.abs(relative["tendon_excursion_m"]))
            ),
            "final_position_m": float(
                np.linalg.norm(relative["uav_position_world_m"][-1])
            ),
            "final_attitude_rad": float(
                relative["uav_quaternion_geodesic_rad"][-1]
            ),
        }
    )


def convergence_metrics(candidate, reference):
    if candidate.config.sim_freq != 2400 or reference.config.sim_freq != 4800:
        raise ValueError("convergence requires 2400 and 4800 Hz traces")
    if candidate.config.case != reference.config.case:
        raise ValueError("convergence cases must match")
    left, right = candidate.control, reference.control

    def vector(name, floor, fraction, *, relative=False):
        a = candidate.relative[name] if relative else left[name]
        b = reference.relative[name] if relative else right[name]
        error = float(np.max(np.linalg.norm(a - b, axis=-1)))
        scale = float(np.max(np.linalg.norm(b, axis=-1)))
        return {"error": error, "limit": max(floor, fraction * scale)}

    def scalar(name, floor, fraction):
        error = float(np.max(np.abs(left[name] - right[name])))
        scale = float(np.max(np.abs(right[name])))
        return {"error": error, "limit": max(floor, fraction * scale)}

    metrics = {
        "time": {
            "error": float(np.max(np.abs(left["time"] - right["time"]))),
            "limit": 1e-12,
        },
        "command": scalar("command", 1e-15, 0.0),
        "actuator_force": scalar("actuator_force", 1e-15, 0.0),
        "position": vector("uav_position_world_m", 2e-5, 5e-4, relative=True),
        "velocity": vector("uav_linear_velocity_world_m_s", 2e-3, 1e-3, relative=True),
        "tip_body": vector("root_to_tip_uav_body_m", 3e-6, 0.02, relative=True),
        "linear_momentum": vector("system_linear_momentum_ns", 5e-5, 0.002),
        "angular_momentum": vector("system_angular_momentum_com_nms", 3e-8, 0.02),
        "mechanical_energy": scalar("mechanical_energy_j", 2e-4, 0.002),
        "tendon_work": scalar("tendon_work_j", 1e-9, 0.02),
        "rotor_work": scalar("rotor_work_j", 2e-5, 0.002),
        "damping": scalar("damping_dissipation_j", 1e-9, 0.02),
    }
    attitude_error = float(
        np.max(
            [
                np.linalg.norm(
                    _rotation_vector(_relative_quaternion(left_q, right_q))
                )
                for left_q, right_q in zip(
                    candidate.relative["uav_quaternion"],
                    reference.relative["uav_quaternion"],
                )
            ]
        )
    )
    attitude_scale = float(
        np.max(reference.relative["uav_quaternion_geodesic_rad"])
    )
    metrics["attitude"] = {
        "error": attitude_error,
        "limit": max(1e-5, 1e-3 * attitude_scale),
    }
    joint_error = float(
        np.max(
            [
                np.linalg.norm(
                    _rotation_vector(_relative_quaternion(left_q, right_q))
                )
                for left_set, right_set in zip(
                    candidate.relative["joint_quaternion"],
                    reference.relative["joint_quaternion"],
                )
                for left_q, right_q in zip(left_set, right_set)
            ]
        )
    )
    joint_scale = float(
        np.max(
            np.linalg.norm(
                reference.relative["joint_rotation_vector_rad"], axis=-1
            )
        )
    )
    metrics["joint_rotation"] = {
        "error": joint_error,
        "limit": max(1e-4, 0.02 * joint_scale),
    }
    composite_error = float(
        np.max(
            [
                np.linalg.norm(
                    _rotation_vector(_relative_quaternion(left_q, right_q))
                )
                for left_q, right_q in zip(
                    candidate.relative["composite_quaternion"],
                    reference.relative["composite_quaternion"],
                )
            ]
        )
    )
    composite_scale = float(
        np.max(
            np.linalg.norm(
                reference.relative["composite_rotation_vector_rad"], axis=1
            )
        )
    )
    metrics["composite_rotation"] = {
        "error": composite_error,
        "limit": max(1e-4, 0.02 * composite_scale),
    }
    combined_reference = right["tendon_work_j"] + right["rotor_work_j"]
    metrics["ledger"] = {
        "error": float(
            np.max(
                np.abs(left["ledger_residual_j"] - right["ledger_residual_j"])
            )
        ),
        "limit": ledger_convergence_limit(
            np.max(np.abs(combined_reference))
        ),
    }
    for name, floor in (
        ("linear_impulse_residual_ns", 5e-5),
        ("angular_impulse_residual_nms", 1e-6),
    ):
        metrics[name] = {
            "error": float(np.max(np.linalg.norm(left[name] - right[name], axis=1))),
            "limit": floor,
        }
    candidate_extrema = np.stack(
        (candidate.physics["tendon_length"].min(0), candidate.physics["tendon_length"].max(0))
    )
    reference_extrema = np.stack(
        (reference.physics["tendon_length"].min(0), reference.physics["tendon_length"].max(0))
    )
    metrics["tendon_length_extrema"] = {
        "error": float(np.max(np.abs(candidate_extrema - reference_extrema))),
        "limit": 2e-8,
    }
    control_velocity_error = float(
        np.max(
            np.abs(
                left["tendon_velocity"] - right["tendon_velocity"]
            )
        )
    )
    control_velocity_scale = float(
        np.max(np.abs(right["tendon_velocity"]))
    )
    metrics["tendon_velocity_control_grid"] = {
        "error": control_velocity_error,
        "limit": max(1e-6, 0.01 * control_velocity_scale),
    }
    for window_name, end_s, absolute_floor, relative_fraction in (
        ("forced", FORCED_END_S, 1e-6, 0.0),
        ("early", EARLY_END_S, 1e-6, 0.02),
        ("full", DURATION_S, 1e-6, 0.02),
    ):
        candidate_window = candidate.physics["tendon_velocity"][
            candidate.physics["time"] <= end_s + 1e-12
        ]
        reference_window = reference.physics["tendon_velocity"][
            reference.physics["time"] <= end_s + 1e-12
        ]
        candidate_velocity_extrema = np.stack(
            (candidate_window.min(0), candidate_window.max(0))
        )
        reference_velocity_extrema = np.stack(
            (reference_window.min(0), reference_window.max(0))
        )
        error = float(
            np.max(
                np.abs(
                    candidate_velocity_extrema - reference_velocity_extrema
                )
            )
        )
        scale = float(np.max(np.abs(reference_window)))
        metrics[f"tendon_velocity_{window_name}_extrema"] = {
            "error": error,
            "limit": max(absolute_floor, relative_fraction * scale),
        }
    metrics["passed"] = all(
        item["error"] <= item["limit"]
        for key, item in metrics.items()
        if key != "passed"
    )
    return MappingProxyType(metrics)


def diagnostic_summary(trace):
    failure = trace.safety.first_failure
    return {
        "case": trace.config.case,
        "sim_freq": trace.config.sim_freq,
        "status": trace.status,
        "completed_steps": trace.safety.completed_steps,
        "failure": None
        if failure is None
        else {
            "stage": failure.stage,
            "physics_step": failure.physics_step,
            "control_tick": failure.control_tick,
            "time_s": failure.time_s,
            "field": failure.field,
            "value": failure.value,
        },
        "latest_command_n": trace.physics["command"][-1].tolist(),
        "response": dict(response_metrics(trace)),
        "impulse": {key: dict(value) for key, value in impulse_metrics(trace).items()},
        "energy": {key: dict(value) for key, value in energy_window_metrics(trace).items()},
        "safety": trace.safety,
    }
