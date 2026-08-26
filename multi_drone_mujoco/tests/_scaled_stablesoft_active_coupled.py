"""C1-B2 active free-base coupling helpers.

The formal model is derived from the active production aviary XML entirely in
memory.  This module does not write XML, traces, logs, or frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Optional

import mujoco
import numpy as np

from multi_drone_mujoco.control.ideal_tendon_tension import (
    resolve_ideal_tendon_actuator_bindings,
)
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    half_cosine_ramp_scale_at_tick,
    sample_ideal_tendon_transmission,
)
from multi_drone_mujoco.tests._scaled_stablesoft_coupled import (
    _body_groups,
    _component_momentum,
    _object_names,
    _remove_floor_and_zero_gravity,
    _slope,
    _verify_transformation,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    composite_rotation_vector,
    joint_rotation_vectors,
    named_joint_addresses,
)
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


CTRL_FREQ = 240
DURATION_S = 1.0
RAMP_DURATION_S = 0.25
FINAL_TENSION_N = 0.002
MAX_TENSION_N = 0.01
TARGET_CORNER = "xp_yp"
JOINT_STIFFNESS = 0.003086
TARGET_DAMPING_RATIO = 0.3
TOTAL_ARM_LENGTH_M = 0.08
TOTAL_SYSTEM_MASS_KG = 0.030
QVEL_HARD_LIMIT = 1000.0
QACC_HARD_LIMIT = 2.0e5


@dataclass(frozen=True)
class C1B2RunConfig:
    sim_freq: int
    ctrl_freq: int = CTRL_FREQ
    duration_s: float = DURATION_S
    ramp_duration_s: float = RAMP_DURATION_S
    final_tension_n: float = FINAL_TENSION_N
    target_corner: str = TARGET_CORNER
    integrator: str = "implicitfast"

    def __post_init__(self):
        if self.sim_freq not in (2400, 4800):
            raise ValueError("C1-B2 sim_freq must be 2400 or 4800")
        if self.ctrl_freq != CTRL_FREQ:
            raise ValueError("C1-B2 ctrl_freq must be 240")
        if self.sim_freq % self.ctrl_freq:
            raise ValueError("sim_freq must be divisible by ctrl_freq")
        if self.duration_s != DURATION_S:
            raise ValueError("C1-B2 duration_s must be 1.0")
        if self.ramp_duration_s != RAMP_DURATION_S:
            raise ValueError("C1-B2 ramp_duration_s must be 0.25")
        if self.final_tension_n != FINAL_TENSION_N:
            raise ValueError("C1-B2 final_tension_n must be 0.002")
        if self.target_corner != TARGET_CORNER:
            raise ValueError("C1-B2 target_corner must be xp_yp")
        if self.integrator != "implicitfast":
            raise ValueError("C1-B2 integrator must be implicitfast")
        if self.ramp_ticks != 60 or self.total_control_ticks != 240:
            raise ValueError("C1-B2 durations must align to the control grid")

    @property
    def sample_stride(self) -> int:
        return self.sim_freq // self.ctrl_freq

    @property
    def ramp_ticks(self) -> int:
        return int(round(self.ramp_duration_s * self.ctrl_freq))

    @property
    def total_control_ticks(self) -> int:
        return int(round(self.duration_s * self.ctrl_freq))

    @property
    def requested_steps(self) -> int:
        return int(round(self.duration_s * self.sim_freq))


@dataclass(frozen=True)
class C1B2Failure:
    stage: str
    physics_step: int
    control_tick: int
    substep: int
    time_s: float
    field: str
    value: object


@dataclass(frozen=True)
class C1B2TopologyProof:
    production_counts: tuple
    free_counts: tuple
    production_gravity: tuple
    free_gravity: tuple
    production_floor_count: int
    free_floor_count: int
    named_topology_match: bool
    mass_inertia_routing_match: bool
    timestep_match: bool
    integrator_match: bool
    free_joint_name: str
    uav_body_names: tuple
    arm_body_names: tuple
    system_mass_kg: float
    tendon_names: tuple
    actuator_names: tuple
    routing_site_body_names: tuple


@dataclass(frozen=True)
class C1B2SafetySummary:
    requested_steps: int
    completed_steps: int
    first_failure: Optional[C1B2Failure]
    maximum_contacts: int
    minimum_tendon_margin_m: float
    maximum_abs_qvel_native: float
    maximum_abs_qacc_native: float
    maximum_root_to_tip_length_m: float
    maximum_force_command_error_n: float
    maximum_length_sign_residual_m: float
    maximum_velocity_sign_residual_m_s: float
    maximum_moment_residual: float
    maximum_generalized_power_residual_w: float
    maximum_actuator_tendon_power_residual_w: float
    maximum_command_tendon_power_residual_w: float
    maximum_abs_rotor_generalized_force: float
    maximum_abs_rotor_power_w: float


@dataclass(frozen=True)
class C1B2Trace:
    config: C1B2RunConfig
    status: str
    topology: C1B2TopologyProof
    bindings: tuple
    target_binding_index: int
    tendon_ranges_m: np.ndarray
    tendon_limited: np.ndarray
    physics: MappingProxyType
    control: MappingProxyType
    safety: C1B2SafetySummary


@dataclass(frozen=True)
class _ResolvedEntities:
    drone_id: int
    drone_qpos_address: int
    drone_dof_address: int
    root_site_id: int
    tip_site_id: int
    joint_addresses: tuple
    tendon_ids: tuple
    uav_body_ids: tuple
    arm_body_ids: tuple
    system_mass: float


class _SteppingFailure(Exception):
    def __init__(self, failure: C1B2Failure):
        self.failure = failure
        super().__init__(str(failure))


def _arm_spec():
    return ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )


def _make_active_production_env(config):
    return BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.zeros((1, 3)),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.MJC,
        sim_freq=config.sim_freq,
        ctrl_freq=config.ctrl_freq,
        gui=False,
        record=False,
        obstacles=False,
        vision_attributes=False,
        act_type=ActionType.RPM,
        render_mode=None,
        scaled_stablesoft_arm=_arm_spec(),
        scaled_stablesoft_tendon_drive=(
            IdealTendonTensionSpec(max_tension=MAX_TENSION_N)
        ),
        integrator=config.integrator,
    )


def _active_counts(model):
    return (
        model.nbody,
        model.njnt,
        model.nq,
        model.nv,
        model.ngeom,
        model.nsite,
        model.ntendon,
        model.nu,
        model.nout,
        model.nsensor,
    )


def _verify_active_free_space(production_model, free_model, manifest):
    passive_proof = _verify_transformation(production_model, free_model)
    production_counts = _active_counts(production_model)
    free_counts = _active_counts(free_model)
    expected_counts = list(production_counts)
    expected_counts[4] -= 1
    if tuple(expected_counts) != free_counts:
        raise AssertionError(
            "active free-space count mismatch: "
            f"production={production_counts}, free={free_counts}"
        )

    actuator_names_before = _object_names(
        production_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        production_model.nu,
    )
    actuator_names_after = _object_names(
        free_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        free_model.nu,
    )
    if actuator_names_before != actuator_names_after:
        raise AssertionError("free-space transform changed actuator names")
    production_floor_count = int(
        mujoco.mj_name2id(
            production_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        >= 0
    )
    free_floor_count = int(
        mujoco.mj_name2id(free_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        >= 0
    )
    if production_floor_count != 1 or free_floor_count != 0:
        raise AssertionError(
            "free-space transform did not remove exactly the production floor"
        )
    for field in (
        "actuator_trntype",
        "actuator_trnid",
        "actuator_gear",
        "actuator_ctrllimited",
        "actuator_ctrlrange",
        "actuator_forcelimited",
    ):
        if not np.array_equal(
            getattr(production_model, field),
            getattr(free_model, field),
        ):
            raise AssertionError(
                f"free-space transform changed actuator field {field}"
            )

    drone_id = mujoco.mj_name2id(
        free_model, mujoco.mjtObj.mjOBJ_BODY, "drone0"
    )
    drone_joint_id = mujoco.mj_name2id(
        free_model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint"
    )
    if drone_id < 0 or drone_joint_id < 0:
        raise AssertionError("active free-space model has no drone0 free base")
    if free_model.jnt_type[drone_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise AssertionError("drone0_joint is not a free joint")

    arm_body_ids = tuple(
        mujoco.mj_name2id(
            free_model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        for name in manifest.segment_body_names
    )
    if any(body_id < 0 for body_id in arm_body_ids):
        raise AssertionError("active manifest arm body is missing")
    uav_body_ids, arm_body_ids = _body_groups(
        free_model, drone_id, arm_body_ids
    )
    system_mass = float(free_model.body_subtreemass[drone_id])
    if not math.isclose(
        system_mass,
        TOTAL_SYSTEM_MASS_KG,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError(f"unexpected active system mass: {system_mass}")

    routing_body_names = []
    subtree = {
        body_id
        for body_id in range(1, free_model.nbody)
        if int(free_model.body_rootid[body_id]) == drone_id
    }
    for site_name in manifest.routing_site_names:
        site_id = mujoco.mj_name2id(
            free_model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if site_id < 0:
            raise AssertionError(f"routing site is missing: {site_name}")
        body_id = int(free_model.site_bodyid[site_id])
        if body_id == 0 or body_id not in subtree or body_id not in arm_body_ids:
            raise AssertionError(
                f"routing site is not internal to the arm subtree: {site_name}"
            )
        routing_body_names.append(
            mujoco.mj_id2name(free_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        )

    return C1B2TopologyProof(
        production_counts=production_counts,
        free_counts=free_counts,
        production_gravity=tuple(
            float(value) for value in production_model.opt.gravity
        ),
        free_gravity=tuple(float(value) for value in free_model.opt.gravity),
        production_floor_count=production_floor_count,
        free_floor_count=free_floor_count,
        named_topology_match=bool(passive_proof["named_topology_match"]),
        mass_inertia_routing_match=bool(
            passive_proof["mass_inertia_routing_match"]
        ),
        timestep_match=bool(passive_proof["timestep_match"]),
        integrator_match=bool(passive_proof["integrator_match"]),
        free_joint_name="drone0_joint",
        uav_body_names=tuple(
            mujoco.mj_id2name(free_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in uav_body_ids
        ),
        arm_body_names=tuple(manifest.segment_body_names),
        system_mass_kg=system_mass,
        tendon_names=tuple(manifest.tendon_names),
        actuator_names=tuple(actuator_names_after),
        routing_site_body_names=tuple(routing_body_names),
    )


def _verify_bindings(model, manifest, bindings):
    if len(bindings) != 4:
        raise AssertionError(f"expected four active bindings, got {len(bindings)}")
    for tendon_name, actuator_name, binding in zip(
        manifest.tendon_names,
        manifest.tendon_actuator_names,
        bindings,
    ):
        if binding.tendon_name != tendon_name:
            raise AssertionError("binding tendon order differs from manifest")
        if binding.actuator_name != actuator_name:
            raise AssertionError("binding actuator order differs from manifest")
        actuator_id = binding.actuator_id
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_TENDON:
            raise AssertionError(f"not a tendon actuator: {actuator_name}")
        if int(model.actuator_trnid[actuator_id, 0]) != binding.tendon_id:
            raise AssertionError(f"transmission mismatch: {actuator_name}")
        np.testing.assert_array_equal(
            model.actuator_gear[actuator_id],
            np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        if not bool(model.actuator_ctrllimited[actuator_id]):
            raise AssertionError(f"control is not limited: {actuator_name}")
        np.testing.assert_allclose(
            model.actuator_ctrlrange[actuator_id],
            [0.0, MAX_TENSION_N],
            rtol=0.0,
            atol=1e-15,
        )
        if bool(model.actuator_forcelimited[actuator_id]):
            raise AssertionError(
                f"unexpected independent force limit: {actuator_name}"
            )
        if int(model.actuator_ctrlnum[actuator_id]) != 1:
            raise AssertionError(f"invalid control width: {actuator_name}")
        if int(model.actuator_outnum[actuator_id]) != 1:
            raise AssertionError(f"invalid output width: {actuator_name}")


def _resolve_entities(model, manifest, bindings):
    drone_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "drone0"
    )
    drone_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint"
    )
    root_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_root_site_name
    )
    tip_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name
    )
    ids = (drone_id, drone_joint_id, root_site_id, tip_site_id)
    if any(object_id < 0 for object_id in ids):
        raise AssertionError(f"failed to resolve active entities: {ids}")
    arm_body_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in manifest.segment_body_names
    )
    uav_body_ids, arm_body_ids = _body_groups(
        model, drone_id, arm_body_ids
    )
    return _ResolvedEntities(
        drone_id=drone_id,
        drone_qpos_address=int(model.jnt_qposadr[drone_joint_id]),
        drone_dof_address=int(model.jnt_dofadr[drone_joint_id]),
        root_site_id=root_site_id,
        tip_site_id=tip_site_id,
        joint_addresses=named_joint_addresses(model, manifest),
        tendon_ids=tuple(binding.tendon_id for binding in bindings),
        uav_body_ids=uav_body_ids,
        arm_body_ids=arm_body_ids,
        system_mass=float(model.body_subtreemass[drone_id]),
    )


def _target_binding_index(bindings, corner):
    matches = [
        index
        for index, binding in enumerate(bindings)
        if binding.tendon_name.endswith(f"_{corner}_tendon")
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one manifest tendon for {corner}, got {matches}"
        )
    return matches[0]


def _command_at_tick(config, bindings, target_index, control_tick):
    scale = half_cosine_ramp_scale_at_tick(
        control_tick,
        config.ramp_ticks,
    )
    command = np.zeros(len(bindings))
    command[target_index] = config.final_tension_n * scale
    return tuple(float(value) for value in command)


def _warning_counts(data):
    return tuple(int(warning.number) for warning in data.warning)


def _raise_failure(
    data,
    *,
    stage,
    physics_step,
    control_tick,
    substep,
    field,
    value,
):
    raise _SteppingFailure(
        C1B2Failure(
            stage=stage,
            physics_step=int(physics_step),
            control_tick=int(control_tick),
            substep=int(substep),
            time_s=float(data.time),
            field=field,
            value=value,
        )
    )


def _require(
    condition,
    data,
    *,
    stage,
    physics_step,
    control_tick,
    substep,
    field,
    value,
):
    if not bool(condition):
        _raise_failure(
            data,
            stage=stage,
            physics_step=physics_step,
            control_tick=control_tick,
            substep=substep,
            field=field,
            value=value,
        )


def _rotor_generalized_force(model, data, entities):
    wrench = np.asarray(data.xfrc_applied[entities.drone_id]).copy()
    generalized = np.zeros(model.nv)
    mujoco.mj_applyFT(
        model,
        data,
        wrench[:3],
        wrench[3:],
        np.asarray(data.xipos[entities.drone_id]),
        entities.drone_id,
        generalized,
    )
    return wrench, generalized, float(generalized @ data.qvel)


def _update_energy(model, data):
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)


def _subtree_values(model, data, entities):
    mujoco.mj_subtreeVel(model, data)
    com = np.asarray(data.subtree_com[entities.drone_id]).copy()
    momentum = (
        entities.system_mass
        * np.asarray(data.subtree_linvel[entities.drone_id]).copy()
    )
    angular = np.asarray(data.subtree_angmom[entities.drone_id]).copy()
    return com, momentum, angular


def _finite_arrays(model, data, transmission):
    return {
        "qpos": data.qpos,
        "qvel": data.qvel,
        "qacc": data.qacc,
        "site_xpos": data.site_xpos,
        "tendon_length": transmission.tendon_length,
        "tendon_velocity": transmission.tendon_velocity,
        "actuator_length": transmission.actuator_length,
        "actuator_velocity": transmission.actuator_velocity,
        "actuator_force": transmission.actuator_force,
        "qfrc_actuator": data.qfrc_actuator,
        "qfrc_damper": data.qfrc_damper,
        "energy": data.energy,
        "actuator_moment": np.asarray(data.actuator_moment).reshape(
            model.nout, model.nv
        ),
    }


def _check_synchronized(
    model,
    data,
    config,
    bindings,
    entities,
    command,
    tendon_ranges,
    tendon_limited,
    accumulator,
    *,
    stage,
    physics_step,
    control_tick,
    substep,
):
    transmission = sample_ideal_tendon_transmission(
        model, data, bindings, command
    )
    context = dict(
        stage=stage,
        physics_step=physics_step,
        control_tick=control_tick,
        substep=substep,
    )
    for field, values in _finite_arrays(model, data, transmission).items():
        _require(
            np.all(np.isfinite(values)),
            data,
            field=f"finite:{field}",
            value=np.asarray(values).shape,
            **context,
        )
    _require(
        math.isfinite(float(data.time)),
        data,
        field="finite:time",
        value=float(data.time),
        **context,
    )
    warning_counts = _warning_counts(data)
    _require(
        not any(warning_counts),
        data,
        field="mujoco_warning",
        value=warning_counts,
        **context,
    )
    contacts = int(data.ncon)
    accumulator["maximum_contacts"] = max(
        accumulator["maximum_contacts"], contacts
    )
    _require(
        contacts == 0,
        data,
        field="contacts",
        value=contacts,
        **context,
    )

    _require(
        np.all(tendon_limited),
        data,
        field="tendon_limited",
        value=tendon_limited.tolist(),
        **context,
    )
    margins = np.minimum(
        transmission.tendon_length - tendon_ranges[:, 0],
        tendon_ranges[:, 1] - transmission.tendon_length,
    )
    accumulator["minimum_tendon_margin_m"] = min(
        accumulator["minimum_tendon_margin_m"],
        float(np.min(margins)),
    )
    for binding, margin, length, limits in zip(
        bindings,
        margins,
        transmission.tendon_length,
        tendon_ranges,
    ):
        tolerance = max(
            1e-9,
            1e-8 * max(abs(float(length)), *np.abs(limits)),
        )
        _require(
            float(margin) >= -tolerance,
            data,
            field=f"tendon_range:{binding.tendon_name}",
            value={"length": float(length), "margin": float(margin)},
            **context,
        )

    command_array = np.asarray(command)
    force_error = float(
        np.max(np.abs(transmission.actuator_force - command_array))
    )
    length_residual = float(
        np.max(
            np.abs(
                transmission.actuator_length + transmission.tendon_length
            )
        )
    )
    velocity_residual = float(
        np.max(
            np.abs(
                transmission.actuator_velocity
                + transmission.tendon_velocity
            )
        )
    )
    moment_residual = float(
        np.max(
            np.abs(data.qfrc_actuator - transmission.expected_qfrc)
        )
    )
    generalized_residual = abs(
        transmission.generalized_power - transmission.actuator_power
    )
    actuator_tendon_residual = abs(
        transmission.actuator_power - transmission.tendon_power
    )
    command_tendon_residual = abs(
        transmission.tendon_power - transmission.command_tendon_power
    )
    residuals = {
        "maximum_force_command_error_n": force_error,
        "maximum_length_sign_residual_m": length_residual,
        "maximum_velocity_sign_residual_m_s": velocity_residual,
        "maximum_moment_residual": moment_residual,
        "maximum_generalized_power_residual_w": generalized_residual,
        "maximum_actuator_tendon_power_residual_w": (
            actuator_tendon_residual
        ),
        "maximum_command_tendon_power_residual_w": command_tendon_residual,
    }
    for name, value in residuals.items():
        accumulator[name] = max(accumulator[name], value)

    _require(
        force_error <= 1e-15,
        data,
        field="force_command_error_n",
        value=force_error,
        **context,
    )
    length_scale = float(np.max(np.abs(transmission.tendon_length)))
    velocity_scale = max(
        1e-12,
        float(np.max(np.abs(transmission.tendon_velocity))),
    )
    _require(
        length_residual <= 1e-12 + 1e-10 * length_scale,
        data,
        field="actuator_length_sign_residual_m",
        value=length_residual,
        **context,
    )
    _require(
        velocity_residual <= 1e-12 + 1e-10 * velocity_scale,
        data,
        field="actuator_velocity_sign_residual_m_s",
        value=velocity_residual,
        **context,
    )
    moment_scale = max(
        1.0,
        float(np.max(np.abs(data.qfrc_actuator))),
        float(np.max(np.abs(transmission.expected_qfrc))),
    )
    _require(
        moment_residual <= 1e-12 + 1e-10 * moment_scale,
        data,
        field="actuator_moment_residual",
        value=moment_residual,
        **context,
    )
    power_scale = max(
        1e-12,
        abs(transmission.generalized_power),
        abs(transmission.actuator_power),
        abs(transmission.tendon_power),
        abs(transmission.command_tendon_power),
    )
    power_limit = 1e-12 + 1e-9 * power_scale
    for field, value in (
        ("generalized_power_residual_w", generalized_residual),
        ("actuator_tendon_power_residual_w", actuator_tendon_residual),
        ("command_tendon_power_residual_w", command_tendon_residual),
    ):
        _require(
            value <= power_limit,
            data,
            field=field,
            value=value,
            **context,
        )

    qvel_max = float(np.max(np.abs(data.qvel)))
    qacc_max = float(np.max(np.abs(data.qacc)))
    accumulator["maximum_abs_qvel_native"] = max(
        accumulator["maximum_abs_qvel_native"], qvel_max
    )
    accumulator["maximum_abs_qacc_native"] = max(
        accumulator["maximum_abs_qacc_native"], qacc_max
    )
    _require(
        qvel_max < QVEL_HARD_LIMIT,
        data,
        field="qvel_hard_limit",
        value=qvel_max,
        **context,
    )
    _require(
        qacc_max < QACC_HARD_LIMIT,
        data,
        field="qacc_hard_limit",
        value=qacc_max,
        **context,
    )

    root_to_tip = np.asarray(
        data.site_xpos[entities.tip_site_id]
        - data.site_xpos[entities.root_site_id]
    )
    root_to_tip_length = float(np.linalg.norm(root_to_tip))
    accumulator["maximum_root_to_tip_length_m"] = max(
        accumulator["maximum_root_to_tip_length_m"], root_to_tip_length
    )
    _require(
        root_to_tip_length <= TOTAL_ARM_LENGTH_M + 1e-9,
        data,
        field="root_to_tip_length_m",
        value=root_to_tip_length,
        **context,
    )

    _require(
        np.array_equal(data.qfrc_applied, np.zeros(model.nv)),
        data,
        field="user_qfrc_applied",
        value=float(np.max(np.abs(data.qfrc_applied))),
        **context,
    )
    _require(
        np.array_equal(data.xfrc_applied, np.zeros((model.nbody, 6))),
        data,
        field="external_xfrc_applied",
        value=float(np.max(np.abs(data.xfrc_applied))),
        **context,
    )
    rotor_wrench, rotor_qfrc, rotor_power = _rotor_generalized_force(
        model, data, entities
    )
    rotor_qfrc_max = float(np.max(np.abs(rotor_qfrc)))
    accumulator["maximum_abs_rotor_generalized_force"] = max(
        accumulator["maximum_abs_rotor_generalized_force"],
        rotor_qfrc_max,
    )
    accumulator["maximum_abs_rotor_power_w"] = max(
        accumulator["maximum_abs_rotor_power_w"], abs(rotor_power)
    )
    _require(
        np.array_equal(rotor_wrench, np.zeros(6)),
        data,
        field="rotor_wrench",
        value=rotor_wrench.tolist(),
        **context,
    )
    _require(
        rotor_qfrc_max <= 1e-15,
        data,
        field="rotor_generalized_force",
        value=rotor_qfrc_max,
        **context,
    )
    _require(
        abs(rotor_power) <= 1e-15,
        data,
        field="rotor_power_w",
        value=rotor_power,
        **context,
    )

    damping_power = float(-data.qfrc_damper @ data.qvel)
    _require(
        damping_power >= -1e-15,
        data,
        field="damping_power_w",
        value=damping_power,
        **context,
    )
    com, linear_momentum, angular_momentum = _subtree_values(
        model, data, entities
    )
    return {
        "transmission": transmission,
        "margins": margins,
        "warning_counts": warning_counts,
        "contacts": contacts,
        "qvel_max": qvel_max,
        "qacc_max": qacc_max,
        "damping_power": damping_power,
        "rotor_wrench": rotor_wrench,
        "rotor_qfrc": rotor_qfrc,
        "rotor_power": rotor_power,
        "com": com,
        "linear_momentum": linear_momentum,
        "angular_momentum": angular_momentum,
    }


def _control_sample(
    model,
    data,
    entities,
    command,
    synchronized,
    cumulative,
    *,
    control_tick,
    physics_step,
):
    qpos_address = entities.drone_qpos_address
    dof_address = entities.drone_dof_address
    quaternion = np.asarray(
        data.qpos[qpos_address + 3:qpos_address + 7]
    ).copy()
    rotation = np.asarray(data.xmat[entities.drone_id]).reshape(3, 3).copy()
    root = np.asarray(data.site_xpos[entities.root_site_id]).copy()
    tip = np.asarray(data.site_xpos[entities.tip_site_id]).copy()
    root_to_tip_world = tip - root
    root_to_tip_body = rotation.T @ root_to_tip_world

    jacp = np.empty((3, model.nv))
    jacr = np.empty((3, model.nv))
    mujoco.mj_jacBodyCom(
        model, data, jacp, jacr, entities.drone_id
    )
    omega_world = jacr @ data.qvel
    omega_body = np.asarray(
        data.qvel[dof_address + 3:dof_address + 6]
    ).copy()
    if not np.allclose(
        omega_world,
        rotation @ omega_body,
        rtol=1e-10,
        atol=1e-12,
    ):
        _raise_failure(
            data,
            stage="control_sample",
            physics_step=physics_step,
            control_tick=control_tick,
            substep=0,
            field="uav_angular_velocity_frame_consistency",
            value={
                "world": omega_world.tolist(),
                "rotated_body": (rotation @ omega_body).tolist(),
            },
        )

    com = synchronized["com"]
    p_uav, h_uav = _component_momentum(
        model, data, entities.uav_body_ids, com
    )
    p_arm, h_arm = _component_momentum(
        model, data, entities.arm_body_ids, com
    )
    joints = joint_rotation_vectors(
        model, data, entities.joint_addresses
    )
    composite = composite_rotation_vector(
        model, data, entities.joint_addresses
    )
    transmission = synchronized["transmission"]
    return {
        "time": float(data.time),
        "tick": int(control_tick),
        "physics_step": int(physics_step),
        "command": np.asarray(command).copy(),
        "actuator_force": transmission.actuator_force.copy(),
        "uav_position": np.asarray(
            data.qpos[qpos_address:qpos_address + 3]
        ).copy(),
        "uav_linear_velocity_world": np.asarray(
            data.qvel[dof_address:dof_address + 3]
        ).copy(),
        "uav_quaternion": quaternion,
        "uav_angular_velocity_body": omega_body,
        "uav_angular_velocity_world": np.asarray(omega_world).copy(),
        "uav_rotation": rotation,
        "root_world": root,
        "tip_world": tip,
        "root_to_tip_world": root_to_tip_world,
        "root_to_tip_uav_body": root_to_tip_body,
        "joint_rotation": joints,
        "composite_rotation": composite,
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "qacc": np.asarray(data.qacc).copy(),
        "qfrc_actuator": np.asarray(data.qfrc_actuator).copy(),
        "system_com": com.copy(),
        "subtree_p": synchronized["linear_momentum"].copy(),
        "subtree_h": synchronized["angular_momentum"].copy(),
        "p_uav": p_uav,
        "p_arm": p_arm,
        "p_total": p_uav + p_arm,
        "h_uav": h_uav,
        "h_arm": h_arm,
        "h_total": h_uav + h_arm,
        "rotor_rpm": np.zeros(4),
        "rotor_wrench": synchronized["rotor_wrench"].copy(),
        "rotor_qfrc": synchronized["rotor_qfrc"].copy(),
        "mechanical_energy": float(np.sum(data.energy)),
        "tendon_work": float(cumulative["tendon_work"]),
        "damping_dissipation": float(cumulative["damping"]),
        "rotor_work": float(cumulative["rotor_work"]),
        "ledger_residual": float(cumulative["ledger_residual"]),
    }


def _physics_sample(
    data,
    command,
    synchronized,
    cumulative,
    *,
    control_tick,
    physics_step,
):
    transmission = synchronized["transmission"]
    return {
        "time": float(data.time),
        "step": int(physics_step),
        "control_tick": int(control_tick),
        "command": np.asarray(command).copy(),
        "actuator_force": transmission.actuator_force.copy(),
        "actuator_length": transmission.actuator_length.copy(),
        "actuator_velocity": transmission.actuator_velocity.copy(),
        "tendon_length": transmission.tendon_length.copy(),
        "tendon_velocity": transmission.tendon_velocity.copy(),
        "system_com": synchronized["com"].copy(),
        "subtree_p": synchronized["linear_momentum"].copy(),
        "subtree_h": synchronized["angular_momentum"].copy(),
        "generalized_power": float(transmission.generalized_power),
        "actuator_power": float(transmission.actuator_power),
        "tendon_power": float(transmission.tendon_power),
        "command_tendon_power": float(
            transmission.command_tendon_power
        ),
        "rotor_power": float(synchronized["rotor_power"]),
        "damping_power": float(synchronized["damping_power"]),
        "position_energy": float(data.energy[0]),
        "kinetic_energy": float(data.energy[1]),
        "mechanical_energy": float(np.sum(data.energy)),
        "tendon_work": float(cumulative["tendon_work"]),
        "damping_dissipation": float(cumulative["damping"]),
        "rotor_work": float(cumulative["rotor_work"]),
        "ledger_residual": float(cumulative["ledger_residual"]),
        "warning_counts": np.asarray(synchronized["warning_counts"], dtype=int),
        "contacts": int(synchronized["contacts"]),
        "minimum_tendon_margin": float(np.min(synchronized["margins"])),
        "qvel_max": float(synchronized["qvel_max"]),
        "qacc_max": float(synchronized["qacc_max"]),
    }


def _freeze_records(records):
    if not records:
        return MappingProxyType({})
    result = {}
    for key in records[0]:
        values = np.asarray([record[key] for record in records])
        values.setflags(write=False)
        result[key] = values
    return MappingProxyType(result)


def _safety_accumulator():
    return {
        "maximum_contacts": 0,
        "minimum_tendon_margin_m": math.inf,
        "maximum_abs_qvel_native": 0.0,
        "maximum_abs_qacc_native": 0.0,
        "maximum_root_to_tip_length_m": 0.0,
        "maximum_force_command_error_n": 0.0,
        "maximum_length_sign_residual_m": 0.0,
        "maximum_velocity_sign_residual_m_s": 0.0,
        "maximum_moment_residual": 0.0,
        "maximum_generalized_power_residual_w": 0.0,
        "maximum_actuator_tendon_power_residual_w": 0.0,
        "maximum_command_tendon_power_residual_w": 0.0,
        "maximum_abs_rotor_generalized_force": 0.0,
        "maximum_abs_rotor_power_w": 0.0,
    }


def _build_trace(
    config,
    topology,
    bindings,
    target_index,
    tendon_ranges,
    tendon_limited,
    physics_records,
    control_records,
    accumulator,
    completed_steps,
    failure,
):
    safety = C1B2SafetySummary(
        requested_steps=config.requested_steps,
        completed_steps=completed_steps,
        first_failure=failure,
        **accumulator,
    )
    tendon_ranges = np.asarray(tendon_ranges).copy()
    tendon_ranges.setflags(write=False)
    tendon_limited = np.asarray(tendon_limited, dtype=bool).copy()
    tendon_limited.setflags(write=False)
    return C1B2Trace(
        config=config,
        status="completed" if failure is None else "failed",
        topology=topology,
        bindings=tuple(bindings),
        target_binding_index=target_index,
        tendon_ranges_m=tendon_ranges,
        tendon_limited=tendon_limited,
        physics=_freeze_records(physics_records),
        control=_freeze_records(control_records),
        safety=safety,
    )


def run_c1_b2(config):
    """Run one formal C1-B2 active free-space trajectory."""
    if not isinstance(config, C1B2RunConfig):
        raise TypeError("config must be C1B2RunConfig")
    env = None
    try:
        env = _make_active_production_env(config)
        production_model = env.model
        production_manifest = env.SCALED_STABLESOFT_ARM_MANIFEST
        if production_manifest is None:
            raise AssertionError("production active build returned no manifest")
        free_xml = _remove_floor_and_zero_gravity(env._generated_xml)
        free_model = mujoco.MjModel.from_xml_string(free_xml)
        free_model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        free_data = mujoco.MjData(free_model)
        topology = _verify_active_free_space(
            production_model, free_model, production_manifest
        )

        env.model = free_model
        env.data = free_data
        bindings = resolve_ideal_tendon_actuator_bindings(
            free_model, production_manifest
        )
        _verify_bindings(free_model, production_manifest, bindings)
        env.SCALED_STABLESOFT_TENDON_BINDINGS = bindings
        entities = _resolve_entities(
            free_model, production_manifest, bindings
        )
        target_index = _target_binding_index(bindings, config.target_corner)

        tendon_ids = np.asarray(entities.tendon_ids, dtype=int)
        tendon_ranges = np.asarray(free_model.tendon_range)[
            tendon_ids
        ].copy()
        tendon_limited = np.asarray(free_model.tendon_limited)[
            tendon_ids
        ].copy()

        env.reset()
        zero_command = (0.0,) * len(bindings)
        env.set_scaled_stablesoft_tension_command(zero_command)
        zero_rpm = np.zeros(4)
        env._physics(zero_rpm, 0)
        free_data.qvel[:] = 0.0
        mujoco.mj_forward(free_model, free_data)
        _update_energy(free_model, free_data)

        accumulator = _safety_accumulator()
        cumulative = {
            "tendon_work": 0.0,
            "damping": 0.0,
            "rotor_work": 0.0,
            "ledger_residual": 0.0,
        }
        physics_records = []
        control_records = []
        completed_steps = 0
        failure = None
        initial_energy = float(np.sum(free_data.energy))

        try:
            initial = _check_synchronized(
                free_model,
                free_data,
                config,
                bindings,
                entities,
                zero_command,
                tendon_ranges,
                tendon_limited,
                accumulator,
                stage="initial_forward",
                physics_step=0,
                control_tick=0,
                substep=0,
            )
            physics_records.append(
                _physics_sample(
                    free_data,
                    zero_command,
                    initial,
                    cumulative,
                    control_tick=0,
                    physics_step=0,
                )
            )

            for control_tick in range(config.total_control_ticks + 1):
                expected_time = control_tick / config.ctrl_freq
                command = _command_at_tick(
                    config, bindings, target_index, control_tick
                )
                written = env.set_scaled_stablesoft_tension_command(command)
                if written != command:
                    raise AssertionError(
                        f"production setter changed command: {written} != {command}"
                    )
                env._physics(zero_rpm, 0)
                mujoco.mj_forward(free_model, free_data)
                _update_energy(free_model, free_data)
                synchronized = _check_synchronized(
                    free_model,
                    free_data,
                    config,
                    bindings,
                    entities,
                    command,
                    tendon_ranges,
                    tendon_limited,
                    accumulator,
                    stage="control_forward",
                    physics_step=completed_steps,
                    control_tick=control_tick,
                    substep=0,
                )
                _require(
                    abs(float(free_data.time) - expected_time) <= 1e-12,
                    free_data,
                    stage="control_forward",
                    physics_step=completed_steps,
                    control_tick=control_tick,
                    substep=0,
                    field="control_time_alignment",
                    value={
                        "actual": float(free_data.time),
                        "expected": expected_time,
                    },
                )
                cumulative["ledger_residual"] = (
                    float(np.sum(free_data.energy)) - initial_energy
                    - (
                        cumulative["tendon_work"]
                        - cumulative["damping"]
                        + cumulative["rotor_work"]
                    )
                )
                control_records.append(
                    _control_sample(
                        free_model,
                        free_data,
                        entities,
                        command,
                        synchronized,
                        cumulative,
                        control_tick=control_tick,
                        physics_step=completed_steps,
                    )
                )
                if control_tick == config.total_control_ticks:
                    break

                previous_actuator_power = synchronized[
                    "transmission"
                ].actuator_power
                previous_damping_power = synchronized["damping_power"]
                previous_rotor_power = synchronized["rotor_power"]

                for substep in range(1, config.sample_stride + 1):
                    next_step = completed_steps + 1
                    env._physics(zero_rpm, 0)
                    _require(
                        np.array_equal(
                            free_data.xfrc_applied,
                            np.zeros((free_model.nbody, 6)),
                        ),
                        free_data,
                        stage="before_step",
                        physics_step=next_step,
                        control_tick=control_tick,
                        substep=substep,
                        field="zero_rotor_wrench",
                        value=float(
                            np.max(np.abs(free_data.xfrc_applied))
                        ),
                    )
                    mujoco.mj_step(free_model, free_data)
                    raw_ok = (
                        math.isfinite(float(free_data.time))
                        and np.all(np.isfinite(free_data.qpos))
                        and np.all(np.isfinite(free_data.qvel))
                    )
                    _require(
                        raw_ok,
                        free_data,
                        stage="after_step",
                        physics_step=next_step,
                        control_tick=control_tick,
                        substep=substep,
                        field="raw_state_finite",
                        value=raw_ok,
                    )
                    warning_counts = _warning_counts(free_data)
                    _require(
                        not any(warning_counts),
                        free_data,
                        stage="after_step",
                        physics_step=next_step,
                        control_tick=control_tick,
                        substep=substep,
                        field="mujoco_warning",
                        value=warning_counts,
                    )

                    mujoco.mj_forward(free_model, free_data)
                    _update_energy(free_model, free_data)
                    synchronized = _check_synchronized(
                        free_model,
                        free_data,
                        config,
                        bindings,
                        entities,
                        command,
                        tendon_ranges,
                        tendon_limited,
                        accumulator,
                        stage="after_forward",
                        physics_step=next_step,
                        control_tick=control_tick,
                        substep=substep,
                    )
                    current_actuator_power = synchronized[
                        "transmission"
                    ].actuator_power
                    current_damping_power = synchronized["damping_power"]
                    current_rotor_power = synchronized["rotor_power"]
                    dt = 1.0 / config.sim_freq
                    cumulative["tendon_work"] += 0.5 * (
                        previous_actuator_power + current_actuator_power
                    ) * dt
                    cumulative["damping"] += 0.5 * (
                        previous_damping_power + current_damping_power
                    ) * dt
                    cumulative["rotor_work"] += 0.5 * (
                        previous_rotor_power + current_rotor_power
                    ) * dt
                    cumulative["ledger_residual"] = (
                        float(np.sum(free_data.energy)) - initial_energy
                        - (
                            cumulative["tendon_work"]
                            - cumulative["damping"]
                            + cumulative["rotor_work"]
                        )
                    )
                    previous_actuator_power = current_actuator_power
                    previous_damping_power = current_damping_power
                    previous_rotor_power = current_rotor_power
                    completed_steps = next_step
                    physics_records.append(
                        _physics_sample(
                            free_data,
                            command,
                            synchronized,
                            cumulative,
                            control_tick=control_tick,
                            physics_step=completed_steps,
                        )
                    )

                next_control_time = (control_tick + 1) / config.ctrl_freq
                _require(
                    abs(float(free_data.time) - next_control_time) <= 1e-12,
                    free_data,
                    stage="after_forward",
                    physics_step=completed_steps,
                    control_tick=control_tick,
                    substep=config.sample_stride,
                    field="control_interval_time_alignment",
                    value={
                        "actual": float(free_data.time),
                        "expected": next_control_time,
                    },
                )
                free_data.time = next_control_time
        except _SteppingFailure as exc:
            failure = exc.failure

        return _build_trace(
            config,
            topology,
            bindings,
            target_index,
            tendon_ranges,
            tendon_limited,
            physics_records,
            control_records,
            accumulator,
            completed_steps,
            failure,
        )
    finally:
        if env is not None:
            env.close()


def _quat_distance(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    dots = np.abs(np.sum(left * right, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))


def activity_metrics(trace):
    control = trace.control
    initial_position = control["uav_position"][0]
    initial_quaternion = control["uav_quaternion"][0]
    initial_body_tip = control["root_to_tip_uav_body"][0]
    composite_bend = np.linalg.norm(
        control["composite_rotation"][:, :2], axis=1
    )
    p_activity = (
        np.linalg.norm(control["p_uav"] - control["p_uav"][0], axis=1)
        + np.linalg.norm(control["p_arm"] - control["p_arm"][0], axis=1)
    )
    h_activity = (
        np.linalg.norm(control["h_uav"] - control["h_uav"][0], axis=1)
        + np.linalg.norm(control["h_arm"] - control["h_arm"][0], axis=1)
    )
    tendon_excursion = trace.physics["tendon_length"] - trace.physics[
        "tendon_length"
    ][0]
    return MappingProxyType(
        {
            "maximum_bend_rad": float(np.max(composite_bend)),
            "maximum_uav_displacement_m": float(
                np.max(
                    np.linalg.norm(
                        control["uav_position"] - initial_position, axis=1
                    )
                )
            ),
            "maximum_uav_orientation_rad": float(
                np.max(
                    _quat_distance(
                        control["uav_quaternion"], initial_quaternion
                    )
                )
            ),
            "maximum_body_tip_excursion_m": float(
                np.max(
                    np.linalg.norm(
                        control["root_to_tip_uav_body"]
                        - initial_body_tip,
                        axis=1,
                    )
                )
            ),
            "maximum_tendon_excursion_m": float(
                np.max(np.abs(tendon_excursion))
            ),
            "linear_component_activity_ns": float(np.max(p_activity)),
            "angular_component_activity_nms": float(np.max(h_activity)),
        }
    )


def conservation_metrics(trace):
    physics = trace.physics
    control = trace.control
    delta_p = physics["subtree_p"] - physics["subtree_p"][0]
    delta_h = physics["subtree_h"] - physics["subtree_h"][0]
    com_delta = physics["system_com"] - physics["system_com"][0]
    tail = physics["time"] >= 0.75 - 1e-12
    p_norm = np.linalg.norm(delta_p, axis=1)
    h_norm = np.linalg.norm(delta_h, axis=1)
    p_residual = control["p_total"] - control["subtree_p"]
    h_residual = control["h_total"] - control["subtree_h"]
    return MappingProxyType(
        {
            "maximum_com_drift_m": float(
                np.max(np.linalg.norm(com_delta, axis=1))
            ),
            "com_tail_speed_m_s": float(
                np.linalg.norm(
                    _slope(physics["time"][tail], physics["system_com"][tail])
                )
            ),
            "maximum_delta_p_ns": float(np.max(p_norm)),
            "maximum_delta_h_nms": float(np.max(h_norm)),
            "delta_p_tail_positive_slope_n": float(
                max(0.0, _slope(physics["time"][tail], p_norm[tail]))
            ),
            "delta_h_tail_positive_slope_nm": float(
                max(0.0, _slope(physics["time"][tail], h_norm[tail]))
            ),
            "maximum_p_partition_residual_ns": float(
                np.max(np.linalg.norm(p_residual, axis=1))
            ),
            "maximum_h_partition_residual_nms": float(
                np.max(np.linalg.norm(h_residual, axis=1))
            ),
        }
    )


def energy_metrics(trace):
    physics = trace.physics
    residual_max = float(np.max(np.abs(physics["ledger_residual"])))
    peak_work = float(np.max(np.abs(physics["tendon_work"])))
    return MappingProxyType(
        {
            "maximum_abs_ledger_residual_j": residual_max,
            "peak_abs_cumulative_tendon_work_j": peak_work,
            "ledger_limit_j": max(5e-10, 0.005 * peak_work),
            "final_tendon_work_j": float(physics["tendon_work"][-1]),
            "final_damping_dissipation_j": float(
                physics["damping_dissipation"][-1]
            ),
            "final_rotor_work_j": float(physics["rotor_work"][-1]),
        }
    )


def convergence_metrics(candidate, reference):
    if candidate.config.sim_freq != 2400 or reference.config.sim_freq != 4800:
        raise ValueError("convergence requires 2400 candidate and 4800 reference")
    left, right = candidate.control, reference.control
    if len(left["time"]) != len(right["time"]):
        raise AssertionError("control grids differ in length")

    def vector(name, floor, scale_name=None):
        error = float(np.max(np.linalg.norm(left[name] - right[name], axis=-1)))
        values = right[scale_name or name]
        scale = float(np.max(np.linalg.norm(values - values[0], axis=-1)))
        return {"error": error, "limit": max(floor, 0.02 * scale)}

    def array(name, floor, excursion=True):
        error = float(np.max(np.abs(left[name] - right[name])))
        values = right[name] - right[name][0] if excursion else right[name]
        scale = float(np.max(np.abs(values)))
        return {"error": error, "limit": max(floor, 0.02 * scale)}

    orientation_error = float(
        np.max(_quat_distance(left["uav_quaternion"], right["uav_quaternion"]))
    )
    orientation_scale = float(
        np.max(
            _quat_distance(
                right["uav_quaternion"], right["uav_quaternion"][0]
            )
        )
    )
    metrics = {
        "time": {
            "error": float(np.max(np.abs(left["time"] - right["time"]))),
            "limit": 1e-12,
        },
        "command": array("command", 1e-15, excursion=False),
        "actuator_force": array(
            "actuator_force", 1e-15, excursion=False
        ),
        "uav_position": vector("uav_position", 1e-7),
        "uav_orientation": {
            "error": orientation_error,
            "limit": max(1e-5, 0.02 * orientation_scale),
        },
        "tip_world": vector("tip_world", 2e-6),
        "tip_body": vector("root_to_tip_uav_body", 2e-6),
        "joint_rotation": vector("joint_rotation", 1e-4),
        "composite_rotation": vector("composite_rotation", 1e-4),
        "subtree_p": vector("subtree_p", 2e-10),
        "subtree_h": vector("subtree_h", 2e-12),
        "p_uav": vector("p_uav", 1e-8),
        "p_arm": vector("p_arm", 1e-8),
        "h_uav": vector("h_uav", 1e-10),
        "h_arm": vector("h_arm", 1e-10),
        "mechanical_energy": array("mechanical_energy", 1e-9),
        "tendon_work": array("tendon_work", 1e-9),
        "damping_dissipation": array("damping_dissipation", 1e-9),
        "ledger_residual": array("ledger_residual", 1e-9),
    }
    # Tendon extrema live on different physics grids.
    for name, floor in (("tendon_length", 2e-8), ("tendon_velocity", 1e-6)):
        left_extrema = np.stack(
            (candidate.physics[name].min(axis=0), candidate.physics[name].max(axis=0))
        )
        right_extrema = np.stack(
            (reference.physics[name].min(axis=0), reference.physics[name].max(axis=0))
        )
        error = float(np.max(np.abs(left_extrema - right_extrema)))
        if name == "tendon_length":
            scale = float(
                np.max(
                    np.abs(
                        reference.physics[name]
                        - reference.physics[name][0]
                    )
                )
            )
        else:
            scale = float(np.max(np.abs(reference.physics[name])))
        metrics[name] = {
            "error": error,
            "limit": max(floor, 0.02 * scale),
        }
    metrics["passed"] = all(
        value["error"] <= value["limit"]
        for key, value in metrics.items()
        if key != "passed"
    )
    return MappingProxyType(metrics)


def diagnostic_summary(trace):
    activity = activity_metrics(trace)
    conservation = conservation_metrics(trace)
    energy = energy_metrics(trace)
    failure = trace.safety.first_failure
    return {
        "status": trace.status,
        "sim_freq": trace.config.sim_freq,
        "completed_steps": trace.safety.completed_steps,
        "failure": None
        if failure is None
        else {
            "stage": failure.stage,
            "physics_step": failure.physics_step,
            "control_tick": failure.control_tick,
            "time": failure.time_s,
            "field": failure.field,
            "value": failure.value,
        },
        "latest_command": (
            trace.physics["command"][-1].tolist()
            if "command" in trace.physics and len(trace.physics["command"])
            else None
        ),
        "activity": dict(activity),
        "conservation": dict(conservation),
        "energy": dict(energy),
        "safety": trace.safety,
    }
