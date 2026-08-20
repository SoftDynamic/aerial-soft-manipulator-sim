"""B4 passive whole-vehicle dynamics helpers and manual diagnostics.

The formal zero-gravity model is derived from the production XML entirely in
memory.  This module never writes XML, traces, frames, or logs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from typing import Callable, Optional
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
    build_scaled_stablesoft_arm_mjcf,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    WarningLatch,
    analyze_decay_signal,
    joint_rotation_vectors,
    named_joint_addresses,
)
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


CTRL_FREQ = 240
JOINT_STIFFNESS = 0.003086
TARGET_DAMPING_RATIO = 0.3
INITIAL_JOINT_BEND = 0.025
INITIAL_TOTAL_BEND = 0.2
RELEASE_DURATION = 5.0
HOVER_DURATION = 1.0
TOTAL_ARM_LENGTH = 0.08
TOTAL_ARM_MASS = 0.003
TOTAL_SUBTREE_MASS = 0.030
MOUNT_Z = -0.015
E_SPRING0 = 8 * 0.5 * JOINT_STIFFNESS * INITIAL_JOINT_BEND**2
E_TOL = max(1.0e-9, 0.01 * E_SPRING0)
QVEL_HARD_LIMIT = 1000.0
QACC_EVIDENCE_LIMIT = 2.0e5

COM_MAX_DRIFT = 1.0e-5
COM_TAIL_SPEED = 2.0e-6
LINEAR_MOMENTUM_MAX = 2.0e-7
LINEAR_MOMENTUM_TAIL_SLOPE = 1.0e-9
ANGULAR_MOMENTUM_MAX = 1.0e-8
ANGULAR_MOMENTUM_TAIL_SLOPE = 1.0e-10

DIRECTION_ANGLES = {
    "+x": 0.0,
    "+y": math.pi / 2,
    "-x": math.pi,
    "-y": -math.pi / 2,
}


@dataclass(frozen=True)
class CoupledConfig:
    sim_freq: int
    direction: str
    gravity: bool = False
    rpm_mode: str = "zero"
    floor: bool = False
    duration: float = RELEASE_DURATION

    @property
    def stride(self) -> int:
        return self.sim_freq // CTRL_FREQ

    @property
    def requested_steps(self) -> int:
        return int(round(self.duration * self.sim_freq))


@dataclass(frozen=True)
class CoupledTrace:
    config: CoupledConfig
    status: str
    failure: Optional[dict]
    completed_steps: int
    timestep: float
    integrator: int
    warning_events: tuple
    time: np.ndarray
    uav_position: np.ndarray
    uav_quaternion: np.ndarray
    uav_linear_velocity: np.ndarray
    uav_angular_velocity: np.ndarray
    composite_com: np.ndarray
    total_linear_momentum: np.ndarray
    total_angular_momentum: np.ndarray
    root_position: np.ndarray
    tip_position: np.ndarray
    root_to_tip: np.ndarray
    tip_relative_uav: np.ndarray
    joint_rotation: np.ndarray
    arm_qvel: np.ndarray
    tendon_lengths: np.ndarray
    position_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    contacts: np.ndarray
    max_abs_qacc: np.ndarray
    control_p_uav: np.ndarray
    control_p_arm: np.ndarray
    control_h_uav: np.ndarray
    control_h_arm: np.ndarray
    tendon_ranges: np.ndarray
    tendon_names: tuple
    maximum_qvel: dict
    qacc_evidence_limit_exceeded: bool
    first_contact: Optional[dict]
    topology: dict
    initial_energy_error: float
    force_overwrite_ok: bool

    @property
    def control_indices(self) -> np.ndarray:
        return np.arange(0, len(self.time), self.config.stride, dtype=int)

    def control(self, name: str) -> np.ndarray:
        values = np.asarray(getattr(self, name))
        return values[self.control_indices]

    @property
    def signed_bend(self) -> np.ndarray:
        axis = 0 if self.config.direction.endswith("x") else 1
        return np.sum(self.joint_rotation[:, :, axis], axis=1)

    @property
    def control_signed_bend(self) -> np.ndarray:
        return self.signed_bend[self.control_indices]


def passive_spec() -> ScaledStableSoftArmSpec:
    return ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )


def _make_env(sim_freq: int, *, initial_z: float) -> BaseAviary:
    return BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, initial_z]]),
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
        scaled_stablesoft_arm=passive_spec(),
        integrator="implicitfast",
    )


def _name(model, object_type, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(object_id)) or "<unnamed>"


def _object_names(model, object_type, count: int) -> tuple:
    return tuple(_name(model, object_type, index) for index in range(count))


def _topology_snapshot(model) -> dict:
    return {
        "counts": (
            model.nbody,
            model.njnt,
            model.nq,
            model.nv,
            model.ngeom,
            model.nsite,
            model.ntendon,
            model.nu,
            model.nsensor,
        ),
        "body_names": _object_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "joint_names": _object_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "site_names": _object_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        "tendon_names": _object_names(model, mujoco.mjtObj.mjOBJ_TENDON, model.ntendon),
        "sensor_names": _object_names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
        "body_mass": np.asarray(model.body_mass).copy(),
        "body_inertia": np.asarray(model.body_inertia).copy(),
        "tendon_range": np.asarray(model.tendon_range).copy(),
        "tendon_adr": np.asarray(model.tendon_adr).copy(),
        "tendon_num": np.asarray(model.tendon_num).copy(),
        "wrap_type": np.asarray(model.wrap_type).copy(),
        "wrap_objid": np.asarray(model.wrap_objid).copy(),
    }


def _remove_floor_and_zero_gravity(xml: str):
    root = ET.fromstring(xml)
    option = root.find("option")
    if option is None:
        raise AssertionError("production XML has no option element")
    option.set("gravity", "0 0 0")
    floors = root.findall(".//geom[@name='floor']")
    if len(floors) != 1:
        raise AssertionError(f"expected one floor geom, found {len(floors)}")
    floor = floors[0]
    parents = [parent for parent in root.iter() if floor in list(parent)]
    if len(parents) != 1:
        raise AssertionError("floor geom parent is ambiguous")
    parents[0].remove(floor)
    return ET.tostring(root, encoding="unicode")


def _compile_energy_model(xml: str):
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    data = mujoco.MjData(model)
    return model, data


def _assert_passive_contract(model) -> None:
    if model.nu != 0 or model.ntendon != 4:
        raise AssertionError(
            f"passive contract violated: nu={model.nu}, ntendon={model.ntendon}"
        )


def _verify_transformation(production_model, transformed_model) -> dict:
    before = _topology_snapshot(production_model)
    after = _topology_snapshot(transformed_model)
    before_counts = before["counts"]
    after_counts = after["counts"]
    count_match = all(
        before_counts[index] == after_counts[index]
        for index in range(len(before_counts))
        if index != 4
    ) and after_counts[4] == before_counts[4] - 1
    named_topology_match = all(
        before[key] == after[key]
        for key in (
            "body_names",
            "joint_names",
            "site_names",
            "tendon_names",
            "sensor_names",
        )
    )
    numeric_topology_match = all(
        np.array_equal(before[key], after[key])
        for key in (
            "body_mass",
            "body_inertia",
            "tendon_range",
            "tendon_adr",
            "tendon_num",
            "wrap_type",
            "wrap_objid",
        )
    )
    proof = {
        "count_match": count_match,
        "named_topology_match": named_topology_match,
        "mass_inertia_routing_match": numeric_topology_match,
        "production_counts": before_counts,
        "free_space_counts": after_counts,
        "production_gravity": np.asarray(production_model.opt.gravity).copy(),
        "free_space_gravity": np.asarray(transformed_model.opt.gravity).copy(),
        "timestep_match": bool(
            production_model.opt.timestep == transformed_model.opt.timestep
        ),
        "integrator_match": bool(
            production_model.opt.integrator == transformed_model.opt.integrator
        ),
    }
    if not all(
        proof[key]
        for key in (
            "count_match",
            "named_topology_match",
            "mass_inertia_routing_match",
            "timestep_match",
            "integrator_match",
        )
    ):
        raise AssertionError(f"free-space transformation changed topology: {proof}")
    return proof


def _manifest(spec):
    return build_scaled_stablesoft_arm_mjcf("drone0", spec, MOUNT_Z).manifest


def _initialize_bend(model, data, manifest, direction: str) -> tuple:
    if direction not in DIRECTION_ANGLES:
        raise ValueError(f"unknown bend direction: {direction}")
    addresses = named_joint_addresses(model, manifest)
    axis = np.zeros(3)
    axis[0 if direction.endswith("x") else 1] = 1.0
    angle = INITIAL_JOINT_BEND * (-1.0 if direction.startswith("-") else 1.0)
    quaternion = np.empty(4)
    mujoco.mju_axisAngle2Quat(quaternion, axis, angle)
    for _, qpos_address, _ in addresses:
        data.qpos[qpos_address:qpos_address + 4] = quaternion
    data.qvel[:] = 0.0
    mujoco.mj_normalizeQuat(model, data.qpos)
    return addresses


def _body_groups(model, drone_id: int, arm_body_ids: tuple) -> tuple:
    subtree = {
        body_id
        for body_id in range(1, model.nbody)
        if int(model.body_rootid[body_id]) == drone_id
        and float(model.body_mass[body_id]) > 0.0
    }
    arm = set(arm_body_ids)
    uav = subtree - arm
    if uav & arm or uav | arm != subtree or uav != {drone_id}:
        raise AssertionError(
            f"invalid UAV/arm mass partition: uav={uav}, arm={arm}, subtree={subtree}"
        )
    return tuple(sorted(uav)), tuple(sorted(arm))


def _component_momentum(model, data, body_ids, total_com):
    p_total = np.zeros(3)
    h_total = np.zeros(3)
    jacp = np.empty((3, model.nv))
    jacr = np.empty((3, model.nv))
    for body_id in body_ids:
        mass = float(model.body_mass[body_id])
        if mass == 0.0:
            continue
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
        velocity = jacp @ data.qvel
        angular_velocity = jacr @ data.qvel
        momentum = mass * velocity
        inertial_rotation = np.asarray(data.ximat[body_id]).reshape(3, 3)
        inertia_world = (
            inertial_rotation
            @ np.diag(np.asarray(model.body_inertia[body_id]))
            @ inertial_rotation.T
        )
        p_total += momentum
        h_total += inertia_world @ angular_velocity + np.cross(
            np.asarray(data.xipos[body_id]) - total_com,
            momentum,
        )
    return p_total, h_total


def _dof_owner(model, dof_index: int, drone_dof_address: int) -> dict:
    joint_id = int(model.dof_jntid[dof_index])
    local_axis = dof_index - int(model.jnt_dofadr[joint_id])
    if joint_id == 0 and local_axis < 3:
        category = "uav_linear"
    elif joint_id == 0:
        category = "uav_angular"
        local_axis -= 3
    else:
        category = "arm_rotational"
    return {
        "dof_index": int(dof_index),
        "joint_name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
        "local_axis": "xyz"[local_axis],
        "category": category,
        "drone_dof_address": int(drone_dof_address),
    }


def _contact_details(model, data) -> Optional[dict]:
    if data.ncon == 0:
        return None
    contact = data.contact[0]
    return {
        "time": float(data.time),
        "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
        "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
    }


def _slope(time, values) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    design = np.column_stack((time - time[0], np.ones(len(time))))
    return np.linalg.lstsq(design, values, rcond=None)[0][0]


def conservation_metrics(trace: CoupledTrace) -> dict:
    tail = trace.time >= trace.time[-1] - 1.0 - 1.0e-12
    com_delta = trace.composite_com - trace.composite_com[0]
    p_norm = np.linalg.norm(trace.total_linear_momentum, axis=1)
    h_norm = np.linalg.norm(trace.total_angular_momentum, axis=1)
    return {
        "max_com_drift": float(np.max(np.linalg.norm(com_delta, axis=1))),
        "com_tail_speed": float(
            np.linalg.norm(
                _slope(trace.time[tail], trace.composite_com[tail])
            )
        ),
        "max_linear_momentum": float(np.max(p_norm)),
        "linear_momentum_tail_positive_slope": float(
            max(0.0, _slope(trace.time[tail], p_norm[tail]))
        ),
        "max_angular_momentum": float(np.max(h_norm)),
        "angular_momentum_tail_positive_slope": float(
            max(0.0, _slope(trace.time[tail], h_norm[tail]))
        ),
    }


def _run(
    model,
    data,
    manifest,
    config: CoupledConfig,
    apply_rpm: Callable[[], bool],
    *,
    topology: Optional[dict] = None,
    formal_contact_free: bool,
    check_energy: bool,
) -> CoupledTrace:
    _assert_passive_contract(model)
    drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
    drone_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint"
    )
    drone_qpos_address = int(model.jnt_qposadr[drone_joint_id])
    drone_dof_address = int(model.jnt_dofadr[drone_joint_id])
    addresses = named_joint_addresses(model, manifest)
    arm_dofs = np.concatenate(
        [np.arange(dof, dof + 3) for _, _, dof in addresses]
    )
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_root_site_name
    )
    tip_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name
    )
    tendon_ids = np.array(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_TENDON, name
            )
            for name in manifest.tendon_names
        ],
        dtype=int,
    )
    arm_body_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in manifest.segment_body_names
    )
    uav_bodies, arm_bodies = _body_groups(model, drone_id, arm_body_ids)
    subtree_mass = float(model.body_subtreemass[drone_id])
    if not math.isclose(
        subtree_mass, TOTAL_SUBTREE_MASS, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise AssertionError(f"unexpected drone subtree mass: {subtree_mass}")

    warning_latch = WarningLatch()
    warning_latch.snapshot(data, stage="initial_forward", step_index=0)
    fields = {
        name: []
        for name in (
            "time",
            "uav_position",
            "uav_quaternion",
            "uav_linear_velocity",
            "uav_angular_velocity",
            "composite_com",
            "total_linear_momentum",
            "total_angular_momentum",
            "root_position",
            "tip_position",
            "root_to_tip",
            "tip_relative_uav",
            "joint_rotation",
            "arm_qvel",
            "tendon_lengths",
            "position_energy",
            "kinetic_energy",
            "total_energy",
            "contacts",
            "max_abs_qacc",
        )
    }
    control_p_uav, control_p_arm = [], []
    control_h_uav, control_h_arm = [], []
    maximum_qvel = {"value": -math.inf, "time": 0.0}
    qacc_evidence = False
    first_contact = None
    force_overwrite_ok = True

    def record(step_index: int):
        nonlocal maximum_qvel, qacc_evidence, first_contact
        mujoco.mj_subtreeVel(model, data)
        com = np.asarray(data.subtree_com[drone_id]).copy()
        fields["time"].append(float(data.time))
        fields["uav_position"].append(
            np.asarray(
                data.qpos[drone_qpos_address:drone_qpos_address + 3]
            ).copy()
        )
        fields["uav_quaternion"].append(
            np.asarray(
                data.qpos[drone_qpos_address + 3:drone_qpos_address + 7]
            ).copy()
        )
        fields["uav_linear_velocity"].append(
            np.asarray(
                data.qvel[drone_dof_address:drone_dof_address + 3]
            ).copy()
        )
        fields["uav_angular_velocity"].append(
            np.asarray(
                data.qvel[drone_dof_address + 3:drone_dof_address + 6]
            ).copy()
        )
        fields["composite_com"].append(com)
        fields["total_linear_momentum"].append(
            subtree_mass * np.asarray(data.subtree_linvel[drone_id]).copy()
        )
        fields["total_angular_momentum"].append(
            np.asarray(data.subtree_angmom[drone_id]).copy()
        )
        root = np.asarray(data.site_xpos[root_id]).copy()
        tip = np.asarray(data.site_xpos[tip_id]).copy()
        fields["root_position"].append(root)
        fields["tip_position"].append(tip)
        fields["root_to_tip"].append(tip - root)
        fields["tip_relative_uav"].append(tip - np.asarray(data.xpos[drone_id]))
        fields["joint_rotation"].append(joint_rotation_vectors(model, data, addresses))
        fields["arm_qvel"].append(np.asarray(data.qvel[arm_dofs]).copy())
        fields["tendon_lengths"].append(np.asarray(data.ten_length[tendon_ids]).copy())
        fields["position_energy"].append(float(data.energy[0]))
        fields["kinetic_energy"].append(float(data.energy[1]))
        fields["total_energy"].append(float(np.sum(data.energy)))
        fields["contacts"].append(int(data.ncon))
        fields["max_abs_qacc"].append(float(np.max(np.abs(data.qacc))))
        qacc_evidence = qacc_evidence or fields["max_abs_qacc"][-1] > QACC_EVIDENCE_LIMIT
        dof_index = int(np.argmax(np.abs(data.qvel)))
        qvel_value = float(abs(data.qvel[dof_index]))
        if qvel_value > maximum_qvel["value"]:
            maximum_qvel = {
                "value": qvel_value,
                "time": float(data.time),
                **_dof_owner(model, dof_index, drone_dof_address),
            }
        if first_contact is None:
            first_contact = _contact_details(model, data)
        if step_index % config.stride == 0:
            p_uav, h_uav = _component_momentum(model, data, uav_bodies, com)
            p_arm, h_arm = _component_momentum(model, data, arm_bodies, com)
            control_p_uav.append(p_uav)
            control_p_arm.append(p_arm)
            control_h_uav.append(h_uav)
            control_h_arm.append(h_arm)

    record(0)
    energy_reference = fields["total_energy"][0]
    initial_energy_error = (
        abs(energy_reference - E_SPRING0) if check_energy else 0.0
    )
    previous_energy = energy_reference
    above_count = 0
    positive_count = 0
    positive_start = previous_energy
    status = "completed"
    failure = None
    completed_steps = 0

    for step_index in range(1, config.requested_steps + 1):
        force_overwrite_ok = bool(apply_rpm()) and force_overwrite_ok
        mujoco.mj_step(model, data)
        warning_latch.snapshot(data, stage="after_step", step_index=step_index)
        raw_ok = (
            math.isfinite(float(data.time))
            and np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
        )
        if not raw_ok:
            status = "raw_nonfinite"
            failure = {
                "stage": "after_step",
                "step": step_index,
                "time": float(data.time),
            }
            break
        mujoco.mj_forward(model, data)
        warning_latch.snapshot(data, stage="after_forward", step_index=step_index)
        synchronized = (
            data.qpos, data.qvel, data.qacc, data.site_xpos,
            data.ten_length, data.energy,
        )
        if not all(np.all(np.isfinite(values)) for values in synchronized):
            status = "synchronized_nonfinite"
            failure = {
                "stage": "after_forward",
                "step": step_index,
                "time": float(data.time),
            }
            break
        record(step_index)
        completed_steps = step_index
        reasons = []
        derived_fields = (
            "uav_position",
            "uav_quaternion",
            "uav_linear_velocity",
            "uav_angular_velocity",
            "composite_com",
            "total_linear_momentum",
            "total_angular_momentum",
            "root_to_tip",
            "tip_relative_uav",
            "joint_rotation",
            "arm_qvel",
            "tendon_lengths",
            "position_energy",
            "kinetic_energy",
            "total_energy",
        )
        if not all(
            np.all(np.isfinite(fields[name][-1]))
            for name in derived_fields
        ):
            reasons.append("synchronized_derived_nonfinite")
        if warning_latch.triggered:
            reasons.append("mujoco_warning")
        if np.max(np.abs(data.qvel)) > QVEL_HARD_LIMIT:
            reasons.append("qvel_hard_limit")
        root_tip_length = float(np.linalg.norm(fields["root_to_tip"][-1]))
        geometry_tolerance = max(
            1.0e-9,
            1.0e-8 * max(TOTAL_ARM_LENGTH, root_tip_length),
        )
        if root_tip_length > TOTAL_ARM_LENGTH + geometry_tolerance:
            reasons.append("root_to_tip_length")
        for tendon_id, length in zip(tendon_ids, fields["tendon_lengths"][-1]):
            lower, upper = model.tendon_range[tendon_id]
            tolerance = max(
                1.0e-9,
                1.0e-8 * max(abs(lower), abs(upper), abs(length)),
            )
            if length < lower - tolerance or length > upper + tolerance:
                reasons.append(
                    "tendon_range:"
                    f"{_name(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)}"
                )
        if formal_contact_free and data.ncon:
            reasons.append("contact")
        current_energy = fields["total_energy"][-1]
        if check_energy:
            delta = current_energy - energy_reference
            increment = current_energy - previous_energy
            above_count = above_count + 1 if delta > E_TOL else 0
            increment_floor = 64 * np.finfo(float).eps * max(
                1.0, abs(current_energy), abs(previous_energy)
            )
            if increment > increment_floor:
                if positive_count == 0:
                    positive_start = previous_energy
                positive_count += 1
            else:
                positive_count = 0
            if increment > E_SPRING0:
                reasons.append("single_step_energy_injection")
            if delta > 5 * E_TOL:
                reasons.append("total_positive_energy_drift")
            if above_count >= config.stride:
                reasons.append("persistent_energy_upper_bound")
            if positive_count >= config.stride and current_energy - positive_start > E_TOL:
                reasons.append("persistent_positive_energy_growth")
            previous_energy = current_energy
        if reasons:
            status = "failed"
            failure = {
                "stage": "after_forward",
                "step": step_index,
                "time": float(data.time),
                "reasons": reasons,
            }
            break
        if not formal_contact_free and data.ncon:
            status = "contact"
            break

    arrays = {
        name: np.asarray(values, dtype=int if name == "contacts" else float)
        for name, values in fields.items()
    }
    return CoupledTrace(
        config=config,
        status=status,
        failure=failure,
        completed_steps=completed_steps,
        timestep=float(model.opt.timestep),
        integrator=int(model.opt.integrator),
        warning_events=tuple(warning_latch.events),
        control_p_uav=np.asarray(control_p_uav),
        control_p_arm=np.asarray(control_p_arm),
        control_h_uav=np.asarray(control_h_uav),
        control_h_arm=np.asarray(control_h_arm),
        tendon_ranges=np.asarray(model.tendon_range[tendon_ids]).copy(),
        tendon_names=tuple(manifest.tendon_names),
        maximum_qvel=maximum_qvel,
        qacc_evidence_limit_exceeded=qacc_evidence,
        first_contact=first_contact,
        topology=topology or {},
        initial_energy_error=float(initial_energy_error),
        force_overwrite_ok=force_overwrite_ok,
        **arrays,
    )


def run_zero_gravity_release(sim_freq: int, direction: str) -> CoupledTrace:
    config = CoupledConfig(sim_freq=sim_freq, direction=direction)
    env = _make_env(sim_freq, initial_z=0.0)
    try:
        free_xml = _remove_floor_and_zero_gravity(env._generated_xml)
        model, data = _compile_energy_model(free_xml)
        topology = _verify_transformation(env.model, model)
        manifest = _manifest(passive_spec())
        _initialize_bend(model, data, manifest, direction)
        mujoco.mj_forward(model, data)
        drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone0")

        def zero_rpm():
            data.xfrc_applied[drone_id] = 0.0
            return bool(np.array_equal(data.xfrc_applied[drone_id], np.zeros(6)))

        return _run(
            model, data, manifest, config, zero_rpm,
            topology=topology, formal_contact_free=True, check_energy=True,
        )
    finally:
        env.close()


def run_hover(sim_freq: int = 2400) -> CoupledTrace:
    config = CoupledConfig(
        sim_freq=sim_freq,
        direction="+x",
        gravity=True,
        rpm_mode="hover",
        floor=True,
        duration=HOVER_DURATION,
    )
    env = _make_env(sim_freq, initial_z=1.0)
    try:
        model, data = _compile_energy_model(env._generated_xml)
        env.model, env.data = model, data
        manifest = _manifest(passive_spec())
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
        rpm = np.full(4, env.HOVER_RPM)
        expected = np.array([0.0, 0.0, env.M * env.G, 0.0, 0.0, 0.0])

        def hover_rpm():
            data.xfrc_applied[drone_id] = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0])
            env._physics(rpm, 0)
            return bool(np.allclose(data.xfrc_applied[drone_id], expected, rtol=0.0, atol=1.0e-14))

        return _run(
            model, data, manifest, config, hover_rpm,
            topology={
                "counts": _topology_snapshot(model)["counts"],
                "gravity": np.asarray(model.opt.gravity).copy(),
                "floor_count": env._generated_xml.count('name="floor"'),
            },
            formal_contact_free=True,
            check_energy=False,
        )
    finally:
        env.close()


def run_open_loop_diagnostic(sim_freq: int, direction: str) -> CoupledTrace:
    config = CoupledConfig(
        sim_freq=sim_freq,
        direction=direction,
        gravity=True,
        rpm_mode="hover",
        floor=True,
    )
    env = _make_env(sim_freq, initial_z=1.0)
    try:
        model, data = _compile_energy_model(env._generated_xml)
        env.model, env.data = model, data
        manifest = _manifest(passive_spec())
        _initialize_bend(model, data, manifest, direction)
        mujoco.mj_forward(model, data)
        rpm = np.full(4, env.HOVER_RPM)

        def hover_rpm():
            env._physics(rpm, 0)
            return True

        return _run(
            model, data, manifest, config, hover_rpm,
            topology={}, formal_contact_free=False, check_energy=False,
        )
    finally:
        env.close()


def _quat_distance(left, right) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    dots = np.abs(np.sum(left * right, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))


def settling_time(trace: CoupledTrace) -> Optional[float]:
    time = trace.control("time")
    bend = trace.control_signed_bend
    qvel = np.max(np.abs(trace.control("arm_qvel")), axis=1)
    final = bend[-1]
    qualified = (np.abs(bend - final) <= max(1.0e-3, 0.02 * abs(final))) & (qvel <= 1.0e-3)
    for index in range(len(time)):
        if time[-1] - time[index] < 0.25:
            break
        if np.all(qualified[index:]):
            return float(time[index])
    return None


def convergence_metrics(candidate: CoupledTrace, reference: CoupledTrace) -> dict:
    if candidate.config.direction != reference.config.direction:
        raise ValueError("convergence traces must use the same direction")
    candidate_time = candidate.control("time")
    reference_time = reference.control("time")
    if len(candidate_time) != len(reference_time):
        raise AssertionError("control trace lengths differ")
    decay_candidate = analyze_decay_signal(candidate_time, candidate.control_signed_bend)
    decay_reference = analyze_decay_signal(reference_time, reference.control_signed_bend)
    candidate_settling = settling_time(candidate)
    reference_settling = settling_time(reference)

    def trajectory(name, absolute_floor, *, relative=0.02, excursion=True):
        cand = candidate.control(name)
        ref = reference.control(name)
        error = float(np.max(np.abs(cand - ref)))
        scale_values = ref - ref[0] if excursion else ref
        scale = float(np.max(np.abs(scale_values)))
        return {"error": error, "scale": scale, "limit": max(absolute_floor, relative * scale)}

    metrics = {
        "time": {"error": float(np.max(np.abs(candidate_time - reference_time))), "limit": 1.0e-9},
        "frequency": {
            "error": abs(decay_candidate.frequency - decay_reference.frequency),
            "scale": abs(decay_reference.frequency),
            "limit": max(0.02, 0.02 * abs(decay_reference.frequency)),
        },
        "settling": {
            "error": abs(candidate_settling - reference_settling),
            "scale": abs(reference_settling),
            "limit": max(1.0 / CTRL_FREQ, 0.02 * abs(reference_settling)),
        },
        "joint_rotation": trajectory("joint_rotation", 1.0e-4, excursion=False),
        "root_to_tip": trajectory("root_to_tip", 2.0e-5),
        "tip_relative_uav": trajectory("tip_relative_uav", 2.0e-5),
        "uav_position": trajectory("uav_position", 1.0e-6),
        "uav_linear_velocity": trajectory("uav_linear_velocity", 1.0e-5, excursion=False),
        "uav_angular_velocity": trajectory("uav_angular_velocity", 1.0e-4, excursion=False),
        "composite_com": trajectory("composite_com", 1.0e-6),
        "tendon_lengths": trajectory("tendon_lengths", 2.0e-8),
        "energy": trajectory("total_energy", 1.0e-9, excursion=False),
    }
    candidate_joints = candidate.control("joint_rotation")
    reference_joints = reference.control("joint_rotation")
    for joint_index in range(reference_joints.shape[1]):
        error = float(
            np.max(
                np.abs(
                    candidate_joints[:, joint_index]
                    - reference_joints[:, joint_index]
                )
            )
        )
        scale = float(np.max(np.abs(reference_joints[:, joint_index])))
        metrics[f"joint_{joint_index + 1:02d}"] = {
            "error": error,
            "scale": scale,
            "limit": max(1.0e-4, 0.02 * scale),
        }
    for name, candidate_values, reference_values, floor in (
        ("p_uav", candidate.control_p_uav, reference.control_p_uav, 1.0e-8),
        ("p_arm", candidate.control_p_arm, reference.control_p_arm, 1.0e-8),
        ("h_uav", candidate.control_h_uav, reference.control_h_uav, 1.0e-10),
        ("h_arm", candidate.control_h_arm, reference.control_h_arm, 1.0e-10),
    ):
        error = float(np.max(np.abs(candidate_values - reference_values)))
        scale = float(np.max(np.abs(reference_values)))
        metrics[name] = {
            "error": error,
            "scale": scale,
            "limit": max(floor, 0.02 * scale),
        }
    orientation_error = float(
        np.max(
            _quat_distance(
                candidate.control("uav_quaternion"),
                reference.control("uav_quaternion"),
            )
        )
    )
    orientation_scale = float(
        np.max(
            _quat_distance(
                reference.control("uav_quaternion"),
                reference.control("uav_quaternion")[0],
            )
        )
    )
    metrics["uav_orientation"] = {
        "error": orientation_error,
        "scale": orientation_scale,
        "limit": max(1.0e-4, 0.02 * orientation_scale),
    }
    peak_count_ok = (
        len(decay_candidate.peak_times) >= 3
        and len(decay_candidate.peak_times)
        == len(decay_reference.peak_times)
    )
    metrics["peak_count"] = {
        "passed": peak_count_ok,
        "candidate": len(decay_candidate.peak_times),
        "reference": len(decay_reference.peak_times),
    }
    if peak_count_ok:
        period = 1.0 / decay_reference.frequency
        peak_time_error = float(
            np.max(
                np.abs(
                    decay_candidate.peak_times - decay_reference.peak_times
                )
            )
        )
        peak_amp_error = float(
            np.max(
                np.abs(
                    decay_candidate.peak_amplitudes
                    - decay_reference.peak_amplitudes
                )
            )
        )
        peak_scale = float(np.max(np.abs(decay_reference.peak_amplitudes)))
        metrics["peak_time"] = {
            "error": peak_time_error,
            "scale": period,
            "limit": max(1.0 / CTRL_FREQ, 0.02 * period),
        }
        metrics["peak_envelope"] = {
            "error": peak_amp_error,
            "scale": peak_scale,
            "limit": max(1.0e-4, 0.02 * peak_scale),
        }
    tendon_min_error = float(
        np.max(
            np.abs(
                np.min(candidate.control("tendon_lengths"), axis=0)
                - np.min(reference.control("tendon_lengths"), axis=0)
            )
        )
    )
    tendon_max_error = float(
        np.max(
            np.abs(
                np.max(candidate.control("tendon_lengths"), axis=0)
                - np.max(reference.control("tendon_lengths"), axis=0)
            )
        )
    )
    tendon_scale = float(
        np.max(np.ptp(reference.control("tendon_lengths"), axis=0))
    )
    metrics["tendon_extrema"] = {
        "error": max(tendon_min_error, tendon_max_error),
        "scale": tendon_scale,
        "limit": max(2.0e-8, 0.02 * tendon_scale),
    }
    candidate_tendons = candidate.control("tendon_lengths")
    reference_tendons = reference.control("tendon_lengths")
    for tendon_index in range(reference_tendons.shape[1]):
        extrema_error = max(
            abs(
                float(np.min(candidate_tendons[:, tendon_index]))
                - float(np.min(reference_tendons[:, tendon_index]))
            ),
            abs(
                float(np.max(candidate_tendons[:, tendon_index]))
                - float(np.max(reference_tendons[:, tendon_index]))
            ),
        )
        scale = float(np.ptp(reference_tendons[:, tendon_index]))
        metrics[f"tendon_{tendon_index + 1}_extrema"] = {
            "error": extrema_error,
            "scale": scale,
            "limit": max(2.0e-8, 0.02 * scale),
        }
    energy_scale = max(
        E_SPRING0,
        float(
            np.max(
                np.abs(
                    reference.control("total_energy")
                    - reference.total_energy[0]
                )
            )
        ),
    )
    metrics["energy"]["scale"] = energy_scale
    metrics["energy"]["limit"] = max(1.0e-9, 0.02 * energy_scale)
    metrics["passed"] = all(
        value.get(
            "passed",
            value.get("error", math.inf) <= value.get("limit", -math.inf),
        )
        for key, value in metrics.items()
        if key != "passed"
    ) and decay_candidate.envelope_valid and decay_reference.envelope_valid
    return metrics


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quaternion_multiply(left, right):
    result = np.empty(4)
    mujoco.mju_mulQuat(result, left, right)
    return result


def _canonical_quaternions(quaternions, angle):
    half = angle / 2
    world_to_canonical = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    canonical_to_world = world_to_canonical.copy()
    canonical_to_world[3] *= -1
    return np.asarray(
        [
            _quaternion_multiply(
                _quaternion_multiply(world_to_canonical, quaternion),
                canonical_to_world,
            )
            for quaternion in quaternions
        ]
    )


def symmetry_metrics(reference: CoupledTrace, other: CoupledTrace) -> dict:
    canonical_angle = -DIRECTION_ANGLES[other.config.direction]
    rotation = _rotation_z(canonical_angle)

    def canonical(values):
        return np.asarray(values) @ rotation.T

    def compare(name, floor, *, relative=0.005, sorted_columns=False):
        ref = reference.control(name)
        value = other.control(name)
        if sorted_columns:
            ref = np.sort(ref, axis=-1)
            value = np.sort(value, axis=-1)
        elif ref.shape[-1] == 3:
            value = canonical(value)
        error = float(np.max(np.abs(value - ref)))
        scale = float(np.max(np.abs(ref - ref[0])))
        return {"error": error, "scale": scale, "limit": max(floor, relative * scale)}

    metrics = {
        "uav_position": compare("uav_position", 1.0e-6),
        "uav_linear_velocity": compare("uav_linear_velocity", 1.0e-6),
        "uav_angular_velocity": compare("uav_angular_velocity", 1.0e-5),
        "composite_com": compare("composite_com", 1.0e-6),
        "root_to_tip": compare("root_to_tip", 1.0e-6),
        "tip_relative_uav": compare("tip_relative_uav", 1.0e-6),
        "joint_rotation": compare("joint_rotation", 1.0e-5),
        "total_linear_momentum": compare("total_linear_momentum", 1.0e-10),
        "total_angular_momentum": compare("total_angular_momentum", 1.0e-12),
        "tendon_lengths": compare("tendon_lengths", 1.0e-9, sorted_columns=True),
        "total_energy": compare("total_energy", 1.0e-9),
    }
    reference_quaternion = reference.control("uav_quaternion")
    other_quaternion = _canonical_quaternions(
        other.control("uav_quaternion"), canonical_angle
    )
    orientation_error = float(
        np.max(_quat_distance(reference_quaternion, other_quaternion))
    )
    orientation_scale = float(
        np.max(_quat_distance(reference_quaternion, reference_quaternion[0]))
    )
    metrics["uav_orientation"] = {
        "error": orientation_error,
        "scale": orientation_scale,
        "limit": max(1.0e-5, 0.005 * orientation_scale),
    }
    for name, reference_values, other_values, floor in (
        ("p_uav", reference.control_p_uav, other.control_p_uav, 1.0e-10),
        ("p_arm", reference.control_p_arm, other.control_p_arm, 1.0e-10),
        ("h_uav", reference.control_h_uav, other.control_h_uav, 1.0e-12),
        ("h_arm", reference.control_h_arm, other.control_h_arm, 1.0e-12),
    ):
        canonical_other = canonical(other_values)
        error = float(np.max(np.abs(canonical_other - reference_values)))
        scale = float(np.max(np.abs(reference_values)))
        metrics[name] = {
            "error": error,
            "scale": scale,
            "limit": max(floor, 0.005 * scale),
        }
    metrics["passed"] = all(
        item["error"] <= item["limit"] for item in metrics.values()
    )
    return metrics


def reaction_metrics(trace: CoupledTrace) -> dict:
    p_residual = (
        trace.control_p_uav
        + trace.control_p_arm
        - trace.control("total_linear_momentum")
    )
    h_residual = (
        trace.control_h_uav
        + trace.control_h_arm
        - trace.control("total_angular_momentum")
    )
    p_activity = np.linalg.norm(
        trace.control_p_uav, axis=1
    ) + np.linalg.norm(trace.control_p_arm, axis=1)
    h_activity = np.linalg.norm(
        trace.control_h_uav, axis=1
    ) + np.linalg.norm(trace.control_h_arm, axis=1)
    p_residual_max = float(np.max(np.linalg.norm(p_residual, axis=1)))
    h_residual_max = float(np.max(np.linalg.norm(h_residual, axis=1)))
    p_active = float(np.max(p_activity))
    h_active = float(np.max(h_activity))
    return {
        "p_activity": p_active,
        "h_activity": h_active,
        "p_residual": p_residual_max,
        "h_residual": h_residual_max,
        "passed": (
            p_active >= 1.0e-6
            and h_active >= 1.0e-7
            and p_residual_max <= min(LINEAR_MOMENTUM_MAX, 0.01 * p_active)
            and h_residual_max <= min(ANGULAR_MOMENTUM_MAX, 0.01 * h_active)
        ),
    }


def diagnostic_summary(trace: CoupledTrace) -> dict:
    interval = trace.time <= min(1.0, trace.time[-1]) + 1.0e-12
    body_z = np.empty((len(trace.time), 3))
    for index, quaternion in enumerate(trace.uav_quaternion):
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        body_z[index] = matrix.reshape(3, 3)[:, 2]
    total_thrust = TOTAL_SUBTREE_MASS * 9.81
    horizontal = total_thrust * np.linalg.norm(body_z[:, :2], axis=1)
    vertical = total_thrust * body_z[:, 2]
    arm_flat_index = int(np.argmax(np.abs(trace.arm_qvel)))
    arm_time_index, arm_dof_index = np.unravel_index(
        arm_flat_index, trace.arm_qvel.shape
    )
    return {
        "sim_freq": trace.config.sim_freq,
        "direction": trace.config.direction,
        "status": trace.status,
        "first_contact": trace.first_contact,
        "final_position": trace.uav_position[-1].tolist(),
        "final_quaternion": trace.uav_quaternion[-1].tolist(),
        "final_linear_velocity": trace.uav_linear_velocity[-1].tolist(),
        "final_angular_velocity": trace.uav_angular_velocity[-1].tolist(),
        "maximum_qvel": trace.maximum_qvel,
        "maximum_arm_qvel": {
            "value": float(abs(trace.arm_qvel[arm_time_index, arm_dof_index])),
            "time": float(trace.time[arm_time_index]),
            "arm_dof_index": int(arm_dof_index),
        },
        "body_z_final": body_z[-1].tolist(),
        "horizontal_thrust_max": float(np.max(horizontal)),
        "vertical_thrust_min": float(np.min(vertical)),
        "horizontal_displacement": float(
            np.max(
                np.linalg.norm(
                    trace.uav_position[:, :2]
                    - trace.uav_position[0, :2],
                    axis=1,
                )
            )
        ),
        "early_horizontal_displacement": float(
            np.max(
                np.linalg.norm(
                    trace.uav_position[interval, :2]
                    - trace.uav_position[0, :2],
                    axis=1,
                )
            )
        ),
        "height_change": float(trace.uav_position[-1, 2] - trace.uav_position[0, 2]),
        "warnings": len(trace.warning_events),
        "tendon_min": np.min(trace.tendon_lengths, axis=0).tolist(),
        "tendon_max": np.max(trace.tendon_lengths, axis=0).tolist(),
    }


def diagnostic_early_comparison(candidate, reference) -> dict:
    contact_times = [
        trace.first_contact["time"]
        for trace in (candidate, reference)
        if trace.first_contact is not None
    ]
    end_time = min([1.0, *contact_times])
    candidate_indices = candidate.control_indices[
        candidate.control("time") <= end_time + 1.0e-12
    ]
    reference_indices = reference.control_indices[
        reference.control("time") <= end_time + 1.0e-12
    ]
    count = min(len(candidate_indices), len(reference_indices))
    candidate_indices = candidate_indices[:count]
    reference_indices = reference_indices[:count]
    position_error = np.linalg.norm(
        candidate.uav_position[candidate_indices]
        - reference.uav_position[reference_indices],
        axis=1,
    )
    linear_velocity_error = np.linalg.norm(
        candidate.uav_linear_velocity[candidate_indices]
        - reference.uav_linear_velocity[reference_indices],
        axis=1,
    )
    angular_velocity_error = np.linalg.norm(
        candidate.uav_angular_velocity[candidate_indices]
        - reference.uav_angular_velocity[reference_indices],
        axis=1,
    )
    orientation_error = _quat_distance(
        candidate.uav_quaternion[candidate_indices],
        reference.uav_quaternion[reference_indices],
    )
    return {
        "direction": candidate.config.direction,
        "interval": [0.0, float(end_time)],
        "samples": int(count),
        "max_position_error": float(np.max(position_error)),
        "max_orientation_error": float(np.max(orientation_error)),
        "max_linear_velocity_error": float(np.max(linear_velocity_error)),
        "max_angular_velocity_error": float(np.max(angular_velocity_error)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-matrix", action="store_true")
    args = parser.parse_args(argv)
    if not args.diagnostic_matrix:
        parser.error("--diagnostic-matrix is required")
    results = {}
    for sim_freq in (2400, 4800):
        for direction in ("+x", "-x", "+y", "-y"):
            trace = run_open_loop_diagnostic(sim_freq, direction)
            results[(sim_freq, direction)] = trace
            print(json.dumps(diagnostic_summary(trace), sort_keys=True))
    for direction in ("+x", "-x", "+y", "-y"):
        print(
            json.dumps(
                {
                    "early_2400_4800": diagnostic_early_comparison(
                        results[(2400, direction)],
                        results[(4800, direction)],
                    )
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
