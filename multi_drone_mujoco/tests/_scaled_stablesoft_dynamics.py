"""Test-only dynamics helpers for the scaled StableSoft arm."""

from dataclasses import dataclass
import math
from typing import Tuple

import mujoco
import numpy as np

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ArmMjcfManifest,
    ScaledStableSoftArmSpec,
    build_scaled_stablesoft_arm_mjcf,
)


MOUNT_Z = -0.015
_SQRT_HALF = math.sqrt(0.5)
_FIXTURE_QUATERNIONS = {
    # The arm's nominal -Z centerline is rotated into the horizontal X/Y axis.
    "x": (_SQRT_HALF, 0.0, _SQRT_HALF, 0.0),
    "y": (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0),
}


@dataclass(frozen=True)
class FixedArmFixture:
    model: mujoco.MjModel
    data: mujoco.MjData
    manifest: ArmMjcfManifest
    sample_stride: int


@dataclass(frozen=True)
class StaticCharacterization:
    plane: str
    timestep: float
    integrator: int
    model_counts: Tuple[int, ...]
    initial_tip_position: np.ndarray
    final_tip_position: np.ndarray
    tip_gravity_displacement: float
    total_bend: float
    total_twist: float
    joint_rotation_vectors: np.ndarray
    joint_main_bends: np.ndarray
    tip_window_drift: float
    max_arm_qvel: float
    joint_angle_window_change: float
    max_contacts: int
    warning_count: int
    all_finite: bool
    initial_energy: float
    maximum_energy: float
    final_energy: float
    root_to_tip_length: float
    tendon_lengths: np.ndarray


def _quaternion_text(quaternion) -> str:
    return " ".join(format(float(value), ".17g") for value in quaternion)


def build_fixed_arm_fixture(
    spec: ScaledStableSoftArmSpec,
    plane: str,
    *,
    sim_freq: int,
    ctrl_freq: int,
    integrator: str,
) -> FixedArmFixture:
    """Compile the production arm fragment under a welded test fixture."""
    if plane not in _FIXTURE_QUATERNIONS:
        raise ValueError("plane must be 'x' or 'y'")
    if sim_freq <= 0 or ctrl_freq <= 0 or sim_freq % ctrl_freq != 0:
        raise ValueError("sim_freq must be positive and divisible by ctrl_freq")
    if integrator not in ("RK4", "implicitfast"):
        raise ValueError("integrator must be RK4 or implicitfast")

    fragments = build_scaled_stablesoft_arm_mjcf("fixture", spec, MOUNT_Z)
    fixture_quaternion = _quaternion_text(_FIXTURE_QUATERNIONS[plane])
    xml = f'''<mujoco model="scaled_stablesoft_fixed_{plane}">
  <option gravity="0 0 -9.81" timestep="{1.0 / sim_freq}" integrator="{integrator}"/>
  <compiler inertiafromgeom="false" autolimits="true"/>
  <worldbody>
    <body name="fixture_mount" quat="{fixture_quaternion}">
{fragments.worldbody_xml}    </body>
  </worldbody>
  <tendon>
{fragments.tendon_xml}
  </tendon>
  <sensor>
{fragments.sensor_xml}
  </sensor>
</mujoco>
'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_forward(model, data)
    return FixedArmFixture(
        model=model,
        data=data,
        manifest=fragments.manifest,
        sample_stride=sim_freq // ctrl_freq,
    )


def named_joint_addresses(model, manifest):
    """Return joint IDs and qpos/qvel addresses without global index literals."""
    result = []
    for name in manifest.ball_joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise AssertionError(name)
        result.append(
            (
                joint_id,
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
        )
    return tuple(result)


def full_mass_matrix(model, data):
    """Expand MuJoCo's packed mass matrix across supported Python APIs."""
    result = np.zeros((model.nv, model.nv), dtype=float)
    try:
        mujoco.mj_fullM(model, data, result)
    except TypeError:
        packed = data.qM if hasattr(data, "qM") else data.M
        mujoco.mj_fullM(model, result, packed)
    return result


def _rotation_vector(quaternion):
    result = np.zeros(3, dtype=float)
    quaternion = np.asarray(quaternion, dtype=float).copy()
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    mujoco.mju_quat2Vel(result, quaternion, 1.0)
    return result


def joint_rotation_vectors(model, data, addresses):
    return np.asarray(
        [
            _rotation_vector(data.qpos[qpos_address:qpos_address + 4])
            for _, qpos_address, _ in addresses
        ]
    )


def composite_rotation_vector(model, data, addresses):
    total = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    product = np.empty(4, dtype=float)
    for _, qpos_address, _ in addresses:
        mujoco.mju_mulQuat(
            product,
            total,
            data.qpos[qpos_address:qpos_address + 4],
        )
        total = product.copy()
    return _rotation_vector(total)


def _warning_count(data):
    return sum(int(warning.number) for warning in data.warning)


def _sample_energy(model, data):
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    return float(np.sum(data.energy))


def run_static_characterization(
    plane: str,
    *,
    physical_duration: float = 5.0,
    steady_window: float = 0.5,
    sim_freq: int = 4800,
    ctrl_freq: int = 240,
    integrator: str = "implicitfast",
) -> StaticCharacterization:
    """Release a horizontal fixed-root arm and record its static response."""
    spec = ScaledStableSoftArmSpec(
        joint_stiffness=0.003086,
        joint_damping=None,
        target_damping_ratio=0.3,
    )
    fixture = build_fixed_arm_fixture(
        spec,
        plane,
        sim_freq=sim_freq,
        ctrl_freq=ctrl_freq,
        integrator=integrator,
    )
    model, data, manifest = fixture.model, fixture.data, fixture.manifest
    addresses = named_joint_addresses(model, manifest)
    arm_dof_addresses = np.concatenate(
        [np.arange(dof_address, dof_address + 3) for _, _, dof_address in addresses]
    )
    tip_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_tip_site_name
    )
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, manifest.arm_root_site_name
    )
    initial_tip = np.asarray(data.site_xpos[tip_id], dtype=float).copy()

    tip_samples = [initial_tip]
    rotation_samples = [joint_rotation_vectors(model, data, addresses)]
    qvel_samples = [np.asarray(data.qvel[arm_dof_addresses]).copy()]
    energy_samples = [_sample_energy(model, data)]
    max_contacts = int(data.ncon)
    all_finite = True
    total_steps = int(round(physical_duration * sim_freq))

    for step_index in range(total_steps):
        mujoco.mj_step(model, data)
        all_finite = all_finite and all(
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
        max_contacts = max(max_contacts, int(data.ncon))
        if _warning_count(data) or not all_finite:
            break
        if (step_index + 1) % fixture.sample_stride == 0:
            mujoco.mj_forward(model, data)
            tip_samples.append(np.asarray(data.site_xpos[tip_id]).copy())
            rotation_samples.append(
                joint_rotation_vectors(model, data, addresses)
            )
            qvel_samples.append(np.asarray(data.qvel[arm_dof_addresses]).copy())
            energy_samples.append(_sample_energy(model, data))
            all_finite = all_finite and all(
                np.all(np.isfinite(values))
                for values in (
                    data.site_xpos,
                    data.ten_length,
                    data.energy,
                )
            )

    mujoco.mj_forward(model, data)
    tips = np.asarray(tip_samples)
    rotations = np.asarray(rotation_samples)
    qvels = np.asarray(qvel_samples)
    energies = np.asarray(energy_samples)
    window_count = int(round(steady_window * ctrl_freq)) + 1
    window_tips = tips[-window_count:]
    window_rotations = rotations[-window_count:]
    window_qvels = qvels[-window_count:]
    final_rotations = joint_rotation_vectors(model, data, addresses)
    total_rotation = composite_rotation_vector(model, data, addresses)
    main_axis = 1 if plane == "x" else 0

    return StaticCharacterization(
        plane=plane,
        timestep=float(model.opt.timestep),
        integrator=int(model.opt.integrator),
        model_counts=(
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
        initial_tip_position=initial_tip,
        final_tip_position=np.asarray(data.site_xpos[tip_id]).copy(),
        tip_gravity_displacement=float(data.site_xpos[tip_id, 2] - initial_tip[2]),
        total_bend=float(np.linalg.norm(total_rotation[:2])),
        total_twist=float(abs(total_rotation[2])),
        joint_rotation_vectors=final_rotations,
        joint_main_bends=final_rotations[:, main_axis].copy(),
        tip_window_drift=float(
            np.max(np.linalg.norm(window_tips - window_tips[-1], axis=1))
        ),
        max_arm_qvel=float(np.max(np.abs(window_qvels))),
        joint_angle_window_change=float(
            np.max(
                np.linalg.norm(
                    window_rotations - window_rotations[-1], axis=2
                )
            )
        ),
        max_contacts=max_contacts,
        warning_count=_warning_count(data),
        all_finite=bool(all_finite),
        initial_energy=float(energies[0]),
        maximum_energy=float(np.max(energies)),
        final_energy=float(energies[-1]),
        root_to_tip_length=float(
            np.linalg.norm(data.site_xpos[tip_id] - data.site_xpos[root_id])
        ),
        tendon_lengths=np.asarray(data.ten_length).copy(),
    )
