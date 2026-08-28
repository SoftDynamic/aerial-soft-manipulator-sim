"""Headless fixed-root closed-loop runner for Gate C2-C2-G."""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from multi_drone_mujoco.control.tendon_compatible_acceleration_controller import (
    TendonCompatibleAccelerationController,
    TendonCompatibleAccelerationControllerConfig,
)
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import IdealTendonTensionSpec
from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_probe_runner import (
    CTRL_FREQ_HZ,
    INTEGRATOR,
    SIM_FREQUENCIES_HZ,
)
from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    JOINT_STIFFNESS,
    MAX_TENSION_N,
    TARGET_DAMPING_RATIO,
    resolve_tendon_actuator_bindings,
    sample_ideal_tendon_transmission,
    write_tension_command,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    WarningLatch,
    build_fixed_arm_fixture,
    composite_rotation_vector,
    joint_rotation_vectors,
    named_joint_addresses,
)
from multi_drone_mujoco.tests._scaled_stablesoft_acceleration_probe_runner import (
    _spring_energy,
)


@dataclass(frozen=True)
class CompatibleClosedLoopConfig:
    sim_freq_hz: int
    controller: TendonCompatibleAccelerationControllerConfig
    direction: tuple
    acceleration_m_s2: float = 0.04
    pulse_duration_s: float = 0.05
    duration_s: float = 0.8

    def __post_init__(self):
        if self.sim_freq_hz not in SIM_FREQUENCIES_HZ:
            raise ValueError(f"sim_freq_hz must be one of {SIM_FREQUENCIES_HZ}")
        if not isinstance(self.controller, TendonCompatibleAccelerationControllerConfig):
            raise TypeError("controller has the wrong type")
        direction = np.asarray(self.direction, dtype=float)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must contain two finite values")
        if not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-15):
            raise ValueError("direction must be a unit two-vector")
        for name in ("acceleration_m_s2", "pulse_duration_s", "duration_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.duration_s <= 2.0 * self.pulse_duration_s:
            raise ValueError("duration must include a post-pulse hold")
        for name, duration in (
            ("pulse_duration_s", self.pulse_duration_s),
            ("duration_s", self.duration_s),
        ):
            if not np.isclose(duration * CTRL_FREQ_HZ, round(duration * CTRL_FREQ_HZ), rtol=0.0, atol=1e-13):
                raise ValueError(f"{name} must align with the control grid")
        object.__setattr__(self, "direction", tuple(direction))

    @property
    def intervals(self):
        return int(round(self.duration_s * CTRL_FREQ_HZ))

    @property
    def pulse_ticks(self):
        return int(round(self.pulse_duration_s * CTRL_FREQ_HZ))

    @property
    def substeps(self):
        return self.sim_freq_hz // CTRL_FREQ_HZ


@dataclass(frozen=True)
class CompatibleClosedLoopTrace:
    config: CompatibleClosedLoopConfig
    tendon_names: tuple
    time_s: np.ndarray
    acceleration_command_m_s2: np.ndarray
    tension_command_n: np.ndarray
    q_reference_m: np.ndarray
    q_velocity_reference_m_s: np.ndarray
    q_actual_m: np.ndarray
    q_velocity_actual_m_s: np.ndarray
    desired_effort_n: np.ndarray
    feedforward_effort_n: np.ndarray
    total_effort_n: np.ndarray
    achieved_effort_n: np.ndarray
    allocator_residual_n: np.ndarray
    allocator_saturated: np.ndarray
    feedforward_out_of_envelope: np.ndarray
    tendon_length_m: np.ndarray
    tendon_velocity_m_s: np.ndarray
    tip_position_m: np.ndarray
    joint_rotation_vectors_rad: np.ndarray
    composite_rotation_vectors_rad: np.ndarray
    common_reference_residual: np.ndarray
    twist_candidate_reference_residual: np.ndarray
    allocator_status: tuple
    fault_latched: np.ndarray
    contact_count: np.ndarray
    warning_events: tuple
    tendon_range_margin_m: np.ndarray
    physics_time_s: np.ndarray
    generalized_power_w: np.ndarray
    actuator_power_w: np.ndarray
    tendon_power_w: np.ndarray
    command_tendon_power_w: np.ndarray
    position_energy_j: np.ndarray
    kinetic_energy_j: np.ndarray
    spring_energy_j: np.ndarray
    total_energy_j: np.ndarray
    tendon_work_j: np.ndarray
    damping_dissipation_j: np.ndarray
    energy_ledger_residual_j: np.ndarray


def _freeze(values, dtype=float):
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _command_at_tick(config, tick):
    sign = 1.0 if tick < config.pulse_ticks else -1.0 if tick < 2 * config.pulse_ticks else 0.0
    return sign * config.acceleration_m_s2 * np.asarray(config.direction)


def run_compatible_closed_loop(config):
    if not isinstance(config, CompatibleClosedLoopConfig):
        raise TypeError("config must be CompatibleClosedLoopConfig")
    fixture = build_fixed_arm_fixture(
        ScaledStableSoftArmSpec(
            joint_stiffness=JOINT_STIFFNESS,
            joint_damping=None,
            target_damping_ratio=TARGET_DAMPING_RATIO,
        ),
        "vertical",
        sim_freq=config.sim_freq_hz,
        ctrl_freq=CTRL_FREQ_HZ,
        integrator=INTEGRATOR,
        tendon_drive=IdealTendonTensionSpec(max_tension=MAX_TENSION_N),
    )
    model, data = fixture.model, fixture.data
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    bindings = tuple(resolve_tendon_actuator_bindings(fixture))
    tendon_names = tuple(item.tendon_name for item in bindings)
    tendon_ids = np.asarray([item.tendon_id for item in bindings], dtype=int)
    tendon_range = np.asarray(model.tendon_range[tendon_ids]).copy()
    drive = IdealTendonTensionSpec(max_tension=MAX_TENSION_N)
    addresses = named_joint_addresses(model, fixture.manifest)
    tip_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, fixture.manifest.arm_tip_site_name
    )
    if tip_id < 0:
        raise AssertionError("arm tip site is missing")

    mujoco.mj_resetData(model, data)
    write_tension_command(fixture, (0.0,) * 4, drive, bindings=bindings)
    mujoco.mj_forward(model, data)
    controller = TendonCompatibleAccelerationController(
        config.controller, tendon_names
    )
    controller.reset(data.ten_length[tendon_ids], data.ten_velocity[tendon_ids])
    controller.enable()
    warning_latch = WarningLatch()
    diagnostic_data = mujoco.MjData(model)

    def diagnostic_sample(command):
        mujoco.mj_copyData(diagnostic_data, model, data)
        mujoco.mj_forward(model, diagnostic_data)
        mujoco.mj_energyPos(model, diagnostic_data)
        mujoco.mj_energyVel(model, diagnostic_data)
        return sample_ideal_tendon_transmission(
            model, diagnostic_data, bindings, command
        )

    records = []
    physics_records = []
    range_margins = []
    contact_counts = []
    tendon_work = 0.0
    damping_dissipation = 0.0
    diagnostic_sample((0.0,) * 4)
    initial_energy = float(np.sum(diagnostic_data.energy))

    for tick in range(config.intervals):
        acceleration = _command_at_tick(config, tick)
        tension = controller.update_bending_acceleration(
            acceleration,
            data.ten_length[tendon_ids],
            data.ten_velocity[tendon_ids],
        )
        written, current_bindings = write_tension_command(
            fixture, tension, drive, bindings=bindings
        )
        if current_bindings != bindings or tuple(written) != tuple(tension):
            raise AssertionError("ideal tension binding or command changed")
        transmission = diagnostic_sample(written)
        allocation = controller.state.allocation
        feedforward = controller.state.static_feedforward
        records.append({
            "time": tick / CTRL_FREQ_HZ,
            "a": acceleration.copy(),
            "tension": np.asarray(tension).copy(),
            "q_ref": np.asarray(controller.state.reference.bending_contraction_reference_m),
            "v_ref": np.asarray(controller.state.reference.bending_velocity_reference_m_s),
            "q": np.asarray(controller.state.bending_contraction_actual_m),
            "v": np.asarray(controller.state.bending_velocity_actual_m_s),
            "desired": np.asarray(controller.state.modal_pd.desired_bending_effort_n),
            "feedforward": (
                np.asarray(feedforward.static_bending_effort_n)
                if feedforward is not None else np.zeros(2)
            ),
            "total": np.asarray(controller.state.desired_total_bending_effort_n),
            "achieved": np.asarray(allocation.achieved_bending_effort_n) if allocation else np.full(2, np.nan),
            "residual": allocation.primary_residual_n if allocation else np.nan,
            "saturated": allocation.status == "saturated_best_effort" if allocation else False,
            "out_of_envelope": controller.state.feedforward_out_of_envelope,
            "tendon_length": transmission.tendon_length.copy(),
            "tendon_velocity": transmission.tendon_velocity.copy(),
            "tip": np.asarray(data.site_xpos[tip_id]).copy(),
            "joints": joint_rotation_vectors(model, data, addresses),
            "composite": composite_rotation_vector(model, data, addresses),
            "common": np.asarray(controller.state.reference_common_residuals),
            "checker": np.asarray(controller.state.reference_twist_candidate_residuals),
            "status": controller.state.status,
            "fault": controller.state.fault_latched,
        })
        previous_power = transmission.actuator_power
        previous_damping = float(
            -diagnostic_data.qfrc_damper @ diagnostic_data.qvel
        )
        for _ in range(config.substeps):
            mujoco.mj_step(model, data)
            if not (
                math.isfinite(float(data.time))
                and np.all(np.isfinite(data.qpos))
                and np.all(np.isfinite(data.qvel))
            ):
                raise AssertionError("non-finite plant state")
            warning_latch.snapshot(data, stage="physics", step_index=len(range_margins) + 1)
            current = diagnostic_sample(written)
            current_damping = float(
                -diagnostic_data.qfrc_damper @ diagnostic_data.qvel
            )
            dt = 1.0 / config.sim_freq_hz
            tendon_work += 0.5 * (previous_power + current.actuator_power) * dt
            damping_dissipation += 0.5 * (previous_damping + current_damping) * dt
            total_energy = float(np.sum(diagnostic_data.energy))
            physics_records.append({
                "time": float(data.time),
                "generalized_power": current.generalized_power,
                "actuator_power": current.actuator_power,
                "tendon_power": current.tendon_power,
                "command_tendon_power": current.command_tendon_power,
                "position_energy": float(diagnostic_data.energy[0]),
                "kinetic_energy": float(diagnostic_data.energy[1]),
                "spring_energy": _spring_energy(
                    model,
                    joint_rotation_vectors(model, diagnostic_data, addresses),
                    addresses,
                ),
                "total_energy": total_energy,
                "tendon_work": tendon_work,
                "damping": damping_dissipation,
                "ledger": total_energy - initial_energy - (
                    tendon_work - damping_dissipation
                ),
            })
            previous_power = current.actuator_power
            previous_damping = current_damping
            contact_counts.append(int(data.ncon))
            lengths = np.asarray(data.ten_length[tendon_ids])
            range_margins.append(
                np.minimum(lengths - tendon_range[:, 0], tendon_range[:, 1] - lengths)
            )

    return CompatibleClosedLoopTrace(
        config=config,
        tendon_names=tendon_names,
        time_s=_freeze([item["time"] for item in records]),
        acceleration_command_m_s2=_freeze([item["a"] for item in records]),
        tension_command_n=_freeze([item["tension"] for item in records]),
        q_reference_m=_freeze([item["q_ref"] for item in records]),
        q_velocity_reference_m_s=_freeze([item["v_ref"] for item in records]),
        q_actual_m=_freeze([item["q"] for item in records]),
        q_velocity_actual_m_s=_freeze([item["v"] for item in records]),
        desired_effort_n=_freeze([item["desired"] for item in records]),
        feedforward_effort_n=_freeze([item["feedforward"] for item in records]),
        total_effort_n=_freeze([item["total"] for item in records]),
        achieved_effort_n=_freeze([item["achieved"] for item in records]),
        allocator_residual_n=_freeze([item["residual"] for item in records]),
        allocator_saturated=_freeze([item["saturated"] for item in records], dtype=bool),
        feedforward_out_of_envelope=_freeze([item["out_of_envelope"] for item in records], dtype=bool),
        tendon_length_m=_freeze([item["tendon_length"] for item in records]),
        tendon_velocity_m_s=_freeze([item["tendon_velocity"] for item in records]),
        tip_position_m=_freeze([item["tip"] for item in records]),
        joint_rotation_vectors_rad=_freeze([item["joints"] for item in records]),
        composite_rotation_vectors_rad=_freeze([item["composite"] for item in records]),
        common_reference_residual=_freeze([item["common"] for item in records]),
        twist_candidate_reference_residual=_freeze([item["checker"] for item in records]),
        allocator_status=tuple(item["status"] for item in records),
        fault_latched=_freeze([item["fault"] for item in records], dtype=bool),
        contact_count=_freeze(contact_counts, dtype=int),
        warning_events=tuple(warning_latch.events),
        tendon_range_margin_m=_freeze(range_margins),
        physics_time_s=_freeze([item["time"] for item in physics_records]),
        generalized_power_w=_freeze([item["generalized_power"] for item in physics_records]),
        actuator_power_w=_freeze([item["actuator_power"] for item in physics_records]),
        tendon_power_w=_freeze([item["tendon_power"] for item in physics_records]),
        command_tendon_power_w=_freeze([item["command_tendon_power"] for item in physics_records]),
        position_energy_j=_freeze([item["position_energy"] for item in physics_records]),
        kinetic_energy_j=_freeze([item["kinetic_energy"] for item in physics_records]),
        spring_energy_j=_freeze([item["spring_energy"] for item in physics_records]),
        total_energy_j=_freeze([item["total_energy"] for item in physics_records]),
        tendon_work_j=_freeze([item["tendon_work"] for item in physics_records]),
        damping_dissipation_j=_freeze([item["damping"] for item in physics_records]),
        energy_ledger_residual_j=_freeze([item["ledger"] for item in physics_records]),
    )
