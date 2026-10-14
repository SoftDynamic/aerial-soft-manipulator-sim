"""Standalone composition runner for one CF2X controlled by PX4 SITL.

Nothing in this module is imported by :mod:`base_aviary`; the optional PX4
path remains an adapter around the existing direct-RPM environment boundary.
"""

from collections import deque
from itertools import count
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import math
from pathlib import Path
import platform
import queue
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from multi_drone_mujoco.backends.actuation import (
    PwmToRotorSpeedModel,
    Px4NormalizedOutputToPwm,
)
from multi_drone_mujoco.backends.cf2x_px4 import (
    cf2x_27g_calibration, validate_cf2x_px4_parameters,
    cf2x_normalized_to_pwm_config,
    cf2x_pwm_to_rpm_config,
)
from multi_drone_mujoco.backends.contracts import SimulatorClock, finite_tuple
from multi_drone_mujoco.backends.lockstep import (
    LockstepBootstrapConfig,
    LockstepConfig,
)
from multi_drone_mujoco.backends.interaction import (
    ArmInteractionCatalog,
    DeterministicInteractionTimeline,
    InteractionCommand,
    InteractionFrame,
    InteractionMode,
    MujocoArmInteractionApplier,
    clamp_norm,
)
from multi_drone_mujoco.backends.px4_environment import (
    PX4_COMPATIBLE_COMMIT,
    Px4EnvironmentMode,
    Px4OwnedProcessLauncher,
    px4_owned_launch_config,
    read_px4_parameter_file,
)
from multi_drone_mujoco.backends.px4_control import (
    Px4FlightControlConfig,
    Px4MavlinkFlightControl,
)
from multi_drone_mujoco.backends.px4_sitl import PX4SITLBackend, Px4BackendInput
from multi_drone_mujoco.backends.sensors import (
    IdealSensorAdapter,
    IdealSensorConfig,
    MujocoTruthSource,
)
from multi_drone_mujoco.backends.transport import (
    MavlinkTransportConfig,
    SimulatorMavlinkTransport,
)

from multi_drone_mujoco.models.parameters import ResolvedUavParameters
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import scaled_stablesoft_baseline_spec, ScaledStableSoftArmSpec
from multi_drone_mujoco.models.components.tendon_drive import BASELINE_MAX_TENSION_N, IdealTendonTensionSpec


WMM_30N_120E_NED_GAUSS = (
    0.33838503045,
    -0.03502778543,
    0.34874449317,
)




@dataclass(frozen=True)
class Px4RunnerConfig:
    """Explicit runtime configuration; network and process launch have no defaults."""

    mode: Px4EnvironmentMode
    bind_host: str
    bind_port: int
    launch_owned_px4: bool = False
    autopilot_dir: Optional[Path] = None
    simulator_host: Optional[str] = None
    parameter_file: Optional[Path] = None
    px4_instance: int = 0
    initial_position_enu_m: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    target_position_enu_m: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    sim_freq_hz: int = 2400
    control_freq_hz: int = 200
    duration_s: Optional[float] = 5.0
    handshake_timeout_s: float = 10.0
    actuator_timeout_s: float = 1.0
    bootstrap_max_frames: int = 1000
    bootstrap_wall_interval_s: float = 0.001
    bootstrap_final_timeout_s: float = 1.0
    gui: bool = False
    log_dir: Path = Path("results/px4")
    random_seed: int = 0
    motor_time_constant_s: float = 0.0
    gcs_bind_host: Optional[str] = None
    gcs_bind_port: Optional[int] = None
    auto_flight: bool = False
    preflight_duration_s: float = 10.0
    offboard_warmup_s: float = 1.0
    armed_ramp_timeout_s: float = 8.0
    release_hover_rpm_fraction: float = 1.1
    landing_duration_s: float = 8.0
    landing_descent_s: float = 3.0
    landing_height_enu_m: float = 0.06
    spool_target_height_offset_m: float = 1.0
    release_transition_s: float = 1.0
    plant: str = "rigid"
    arm_source: str = "manual"
    arm_acceleration_m_s2: Tuple[float, float] = (0.0, 0.0)
    visual_mode: str = "mesh"
    plant_step_freq_hz: Optional[int] = None
    interaction_mode: str = "disabled"
    interaction_script: Optional[Path] = None
    interaction_target_body: Optional[str] = None
    interaction_target_site: Optional[str] = None
    interaction_target_policy: str = "arm_segments"
    interaction_force_step_n: float = 0.0005
    interaction_max_force_n: float = 0.001
    interaction_spring_n_per_m: float = 0.25
    interaction_damping_n_s_per_m: float = 0.01
    interaction_force_release: str = "immediate"
    tendon_acceleration_step_m_s2: float = 0.005
    tendon_acceleration_limit_m_s2: float = 0.04
    interaction_overlay: bool = True
    allow_unverified_force: bool = False
    joint_torque_step_nm: float = 10e-6
    joint_torque_limit_nm: float = 50e-6
    allow_unverified_joint_torque: bool = False
    uav_parameters: ResolvedUavParameters = field(default_factory=ResolvedUavParameters.resolve)
    arm_spec: ScaledStableSoftArmSpec = field(default_factory=scaled_stablesoft_baseline_spec)
    max_tension_n: float = BASELINE_MAX_TENSION_N
    diagnostics: Optional[dict] = None
    effective_config: Optional[dict] = None

    def __post_init__(self) -> None:
        IdealTendonTensionSpec(self.max_tension_n)
        if self.duration_s is None and not self.gui:
            raise ValueError("headless PX4 requires a finite duration")
        if self.gui and self.plant != "rigid" and self.interaction_mode == "disabled":
            object.__setattr__(self, "interaction_mode", "mouse_force")
        if self.diagnostics and self.diagnostics.get("trial") is not None:
            d = self.diagnostics
            if d["trial"] not in ("baseline", "pulse"):
                raise ValueError("diagnostic trial must be baseline or pulse")
            if self.gui or self.arm_source != "manual" or self.plant != "active_arm" or not self.launch_owned_px4 or not self.auto_flight:
                raise ValueError("direct-tension diagnostic requires headless owned auto-flight SITL with manual active arm")
            for key in ("baseline_s", "stable_window_s", "position_threshold_m", "speed_threshold_m_s", "pulse_tension_n", "pulse_duration_s", "recovery_s"):
                if not math.isfinite(d[key]) or d[key] <= 0: raise ValueError(f"invalid diagnostic {key}")
            if type(d["pulse_tendon_index"]) is not int or not 0 <= d["pulse_tendon_index"] < 4:
                raise ValueError("pulse tendon index must be 0..3")
            if d["pulse_tension_n"] > self.max_tension_n or d["stable_window_s"] > min(d["baseline_s"], d["recovery_s"]):
                raise ValueError("diagnostic limits/windows inconsistent")
            if type(d["sample_hz"]) is not int or not 1 <= d["sample_hz"] <= 20 or type(d["max_samples"]) is not int or not 1 <= d["max_samples"] <= 1024:
                raise ValueError("diagnostic storage must be bounded: 1..20 Hz, 1..1024 samples")
            required = d["baseline_s"] + (d["pulse_duration_s"] + d["recovery_s"] if d["trial"] == "pulse" else 0)
            if self.duration_s is None or self.duration_s < required:
                raise ValueError("duration must cover the diagnostic observation window")
        try:
            mode = Px4EnvironmentMode(self.mode)
        except ValueError as exc:
            raise ValueError("mode must be local or remote") from exc
        if mode is Px4EnvironmentMode.OFFLINE:
            raise ValueError("the real PX4 runner does not support offline mode")
        object.__setattr__(self, "mode", mode)
        if self.plant not in ("rigid", "passive_arm", "active_arm"):
            raise ValueError("plant must be rigid, passive_arm, or active_arm")
        if self.arm_source not in ("manual", "controller"):
            raise ValueError("arm_source must be manual or controller")
        if self.arm_source == "controller" and self.plant != "active_arm":
            raise ValueError("arm controller requires plant='active_arm'")
        if self.interaction_mode == "tendon_modal" and self.arm_source != "controller":
            raise ValueError("tendon_modal input requires arm_source='controller'")
        if self.visual_mode not in ("mesh", "cylinder"):
            raise ValueError("visual_mode must be mesh or cylinder")
        object.__setattr__(self, "arm_acceleration_m_s2", finite_tuple(
            "arm_acceleration_m_s2", self.arm_acceleration_m_s2, 2
        ))
        interaction_modes = (
            "disabled", "replay", "mouse_force", "keyboard_force", "tendon_modal",
            "joint_torque",
        )
        if self.interaction_mode not in interaction_modes:
            raise ValueError(
                "interaction_mode must be disabled, replay, mouse_force, "
                "keyboard_force, tendon_modal, or joint_torque"
            )
        if self.interaction_mode != "disabled" and self.plant == "rigid":
            raise ValueError("interaction requires passive_arm or active_arm")
        if self.interaction_mode == "tendon_modal" and self.plant != "active_arm":
            raise ValueError("tendon_modal interaction requires plant='active_arm'")
        if self.interaction_mode in (
            "mouse_force", "keyboard_force", "tendon_modal", "joint_torque"
        ) and not self.gui:
            raise ValueError(f"{self.interaction_mode} interaction requires --viewer")
        if self.interaction_mode == "replay" and self.interaction_script is None:
            raise ValueError("replay interaction requires interaction_script")
        if self.interaction_mode != "replay" and self.interaction_script is not None:
            raise ValueError("interaction_script is only valid in replay mode")
        if self.interaction_force_release != "immediate":
            raise ValueError("only immediate interaction force release is supported")
        if self.interaction_target_policy not in ("arm_segments", "single"):
            raise ValueError("interaction_target_policy must be arm_segments or single")
        for name in (
            "interaction_force_step_n", "interaction_max_force_n",
            "interaction_spring_n_per_m", "tendon_acceleration_step_m_s2",
            "tendon_acceleration_limit_m_s2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        damping = float(self.interaction_damping_n_s_per_m)
        if not math.isfinite(damping) or damping < 0.0:
            raise ValueError(
                "interaction_damping_n_s_per_m must be finite and non-negative"
            )
        object.__setattr__(self, "interaction_damping_n_s_per_m", damping)
        if self.interaction_max_force_n > 0.005 and not self.allow_unverified_force:
            raise ValueError(
                "interaction_max_force_n above the verified-default 5 mN requires "
                "allow_unverified_force=True"
            )
        for name in ("joint_torque_step_nm", "joint_torque_limit_nm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.joint_torque_step_nm > self.joint_torque_limit_nm:
            raise ValueError("joint_torque_step_nm cannot exceed joint_torque_limit_nm")
        if (
            self.joint_torque_limit_nm > 50e-6
            and not self.allow_unverified_joint_torque
        ):
            raise ValueError(
                "joint torque above the diagnostic 50 uN m limit requires "
                "allow_unverified_joint_torque=True"
            )
        if self.tendon_acceleration_limit_m_s2 > 0.04:
            raise ValueError(
                "tendon_acceleration_limit_m_s2 cannot exceed the frozen 0.04 m/s^2"
            )
        if not isinstance(self.bind_host, str) or not self.bind_host.strip():
            raise ValueError("bind_host must be a non-empty string")
        if isinstance(self.bind_port, bool) or not isinstance(self.bind_port, int) or not 1 <= self.bind_port <= 65535:
            raise ValueError("bind_port must be in [1, 65535]")
        for name in ("sim_freq_hz", "control_freq_hz", "bootstrap_max_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sim_freq_hz % self.control_freq_hz:
            raise ValueError("sim_freq_hz must be divisible by control_freq_hz")
        plant_frequency = (
            (1200 if self.plant == "active_arm" else self.control_freq_hz)
            if self.plant_step_freq_hz is None else self.plant_step_freq_hz
        )
        if isinstance(plant_frequency, bool) or not isinstance(plant_frequency, int) or plant_frequency <= 0:
            raise ValueError("plant_step_freq_hz must be a positive integer")
        if self.sim_freq_hz % plant_frequency:
            raise ValueError("sim_freq_hz must be divisible by plant_step_freq_hz")
        if plant_frequency % self.control_freq_hz:
            raise ValueError("plant_step_freq_hz must be divisible by control_freq_hz")
        if self.plant == "active_arm" and plant_frequency % 240:
            raise ValueError("active_arm plant_step_freq_hz must be divisible by 240 Hz")
        object.__setattr__(self, "plant_step_freq_hz", plant_frequency)
        for name in (
            "handshake_timeout_s", "actuator_timeout_s",
            "bootstrap_final_timeout_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.duration_s is not None and (not math.isfinite(self.duration_s) or self.duration_s <= 0):
            raise ValueError("duration_s must be finite and positive")
        if self.auto_flight and (mode is not Px4EnvironmentMode.LOCAL or not self.launch_owned_px4):
            raise ValueError("automatic arming requires local owned SITL")
        interval = float(self.bootstrap_wall_interval_s)
        if not math.isfinite(interval) or interval < 0.0:
            raise ValueError("bootstrap_wall_interval_s must be finite and non-negative")
        object.__setattr__(self, "bootstrap_wall_interval_s", interval)
        motor_tau = float(self.motor_time_constant_s)
        if not math.isfinite(motor_tau) or motor_tau < 0.0:
            raise ValueError("motor_time_constant_s must be finite and non-negative")
        object.__setattr__(self, "motor_time_constant_s", motor_tau)
        for name in (
            "preflight_duration_s", "offboard_warmup_s", "armed_ramp_timeout_s"
            , "landing_duration_s", "landing_descent_s", "release_transition_s"
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.offboard_warmup_s >= self.preflight_duration_s:
            raise ValueError("offboard_warmup_s must be shorter than preflight_duration_s")
        if (self.gcs_bind_host is None) != (self.gcs_bind_port is None):
            raise ValueError("gcs_bind_host and gcs_bind_port must be provided together")
        if self.gcs_bind_host is not None:
            if not isinstance(self.gcs_bind_host, str) or not self.gcs_bind_host.strip():
                raise ValueError("gcs_bind_host must be a non-empty string")
            if isinstance(self.gcs_bind_port, bool) or not isinstance(self.gcs_bind_port, int) or not 1 <= self.gcs_bind_port <= 65535:
                raise ValueError("gcs_bind_port must be in [1, 65535]")
        if self.auto_flight and self.gcs_bind_host is None:
            raise ValueError("auto_flight requires an explicit GCS bind endpoint")
        release_fraction = float(self.release_hover_rpm_fraction)
        calibration = cf2x_27g_calibration(parameters=self.uav_parameters, payload_mass_kg=self.arm_spec.total_mass if self.plant != "rigid" else 0.0)
        max_fraction = calibration.max_rpm / calibration.hover_rpm
        if not math.isfinite(release_fraction) or not 0.0 < release_fraction <= max_fraction:
            raise ValueError(
                f"release_hover_rpm_fraction must be in (0, {max_fraction:g}]"
            )
        object.__setattr__(self, "release_hover_rpm_fraction", release_fraction)
        landing_height = float(self.landing_height_enu_m)
        if not math.isfinite(landing_height) or landing_height <= 0.0:
            raise ValueError("landing_height_enu_m must be finite and positive")
        object.__setattr__(self, "landing_height_enu_m", landing_height)
        spool_offset = float(self.spool_target_height_offset_m)
        if not math.isfinite(spool_offset) or spool_offset <= 0.0:
            raise ValueError("spool_target_height_offset_m must be finite and positive")
        object.__setattr__(self, "spool_target_height_offset_m", spool_offset)
        if self.landing_descent_s > self.landing_duration_s:
            raise ValueError("landing_descent_s cannot exceed landing_duration_s")
        object.__setattr__(self, "initial_position_enu_m", finite_tuple(
            "initial_position_enu_m", self.initial_position_enu_m, 3
        ))
        object.__setattr__(self, "target_position_enu_m", finite_tuple(
            "target_position_enu_m", self.target_position_enu_m, 3
        ))
        object.__setattr__(self, "log_dir", Path(self.log_dir))
        if self.autopilot_dir is not None:
            object.__setattr__(self, "autopilot_dir", Path(self.autopilot_dir))
        if self.parameter_file is not None:
            parameter_file = Path(self.parameter_file)
            if not parameter_file.is_file():
                raise ValueError(f"parameter_file is not a file: {parameter_file}")
            object.__setattr__(self, "parameter_file", parameter_file)
        if self.interaction_script is not None:
            interaction_script = Path(self.interaction_script)
            if not interaction_script.is_file():
                raise ValueError(
                    f"interaction_script is not a file: {interaction_script}"
                )
            object.__setattr__(self, "interaction_script", interaction_script)
        if self.launch_owned_px4:
            if mode is not Px4EnvironmentMode.LOCAL:
                raise ValueError("owned PX4 launch is only valid in local mode")
            if self.autopilot_dir is None:
                raise ValueError("owned PX4 launch requires autopilot_dir")
            if not self.simulator_host:
                raise ValueError("owned PX4 launch requires simulator_host")
            expected_port = 4560 + self.px4_instance
            if self.bind_port != expected_port:
                raise ValueError(
                    f"owned PX4 instance {self.px4_instance} connects to port "
                    f"{expected_port}, not {self.bind_port}"
                )


@dataclass
class Px4RunMetrics:
    """Bounded-run evidence written as JSON after every exit path."""

    outcome: str = "not_started"
    physics_steps: int = 0
    bootstrap_frames: int = 0
    simulation_duration_s: float = 0.0
    wall_duration_s: float = 0.0
    position_rmse_m: float = 0.0
    max_position_error_m: float = 0.0
    max_abs_rpy_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_rpm: float = 0.0
    saturation_ratio: float = 0.0
    timeout_count: int = 0
    disconnect_count: int = 0
    sent_packets: int = 0
    received_packets: int = 0
    peer_system_id: Optional[int] = None
    peer_component_id: Optional[int] = None
    protocol_version: Optional[str] = None
    heartbeat_count: int = 0
    estimator_health_flags: Optional[int] = None
    estimator_timeout_flags: Optional[int] = None
    final_armed: bool = False
    landed: bool = False
    landing_simulation_duration_s: float = 0.0
    flight_warnings: Tuple[str, ...] = ()
    max_arm_tip_relative_displacement_m: float = 0.0
    max_contact_count: int = 0
    max_tendon_tension_n: float = 0.0
    tendon_saturation_ratio: float = 0.0
    max_tendon_modal_error_m: float = 0.0
    final_tendon_reference_m: Tuple[float, float] = (0.0, 0.0)
    final_tendon_actual_m: Tuple[float, float] = (0.0, 0.0)
    tendon_fault: Optional[str] = None
    interaction_command_count: int = 0
    interaction_release_count: int = 0
    interaction_clipped_count: int = 0
    max_external_force_n: float = 0.0
    max_generated_torque_nm: float = 0.0
    max_joint_torque_nm: float = 0.0
    last_interaction_release_s: Optional[float] = None
    interaction_recovery_time_s: Optional[float] = None
    min_altitude_m: Optional[float] = None
    max_tilt_rad: float = 0.0
    exit_detail: str = ""
    final_position_enu_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    log_path: Optional[str] = None
    baseline_status: str = "not_observed"
    baseline_mean_position_error_m: float = 0.0
    baseline_max_position_error_m: float = 0.0
    baseline_max_speed_m_s: float = 0.0
    diagnostic_peak_position_error_m: float = 0.0
    diagnostic_peak_speed_m_s: float = 0.0
    diagnostic_peak_attitude_rad: float = 0.0
    diagnostic_max_rotor_saturation: float = 0.0
    diagnostic_final_position_error_m: Optional[float] = None
    diagnostic_final_speed_m_s: Optional[float] = None
    recovery_status: str = "not_observed"
    pulse_start_s: Optional[float] = None
    pulse_release_s: Optional[float] = None
    actual_pulse_max_tension_n: float = 0.0
    plant_release_timestamp_us: Optional[int] = None
    plant_release_sequence: Optional[int] = None
    final_arm_tip_displacement_m: Optional[float] = None
    final_tendon_velocity_m_s: Tuple[float, ...] = ()


class Px4MujocoRunner:
    """Own the PX4 adapter lifecycle while treating ``BaseAviary`` as a plant."""

    def __init__(
        self,
        config: Px4RunnerConfig,
        aviary,
        backend: PX4SITLBackend,
        truth_source: MujocoTruthSource,
        launcher: Optional[Px4OwnedProcessLauncher] = None,
        flight_control: Optional[Px4MavlinkFlightControl] = None,
    ):
        self.config = config
        self.aviary = aviary
        self.backend = backend
        self.truth_source = truth_source
        self.launcher = launcher
        self.flight_control = flight_control
        self._calibration = cf2x_27g_calibration(parameters=aviary.UAV_PARAMETERS, payload_mass_kg=aviary.ARM_MASS + aviary.PAYLOAD_MASS)
        self._closed = False
        self._last_state = None
        self._final_actuator_armed = False
        self._last_rpm = np.zeros(4, dtype=float)
        self._tendon_controller = None
        self._tendon_ids = None
        self._tendon_updates = 0
        self._tendon_saturated_updates = 0
        self._started_utc = None
        self._interaction_timeline = None
        self._interaction_catalog = None
        self._force_applier = None
        self._last_interaction_command = None
        self._interaction_viewer = None
        self._interaction_events = deque(maxlen=256)

    @classmethod
    def create_default(cls, config: Px4RunnerConfig) -> "Px4MujocoRunner":
        """Create the rigid-CF2X plant and all replaceable PX4 adapters."""
        from multi_drone_mujoco.envs.base_aviary import BaseAviary
        from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
            scaled_stablesoft_baseline_spec,
            ScaledStableSoftVisualSpec,
        )
        from multi_drone_mujoco.models.components.tendon_drive import (
            IdealTendonTensionSpec,
        )
        from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics

        arm_spec = None
        tendon_drive = None
        integrator = "RK4"
        if config.plant != "rigid":
            arm_spec = config.arm_spec
            integrator = "implicitfast"
        if config.plant == "active_arm":
            tendon_drive = IdealTendonTensionSpec(max_tension=config.max_tension_n)
        aviary = BaseAviary(
            drone_model=DroneModel.CF2X,
            uav_parameters=config.uav_parameters,
            num_drones=1,
            initial_xyzs=np.asarray([config.initial_position_enu_m]),
            physics=Physics.MJC,
            sim_freq=config.sim_freq_hz,
            ctrl_freq=config.plant_step_freq_hz,
            gui=(config.gui and config.interaction_mode == "disabled"),
            act_type=ActionType.RPM,
            scaled_stablesoft_arm=arm_spec,
            scaled_stablesoft_visual=(
                ScaledStableSoftVisualSpec(mode=config.visual_mode) if arm_spec else None
            ),
            uav_source="px4", arm_source=config.arm_source,
            scaled_stablesoft_tendon_drive=tendon_drive,
            integrator=integrator,
        )
        calibration = cf2x_27g_calibration(parameters=aviary.UAV_PARAMETERS, payload_mass_kg=aviary.ARM_MASS + aviary.PAYLOAD_MASS)
        if config.parameter_file is not None:
            validate_cf2x_px4_parameters(read_px4_parameter_file(config.parameter_file), calibration)
        elif config.launch_owned_px4 or config.uav_parameters != ResolvedUavParameters.resolve() or config.arm_spec != scaled_stablesoft_baseline_spec():
            raise ValueError("owned or modified PX4 plant requires a physical parameter file")
        transport = SimulatorMavlinkTransport(MavlinkTransportConfig(
            bind_host=config.bind_host,
            bind_port=config.bind_port,
            heartbeat_timeout_s=config.handshake_timeout_s,
            source_system=255,
            source_component=190,
            max_reconnect_attempts=0,
            tcp_role="listen",
        ))
        sensor_adapter = IdealSensorAdapter(IdealSensorConfig(
            origin_latitude_deg=30.0,
            origin_longitude_deg=120.0,
            origin_altitude_m=100.0,
            gravity_m_s2=9.81,
            magnetic_field_ned_gauss=WMM_30N_120E_NED_GAUSS,
            imu_rate_hz=float(config.control_freq_hz),
            gps_rate_hz=10.0,
            barometer_rate_hz=50.0,
            magnetometer_rate_hz=50.0,
        ))
        backend = PX4SITLBackend(
            sensor_adapter=sensor_adapter,
            transport=transport,
            normalized_to_pwm=Px4NormalizedOutputToPwm(
                cf2x_normalized_to_pwm_config(calibration)
            ),
            pwm_to_rotor_speed=PwmToRotorSpeedModel(
                cf2x_pwm_to_rpm_config(config.motor_time_constant_s, calibration)
            ),
            lockstep_config=LockstepConfig(
                actuator_timeout_s=config.actuator_timeout_s,
                safe_normalized_values=(0.0, 0.0, 0.0, 0.0),
            ),
        )
        launcher = None
        if config.launch_owned_px4:
            parameters = (
                read_px4_parameter_file(config.parameter_file)
                if config.parameter_file is not None else {}
            )
            launcher = Px4OwnedProcessLauncher(px4_owned_launch_config(
                config.autopilot_dir,
                config.simulator_host,
                instance=config.px4_instance,
                startup_timeout_s=config.handshake_timeout_s,
                parameter_overrides=parameters,
            ))
        flight_control = None
        if config.gcs_bind_host is not None:
            flight_control = Px4MavlinkFlightControl(Px4FlightControlConfig(
                config.gcs_bind_host, config.gcs_bind_port
            ))
        return cls(
            config, aviary, backend, MujocoTruthSource(), launcher, flight_control
        )

    def run(self) -> Px4RunMetrics:
        """Run a bounded disarmed/externally-commanded lockstep experiment."""
        if self._closed:
            raise RuntimeError("runner is closed")
        metrics = Px4RunMetrics()
        started = time.monotonic()
        self._started_utc = datetime.now(timezone.utc).isoformat()
        squared_error_sum = 0.0
        saturated_values = 0
        rotor_values = 0
        max_rpy = np.zeros(3)
        recovery_candidate_s = None
        initial_arm_tip_relative = None
        self._diagnostic_samples = deque(maxlen=(self.config.diagnostics or {}).get("max_samples", 256))
        self._baseline_sum = 0.0
        self._baseline_count = 0
        self._baseline_stable = True
        self._diagnostic_done = False
        self._diagnostic_phase = "startup"
        dt_s = 1.0 / self.config.control_freq_hz
        plant_dt_s = 1.0 / self.config.plant_step_freq_hz
        try:
            self.aviary.reset(seed=self.config.random_seed)
            self._setup_interaction()
            self._setup_tendon_controller()
            if (self.config.diagnostics or {}).get("trial"):
                if len(self.aviary.SCALED_STABLESOFT_TENDON_BINDINGS) != 4 or self.launcher is None:
                    raise RuntimeError("diagnostic requires four named tendon actuators and owned SITL")
            self._open_interactive_viewer()
            initial_arm_tip_relative = self._arm_tip_relative()
            self.truth_source.reset(self.aviary)
            transport = self.backend.transport
            if self.flight_control is not None:
                self.flight_control.open()
            open_listener = getattr(transport, "open_listener", None)
            accept_peer = getattr(transport, "accept_peer", None)
            if open_listener is not None:
                open_listener()
                if self.launcher is not None:
                    self.launcher.start()
                accept_peer(self.config.handshake_timeout_s)
            else:
                if self.launcher is not None:
                    self.launcher.start()
                transport.connect()
            self.backend.reset(SimulatorClock(0, 0))

            initial_truth = self.truth_source.sample(self.aviary, dt_s)
            start_timestamp_us = 1_000_000

            def startup_frames():
                for index in range(self.config.bootstrap_max_frames):
                    clock = self._clock(index, start_timestamp_us)
                    yield self.backend.sensor_adapter.sample(initial_truth, clock)

            self.backend.coordinator.bootstrap(
                startup_frames(),
                LockstepBootstrapConfig(
                    max_frames=self.config.bootstrap_max_frames,
                    wall_interval_s=self.config.bootstrap_wall_interval_s,
                    final_timeout_s=self.config.bootstrap_final_timeout_s,
                ),
            )
            last_boot_sequence = self.backend.coordinator.last_sent_sequence
            metrics.bootstrap_frames = last_boot_sequence + 1
            if self.config.auto_flight:
                self._prepare_auto_flight(initial_truth, dt_s)
                last_boot_sequence = self.backend.coordinator.last_sent_sequence
                metrics.plant_release_timestamp_us = self.backend.coordinator.last_sent_timestamp_us
                metrics.plant_release_sequence = last_boot_sequence
            steps = None if self.config.duration_s is None else int(round(self.config.duration_s * self.config.plant_step_freq_hz))
            px4_stride = self.config.plant_step_freq_hz // self.config.control_freq_hz
            tendon_stride = (
                self.config.plant_step_freq_hz // 240
                if self.config.plant == "active_arm" else None
            )
            px4_sequence = last_boot_sequence
            target = np.asarray(self.config.target_position_enu_m)
            calibration = self._calibration
            for offset in (count(1) if steps is None else range(1, steps + 1)):
                # Retained rigid native viewer shares pause/close semantics.
                while self.config.gui and self._interaction_viewer is None and not self._poll_native_viewer():
                    if self._native_exit_requested: break
                    self.aviary.clear_automatic_commands()
                    self._last_rpm[:] = 0
                    time.sleep(1 / 60)
                if getattr(self, "_native_exit_requested", False):
                    metrics.outcome = "completed"
                    metrics.exit_detail = "viewer requested landing"
                    break
                elapsed_s = offset * plant_dt_s
                if self._interaction_viewer is not None:
                    while True:
                        interaction = self._interaction_viewer.poll(elapsed_s)
                        if self._interaction_viewer.exit_requested:
                            metrics.outcome = "completed"
                            metrics.exit_detail = "viewer requested landing"
                            break
                        if self._interaction_viewer.consume_advance_permission():
                            break
                        self.aviary.clear_automatic_commands()
                        self._last_rpm[:] = 0.0
                        time.sleep(1.0 / 60.0)
                    if self._interaction_viewer.exit_requested:
                        break
                    if self._interaction_timeline is not None:
                        interaction = self._sample_interaction(elapsed_s, metrics)
                    else:
                        interaction = self._prepare_interaction_command(interaction)
                        self._record_interaction_command(interaction, elapsed_s, metrics)
                else:
                    interaction = self._sample_interaction(elapsed_s, metrics)
                if interaction.released:
                    recovery_candidate_s = None
                    metrics.interaction_recovery_time_s = None
                if (offset - 1) % px4_stride == 0:
                    px4_sequence += 1
                    clock = self._clock(px4_sequence, start_timestamp_us)
                    if self.config.auto_flight:
                        self.flight_control.send_position_setpoint_enu(
                            clock.timestamp_us,
                            self._flight_target_local_enu(elapsed_s),
                        )
                    truth = self.truth_source.sample(self.aviary, dt_s)
                    result = self.backend.step(Px4BackendInput(truth, clock, dt_s))
                    if self.flight_control is not None:
                        self.flight_control.poll()
                    self._last_rpm = np.asarray(result.rotor_speed.rpm)
                    self._final_actuator_armed = result.rotor_speed.armed
                    self.aviary.set_rotor_rpm_command(
                        self._last_rpm, armed=result.rotor_speed.armed
                    )
                    if not result.rotor_speed.armed:
                        self.aviary.clear_automatic_commands()
                    metrics.max_rpm = max(
                        metrics.max_rpm, float(np.max(self._last_rpm))
                    )
                    saturated_values += int(np.count_nonzero(
                        self._last_rpm >= calibration.max_rpm * (1.0 - 1e-9)
                    ))
                    rotor_values += self._last_rpm.size
                    if not result.advance_physics:
                        self.aviary.clear_automatic_commands()
                        metrics.outcome = result.health.state.value
                        metrics.exit_detail = result.health.detail
                        break
                if tendon_stride is not None and (offset - 1) % tendon_stride == 0:
                    self._update_tendon_controller(elapsed_s, metrics, interaction)
                self._diagnostic_tension((offset - 1) * plant_dt_s, metrics)
                try:
                    if self._force_applier is not None and interaction.source != "native_mouse_wrench":
                        self._force_applier.apply(interaction)
                    self.aviary.step()
                finally:
                    if self._force_applier is not None:
                        self._force_applier.clear()
                    if self._interaction_viewer is not None:
                        self._interaction_viewer.after_step()
                        self._interaction_viewer.render_only()
                if self.config.gui and self._interaction_viewer is None:
                    self.aviary.render()
                position = np.asarray(self.aviary.pos[0], dtype=float)
                error = float(np.linalg.norm(position - target))
                speed = float(np.linalg.norm(np.asarray(self.aviary.vel[0], dtype=float)))
                altitude = float(position[2])
                metrics.min_altitude_m = (
                    altitude if metrics.min_altitude_m is None
                    else min(metrics.min_altitude_m, altitude)
                )
                squared_error_sum += error * error
                metrics.max_position_error_m = max(
                    metrics.max_position_error_m, error
                )
                max_rpy = np.maximum(
                    max_rpy, np.abs(np.asarray(self.aviary.rpy[0], dtype=float))
                )
                metrics.max_tilt_rad = max(
                    metrics.max_tilt_rad,
                    float(np.linalg.norm(np.asarray(self.aviary.rpy[0, :2], dtype=float))),
                )
                recovery_candidate_s = self._update_recovery(elapsed_s, error, speed, recovery_candidate_s, metrics)
                metrics.physics_steps += 1
                arm_tip_relative = self._arm_tip_relative()
                if arm_tip_relative is not None and initial_arm_tip_relative is not None:
                    metrics.max_arm_tip_relative_displacement_m = max(
                        metrics.max_arm_tip_relative_displacement_m,
                        float(np.linalg.norm(arm_tip_relative - initial_arm_tip_relative)),
                    )
                metrics.max_contact_count = max(
                    metrics.max_contact_count,
                    int(self.aviary.data.ncon),
                )
                self._measure_diagnostic(elapsed_s, error, speed, arm_tip_relative, initial_arm_tip_relative, metrics)
                if self._diagnostic_done:
                    metrics.outcome = "completed"
                    break
                if metrics.physics_steps % max(1, self.config.plant_step_freq_hz // 50) == 0:
                    flight_health = (
                        self.flight_control.health
                        if self.flight_control is not None else None
                    )
                    tendon_state = (
                        self._tendon_controller.state
                        if self._tendon_controller is not None else None
                    )
                    self._last_state = {
                        "time_s": metrics.physics_steps * plant_dt_s,
                        "position_enu_m": tuple(float(value) for value in self.aviary.pos[0]),
                        "rpy_enu_flu_rad": tuple(float(value) for value in self.aviary.rpy[0]),
                        "rpm": tuple(float(value) for value in self._last_rpm),
                        "normalized_motor_output": (
                            None
                            if self.backend.last_normalized_command is None
                            else self.backend.last_normalized_command.values
                        ),
                        "pwm_us": (
                            None
                            if self.backend.last_pwm_command is None
                            else self.backend.last_pwm_command.pulse_width_us
                        ),
                        "arm_tip_relative_m": (
                            None if arm_tip_relative is None
                            else tuple(float(value) for value in arm_tip_relative)
                        ),
                        "tendon_reference_m": (
                            None if tendon_state is None else
                            tendon_state.reference.bending_contraction_reference_m
                        ),
                        "tendon_actual_m": (
                            None if tendon_state is None else
                            tendon_state.bending_contraction_actual_m
                        ),
                        "tendon_tension_n": (
                            None if tendon_state is None else
                            tendon_state.tension_command_n
                        ),
                        "interaction": interaction.as_dict(),
                        "px4_local_position_ned_m": (
                            flight_health.local_position_ned_m
                            if flight_health is not None else None
                        ),
                    }
            else:
                metrics.outcome = "completed"
            metrics.simulation_duration_s = metrics.physics_steps * plant_dt_s
            if metrics.physics_steps:
                metrics.position_rmse_m = math.sqrt(
                    squared_error_sum / metrics.physics_steps
                )
            metrics.saturation_ratio = (
                saturated_values / rotor_values if rotor_values else 0.0
            )
            metrics.max_abs_rpy_rad = tuple(float(value) for value in max_rpy)
            if self._tendon_controller is not None:
                tendon_state = self._tendon_controller.state
                metrics.final_tendon_reference_m = tuple(
                    tendon_state.reference.bending_contraction_reference_m
                )
                metrics.final_tendon_actual_m = tuple(
                    tendon_state.bending_contraction_actual_m
                )
                metrics.tendon_fault = tendon_state.fault_reason
                metrics.tendon_saturation_ratio = (
                    self._tendon_saturated_updates / self._tendon_updates
                    if self._tendon_updates else 0.0
                )
            if self.config.auto_flight and metrics.outcome == "completed":
                if self._interaction_viewer is not None:
                    self._interaction_viewer.close()
                    self._interaction_viewer = None
                if getattr(self.aviary, "_viewer", None) is not None:
                    self.aviary._viewer.close()
                    self.aviary._viewer = None
                landing_steps = self._land_and_disarm(dt_s)
                metrics.landing_simulation_duration_s = landing_steps * plant_dt_s
                metrics.landed = not self._final_actuator_armed
            metrics.final_position_enu_m = tuple(
                float(value) for value in self.aviary.pos[0]
            )
        except KeyboardInterrupt:
            metrics.outcome = "interrupted"
            metrics.exit_detail = "KeyboardInterrupt"
        except Exception as exc:
            metrics.outcome = "error"
            metrics.exit_detail = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            metrics.wall_duration_s = time.monotonic() - started
            health = self.backend.transport.health
            metrics.timeout_count = health.timeout_count
            metrics.disconnect_count = health.disconnect_count
            metrics.sent_packets = health.sent_packets
            metrics.received_packets = health.received_packets
            metrics.peer_system_id = health.peer_system_id
            metrics.peer_component_id = health.peer_component_id
            metrics.protocol_version = health.protocol_version
            if self.flight_control is not None:
                flight_health = self.flight_control.health
                metrics.heartbeat_count = flight_health.heartbeat_count
                metrics.estimator_health_flags = flight_health.estimator_health_flags
                metrics.estimator_timeout_flags = flight_health.estimator_timeout_flags
                metrics.final_armed = self._final_actuator_armed
                metrics.flight_warnings = flight_health.warnings
            self.close()
            metrics.log_path = str(self._write_log(metrics))
        return metrics

    def reset(self) -> None:
        """Reset only the plant before a new runner lifecycle is connected."""
        if self._closed:
            raise RuntimeError("runner is closed")
        self.aviary.reset(seed=self.config.random_seed)
        if self._force_applier is not None:
            self._force_applier.clear()
        if self._interaction_timeline is not None:
            self._interaction_timeline.reset()
        if self._interaction_viewer is not None:
            self._interaction_viewer.reset()
        self._last_interaction_command = None
        self.truth_source.reset(self.aviary)
        self._setup_tendon_controller()

    def pause(self) -> None:
        self.backend.pause()

    def resume(self) -> None:
        self.backend.resume()

    def request_single_step(self) -> None:
        self.backend.request_single_step()

    def close(self) -> None:
        if self._closed:
            return
        if self._force_applier is not None:
            self._force_applier.clear()
        if self._interaction_viewer is not None:
            self._interaction_viewer.close()
            self._interaction_viewer = None
        if (self.config.diagnostics or {}).get("trial"):
            self.aviary.set_scaled_stablesoft_tension_command((0.0,) * 4)
        self.aviary.clear_automatic_commands()
        self._last_rpm[:] = 0.0
        try:
            self.backend.close()
        finally:
            try:
                if self.launcher is not None:
                    self.launcher.close()
            finally:
                try:
                    if self.flight_control is not None:
                        self.flight_control.close()
                finally:
                    self.aviary.close()
                    self._closed = True

    def _setup_interaction(self) -> None:
        """Resolve arm names once and prepare a simulation-time replay."""
        self._interaction_timeline = None
        self._interaction_catalog = None
        self._force_applier = None
        self._last_interaction_command = None
        self._interaction_events = deque(maxlen=256)
        if self.config.interaction_mode == "disabled":
            return
        manifest = self.aviary.SCALED_STABLESOFT_ARM_MANIFEST
        self._interaction_catalog = ArmInteractionCatalog.discover(
            self.aviary.model, manifest
        )
        body_name, site_name = self._resolve_interaction_target()
        selected = self._interaction_catalog.segment_for_body(body_name)
        legacy_single_target = (
            self.config.interaction_target_body is not None
            or self.config.interaction_target_site is not None
        )
        allowed_body_ids = (
            (selected.body_id,)
            if self.config.interaction_target_policy == "single" or legacy_single_target
            else self._interaction_catalog.body_ids
        )
        self._force_applier = MujocoArmInteractionApplier(
            self.aviary.model,
            self.aviary.data,
            self._interaction_catalog,
            allowed_body_ids=allowed_body_ids,
            default_site_name=site_name,
        )
        # The compatibility properties default to the selected legacy target,
        # while the arm_segments policy retains all manifest bodies as legal.
        self._force_applier.default_segment = selected
        if self.config.interaction_mode == "replay":
            self._interaction_timeline = DeterministicInteractionTimeline.from_json(
                self.config.interaction_script
            )
            for event in self._interaction_timeline.events:
                if event.mode is InteractionMode.TENDON_MODAL and self.config.arm_source != "controller":
                    raise ValueError("tendon-modal replay requires arm_source='controller'")
                if (
                    event.mode is InteractionMode.TENDON_MODAL
                    and self.config.plant != "active_arm"
                ):
                    raise ValueError(
                        "tendon-modal replay events require plant='active_arm'"
                    )
                if (
                    event.mode is InteractionMode.JOINT_TORQUE
                    and self.config.plant == "rigid"
                ):
                    raise ValueError("joint-torque replay requires an arm plant")

    def _open_interactive_viewer(self) -> None:
        if not self.config.gui:
            return
        if self.config.plant == "rigid":
            self._native_events = queue.SimpleQueue()
            self._native_paused = False
            self.aviary._viewer_key_callback = self._native_events.put
            return
        from multi_drone_mujoco.backends.px4_interactive_viewer import (
            Px4InteractionViewer,
        )

        self._interaction_viewer = Px4InteractionViewer(
            self.aviary, self._force_applier, self.config, self
        )

    def _poll_native_viewer(self):
        import glfw
        viewer = getattr(self.aviary, "_viewer", None)
        self._native_exit_requested = viewer is not None and not viewer.is_running()
        if viewer is not None and not self._native_exit_requested: viewer.sync()
        single_step = False
        while not self._native_events.empty():
            key = self._native_events.get_nowait()
            if key in (glfw.KEY_ESCAPE, glfw.KEY_L, glfw.KEY_Q, glfw.KEY_BACKSPACE):
                self._native_exit_requested = True
            elif key == glfw.KEY_SPACE: self._native_paused = not self._native_paused
            elif key == glfw.KEY_PERIOD and self._native_paused: single_step = True
        return not self._native_paused or single_step

    def _resolve_interaction_target(self) -> Tuple[str, Optional[str]]:
        """Return a terminal body/site pair without numeric-ID configuration."""
        import mujoco

        manifest = self.aviary.SCALED_STABLESOFT_ARM_MANIFEST
        body_name = self.config.interaction_target_body
        site_name = self.config.interaction_target_site
        if site_name is None and body_name is None:
            site_name = manifest.arm_tip_site_name
        if site_name is None:
            segment = self._interaction_catalog.segment_for_body(body_name)
            return segment.body_name, None
        site_id = mujoco.mj_name2id(
            self.aviary.model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if site_id < 0:
            raise ValueError(f"MuJoCo interaction site not found: {site_name}")
        actual_body_id = int(self.aviary.model.site_bodyid[site_id])
        actual_body_name = mujoco.mj_id2name(
            self.aviary.model, mujoco.mjtObj.mjOBJ_BODY, actual_body_id
        )
        body_name = body_name or actual_body_name
        if body_name != actual_body_name:
            raise ValueError(
                f"interaction site {site_name} belongs to {actual_body_name}, "
                f"not {body_name}"
            )
        return body_name, site_name

    def _sample_interaction(
        self, elapsed_s: float, metrics: Px4RunMetrics
    ) -> InteractionCommand:
        if self._interaction_timeline is None:
            command = InteractionCommand(
                timestamp_us=round(elapsed_s * 1_000_000),
                sequence=metrics.physics_steps + 1,
                mode=InteractionMode.DISABLED,
                target_body=(
                    self._force_applier.body_name
                    if self._force_applier is not None else None
                ),
                target_site=(
                    self._force_applier.site_name
                    if self._force_applier is not None else None
                ),
                frame=InteractionFrame.WORLD_ENU,
                source="none",
            )
        else:
            command = self._interaction_timeline.sample(
                elapsed_s,
                target_body=self._force_applier.body_name,
                target_site=self._force_applier.site_name,
                point_world_m=self._force_applier.point_world_m,
            )
        if command.mode is InteractionMode.EXTERNAL_FORCE:
            force, clipped = clamp_norm(
                command.force_n, self.config.interaction_max_force_n
            )
            command = replace(
                command,
                force_n=tuple(float(value) for value in force),
                clipped=command.clipped or clipped,
            )
            metrics.max_external_force_n = max(
                metrics.max_external_force_n, float(np.linalg.norm(force))
            )
        elif command.mode is InteractionMode.TENDON_MODAL:
            acceleration, clipped = clamp_norm(
                command.tendon_acceleration_m_s2,
                self.config.tendon_acceleration_limit_m_s2,
            )
            command = replace(
                command,
                tendon_acceleration_m_s2=tuple(
                    float(value) for value in acceleration
                ),
                clipped=command.clipped or clipped,
            )
        elif command.mode is InteractionMode.JOINT_TORQUE:
            torque = command.joint_torque_nm
            clipped = abs(torque) > self.config.joint_torque_limit_nm
            torque = max(
                -self.config.joint_torque_limit_nm,
                min(self.config.joint_torque_limit_nm, torque),
            )
            command = replace(
                command,
                joint_torque_nm=torque,
                clipped=command.clipped or clipped,
            )
        command = self._prepare_interaction_command(command)
        return self._record_interaction_command(command, elapsed_s, metrics)

    def _prepare_interaction_command(
        self, command: InteractionCommand
    ) -> InteractionCommand:
        if self._force_applier is None:
            return command
        if command.source == "native_mouse_wrench":
            return command
        return self._force_applier.enrich(command)

    def _record_interaction_command(
        self,
        command: InteractionCommand,
        elapsed_s: float,
        metrics: Px4RunMetrics,
    ) -> InteractionCommand:
        if command.mode is InteractionMode.EXTERNAL_FORCE:
            metrics.max_external_force_n = max(
                metrics.max_external_force_n,
                float(np.linalg.norm(command.force_n)),
            )
            metrics.max_generated_torque_nm = max(
                metrics.max_generated_torque_nm,
                float(np.linalg.norm(command.generated_torque_nm)),
            )
        elif command.mode is InteractionMode.JOINT_TORQUE:
            metrics.max_joint_torque_nm = max(
                metrics.max_joint_torque_nm, abs(command.joint_torque_nm)
            )
        if command.active:
            metrics.interaction_command_count += 1
            event = {
                "start_s": elapsed_s,
                "duration_s": 1.0 / self.config.plant_step_freq_hz,
                "mode": command.mode.value,
            }
            if command.mode is InteractionMode.EXTERNAL_FORCE:
                event["force_n"] = command.force_n
                event["target_body"] = command.target_body
                event["target_site"] = command.target_site
                event["point_world_m"] = command.point_world_m
            elif command.mode is InteractionMode.TENDON_MODAL:
                event["tendon_acceleration_m_s2"] = (
                    command.tendon_acceleration_m_s2
                )
            else:
                event["diagnostic_label"] = "DIAGNOSTIC JOINT TORQUE"
                event["target_joint"] = command.target_joint
                event["target_dof_axis"] = command.target_dof_axis
                event["joint_torque_nm"] = command.joint_torque_nm
            previous = self._interaction_events[-1] if self._interaction_events else None
            same_payload = (
                previous is not None
                and {
                    key: value for key, value in previous.items()
                    if key not in ("start_s", "duration_s", "point_world_m")
                } == {
                    key: value for key, value in event.items()
                    if key not in ("start_s", "duration_s", "point_world_m")
                }
            )
            contiguous = (
                same_payload
                and abs(
                    previous["start_s"] + previous["duration_s"] - elapsed_s
                ) <= 1e-12
            )
            if contiguous:
                previous["duration_s"] += event["duration_s"]
            else:
                if self.config.interaction_mode == "replay" or (self.config.diagnostics or {}).get("trial"):
                    self._interaction_events.append(event)
        if command.released:
            metrics.interaction_release_count += 1
            metrics.last_interaction_release_s = elapsed_s
            metrics.interaction_recovery_time_s = None
        if command.clipped:
            metrics.interaction_clipped_count += 1
        self._last_interaction_command = command
        return command

    def _prepare_auto_flight(self, truth, dt_s: float) -> None:
        """Initialize EKF, establish offboard stream, arm, then spool safely."""
        control = self.flight_control
        preflight_steps = int(round(
            self.config.preflight_duration_s * self.config.control_freq_hz
        ))
        warmup_steps = int(round(
            self.config.offboard_warmup_s * self.config.control_freq_hz
        ))
        mode_requested = False
        for index in range(preflight_steps):
            sequence = self.backend.coordinator.last_sent_sequence + 1
            clock = self._clock(sequence)
            if index >= preflight_steps - warmup_steps:
                control.send_position_setpoint_enu(
                    clock.timestamp_us, self._spool_target_local_enu()
                )
            if not mode_requested and index >= preflight_steps - warmup_steps // 2:
                control.request_offboard_mode()
                mode_requested = True
            result = self.backend.step(Px4BackendInput(truth, clock, dt_s))
            if not result.advance_physics:
                raise RuntimeError(
                    f"PX4 preflight lockstep failed: {result.health.detail}"
                )
            control.poll()
        health = control.health
        if health.heartbeat_count == 0:
            raise RuntimeError("PX4 GCS channel received no heartbeat during preflight")
        if health.estimator_timeout_flags not in (None, 0):
            raise RuntimeError(
                "PX4 estimator reports timeout flags: "
                f"{health.estimator_timeout_flags}"
            )
        control.request_offboard_mode()
        control.request_arm(True)
        ramp_steps = int(round(
            self.config.armed_ramp_timeout_s * self.config.control_freq_hz
        ))
        release_rpm = (
            self.config.release_hover_rpm_fraction * self._plant_hover_rpm()
        )
        max_ramp_rpm = 0.0
        for index in range(ramp_steps):
            sequence = self.backend.coordinator.last_sent_sequence + 1
            clock = self._clock(sequence)
            control.send_position_setpoint_enu(
                clock.timestamp_us, self._spool_target_local_enu()
            )
            if index and index % self.config.control_freq_hz == 0:
                control.request_offboard_mode()
                control.request_arm(True)
            result = self.backend.step(Px4BackendInput(truth, clock, dt_s))
            if not result.advance_physics:
                raise RuntimeError(f"PX4 armed ramp failed: {result.health.detail}")
            control.poll()
            max_ramp_rpm = max(max_ramp_rpm, max(result.rotor_speed.rpm))
            if result.rotor_speed.armed and min(result.rotor_speed.rpm) >= release_rpm:
                self._last_rpm = np.asarray(result.rotor_speed.rpm)
                self._final_actuator_armed = result.rotor_speed.armed
                self.aviary.set_rotor_rpm_command(self._last_rpm, armed=result.rotor_speed.armed)
                return
        raise RuntimeError(
            "PX4 did not reach the bounded release thrust during armed ramp; "
            f"required {release_rpm:g} RPM on every rotor, observed max "
            f"{max_ramp_rpm:g} RPM"
        )

    def _land_and_disarm(self, dt_s: float) -> int:
        """Ramp the offboard altitude to a safe height, then disarm on ground."""
        control = self.flight_control
        control.request_auto_land()
        if self.config.arm_source == "controller":
            self.aviary.set_scaled_stablesoft_tension_command((0.0,) * 4)
        plant_dt_s = 1.0 / self.config.plant_step_freq_hz
        px4_stride = self.config.plant_step_freq_hz // self.config.control_freq_hz
        initial_target = np.asarray(self._target_local_enu(), dtype=float)
        landing_target = np.asarray((
            initial_target[0],
            initial_target[1],
            self.config.landing_height_enu_m
            - self.config.initial_position_enu_m[2],
        ))
        descent_steps = int(round(
            self.config.landing_descent_s * self.config.plant_step_freq_hz
        ))
        max_steps = int(round(
            self.config.landing_duration_s * self.config.plant_step_freq_hz
        ))
        disarm_requested = False
        for index in range(max_steps):
            fraction = min(1.0, (index + 1) / max(1, descent_steps))
            setpoint = initial_target + fraction * (landing_target - initial_target)
            altitude = float(self.aviary.pos[0, 2])
            vertical_speed = abs(float(self.aviary.vel[0, 2]))
            if (
                altitude <= self.config.landing_height_enu_m + 0.04
                and vertical_speed <= 0.25
            ):
                control.request_arm(False)
                disarm_requested = True
            elif disarm_requested and index % max(1, self.config.plant_step_freq_hz // 4) == 0:
                control.request_arm(False)
            if index % px4_stride == 0:
                sequence = self.backend.coordinator.last_sent_sequence + 1
                clock = self._clock(sequence)
                truth = self.truth_source.sample(self.aviary, dt_s)
                result = self.backend.step(Px4BackendInput(truth, clock, dt_s))
                if not result.advance_physics:
                    self.aviary.clear_automatic_commands()
                    raise RuntimeError(f"PX4 landing lockstep failed: {result.health.detail}")
                self._last_rpm = np.asarray(result.rotor_speed.rpm)
                self._final_actuator_armed = result.rotor_speed.armed
                self.aviary.set_rotor_rpm_command(self._last_rpm, armed=result.rotor_speed.armed)
                if not result.rotor_speed.armed:
                    self.aviary.clear_automatic_commands()
                control.poll()
            self.aviary.step()
            if self._interaction_viewer is not None:
                self._interaction_viewer.render_only()
            # Landing/cleanup does not require a live viewer.
            if (index + 1) % max(1, self.config.plant_step_freq_hz // 20) == 0:
                self._last_state = {
                    "phase": "landing",
                    "time_s": (index + 1) * plant_dt_s,
                    "position_enu_m": tuple(float(value) for value in self.aviary.pos[0]),
                    "velocity_enu_m_s": tuple(float(value) for value in self.aviary.vel[0]),
                    "setpoint_local_enu_m": tuple(float(value) for value in setpoint),
                    "rpm": tuple(float(value) for value in self._last_rpm),
                }
            if disarm_requested and not result.rotor_speed.armed:
                return index + 1
        raise RuntimeError(
            "PX4 did not reach the landing/disarm condition within "
            f"{self.config.landing_duration_s:g} s of simulation time; final "
            f"altitude={float(self.aviary.pos[0, 2]):g} m, vertical speed="
            f"{float(self.aviary.vel[0, 2]):g} m/s"
        )

    def _target_local_enu(self) -> Tuple[float, float, float]:
        """Convert the runner's absolute plant target to PX4 local-origin ENU."""
        return tuple(
            target - initial for target, initial in zip(
                self.config.target_position_enu_m,
                self.config.initial_position_enu_m,
            )
        )

    def _clock(self, sequence: int, origin_us: int = 1_000_000) -> SimulatorClock:
        """Generate drift-free integer timestamps for rational control periods."""
        timestamp = origin_us + round(
            sequence * 1_000_000 / self.config.control_freq_hz
        )
        return SimulatorClock(timestamp, sequence)

    def _plant_hover_rpm(self) -> float:
        return float(self.aviary.HOVER_RPM)

    def _arm_tip_relative(self):
        manifest = self.aviary.SCALED_STABLESOFT_ARM_MANIFEST
        if manifest is None:
            return None
        import mujoco
        tip_id = mujoco.mj_name2id(
            self.aviary.model, mujoco.mjtObj.mjOBJ_SITE,
            manifest.arm_tip_site_name,
        )
        if tip_id < 0:
            raise RuntimeError("scaled StableSoft arm tip site is missing")
        return np.asarray(self.aviary.data.site_xpos[tip_id], dtype=float) - np.asarray(
            self.aviary.pos[0], dtype=float
        )


    def _update_recovery(self, time_s, error, speed, candidate_s, metrics):
        d = self.config.diagnostics or {}
        if metrics.last_interaction_release_s is None:
            return None
        if not (error <= d.get("position_threshold_m", 0.10) and speed <= d.get("speed_threshold_m_s", 0.10)):
            return None
        if candidate_s is None: candidate_s = time_s
        if metrics.interaction_recovery_time_s is None and time_s - candidate_s >= d.get("stable_window_s", 1.0) - 1.0 / self.config.plant_step_freq_hz - 1e-12:
            metrics.interaction_recovery_time_s = candidate_s - metrics.last_interaction_release_s
        return candidate_s

    def _diagnostic_tension(self, time_s, metrics):
        """Explicit headless test intervention; rotor ownership remains PX4."""
        d = self.config.diagnostics or {}
        if d.get("trial") is None:
            return
        tension = [0.0] * 4
        if metrics.baseline_status == "passed" and d["trial"] == "pulse":
            if time_s < d["baseline_s"] + d["pulse_duration_s"] - 1e-12:
                if not self._final_actuator_armed:
                    raise RuntimeError("PX4 disarmed before diagnostic pulse")
                self._diagnostic_phase = "disturbance"
                tension[d["pulse_tendon_index"]] = d["pulse_tension_n"]
                if metrics.pulse_start_s is None: metrics.pulse_start_s = time_s
            else:
                self._diagnostic_phase = "recovery"
                if metrics.pulse_release_s is None:
                    metrics.pulse_release_s = time_s
                    metrics.last_interaction_release_s = time_s
                    metrics.interaction_recovery_time_s = None
        else:
            self._diagnostic_phase = "baseline"
        self.aviary.set_scaled_stablesoft_tension_command(tension)

    def _measure_diagnostic(self, time_s, error, speed, tip, initial_tip, metrics):
        """Reuse step measurements for peaks; save only a bounded 20 Hz trace."""
        d = self.config.diagnostics or {}
        if d.get("trial") is None: return
        phase = self._diagnostic_phase
        limits_ok = error <= d["position_threshold_m"] and speed <= d["speed_threshold_m_s"]
        if phase == "baseline":
            metrics.baseline_status = "incomplete"
            self._baseline_sum += error
            self._baseline_count += 1
            metrics.baseline_mean_position_error_m = self._baseline_sum / self._baseline_count
            metrics.baseline_max_position_error_m = max(metrics.baseline_max_position_error_m, error)
            metrics.baseline_max_speed_m_s = max(metrics.baseline_max_speed_m_s, speed)
            if time_s > d["baseline_s"] - d["stable_window_s"] + 1e-12:
                self._baseline_stable &= limits_ok and self._final_actuator_armed
            if time_s >= d["baseline_s"] - 1e-12:
                self._baseline_rpy = self.aviary.rpy[0].copy()
                metrics.baseline_status = "passed" if self._baseline_stable else "未在规定窗口内满足悬停判据"
                self._diagnostic_done = not self._baseline_stable or d["trial"] == "baseline"
        else:
            metrics.diagnostic_peak_position_error_m = max(metrics.diagnostic_peak_position_error_m, error)
            metrics.diagnostic_peak_speed_m_s = max(metrics.diagnostic_peak_speed_m_s, speed)
            metrics.diagnostic_peak_attitude_rad = max(metrics.diagnostic_peak_attitude_rad, float(np.linalg.norm((self.aviary.rpy[0] - self._baseline_rpy + np.pi) % (2 * np.pi) - np.pi)))
            metrics.diagnostic_max_rotor_saturation = max(metrics.diagnostic_max_rotor_saturation, float(np.count_nonzero(self._last_rpm >= self.aviary.MAX_RPM * (1 - 1e-9)) / 4))
        bindings = self.aviary.SCALED_STABLESOFT_TENDON_BINDINGS
        tensions = tuple(float(self.aviary.data.actuator_force[b.output_address]) for b in bindings)
        if phase == "disturbance": metrics.actual_pulse_max_tension_n = max(metrics.actual_pulse_max_tension_n, max(tensions))
        metrics.diagnostic_final_position_error_m = error
        metrics.diagnostic_final_speed_m_s = speed
        displacement = None if tip is None else float(np.linalg.norm(tip - initial_tip))
        metrics.final_arm_tip_displacement_m = displacement
        metrics.final_tendon_velocity_m_s = tuple(float(self.aviary.data.ten_velocity[b.tendon_id]) for b in bindings)
        if phase == "recovery":
            metrics.recovery_status = "incomplete"
            if time_s >= metrics.pulse_release_s + d["recovery_s"] - 1e-12:
                metrics.recovery_status = "passed" if metrics.interaction_recovery_time_s is not None else "未在规定恢复窗口内达标"
                self._diagnostic_done = True
        stride = max(1, round(self.config.plant_step_freq_hz / d["sample_hz"]))
        if metrics.physics_steps % stride == 0 or self._diagnostic_done:
            health = self.flight_control.health if self.flight_control else None
            self._diagnostic_samples.append({
                "time_s": time_s, "phase": phase, "armed": self._final_actuator_armed,
                "position_enu_m": self.aviary.pos[0].tolist(), "velocity_enu_m_s": self.aviary.vel[0].tolist(),
                "rpy_rad": self.aviary.rpy[0].tolist(), "target_enu_m": self.config.target_position_enu_m,
                "position_error_m": error, "rpm": self._last_rpm.tolist(),
                "rotor_commands_n": self.aviary.data.ctrl[self.aviary.ROTOR_CONTROL_ADDRESSES[0]].tolist(),
                "tendon_tensions_n": tensions, "tendon_commands_n": [float(self.aviary.data.ctrl[b.control_address]) for b in bindings],
                "tendon_lengths_m": [float(self.aviary.data.ten_length[b.tendon_id]) for b in bindings],
                "tendon_velocities_m_s": metrics.final_tendon_velocity_m_s, "arm_tip_displacement_m": displacement,
                "pulse_active": phase == "disturbance", "pulse_released": metrics.pulse_release_s is not None,
                "px4_local_position_ned_m": health.local_position_ned_m if health else None,
                "px4_custom_mode": health.custom_mode if health else None,
                "px4_system_status": health.system_status if health else None,
                "estimator_health_flags": health.estimator_health_flags if health else None,
                "estimator_timeout_flags": health.estimator_timeout_flags if health else None,
            })

    def _setup_tendon_controller(self) -> None:
        self._tendon_controller = None
        if self.config.arm_source != "controller":
            return
        from multi_drone_mujoco.control.scaled_stablesoft_bending_control import (
            initialize_scaled_stablesoft_feedback,
        )
        self._tendon_controller, self._tendon_ids = initialize_scaled_stablesoft_feedback(self.aviary)

    def _update_tendon_controller(
        self,
        elapsed_s: float,
        metrics: Px4RunMetrics,
        interaction: Optional[InteractionCommand] = None,
    ) -> None:
        if self._tendon_controller is None:
            return
        interactive_tendon = (
            interaction is not None
            and interaction.mode is InteractionMode.TENDON_MODAL
            and interaction.active
        )
        if not self._final_actuator_armed:
            self.aviary.set_scaled_stablesoft_tension_command((0.0,) * 4)
            return
        from multi_drone_mujoco.control.scaled_stablesoft_bending_control import (
            update_scaled_stablesoft_feedback,
        )
        command = interaction.tendon_acceleration_m_s2 if interactive_tendon else self.config.arm_acceleration_m_s2
        tension = update_scaled_stablesoft_feedback(
            self.aviary, self._tendon_controller, self._tendon_ids, command
        )
        self._tendon_updates += 1
        maximum = max(tension)
        metrics.max_tendon_tension_n = max(metrics.max_tendon_tension_n, maximum)
        if maximum >= self.config.max_tension_n * (1.0 - 1e-12):
            self._tendon_saturated_updates += 1
        state = self._tendon_controller.state
        error = np.asarray(state.reference.bending_contraction_reference_m) - np.asarray(
            state.bending_contraction_actual_m
        )
        metrics.max_tendon_modal_error_m = max(
            metrics.max_tendon_modal_error_m, float(np.linalg.norm(error))
        )


    def _spool_target_local_enu(self) -> Tuple[float, float, float]:
        """Temporary demand used only while the MuJoCo plant is held fixed."""
        target = list(self._target_local_enu())
        target[2] += self.config.spool_target_height_offset_m
        return tuple(target)

    def _flight_target_local_enu(self, elapsed_s: float) -> Tuple[float, float, float]:
        """Blend out the held-plant spool demand without a thrust discontinuity."""
        fraction = min(1.0, max(0.0, elapsed_s / self.config.release_transition_s))
        target = list(self._target_local_enu())
        target[2] += (1.0 - fraction) * self.config.spool_target_height_offset_m
        return tuple(target)

    def _write_log(self, metrics: Px4RunMetrics) -> Path:
        log_dir = self.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = log_dir / f"px4-run-{timestamp}.json"
        metrics.log_path = str(path)
        payload: Dict[str, Any] = {
            "schema_version": 5,
            "started_utc": self._started_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "project_git_commit": self._git_value(Path.cwd()),
            "px4_git_commit": (
                self._git_value(self.config.autopilot_dir)
                if self.config.autopilot_dir is not None else None
            ),
            "expected_px4_git_commit": PX4_COMPATIBLE_COMMIT,
            "python_version": platform.python_version(),
            "mujoco_version": self._package_version("mujoco"),
            "pymavlink_version": self._package_version("pymavlink"),
            "runner_config": self._serializable_config(),
            "events": list(self._interaction_events),
            "effective_config": self.config.effective_config,
            "diagnostic_samples": list(getattr(self, "_diagnostic_samples", ())),
            "command_line": tuple(sys.argv),
            "parameter_file_sha256": self._file_sha256(
                self.config.parameter_file
            ),
            "interaction_script_sha256": self._file_sha256(
                self.config.interaction_script
            ),
            "model_xml_sha256": hashlib.sha256(
                self.aviary._generated_xml.encode("utf-8")
            ).hexdigest(),
            "px4_launch_command": (
                self.launcher.command_display if self.launcher is not None else None
            ),
            "px4_launch_environment": (
                dict(self.launcher.config.environment)
                if self.launcher is not None else None
            ),
            "airframe": 10016,
            "plant_calibration": asdict(self._calibration),
            "metrics": asdict(metrics),
            "last_state": self._last_state,
            "final_rpm": tuple(float(value) for value in self._last_rpm),
            "final_tendon_tension_n": tuple(
                float(self.aviary.data.ctrl[b.control_address])
                for b in self.aviary.SCALED_STABLESOFT_TENDON_BINDINGS
            ),
            "final_actuator_armed": self._final_actuator_armed,
            "px4_log_tail": (
                self.launcher.log_text if self.launcher is not None else None
            ),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def _serializable_config(self) -> Dict[str, Any]:
        values = asdict(self.config)
        for name in (
            "autopilot_dir", "parameter_file", "log_dir", "interaction_script"
        ):
            if values[name] is not None:
                values[name] = str(values[name])
        values["mode"] = self.config.mode.value
        return values

    @staticmethod
    def _git_value(directory: Path) -> Optional[str]:
        try:
            result = subprocess.run(
                ("git", "-C", str(directory), "rev-parse", "HEAD"),
                check=True, capture_output=True, text=True, timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    @staticmethod
    def _file_sha256(path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _package_version(name: str) -> Optional[str]:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
