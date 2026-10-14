"""Frozen PX4 adapter calibration for the repository's rigid 27 g CF2X."""

from dataclasses import dataclass
import math
from typing import Tuple

from multi_drone_mujoco.backends.actuation import (
    NormalizedToPwmConfig,
    PwmToRotorSpeedConfig,
)
from multi_drone_mujoco.models.parameters import DRONE_PARAMS, ResolvedUavParameters
from multi_drone_mujoco.utils.enums import DroneModel


# Simulator rotor index -> zero-based PX4 Motor channel.
# PX4 10016 order is FR, RL, FL, RR in FRD; MuJoCo order is FL, RL, RR, FR
# in FLU.  This permutation is derived from positions, not display names.
CF2X_PX4_MOTOR_ORDER: Tuple[int, int, int, int] = (2, 1, 3, 0)
CF2X_ROTOR_DIRECTIONS: Tuple[int, int, int, int] = (-1, 1, -1, 1)


@dataclass(frozen=True)
class Cf2xPx4Calibration:
    mass_kg: float
    inertia_kg_m2: Tuple[float, float, float]
    arm_length_m: float
    kf_n_per_rpm2: float
    km_nm_per_rpm2: float
    max_rpm: float
    hover_rpm: float
    max_thrust_n: float
    pwm_disarmed_us: float
    pwm_min_us: float
    pwm_max_us: float
    rpm_idle: float
    hover_pwm_us: float
    hover_normalized: float
    px4_hover_thrust: float
    px4_moment_ratio_m: float


def cf2x_27g_calibration(gravity_m_s2: float = 9.81, *, parameters=None, payload_mass_kg=0.0) -> Cf2xPx4Calibration:
    """Derive every actuator endpoint from the frozen MuJoCo plant."""
    gravity = float(gravity_m_s2)
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity_m_s2 must be finite and positive")
    params = parameters or ResolvedUavParameters.resolve()
    defaults = DRONE_PARAMS[DroneModel.CF2X]
    for name in ("mass", "kf", "km", "thrust2weight_ratio"):
        if params[name] != defaults[name]:
            raise ValueError(f"PX4 CF2X motor calibration does not support modified {name}; use manual UAV mode")
    if not math.isfinite(payload_mass_kg) or payload_mass_kg < 0:
        raise ValueError("payload_mass_kg must be finite and non-negative")
    mass = params["mass"] + payload_mass_kg
    kf = params["kf"]
    km = params["km"]
    hover_rpm = math.sqrt(mass * gravity / (4.0 * kf))
    max_rpm = math.sqrt(
        params["thrust2weight_ratio"] * params["mass"] * gravity / (4.0 * kf)
    )
    rpm_idle = 0.0
    pwm_min = 1000.0
    pwm_max = 2000.0
    hover_normalized = (hover_rpm - rpm_idle) / (max_rpm - rpm_idle)
    px4_hover_thrust = mass / (params["thrust2weight_ratio"] * params["mass"])
    if px4_hover_thrust >= 1:
        raise ValueError("PX4 payload exceeds available hover thrust")
    hover_pwm = pwm_min + hover_normalized * (pwm_max - pwm_min)
    return Cf2xPx4Calibration(
        mass_kg=mass,
        inertia_kg_m2=(params["ixx"], params["iyy"], params["izz"]),
        arm_length_m=params["arm_length"],
        kf_n_per_rpm2=kf,
        km_nm_per_rpm2=km,
        max_rpm=max_rpm,
        hover_rpm=hover_rpm,
        max_thrust_n=4.0 * kf * max_rpm**2,
        pwm_disarmed_us=900.0,
        pwm_min_us=pwm_min,
        pwm_max_us=pwm_max,
        rpm_idle=rpm_idle,
        hover_pwm_us=hover_pwm,
        hover_normalized=hover_normalized,
        px4_hover_thrust=px4_hover_thrust,
        px4_moment_ratio_m=km / kf,
    )


def cf2x_normalized_to_pwm_config(calibration=None) -> NormalizedToPwmConfig:
    item = calibration or cf2x_27g_calibration()
    return NormalizedToPwmConfig(
        pwm_disarmed_us=item.pwm_disarmed_us,
        pwm_min_us=item.pwm_min_us,
        pwm_max_us=item.pwm_max_us,
        normalized_min=0.0,
        normalized_max=1.0,
        motor_order=CF2X_PX4_MOTOR_ORDER,
        motor_directions=CF2X_ROTOR_DIRECTIONS,
    )


def cf2x_pwm_to_rpm_config(
    motor_time_constant_s: float = 0.0, calibration=None,
) -> PwmToRotorSpeedConfig:
    item = calibration or cf2x_27g_calibration()
    return PwmToRotorSpeedConfig(
        pwm_disarmed_us=item.pwm_disarmed_us,
        pwm_min_us=item.pwm_min_us,
        pwm_max_us=item.pwm_max_us,
        rpm_disarmed=0.0,
        rpm_idle=item.rpm_idle,
        rpm_max=item.max_rpm,
        motor_time_constant_s=motor_time_constant_s,
        motor_directions=CF2X_ROTOR_DIRECTIONS,
    )


def validate_cf2x_px4_parameters(values, calibration):
    """Check PX4 FRD motor geometry against MuJoCo FLU order and RPM model."""
    a = calibration.arm_length_m / math.sqrt(2)
    expected = {"CA_ROTOR_COUNT": 4, "MPC_THR_HOVER": calibration.px4_hover_thrust,
                "THR_MDL_FAC": 1.0}
    # PX4 FR, RL, FL, RR. KM includes FRD/FLU moment conversion.
    for i, (x, y, sign) in enumerate(((a, a, -1), (-a, -a, -1),
                                      (a, -a, 1), (-a, a, 1))):
        expected.update({f"CA_ROTOR{i}_PX": x, f"CA_ROTOR{i}_PY": y,
                         f"CA_ROTOR{i}_KM": sign * calibration.px4_moment_ratio_m})
    for i in range(4):
        for suffix, default in (("PZ", 0.0), ("AX", 0.0), ("AY", 0.0), ("AZ", -1.0)):
            name = f"CA_ROTOR{i}_{suffix}"
            if name in values: expected[name] = default
    for name, value in expected.items():
        # Verified file uses an approximate diagonal rotor coordinate (1.011 um
        # residual at L=0.0397 m). Preserve it, with a 1.1 um geometry allowance.
        tolerance = 1.1e-6 if name.endswith(("_PX", "_PY")) else 1e-9
        if name not in values or not math.isclose(float(values[name]), value, rel_tol=2e-6, abs_tol=tolerance):
            raise ValueError(f"PX4 parameter {name} incompatible with resolved physical model: expected {value:g}, got {values.get(name)}")
