"""Frozen fixed-root tuning for StableSoft tendon-modal bending control."""

from multi_drone_mujoco.control.bounded_tendon_allocator import (
    BoundedTendonAllocatorConfig,
)
from multi_drone_mujoco.control.tendon_bending_reference import (
    BendingReferenceConfig,
)
from multi_drone_mujoco.control.tendon_compatible_acceleration_controller import (
    TendonCompatibleAccelerationController,
    TendonCompatibleAccelerationControllerConfig,
)
from multi_drone_mujoco.control.tendon_modal_pd import BendingModalPDConfig
from multi_drone_mujoco.control.tendon_modal_static_feedforward import (
    BendingModalStaticFeedforwardConfig,
)


CONTROL_FREQUENCY_HZ = 240
MAX_ACCELERATION_M_S2 = 0.04
MAX_VELOCITY_M_S = 0.002
MAX_REFERENCE_L1_M = 0.0003
MAX_CALIBRATED_ACTUAL_L1_M = 0.0004
MAX_TENSION_N = 0.01
KP_N_PER_M = 300.0
KD_N_S_PER_M = 0.02
STATIC_STIFFNESS_N_PER_M = (52.58, 52.58)
ZERO_PRELOAD_N = (0.0, 0.0, 0.0, 0.0)


def scaled_stablesoft_bending_controller_config(control_frequency_hz=CONTROL_FREQUENCY_HZ):
    """Return the validated 240 Hz, zero-preload controller configuration."""
    if control_frequency_hz != CONTROL_FREQUENCY_HZ:
        raise ValueError(
            f"validated control_frequency_hz must be {CONTROL_FREQUENCY_HZ}"
        )
    return TendonCompatibleAccelerationControllerConfig(
        reference=BendingReferenceConfig(
            control_timestep_s=1.0 / CONTROL_FREQUENCY_HZ,
            max_acceleration_m_s2=MAX_ACCELERATION_M_S2,
            max_velocity_m_s=MAX_VELOCITY_M_S,
            max_contraction_m=MAX_REFERENCE_L1_M,
        ),
        modal_pd=BendingModalPDConfig(
            kp_n_per_m=KP_N_PER_M,
            kd_n_s_per_m=KD_N_S_PER_M,
        ),
        allocator=BoundedTendonAllocatorConfig(
            max_tension_n=MAX_TENSION_N,
            bias_tension_n=ZERO_PRELOAD_N,
        ),
        static_feedforward=BendingModalStaticFeedforwardConfig(
            stiffness_n_per_m=STATIC_STIFFNESS_N_PER_M,
            max_reference_l1_m=MAX_REFERENCE_L1_M,
            max_actual_l1_m=MAX_CALIBRATED_ACTUAL_L1_M,
        ),
    )


def make_scaled_stablesoft_bending_controller(
    tendon_names,
    control_frequency_hz=CONTROL_FREQUENCY_HZ,
):
    """Create a controller using manifest-resolved tendon names."""
    return TendonCompatibleAccelerationController(
        scaled_stablesoft_bending_controller_config(control_frequency_hz),
        tendon_names,
    )
