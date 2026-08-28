"""Compatible two-mode acceleration control with bounded tendon allocation.

This controller is additive: the legacy four-channel controller and every
existing actuation API remain unchanged.  Its public two-vector acceleration
is ``(q_x_ddot, q_y_ddot)`` in tendon-modal m/s^2, not Cartesian, curvature,
or spool acceleration.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from multi_drone_mujoco.control.bounded_tendon_allocator import (
    AllocationNumericalError,
    BoundedTendonAllocation,
    BoundedTendonAllocator,
    BoundedTendonAllocatorConfig,
)
from multi_drone_mujoco.control.tendon_bending_reference import (
    BendingModalReferenceIntegrator,
    BendingReferenceConfig,
    BendingReferenceState,
)
from multi_drone_mujoco.control.tendon_modal_coordinates import (
    BendingProjection,
    TendonModalCoordinates,
)
from multi_drone_mujoco.control.tendon_modal_pd import (
    BendingModalPD,
    BendingModalPDConfig,
    BendingModalPDState,
)


_MAX_TENSION_N = 0.01


def _vector(name, values, size):
    try:
        result = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite values") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _nonnegative_lengths(values):
    result = _vector("tendon_length_m", values, 4)
    if np.any(result < 0.0):
        raise ValueError("tendon_length_m values must be non-negative")
    return result


@dataclass(frozen=True)
class TendonCompatibleAccelerationControllerConfig:
    reference: BendingReferenceConfig
    modal_pd: BendingModalPDConfig
    allocator: BoundedTendonAllocatorConfig

    def __post_init__(self):
        if not isinstance(self.reference, BendingReferenceConfig):
            raise TypeError("reference must be BendingReferenceConfig")
        if not isinstance(self.modal_pd, BendingModalPDConfig):
            raise TypeError("modal_pd must be BendingModalPDConfig")
        if not isinstance(self.allocator, BoundedTendonAllocatorConfig):
            raise TypeError("allocator must be BoundedTendonAllocatorConfig")
        if self.allocator.max_tension_n != _MAX_TENSION_N:
            raise ValueError("allocator max_tension_n must remain exactly 0.01 N")
        if self.allocator.bias_tension_n != (0.0, 0.0, 0.0, 0.0):
            raise ValueError("Gate C2-C2 v1 fixed preload must remain zero")


@dataclass(frozen=True)
class TendonCompatibleAccelerationControllerState:
    control_tick: int
    status: str
    enabled: bool
    fault_latched: bool
    fault_reason: str | None
    fault_tick: int | None
    neutral_tendon_length_m: tuple
    tendon_length_m: tuple
    tendon_velocity_m_s: tuple
    contraction_actual_m: tuple
    contraction_velocity_actual_m_s: tuple
    bending_contraction_actual_m: tuple
    bending_velocity_actual_m_s: tuple
    requested_tendon_acceleration_m_s2: tuple
    projected_tendon_acceleration_m_s2: tuple
    input_projection_residual_m_s2: tuple
    reference: BendingReferenceState
    modal_pd: BendingModalPDState
    allocation: BoundedTendonAllocation | None
    tension_command_n: tuple
    reference_common_residuals: tuple
    reference_twist_candidate_residuals: tuple


class CompatibleControllerNumericalError(RuntimeError):
    """An internal numerical invariant failed and requires explicit clear."""


class TendonCompatibleAccelerationController:
    """Compose reference integration, explicit modal PD, and allocation.

    The object starts disabled.  ``reset`` clears a latched fault and replaces
    the neutral tendon lengths, but does not implicitly enable control.
    Disabled and fault-latched updates always return four exact zeros.
    """

    def __init__(self, config, tendon_names):
        if not isinstance(config, TendonCompatibleAccelerationControllerConfig):
            raise TypeError(
                "config must be TendonCompatibleAccelerationControllerConfig"
            )
        self.config = config
        self.coordinates = TendonModalCoordinates.from_tendon_names(tendon_names)
        self.reference = BendingModalReferenceIntegrator(
            config.reference, self.coordinates
        )
        self.modal_pd = BendingModalPD(config.modal_pd)
        self.allocator = BoundedTendonAllocator(config.allocator, self.coordinates)
        self._state = None
        self._enabled = False
        self._fault_reason = None
        self._fault_tick = None
        self._control_tick = 0

    @property
    def is_reset(self):
        return self._state is not None

    @property
    def state(self):
        if self._state is None:
            raise RuntimeError("controller must be reset before reading state")
        return self._state

    @property
    def fault_latched(self):
        return self._fault_reason is not None

    def enable(self):
        if self._state is None:
            raise RuntimeError("controller must be reset before enable")
        self._enabled = True

    def disable(self):
        self._enabled = False

    def clear_fault(self):
        """Explicitly clear a latch while preserving the compatible reference."""
        if self._state is None:
            raise RuntimeError("controller must be reset before clear_fault")
        self._fault_reason = None
        self._fault_tick = None

    def reset(self, tendon_length_m, tendon_velocity_m_s):
        lengths = _nonnegative_lengths(tendon_length_m)
        velocities = _vector("tendon_velocity_m_s", tendon_velocity_m_s, 4)
        self._enabled = False
        self._fault_reason = None
        self._fault_tick = None
        self._control_tick = 0
        reference = self.reference.reset(velocities)
        modal_pd = self._evaluate_pd(lengths, velocities, lengths, reference)
        allocation = self.allocator.allocate((0.0, 0.0))
        zeros = (0.0,) * 4
        self._state = self._make_state(
            status="disabled",
            lengths=lengths,
            velocities=velocities,
            neutral=lengths,
            requested=zeros,
            projected=zeros,
            residual=zeros,
            reference=reference,
            modal_pd=modal_pd,
            allocation=allocation,
            command=zeros,
        )
        return self._state

    def _evaluate_pd(self, lengths, velocities, neutral, reference):
        contraction = neutral - lengths
        contraction_velocity = -velocities
        actual_q = self.coordinates.tendon_to_bending(contraction)
        actual_v = self.coordinates.tendon_to_bending(contraction_velocity)
        self.modal_pd.update(
            reference.bending_contraction_reference_m,
            reference.bending_velocity_reference_m_s,
            actual_q,
            actual_v,
        )
        return self.modal_pd.state

    def _validate_reference(self, reference):
        values = (
            *reference.common_residuals,
            *reference.twist_candidate_residuals,
        )
        scale = max(
            1.0,
            float(np.linalg.norm(reference.tendon_acceleration_reference_m_s2)),
            float(np.linalg.norm(reference.tendon_velocity_reference_m_s)),
            float(np.linalg.norm(reference.tendon_contraction_reference_m)),
        )
        tolerance = 256.0 * np.finfo(float).eps * scale
        if max(abs(value) for value in values) > tolerance:
            raise CompatibleControllerNumericalError(
                "compatible reference has common/twist-candidate residual"
            )

    def _make_state(
        self, *, status, lengths, velocities, neutral, requested, projected,
        residual, reference, modal_pd, allocation, command,
    ):
        contraction = neutral - lengths
        contraction_velocity = -velocities
        return TendonCompatibleAccelerationControllerState(
            control_tick=self._control_tick,
            status=status,
            enabled=self._enabled,
            fault_latched=self.fault_latched,
            fault_reason=self._fault_reason,
            fault_tick=self._fault_tick,
            neutral_tendon_length_m=tuple(neutral),
            tendon_length_m=tuple(lengths),
            tendon_velocity_m_s=tuple(velocities),
            contraction_actual_m=tuple(contraction),
            contraction_velocity_actual_m_s=tuple(contraction_velocity),
            bending_contraction_actual_m=tuple(
                self.coordinates.tendon_to_bending(contraction)
            ),
            bending_velocity_actual_m_s=tuple(
                self.coordinates.tendon_to_bending(contraction_velocity)
            ),
            requested_tendon_acceleration_m_s2=tuple(requested),
            projected_tendon_acceleration_m_s2=tuple(projected),
            input_projection_residual_m_s2=tuple(residual),
            reference=reference,
            modal_pd=modal_pd,
            allocation=allocation,
            tension_command_n=tuple(command),
            reference_common_residuals=reference.common_residuals,
            reference_twist_candidate_residuals=reference.twist_candidate_residuals,
        )

    def _update(self, bending_acceleration, input_projection, lengths, velocities):
        if self._state is None:
            raise RuntimeError("controller must be reset before update")
        lengths = _nonnegative_lengths(lengths)
        velocities = _vector("tendon_velocity_m_s", velocities, 4)
        requested = input_projection.requested
        projected = input_projection.projected
        residual = input_projection.residual
        neutral = np.asarray(self.state.neutral_tendon_length_m)

        if not self._enabled or self.fault_latched:
            self._control_tick += 1
            status = "fault_latched" if self.fault_latched else "disabled"
            modal_pd = self._evaluate_pd(
                lengths, velocities, neutral, self.reference.state
            )
            self._state = self._make_state(
                status=status, lengths=lengths, velocities=velocities,
                neutral=neutral, requested=requested, projected=projected,
                residual=residual, reference=self.reference.state,
                modal_pd=modal_pd, allocation=None, command=(0.0,) * 4,
            )
            return self._state.tension_command_n

        self._control_tick += 1
        try:
            reference = self.reference.update_bending(bending_acceleration)
            self._validate_reference(reference)
            modal_pd = self._evaluate_pd(lengths, velocities, neutral, reference)
            allocation = self.allocator.allocate(
                modal_pd.desired_bending_effort_n
            )
            command = allocation.tension_n
            status = allocation.status
        except (
            AllocationNumericalError,
            CompatibleControllerNumericalError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            self._fault_reason = f"{type(exc).__name__}: {exc}"
            self._fault_tick = self._control_tick
            reference = self.reference.state
            modal_pd = self.modal_pd.state
            allocation = None
            command = (0.0,) * 4
            status = "fault_latched"

        self._state = self._make_state(
            status=status, lengths=lengths, velocities=velocities,
            neutral=neutral, requested=requested, projected=projected,
            residual=residual, reference=reference, modal_pd=modal_pd,
            allocation=allocation, command=command,
        )
        return self._state.tension_command_n

    def update_bending_acceleration(
        self, bending_acceleration_m_s2, tendon_length_m, tendon_velocity_m_s
    ):
        bending = _vector(
            "bending_acceleration_m_s2", bending_acceleration_m_s2, 2
        )
        tendon = self.coordinates.bending_to_tendon(bending)
        projection = self.coordinates.project_to_bending(tendon)
        return self._update(bending, projection, tendon_length_m, tendon_velocity_m_s)

    def update_tendon_acceleration(
        self, tendon_acceleration_m_s2, tendon_length_m, tendon_velocity_m_s
    ):
        projection = self.coordinates.project_to_bending(tendon_acceleration_m_s2)
        return self._update(
            projection.bending_coordinates,
            projection,
            tendon_length_m,
            tendon_velocity_m_s,
        )
