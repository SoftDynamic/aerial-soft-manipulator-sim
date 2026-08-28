"""External modal controller source for the unchanged active-hover runner."""

import numpy as np

from multi_drone_mujoco.control.bounded_tendon_allocator import (
    AllocationNumericalError,
)
from multi_drone_mujoco.control.scaled_stablesoft_bending_control import (
    make_scaled_stablesoft_bending_controller,
)
from multi_drone_mujoco.control.tendon_compatible_acceleration_controller import (
    TendonCompatibleAccelerationController,
)


class ModalAccelerationHoverCommandSource:
    """Own one controller and expose its four-tension command each 240 Hz tick."""

    def __init__(
        self,
        acceleration_commands_m_s2,
        *,
        enabled=True,
        disable_at_tick=None,
        enable_at_tick=None,
        reset_at_tick=None,
        fault_at_tick=None,
        clear_fault_at_tick=None,
        controller_config=None,
    ):
        commands = np.asarray(acceleration_commands_m_s2, dtype=float)
        if commands.ndim != 2 or commands.shape[1] != 2:
            raise ValueError("acceleration_commands_m_s2 must have shape (N, 2)")
        if not np.all(np.isfinite(commands)):
            raise ValueError("acceleration_commands_m_s2 must be finite")
        self.commands = commands.copy()
        self.enabled = bool(enabled)
        self.disable_at_tick = disable_at_tick
        self.enable_at_tick = enable_at_tick
        self.reset_at_tick = reset_at_tick
        self.fault_at_tick = fault_at_tick
        self.clear_fault_at_tick = clear_fault_at_tick
        self.controller_config = controller_config
        self.controller = None
        self._last_acceleration = np.zeros(2)

    def reset(self, tendon_names, tendon_length_m, tendon_velocity_m_s):
        self.controller = (
            make_scaled_stablesoft_bending_controller(tendon_names)
            if self.controller_config is None
            else TendonCompatibleAccelerationController(
                self.controller_config, tendon_names
            )
        )
        self.controller.reset(tendon_length_m, tendon_velocity_m_s)
        if self.enabled:
            self.controller.enable()

    def update(self, tick, tendon_length_m, tendon_velocity_m_s):
        if self.controller is None:
            raise RuntimeError("command source must be reset before update")
        self._last_acceleration = (
            self.commands[tick].copy()
            if tick < len(self.commands)
            else np.zeros(2)
        )
        if tick == self.reset_at_tick:
            self.controller.reset(tendon_length_m, tendon_velocity_m_s)
        if tick == self.disable_at_tick:
            self.controller.disable()
        if tick == self.enable_at_tick:
            self.controller.enable()
        if tick == self.clear_fault_at_tick:
            self.controller.clear_fault()
        if tick != self.fault_at_tick:
            return self.controller.update_bending_acceleration(
                self._last_acceleration,
                tendon_length_m,
                tendon_velocity_m_s,
            )

        original_allocate = self.controller.allocator.allocate

        def injected_failure(_):
            raise AllocationNumericalError("injected modal-hover fault")

        self.controller.allocator.allocate = injected_failure
        try:
            return self.controller.update_bending_acceleration(
                self._last_acceleration,
                tendon_length_m,
                tendon_velocity_m_s,
            )
        finally:
            self.controller.allocator.allocate = original_allocate

    def diagnostics(self):
        state = self.controller.state
        allocation = state.allocation
        feedforward = state.static_feedforward
        return {
            "modal_acceleration_command_m_s2": self._last_acceleration.copy(),
            "modal_reference_m": np.asarray(
                state.reference.bending_contraction_reference_m
            ),
            "modal_velocity_reference_m_s": np.asarray(
                state.reference.bending_velocity_reference_m_s
            ),
            "modal_actual_m": np.asarray(state.bending_contraction_actual_m),
            "modal_velocity_actual_m_s": np.asarray(
                state.bending_velocity_actual_m_s
            ),
            "modal_pd_effort_n": np.asarray(
                state.modal_pd.desired_bending_effort_n
            ),
            "modal_feedforward_effort_n": (
                np.asarray(feedforward.static_bending_effort_n)
                if feedforward is not None else np.zeros(2)
            ),
            "modal_total_effort_n": np.asarray(
                state.desired_total_bending_effort_n
            ),
            "allocator_residual_n": float(
                allocation.primary_residual_n if allocation is not None else 0.0
            ),
            "allocator_saturated": bool(
                allocation is not None
                and allocation.status == "saturated_best_effort"
            ),
            "controller_enabled": bool(state.enabled),
            "controller_fault_latched": bool(state.fault_latched),
            "feedforward_out_of_envelope": bool(
                state.feedforward_out_of_envelope
            ),
        }


def pulse_hold_commands(
    direction, acceleration_m_s2, pulse_ticks, total_ticks=721
):
    """Return accelerate/decelerate/hold commands on the exact control grid."""
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (2,) or not np.isclose(
        np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-15
    ):
        raise ValueError("direction must be a unit two-vector")
    if not isinstance(pulse_ticks, int) or pulse_ticks <= 0:
        raise ValueError("pulse_ticks must be a positive integer")
    if total_ticks <= 2 * pulse_ticks:
        raise ValueError("total_ticks must include a hold")
    commands = np.zeros((total_ticks, 2))
    commands[:pulse_ticks] = acceleration_m_s2 * direction
    commands[pulse_ticks:2 * pulse_ticks] = -acceleration_m_s2 * direction
    return commands
