"""Optional controller backends and typed simulator boundary contracts.

This package has no PX4 or MAVLink import-time dependency.
"""

from multi_drone_mujoco.backends.contracts import (
    ControllerBackend,
    ControllerHealth,
    ControllerStepResult,
    ControllerState,
    NormalizedMotorCommand,
    PwmMotorCommand,
    RotorSpeedCommand,
    SimulatorClock,
)
from multi_drone_mujoco.backends.watchdog import CommandWatchdog

__all__ = [
    "CommandWatchdog",
    "ControllerBackend",
    "ControllerHealth",
    "ControllerStepResult",
    "ControllerState",
    "NormalizedMotorCommand",
    "PwmMotorCommand",
    "RotorSpeedCommand",
    "SimulatorClock",
]
