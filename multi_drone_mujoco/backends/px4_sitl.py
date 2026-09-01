"""Composition root for the optional PX4 SITL controller backend."""

from dataclasses import dataclass
import math

from multi_drone_mujoco.backends.actuation import (
    PwmToRotorSpeedModel,
    Px4NormalizedOutputToPwm,
)
from multi_drone_mujoco.backends.contracts import (
    ControllerHealth,
    ControllerState,
    ControllerStepResult,
    NormalizedMotorCommand,
    SimulatorClock,
)
from multi_drone_mujoco.backends.lockstep import (
    LockstepConfig,
    LockstepCoordinator,
    LockstepStatus,
)
from multi_drone_mujoco.backends.sensors import (
    IdealSensorAdapter,
    MujocoTruthFrame,
)
from multi_drone_mujoco.backends.transport import (
    SimulatorTransport,
    TransportState,
)


@dataclass(frozen=True)
class Px4BackendInput:
    """Truth and simulation interval consumed by one PX4 backend cycle."""

    truth: MujocoTruthFrame
    clock: SimulatorClock
    dt_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.truth, MujocoTruthFrame):
            raise ValueError("truth must be MujocoTruthFrame")
        if not isinstance(self.clock, SimulatorClock):
            raise ValueError("clock must be SimulatorClock")
        dt_s = float(self.dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        object.__setattr__(self, "dt_s", dt_s)


class PX4SITLBackend:
    """Optional PX4 controller pipeline; owns transport, never a PX4 process."""

    def __init__(
        self,
        sensor_adapter: IdealSensorAdapter,
        transport: SimulatorTransport,
        normalized_to_pwm: Px4NormalizedOutputToPwm,
        pwm_to_rotor_speed: PwmToRotorSpeedModel,
        lockstep_config: LockstepConfig,
    ):
        self.sensor_adapter = sensor_adapter
        self.transport = transport
        self.normalized_to_pwm = normalized_to_pwm
        self.pwm_to_rotor_speed = pwm_to_rotor_speed
        self.coordinator = LockstepCoordinator(transport, lockstep_config)
        self._health = ControllerHealth(ControllerState.DISCONNECTED, 0, -1)
        self._closed = False
        self._last_normalized_command = None
        self._last_pwm_command = None

    @property
    def health(self) -> ControllerHealth:
        return self._health

    @property
    def last_normalized_command(self):
        return self._last_normalized_command

    @property
    def last_pwm_command(self):
        return self._last_pwm_command

    def reset(self, clock: SimulatorClock) -> None:
        if self._closed:
            raise RuntimeError("PX4 backend is closed")
        self.sensor_adapter.reset()
        self.pwm_to_rotor_speed.reset()
        self.coordinator.reset()
        self._last_normalized_command = None
        self._last_pwm_command = None
        if self.transport.health.state != TransportState.CONNECTED:
            self._health = ControllerHealth(
                ControllerState.CONNECTING, clock.timestamp_us, -1
            )
            self.coordinator.connect()
        self._health = ControllerHealth(
            ControllerState.READY, clock.timestamp_us, -1
        )

    def step(self, backend_input: Px4BackendInput) -> ControllerStepResult:
        if self._closed:
            raise RuntimeError("PX4 backend is closed")
        if not self.coordinator.will_exchange:
            safe_normalized = NormalizedMotorCommand(
                self.coordinator.config.safe_normalized_values,
                False,
                backend_input.clock.timestamp_us,
                backend_input.clock.sequence,
            )
            pwm = self.normalized_to_pwm.convert(safe_normalized)
            self._last_normalized_command = safe_normalized
            self._last_pwm_command = pwm
            rotor_speed = self.pwm_to_rotor_speed.convert(pwm, backend_input.dt_s)
            self._health = ControllerHealth(
                ControllerState.READY,
                backend_input.clock.timestamp_us,
                backend_input.clock.sequence,
                "simulation is paused",
            )
            return ControllerStepResult(rotor_speed, False, self._health)
        frame = self.sensor_adapter.sample(
            backend_input.truth, backend_input.clock
        )
        outcome = self.coordinator.exchange(frame)
        pwm = self.normalized_to_pwm.convert(outcome.command)
        self._last_normalized_command = outcome.command
        self._last_pwm_command = pwm
        rotor_speed = self.pwm_to_rotor_speed.convert(pwm, backend_input.dt_s)
        state = self._controller_state(outcome.status)
        self._health = ControllerHealth(
            state,
            backend_input.clock.timestamp_us,
            backend_input.clock.sequence,
            outcome.detail,
        )
        return ControllerStepResult(
            rotor_speed=rotor_speed,
            advance_physics=outcome.advance_physics,
            health=self._health,
        )

    def pause(self) -> None:
        self.coordinator.pause()

    def resume(self) -> None:
        self.coordinator.resume()

    def request_single_step(self) -> None:
        self.coordinator.request_single_step()

    def reconnect(self) -> None:
        self.coordinator.reconnect()

    def close(self) -> None:
        if not self._closed:
            self.coordinator.close()
            self._closed = True
            self._health = ControllerHealth(
                ControllerState.CLOSED,
                self._health.timestamp_us,
                self._health.last_sequence,
            )

    @staticmethod
    def _controller_state(status: LockstepStatus) -> ControllerState:
        if status == LockstepStatus.READY or status == LockstepStatus.PAUSED:
            return ControllerState.READY
        if status == LockstepStatus.TIMED_OUT:
            return ControllerState.TIMED_OUT
        if status == LockstepStatus.DISCONNECTED:
            return ControllerState.DISCONNECTED
        if status == LockstepStatus.CLOSED:
            return ControllerState.CLOSED
        return ControllerState.ERROR
