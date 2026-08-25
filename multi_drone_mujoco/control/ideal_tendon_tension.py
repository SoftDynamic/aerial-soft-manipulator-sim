"""Runtime binding and command writes for ideal tendon tension actuators."""

from dataclasses import dataclass
from typing import Tuple

import mujoco

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ArmMjcfManifest,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
    validate_ideal_tendon_tension_command,
)


@dataclass(frozen=True)
class IdealTendonActuatorBinding:
    """Manifest-ordered runtime addresses for one tendon actuator."""

    tendon_name: str
    actuator_name: str
    tendon_id: int
    actuator_id: int
    control_address: int
    output_address: int


def _named_id(model, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(object_id)


def resolve_ideal_tendon_actuator_bindings(
    model,
    manifest: ArmMjcfManifest,
) -> Tuple[IdealTendonActuatorBinding, ...]:
    """Resolve one-to-one tendon actuators in manifest command order."""
    tendon_names = tuple(manifest.tendon_names)
    actuator_names = tuple(manifest.tendon_actuator_names)
    if len(tendon_names) != len(actuator_names):
        raise ValueError(
            "Tendon and actuator manifest lengths must match"
        )

    bindings = []
    for tendon_name, actuator_name in zip(tendon_names, actuator_names):
        tendon_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tendon_name,
        )
        actuator_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if (
            model.actuator_trntype[actuator_id]
            != mujoco.mjtTrn.mjTRN_TENDON
        ):
            raise ValueError(
                f"Actuator is not tendon-transmitted: {actuator_name}"
            )
        if int(model.actuator_trnid[actuator_id, 0]) != tendon_id:
            raise ValueError(
                f"Actuator tendon transmission mismatch: {actuator_name}"
            )
        if int(model.actuator_ctrlnum[actuator_id]) != 1:
            raise ValueError(
                f"Actuator must have one control: {actuator_name}"
            )
        if int(model.actuator_outnum[actuator_id]) != 1:
            raise ValueError(
                f"Actuator must have one output: {actuator_name}"
            )
        bindings.append(
            IdealTendonActuatorBinding(
                tendon_name=tendon_name,
                actuator_name=actuator_name,
                tendon_id=tendon_id,
                actuator_id=actuator_id,
                control_address=int(
                    model.actuator_ctrladr[actuator_id]
                ),
                output_address=int(
                    model.actuator_outadr[actuator_id]
                ),
            )
        )
    return tuple(bindings)


def write_ideal_tendon_tension_command(
    data,
    bindings,
    tendon_tension_cmd_n,
    spec: IdealTendonTensionSpec,
) -> Tuple[float, ...]:
    """Validate newtons and write through resolved control addresses."""
    bindings = tuple(bindings)
    values = validate_ideal_tendon_tension_command(
        tendon_tension_cmd_n,
        spec,
        expected_size=len(bindings),
    )
    for value, binding in zip(values, bindings):
        data.ctrl[binding.control_address] = value
    return values
