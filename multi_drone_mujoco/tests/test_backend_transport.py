"""Gate P4 tests for no-network transport behavior."""

import sys
import types

import pytest

from multi_drone_mujoco.backends.contracts import NormalizedMotorCommand
from multi_drone_mujoco.backends.sensors import SensorFrame
from multi_drone_mujoco.backends.transport import (
    FakeTransport,
    MavlinkTransportConfig,
    SimulatorMavlinkTransport,
    SimulatorTransport,
    TransportDisconnected,
    TransportState,
    TransportTimeout,
    _mavlink_int16_milli,
)


def sensor(timestamp=10, sequence=1):
    return SensorFrame(
        timestamp, sequence, (0, 0, 0), (0, 0, 0), (1, 0, 0, 0),
        (0, 0, 0), (0, 0, -9.81), (0, 0, 0), 30, 120, 100, 0,
        (0.2, 0, 0.4), True, True, True, True,
    )


def actuator(timestamp=10, sequence=2):
    return NormalizedMotorCommand((0.1, 0.2, 0.3, 0.4), True, timestamp, sequence)


def test_fake_transport_handshake_packets_sequence_and_close():
    transport = FakeTransport([actuator()])
    assert isinstance(transport, SimulatorTransport)
    transport.connect()
    transport.send_sensor_frame(sensor())
    assert transport.receive_actuator(0.01) == actuator()
    assert transport.health.state == TransportState.CONNECTED
    assert transport.health.sent_packets == transport.health.received_packets == 1
    transport.close()
    transport.close()
    with pytest.raises(TransportDisconnected):
        transport.send_sensor_frame(sensor())


def test_fake_timeout_disconnect_and_explicit_reconnect():
    transport = FakeTransport()
    with pytest.raises(TransportDisconnected):
        transport.receive_actuator(0)
    transport.connect()
    with pytest.raises(TransportTimeout, match="no fake"):
        transport.receive_actuator(0.001)
    transport.disconnect("wire lost")
    assert transport.health.detail == "wire lost"
    with pytest.raises(TransportDisconnected):
        transport.send_sensor_frame(sensor())
    transport.connect()
    transport.queue_actuator(actuator())
    assert transport.receive_actuator(0) == actuator()


def test_mavlink_network_configuration_is_validated():
    with pytest.raises(ValueError, match="host"):
        MavlinkTransportConfig("", 4560, 1, 1, 1, 0)
    with pytest.raises(ValueError, match="port"):
        MavlinkTransportConfig("remote", 0, 1, 1, 1, 0)
    config = MavlinkTransportConfig("px4.example", 4560, 2, 200, 190, 3)
    assert config.bind_host == "px4.example"
    assert config.bind_port == 4560
    transport = SimulatorMavlinkTransport(config)
    assert transport.health.state == TransportState.DISCONNECTED
    assert transport.endpoint == "tcpin:px4.example:4560"
    client = SimulatorMavlinkTransport(
        MavlinkTransportConfig("px4.example", 4560, 2, 200, 190, 3, "connect")
    )
    assert client.endpoint == "tcp:px4.example:4560"
    # Construction/import of core modules opens no socket and imports no pymavlink.
    transport.close()


def test_missing_optional_mavlink_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymavlink", None)
    config = MavlinkTransportConfig("127.0.0.1", 4560, 0.01, 200, 190, 0)
    transport = SimulatorMavlinkTransport(config)
    with pytest.raises(RuntimeError, match="optional dependency"):
        transport.connect()
    assert transport.health.state == TransportState.ERROR


def test_standard_tcp_listener_waits_for_peer_not_peer_heartbeat(monkeypatch):
    class FakeMav:
        def __init__(self):
            self.heartbeats = 0

        def heartbeat_send(self, *args):
            self.heartbeats += 1

    class FakeConnection:
        def __init__(self):
            self.port = None
            self.mav = FakeMav()
            self.recv_calls = 0

        def recv(self):
            self.recv_calls += 1
            if self.recv_calls == 2:
                self.port = object()
            return b""

        def close(self):
            pass

    connection = FakeConnection()
    mavutil = types.SimpleNamespace(
        mavlink_connection=lambda *args, **kwargs: connection,
        mavlink=types.SimpleNamespace(
            MAV_TYPE_GCS=6,
            MAV_AUTOPILOT_INVALID=8,
            MAV_STATE_ACTIVE=4,
        ),
    )
    module = types.ModuleType("pymavlink")
    module.mavutil = mavutil
    monkeypatch.setitem(sys.modules, "pymavlink", module)
    transport = SimulatorMavlinkTransport(
        MavlinkTransportConfig("127.0.0.1", 4560, 0.1, 200, 190, 0)
    )
    transport.connect()
    assert connection.recv_calls == 2
    assert connection.mav.heartbeats == 1
    assert transport.health.state == TransportState.CONNECTED
    assert transport.health.handshake_duration_s is not None
    assert transport.health.protocol_version == "unknown"


def test_listener_can_bind_before_px4_is_started(monkeypatch):
    class FakeMav:
        def heartbeat_send(self, *args):
            pass

    class FakeConnection:
        port = None
        mav = FakeMav()
        WIRE_PROTOCOL_VERSION = "2.0"

        def recv(self):
            self.port = object()

        def close(self):
            pass

    connection = FakeConnection()
    mavutil = types.SimpleNamespace(
        mavlink_connection=lambda *args, **kwargs: connection,
        mavlink=types.SimpleNamespace(
            MAV_TYPE_GCS=6,
            MAV_AUTOPILOT_INVALID=8,
            MAV_STATE_ACTIVE=4,
        ),
    )
    module = types.ModuleType("pymavlink")
    module.mavutil = mavutil
    monkeypatch.setitem(sys.modules, "pymavlink", module)
    transport = SimulatorMavlinkTransport(
        MavlinkTransportConfig("127.0.0.1", 4560, 0.1, 200, 190, 0)
    )
    transport.open_listener()
    assert transport.health.state == TransportState.LISTENING
    assert "127.0.0.1:4560" in transport.health.detail
    transport.accept_peer()
    assert transport.health.state == TransportState.CONNECTED
    assert transport.health.protocol_version == "2.0"


def test_hil_state_int16_acceleration_is_saturated_not_wrapped():
    assert _mavlink_int16_milli(0.0) == 0
    assert _mavlink_int16_milli(100.0) == 32767
    assert _mavlink_int16_milli(-100.0) == -32768
