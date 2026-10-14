"""Gate P5 lockstep ordering, timeout, pause, and replay tests."""

import pytest

from multi_drone_mujoco.backends.contracts import NormalizedMotorCommand
from multi_drone_mujoco.backends.lockstep import (
    LockstepBootstrapConfig,
    LockstepConfig,
    LockstepCoordinator,
    LockstepStatus,
)
from multi_drone_mujoco.backends.sensors import SensorFrame
from multi_drone_mujoco.backends.transport import FakeTransport


def frame(timestamp, sequence):
    return SensorFrame(
        timestamp, sequence, (0, 0, 0), (0, 0, 0), (1, 0, 0, 0),
        (0, 0, 0), (0, 0, -9.81), (0, 0, 0), 30, 120, 100, 0,
        (0.2, 0, 0.4), True, True, True, True,
    )


def packet(timestamp, sequence, value=0.5):
    return NormalizedMotorCommand((value,) * 4, True, timestamp, sequence)


def coordinator(packets=()):
    transport = FakeTransport(packets)
    item = LockstepCoordinator(transport, LockstepConfig(0.01, (0, 0, 0, 0)))
    item.connect()
    return item, transport


def test_normal_lockstep_matches_time_and_advances_once():
    item, transport = coordinator([packet(1000, 7)])
    result = item.exchange(frame(1000, 1))
    assert result.status == LockstepStatus.READY
    assert result.advance_physics
    assert result.command.sequence == 7
    assert transport.sent_sensor_frames == [frame(1000, 1)]
    with pytest.raises(ValueError, match="timestamp"):
        item.exchange(frame(1000, 2))


def test_delayed_stale_duplicate_and_out_of_order_packets_do_not_advance_twice():
    packets = [packet(500, 1), packet(500, 1), packet(1000, 2)]
    item, _ = coordinator(packets)
    result = item.exchange(frame(1000, 1))
    assert result.status == LockstepStatus.READY
    assert result.command.timestamp_us == 1000
    timeout = item.exchange(frame(2000, 2))
    assert timeout.status == LockstepStatus.TIMED_OUT
    assert not timeout.advance_physics and not timeout.command.armed


def test_future_packet_is_rejected_not_silently_applied():
    item, _ = coordinator([packet(2000, 1)])
    result = item.exchange(frame(1000, 1))
    assert result.status == LockstepStatus.FUTURE_PACKET
    assert not result.advance_physics
    assert "exceeds" in result.detail


def test_disconnect_reconnect_and_lost_packet_safe_stop():
    item, transport = coordinator()
    transport.disconnect()
    disconnected = item.exchange(frame(1000, 1))
    assert disconnected.status == LockstepStatus.DISCONNECTED
    assert not disconnected.command.armed
    item.reconnect()
    transport.queue_actuator(packet(1000, 1))
    assert item.exchange(frame(1000, 1)).advance_physics


def test_pause_and_exactly_one_requested_single_step():
    item, transport = coordinator([packet(1000, 1)])
    item.pause()
    paused = item.exchange(frame(1000, 1))
    assert paused.status == LockstepStatus.PAUSED
    assert not transport.sent_sensor_frames
    item.request_single_step()
    assert item.exchange(frame(1000, 1)).advance_physics
    again = item.exchange(frame(2000, 2))
    assert again.status == LockstepStatus.PAUSED
    item.resume()


def test_reset_close_and_deterministic_replay():
    def replay():
        item, _ = coordinator([packet(1000, 9, 0.2), packet(2000, 10, 0.4)])
        outcomes = [item.exchange(frame(1000, 1)), item.exchange(frame(2000, 2))]
        item.reset()
        item.close()
        item.close()
        return outcomes

    assert replay() == replay()


def test_bootstrap_sends_without_advancing_physics_and_bounds_frame_order():
    class DelayedFake(FakeTransport):
        def __init__(self):
            super().__init__()
            self.polls = 0

        def poll_actuator(self):
            self.polls += 1
            if self.polls == 3:
                return packet(3000, 9, 0.0)
            return None

    transport = DelayedFake()
    item = LockstepCoordinator(transport, LockstepConfig(0.01, (0, 0, 0, 0)))
    item.connect()
    result = item.bootstrap(
        (frame(i * 1000, i) for i in range(1, 6)),
        LockstepBootstrapConfig(5, 0.0, 0.01),
    )
    assert result.timestamp_us == 3000
    assert [value.timestamp_us for value in transport.sent_sensor_frames] == [
        1000, 2000, 3000,
    ]
    assert item._last_advanced_timestamp_us == -1
    with pytest.raises(ValueError, match="last sent"):
        item.exchange(frame(3000, 4))


def test_long_sequence_deduplication_is_bounded_and_expired_stale_stays_rejected():
    item, transport = coordinator()
    for i in range(1, 10001):
        transport.queue_actuator(packet(i * 1000, i))
        result = item.exchange(frame(i * 1000, i))
        assert result.advance_physics and result.command.sequence == i
        assert len(item._seen_packets) <= 512
        assert len(item._packet_history) <= 512
    assert (1000, 1) not in item._seen_packets
    transport.queue_actuator(packet(1000, 1))
    transport.queue_actuator(packet(10000000, 10000))
    transport.queue_actuator(packet(10001000, 10001))
    result = item.exchange(frame(10001000, 10001))
    assert result.advance_physics and result.command.sequence == 10001
    transport.queue_actuator(packet(10003000, 10003))
    result = item.exchange(frame(10002000, 10002))
    assert result.status == LockstepStatus.FUTURE_PACKET and not result.advance_physics
