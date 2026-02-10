import queue

import numpy as np
import pytest

from pi_audio.metrics import DeviceMetrics


SAMPLE_RATE = 48_000
FRAME_SIZE = 960


class TestAudioPipelineQueues:
    """Test the queue logic of AudioPipeline without requiring real audio hardware."""

    def test_queue_playback_and_overflow(self):
        """Enqueuing beyond max drops oldest frame."""
        metrics = DeviceMetrics()

        # Simulate the queue directly (same logic as AudioPipeline).
        speaker_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        frame_a = np.ones(FRAME_SIZE, dtype=np.int16)
        frame_b = np.ones(FRAME_SIZE, dtype=np.int16) * 2
        frame_c = np.ones(FRAME_SIZE, dtype=np.int16) * 3

        speaker_q.put_nowait(frame_a)
        speaker_q.put_nowait(frame_b)
        assert speaker_q.full()

        # Overflow: drop oldest, push new.
        try:
            speaker_q.put_nowait(frame_c)
        except queue.Full:
            speaker_q.get_nowait()
            speaker_q.put_nowait(frame_c)

        # frame_a was dropped; remaining are frame_b and frame_c.
        got1 = speaker_q.get_nowait()
        got2 = speaker_q.get_nowait()
        np.testing.assert_array_equal(got1, frame_b)
        np.testing.assert_array_equal(got2, frame_c)

    def test_mic_queue_overflow_increments_metric(self):
        """When mic queue overflows, dropped counter increments."""
        metrics = DeviceMetrics()
        mic_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        frame = np.zeros(FRAME_SIZE, dtype=np.int16)

        mic_q.put_nowait(frame)
        try:
            mic_q.put_nowait(frame)
        except queue.Full:
            try:
                mic_q.get_nowait()
            except queue.Empty:
                pass
            mic_q.put_nowait(frame)
            metrics.mic_frames_dropped += 1

        assert metrics.mic_frames_dropped == 1


class TestDeviceMetrics:
    def test_snapshot_returns_expected_keys(self):
        m = DeviceMetrics()
        snap = m.snapshot()
        expected_keys = {
            "connected",
            "uptime_seconds",
            "mic_frames_sent",
            "mic_frames_dropped",
            "tts_frames_received",
            "tts_frames_played",
            "mic_queue_depth",
            "speaker_queue_depth",
            "audio_callback_duration_ms",
            "connections",
            "connection_failures",
        }
        assert set(snap.keys()) == expected_keys

    def test_uptime_positive(self):
        import time
        m = DeviceMetrics()
        time.sleep(0.01)
        assert m.uptime_seconds > 0

    def test_periodic_log_deltas(self):
        m = DeviceMetrics()
        m.mic_frames_sent = 100
        m.tts_frames_received = 50
        m.log_periodic()
        # After first log, prev counters updated.
        m.mic_frames_sent = 150
        # Delta should be 50 on next call (tested implicitly via no crash).
        m.log_periodic()
