import numpy as np
import pytest


SAMPLE_RATE = 48_000
FRAME_SIZE = 960


class TestEchoCanceller:
    def test_passthrough_without_speexdsp(self, monkeypatch):
        """When speexdsp is unavailable, mic passes through unchanged."""
        import pi_audio.aec as aec_mod

        # Force import failure.
        original_init = aec_mod.EchoCanceller._init_echo

        def _broken_init(self):
            self._echo = None

        monkeypatch.setattr(aec_mod.EchoCanceller, "_init_echo", _broken_init)

        ec = aec_mod.EchoCanceller(SAMPLE_RATE, FRAME_SIZE)
        mic = np.random.randint(-1000, 1000, size=FRAME_SIZE, dtype=np.int16)
        ref = np.zeros(FRAME_SIZE, dtype=np.int16)
        out = ec.process(mic, ref)
        np.testing.assert_array_equal(out, mic)

    def test_output_shape_and_dtype(self, monkeypatch):
        """Output is int16 with correct frame size, even in passthrough."""
        import pi_audio.aec as aec_mod

        def _broken_init(self):
            self._echo = None

        monkeypatch.setattr(aec_mod.EchoCanceller, "_init_echo", _broken_init)

        ec = aec_mod.EchoCanceller(SAMPLE_RATE, FRAME_SIZE)
        mic = np.zeros(FRAME_SIZE, dtype=np.int16)
        ref = np.zeros(FRAME_SIZE, dtype=np.int16)
        out = ec.process(mic, ref)
        assert out.shape == (FRAME_SIZE,)
        assert out.dtype == np.int16

    def test_filter_tail_calculation(self):
        """Verify 100ms filter tail = 4800 samples at 48kHz."""
        from pi_audio.aec import _FILTER_TAIL_MS, EchoCanceller

        tail = SAMPLE_RATE * _FILTER_TAIL_MS // 1000
        assert tail == 4800
