from abc import ABC, abstractmethod

import numpy as np


class VAD(ABC):
    """Base class for Voice Activity Detection."""
    
    @abstractmethod
    def is_speech(self, pcm: np.ndarray) -> bool:
        """Determine if audio chunk contains speech."""
        pass


class EnergyVAD(VAD):
    """
    Simple energy-based VAD.
    Fast and lightweight, good for clear audio.
    """
    
    def __init__(self, threshold: int = 500):
        self.threshold = threshold
    
    def is_speech(self, pcm: np.ndarray) -> bool:
        """Check if RMS energy exceeds threshold."""
        energy = np.sqrt(np.mean(pcm.astype(np.float32) ** 2))
        return energy > self.threshold


class SileroVAD(VAD):
    """
    Silero VAD - more accurate but heavier.
    Requires torch and silero-vad packages.
    """
    
    def __init__(self, threshold: float = 0.5, sample_rate: int = 48000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            # Lazy load to avoid import errors if not installed
            import torch
            self._model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
        return self._model
    
    def is_speech(self, pcm: np.ndarray) -> bool:
        """Run Silero VAD on audio chunk."""
        import torch
        
        # Silero expects float32, normalized to [-1, 1]
        audio = pcm.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio)
        
        confidence = self.model(audio_tensor, self.sample_rate).item()
        return confidence > self.threshold


def create_vad(mode: str = "energy", **kwargs) -> VAD:
    """Factory function to create VAD instance."""
    if mode == "energy":
        return EnergyVAD(threshold=kwargs.get("threshold", 500))
    elif mode == "silero":
        return SileroVAD(
            threshold=kwargs.get("threshold", 0.5),
            sample_rate=kwargs.get("sample_rate", 48000)
        )
    else:
        raise ValueError(f"Unknown VAD mode: {mode}")
