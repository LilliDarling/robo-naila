"""Voice Manager for multi-voice TTS support

Manages multiple Piper voice models, allowing dynamic voice switching
and efficient voice loading/unloading.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class VoiceConfig:
    """Configuration for a single voice"""
    name: str
    model_path: Path
    language: str = "en_US"
    sample_rate: int = 22050
    description: str = ""
    speaker_id: int = 0


class VoiceManager:
    """Manages multiple TTS voice models"""

    def __init__(self):
        self._voices: Dict[str, any] = {}  # name -> PiperVoice
        self._voice_configs: Dict[str, VoiceConfig] = {}
        self._current_voice: Optional[str] = None
        self._use_cuda: bool = False

    def register_voice(self, config: VoiceConfig):
        """Register a voice configuration without loading it

        Args:
            config: Voice configuration
        """
        self._voice_configs[config.name] = config
        logger.info(f"Registered voice: {config.name} ({config.description})")

    async def load_voice(self, voice_name: str, use_cuda: bool = False) -> bool:
        """Load a specific voice model

        Args:
            voice_name: Name of the voice to load
            use_cuda: Whether to use CUDA acceleration

        Returns:
            True if loaded successfully, False otherwise
        """
        if voice_name in self._voices:
            logger.info(f"Voice '{voice_name}' already loaded")
            return True

        if voice_name not in self._voice_configs:
            logger.error(f"Voice '{voice_name}' not registered")
            return False

        config = self._voice_configs[voice_name]

        try:
            from piper import PiperVoice

            # Load model in executor (blocking operation)
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                lambda: PiperVoice.load(
                    str(config.model_path),
                    use_cuda=use_cuda
                )
            )

            self._voices[voice_name] = model
            self._use_cuda = use_cuda

            logger.info(
                f"Loaded voice: {voice_name} "
                f"(sample_rate={config.sample_rate}Hz, "
                f"model={config.model_path.name})"
            )

            # Set as current if first voice loaded
            if self._current_voice is None:
                self._current_voice = voice_name

            return True

        except FileNotFoundError:
            logger.error(f"Model file not found: {config.model_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load voice '{voice_name}': {e}")
            return False

    def unload_voice(self, voice_name: str):
        """Unload a specific voice model

        Args:
            voice_name: Name of the voice to unload
        """
        if voice_name in self._voices:
            del self._voices[voice_name]
            logger.info(f"Unloaded voice: {voice_name}")

            # Clear current voice if it was unloaded
            if self._current_voice == voice_name:
                self._current_voice = None

    def unload_all_voices(self):
        """Unload all voice models"""
        voice_names = list(self._voices.keys())
        for voice_name in voice_names:
            self.unload_voice(voice_name)

    def set_current_voice(self, voice_name: str) -> bool:
        """Set the current active voice

        Args:
            voice_name: Name of the voice to make current

        Returns:
            True if successful, False if voice not loaded
        """
        if voice_name not in self._voices:
            logger.error(f"Cannot set current voice: '{voice_name}' not loaded")
            return False

        self._current_voice = voice_name
        logger.info(f"Current voice set to: {voice_name}")
        return True

    def get_current_voice(self) -> Optional[any]:
        """Get the current active voice model

        Returns:
            PiperVoice model or None if no voice is loaded
        """
        if self._current_voice is None:
            return None
        return self._voices.get(self._current_voice)

    def get_current_voice_name(self) -> Optional[str]:
        """Get the name of the current active voice

        Returns:
            Voice name or None if no voice is loaded
        """
        return self._current_voice

    def get_voice(self, voice_name: str) -> Optional[any]:
        """Get a specific voice model

        Args:
            voice_name: Name of the voice

        Returns:
            PiperVoice model or None if not loaded
        """
        return self._voices.get(voice_name)

    def get_voice_config(self, voice_name: str) -> Optional[VoiceConfig]:
        """Get configuration for a specific voice

        Args:
            voice_name: Name of the voice

        Returns:
            VoiceConfig or None if not registered
        """
        return self._voice_configs.get(voice_name)

    def get_current_voice_config(self) -> Optional[VoiceConfig]:
        """Get configuration for the current voice

        Returns:
            VoiceConfig or None if no voice is current
        """
        if self._current_voice is None:
            return None
        return self._voice_configs.get(self._current_voice)

    def is_voice_loaded(self, voice_name: str) -> bool:
        """Check if a voice is loaded

        Args:
            voice_name: Name of the voice

        Returns:
            True if loaded, False otherwise
        """
        return voice_name in self._voices

    def get_loaded_voices(self) -> List[str]:
        """Get list of currently loaded voice names

        Returns:
            List of voice names
        """
        return list(self._voices.keys())

    def get_registered_voices(self) -> List[str]:
        """Get list of all registered voice names

        Returns:
            List of voice names
        """
        return list(self._voice_configs.keys())

    def get_status(self) -> Dict:
        """Get voice manager status

        Returns:
            Status dictionary
        """
        return {
            "current_voice": self._current_voice,
            "loaded_voices": self.get_loaded_voices(),
            "registered_voices": self.get_registered_voices(),
            "use_cuda": self._use_cuda,
            "voice_count": {
                "loaded": len(self._voices),
                "registered": len(self._voice_configs)
            }
        }
