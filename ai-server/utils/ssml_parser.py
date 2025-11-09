"""SSML (Speech Synthesis Markup Language) Parser for TTS

Parses SSML markup and extracts text and synthesis directives.
Supports a subset of SSML tags relevant to Piper TTS capabilities.

Supported SSML tags:
- <speak>: Root element
- <break>: Pauses (time attribute)
- <prosody>: Rate, pitch, volume adjustments
- <emphasis>: Emphasis levels (strong, moderate, reduced)
- <say-as>: Interpret text as specific type (number, date, etc.)
- <phoneme>: Phonetic pronunciation
- <voice>: Voice selection (multi-voice mode)
- <emotion>: Emotion preset application
"""


import contextlib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

from utils import get_logger


logger = get_logger(__name__)


@dataclass
class SSMLSegment:
    """A segment of synthesized text with its parameters"""
    text: str
    length_scale: Optional[float] = None  # Speaking rate
    noise_scale: Optional[float] = None   # Pitch variation
    noise_w: Optional[float] = None       # Energy variation
    voice: Optional[str] = None           # Voice name
    emotion: Optional[str] = None         # Emotion preset
    pause_before: float = 0.0             # Pause before segment (seconds)
    pause_after: float = 0.0              # Pause after segment (seconds)


class SSMLParser:
    """Parse SSML markup for TTS synthesis"""

    # Prosody rate mappings (relative to normal)
    PROSODY_RATE = {
        "x-slow": 1.5,
        "slow": 1.2,
        "medium": 1.0,
        "fast": 0.85,
        "x-fast": 0.7,
    }

    # Emphasis level mappings (affects pitch/energy variation)
    EMPHASIS_LEVELS = {
        "strong": {"noise_scale": 0.9, "noise_w": 0.95},
        "moderate": {"noise_scale": 0.7, "noise_w": 0.8},
        "reduced": {"noise_scale": 0.3, "noise_w": 0.4},
        "none": {"noise_scale": 0.5, "noise_w": 0.6},
    }

    def __init__(self):
        self.default_params = {
            "length_scale": 1.0,
            "noise_scale": 0.5,
            "noise_w": 0.6,
        }

    def parse(self, ssml_text: str) -> List[SSMLSegment]:
        """Parse SSML text into segments

        Args:
            ssml_text: SSML markup text

        Returns:
            List of SSMLSegment objects
        """
        try:
            # Ensure SSML has <speak> root element
            if not ssml_text.strip().startswith("<speak"):
                ssml_text = f"<speak>{ssml_text}</speak>"

            # Parse XML
            root = ET.fromstring(ssml_text)

            # Extract segments
            segments = self._process_element(root, self.default_params.copy())

            # Merge consecutive text-only segments
            segments = self._merge_segments(segments)

            return segments

        except ET.ParseError as e:
            logger.error("ssml_parse_error", error=str(e), error_type=type(e).__name__)
            # Fallback: treat as plain text
            return [SSMLSegment(text=self._strip_tags(ssml_text))]
        except Exception as e:
            logger.error("ssml_processing_error", error=str(e), error_type=type(e).__name__)
            return [SSMLSegment(text=self._strip_tags(ssml_text))]

    def _process_element(
        self,
        element: ET.Element,
        params: dict,
        pause_before: float = 0.0
    ) -> List[SSMLSegment]:
        """Process an XML element and its children

        Args:
            element: XML element to process
            params: Current synthesis parameters
            pause_before: Pause time before this element

        Returns:
            List of segments
        """
        segments = []
        current_pause_before = pause_before

        # Process element's own text
        if element.text and element.text.strip():
            segment = SSMLSegment(
                text=element.text.strip(),
                length_scale=params.get("length_scale"),
                noise_scale=params.get("noise_scale"),
                noise_w=params.get("noise_w"),
                voice=params.get("voice"),
                emotion=params.get("emotion"),
                pause_before=current_pause_before
            )
            segments.append(segment)
            current_pause_before = 0.0

        # Process child elements
        for child in element:
            child_params = params.copy()
            child_pause_before = 0.0

            # Handle different SSML tags
            if child.tag == "break":
                # Pause/break
                pause_time = self._parse_break(child)
                if segments:
                    segments[-1].pause_after = pause_time
                else:
                    child_pause_before = pause_time

            elif child.tag == "prosody":
                # Prosody adjustments (rate, pitch, volume)
                child_params = self._apply_prosody(child, child_params)

            elif child.tag == "emphasis":
                # Emphasis level
                child_params = self._apply_emphasis(child, child_params)

            elif child.tag == "voice":
                if voice_name := child.get("name"):
                    child_params["voice"] = voice_name

            elif child.tag == "emotion":
                # Emotion preset (custom tag)
                emotion = child.get("name")
                if emotion:
                    child_params["emotion"] = emotion

            elif child.tag == "say-as":
                # Interpret as specific type (handled by text normalizer)
                pass  # Text normalizer handles this

            elif child.tag == "phoneme":
                # Phonetic pronunciation (not supported by Piper)
                logger.warning("Phoneme tag not supported, using alphabet text")

            # Recursively process child
            child_segments = self._process_element(child, child_params, child_pause_before)
            segments.extend(child_segments)

            # Process tail text (text after child element)
            if child.tail and child.tail.strip():
                segment = SSMLSegment(
                    text=child.tail.strip(),
                    length_scale=params.get("length_scale"),
                    noise_scale=params.get("noise_scale"),
                    noise_w=params.get("noise_w"),
                    voice=params.get("voice"),
                    emotion=params.get("emotion")
                )
                segments.append(segment)

        return segments

    def _parse_break(self, element: ET.Element) -> float:
        """Parse break/pause duration

        Args:
            element: Break element

        Returns:
            Pause duration in seconds
        """
        time_str = element.get("time", "0s")
        strength = element.get("strength")

        # Parse time attribute (e.g., "500ms", "1s")
        if time_str:
            if match := re.match(r"(\d+(?:\.\d+)?)(ms|s)", time_str):
                value = float(match[1])
                unit = match[2]
                return value / 1000.0 if unit == "ms" else value
        # Parse strength attribute
        if strength:
            strength_map = {
                "none": 0.0,
                "x-weak": 0.1,
                "weak": 0.25,
                "medium": 0.5,
                "strong": 0.75,
                "x-strong": 1.0,
            }
            return strength_map.get(strength, 0.5)

        return 0.0

    def _apply_prosody(self, element: ET.Element, params: dict) -> dict:
        """Apply prosody adjustments

        Args:
            element: Prosody element
            params: Current parameters

        Returns:
            Updated parameters
        """
        new_params = params.copy()

        if rate := element.get("rate"):
            if rate in self.PROSODY_RATE:
                new_params["length_scale"] = self.PROSODY_RATE[rate]
            elif rate.endswith("%"):
                # Percentage adjustment
                with contextlib.suppress(ValueError):
                    percentage = float(rate[:-1]) / 100.0
                    new_params["length_scale"] = params.get("length_scale", 1.0) / percentage
        if pitch := element.get("pitch"):
            if pitch in ["x-low", "low"]:
                new_params["noise_scale"] = 0.3
            elif pitch in ["high", "x-high"]:
                new_params["noise_scale"] = 0.8
        if volume := element.get("volume"):
            if volume in ["x-soft", "soft"]:
                new_params["noise_w"] = 0.3
            elif volume in ["loud", "x-loud"]:
                new_params["noise_w"] = 0.9
        return new_params

    def _apply_emphasis(self, element: ET.Element, params: dict) -> dict:
        """Apply emphasis level

        Args:
            element: Emphasis element
            params: Current parameters

        Returns:
            Updated parameters
        """
        new_params = params.copy()
        level = element.get("level", "moderate")

        if level in self.EMPHASIS_LEVELS:
            new_params |= self.EMPHASIS_LEVELS[level]

        return new_params

    def _merge_segments(self, segments: List[SSMLSegment]) -> List[SSMLSegment]:
        """Merge consecutive segments with identical parameters

        Args:
            segments: List of segments

        Returns:
            Merged segments
        """
        if not segments:
            return []

        merged = [segments[0]]

        for segment in segments[1:]:
            last = merged[-1]

            # Check if parameters match
            if (last.length_scale == segment.length_scale and
                last.noise_scale == segment.noise_scale and
                last.noise_w == segment.noise_w and
                last.voice == segment.voice and
                last.emotion == segment.emotion and
                last.pause_after == 0.0 and
                segment.pause_before == 0.0):

                # Merge text
                last.text = f"{last.text} {segment.text}"
            else:
                merged.append(segment)

        return merged

    def _strip_tags(self, text: str) -> str:
        """Strip all SSML tags, leaving only text

        Args:
            text: SSML text

        Returns:
            Plain text
        """
        return re.sub(r"<[^>]+>", "", text).strip()

    def is_ssml(self, text: str) -> bool:
        """Check if text contains SSML markup

        Args:
            text: Text to check

        Returns:
            True if text appears to be SSML
        """
        text = text.strip()
        return text.startswith("<speak") or ("<" in text and ">" in text)
