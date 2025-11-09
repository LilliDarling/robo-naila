"""Emotion and tone presets for TTS synthesis

Provides pre-configured parameter sets for different emotional tones and contexts.
Each preset adjusts LENGTH_SCALE, NOISE_SCALE, and NOISE_W to convey specific emotions.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EmotionPreset:
    """Preset configuration for emotional tone"""
    name: str
    description: str
    length_scale: float  # Speaking rate
    noise_scale: float   # Pitch variation
    noise_w: float       # Energy variation


# Emotion presets organized by category
EMOTION_PRESETS: Dict[str, EmotionPreset] = {
    # ========================================================================
    # NEUTRAL/STANDARD
    # ========================================================================
    "neutral": EmotionPreset(
        name="neutral",
        description="Balanced, natural speech",
        length_scale=1.0,
        noise_scale=0.5,
        noise_w=0.6
    ),

    # ========================================================================
    # POSITIVE EMOTIONS
    # ========================================================================
    "happy": EmotionPreset(
        name="happy",
        description="Upbeat, cheerful tone",
        length_scale=0.95,    # Slightly faster
        noise_scale=0.8,      # More expressive
        noise_w=0.85          # More dynamic
    ),

    "excited": EmotionPreset(
        name="excited",
        description="Energetic, enthusiastic",
        length_scale=0.85,    # Faster
        noise_scale=0.9,      # Very expressive
        noise_w=0.95          # Very dynamic
    ),

    "cheerful": EmotionPreset(
        name="cheerful",
        description="Pleasant, friendly",
        length_scale=0.95,
        noise_scale=0.75,
        noise_w=0.8
    ),

    "proud": EmotionPreset(
        name="proud",
        description="Confident, accomplished",
        length_scale=1.0,
        noise_scale=0.65,
        noise_w=0.7
    ),

    # ========================================================================
    # CALM/SOOTHING
    # ========================================================================
    "calm": EmotionPreset(
        name="calm",
        description="Peaceful, soothing",
        length_scale=1.1,     # Slower
        noise_scale=0.3,      # Less variation
        noise_w=0.4           # Stable
    ),

    "gentle": EmotionPreset(
        name="gentle",
        description="Soft, tender",
        length_scale=1.15,
        noise_scale=0.35,
        noise_w=0.45
    ),

    "reassuring": EmotionPreset(
        name="reassuring",
        description="Comforting, supportive",
        length_scale=1.05,
        noise_scale=0.4,
        noise_w=0.5
    ),

    # ========================================================================
    # SERIOUS/FORMAL
    # ========================================================================
    "serious": EmotionPreset(
        name="serious",
        description="Professional, grave",
        length_scale=1.05,
        noise_scale=0.35,
        noise_w=0.4
    ),

    "formal": EmotionPreset(
        name="formal",
        description="Professional, business-like",
        length_scale=1.0,
        noise_scale=0.3,
        noise_w=0.35
    ),

    "authoritative": EmotionPreset(
        name="authoritative",
        description="Commanding, confident",
        length_scale=0.95,
        noise_scale=0.4,
        noise_w=0.5
    ),

    # ========================================================================
    # NEGATIVE EMOTIONS (CONVEYED EMPATHETICALLY)
    # ========================================================================
    "sad": EmotionPreset(
        name="sad",
        description="Melancholic, downcast (empathetic delivery)",
        length_scale=1.15,    # Slower
        noise_scale=0.25,     # Less variation
        noise_w=0.3           # Low energy
    ),

    "apologetic": EmotionPreset(
        name="apologetic",
        description="Sorry, regretful",
        length_scale=1.1,
        noise_scale=0.35,
        noise_w=0.4
    ),

    "concerned": EmotionPreset(
        name="concerned",
        description="Worried, caring",
        length_scale=1.05,
        noise_scale=0.5,
        noise_w=0.55
    ),

    "disappointed": EmotionPreset(
        name="disappointed",
        description="Let down, unfulfilled",
        length_scale=1.1,
        noise_scale=0.3,
        noise_w=0.35
    ),

    # ========================================================================
    # URGENT/ALERT
    # ========================================================================
    "urgent": EmotionPreset(
        name="urgent",
        description="Time-sensitive, pressing",
        length_scale=0.9,     # Faster
        noise_scale=0.7,      # Expressive
        noise_w=0.8           # Energetic
    ),

    "alert": EmotionPreset(
        name="alert",
        description="Attention-getting, important",
        length_scale=0.95,
        noise_scale=0.65,
        noise_w=0.75
    ),

    "warning": EmotionPreset(
        name="warning",
        description="Cautionary, serious alert",
        length_scale=1.0,
        noise_scale=0.5,
        noise_w=0.6
    ),

    # ========================================================================
    # THINKING/UNCERTAIN
    # ========================================================================
    "thoughtful": EmotionPreset(
        name="thoughtful",
        description="Contemplative, considering",
        length_scale=1.1,
        noise_scale=0.45,
        noise_w=0.5
    ),

    "uncertain": EmotionPreset(
        name="uncertain",
        description="Hesitant, unsure",
        length_scale=1.05,
        noise_scale=0.55,
        noise_w=0.6
    ),

    "curious": EmotionPreset(
        name="curious",
        description="Inquisitive, interested",
        length_scale=0.95,
        noise_scale=0.7,
        noise_w=0.75
    ),

    # ========================================================================
    # ROBOTIC/MONOTONE
    # ========================================================================
    "robotic": EmotionPreset(
        name="robotic",
        description="Mechanical, emotionless",
        length_scale=1.0,
        noise_scale=0.2,
        noise_w=0.3
    ),

    "monotone": EmotionPreset(
        name="monotone",
        description="Flat, minimal variation",
        length_scale=1.0,
        noise_scale=0.15,
        noise_w=0.25
    ),

    # ========================================================================
    # SPECIAL CASES
    # ========================================================================
    "whispering": EmotionPreset(
        name="whispering",
        description="Quiet, secretive",
        length_scale=1.2,     # Slower
        noise_scale=0.25,     # Minimal variation
        noise_w=0.2           # Very low energy
    ),

    "storytelling": EmotionPreset(
        name="storytelling",
        description="Narrative, engaging",
        length_scale=1.0,
        noise_scale=0.7,
        noise_w=0.75
    ),

    "teaching": EmotionPreset(
        name="teaching",
        description="Educational, clear",
        length_scale=1.1,
        noise_scale=0.4,
        noise_w=0.5
    ),

    "announcing": EmotionPreset(
        name="announcing",
        description="Public announcement style",
        length_scale=1.0,
        noise_scale=0.5,
        noise_w=0.6
    ),
}


# Emotion categories for easy discovery
EMOTION_CATEGORIES = {
    "positive": ["happy", "excited", "cheerful", "proud"],
    "calm": ["calm", "gentle", "reassuring"],
    "serious": ["serious", "formal", "authoritative"],
    "negative": ["sad", "apologetic", "concerned", "disappointed"],
    "urgent": ["urgent", "alert", "warning"],
    "thinking": ["thoughtful", "uncertain", "curious"],
    "robotic": ["robotic", "monotone"],
    "special": ["whispering", "storytelling", "teaching", "announcing"],
}


def get_emotion_preset(emotion: str) -> Optional[EmotionPreset]:
    """Get emotion preset by name

    Args:
        emotion: Emotion name (case-insensitive)

    Returns:
        EmotionPreset or None if not found
    """
    return EMOTION_PRESETS.get(emotion.lower())


def get_emotion_parameters(emotion: str) -> Optional[Dict[str, float]]:
    """Get synthesis parameters for an emotion

    Args:
        emotion: Emotion name (case-insensitive)

    Returns:
        Dict with length_scale, noise_scale, noise_w or None if not found
    """
    preset = get_emotion_preset(emotion)
    if preset:
        return {
            "length_scale": preset.length_scale,
            "noise_scale": preset.noise_scale,
            "noise_w": preset.noise_w
        }
    return None


def list_emotions() -> list[str]:
    """Get list of all available emotion names

    Returns:
        List of emotion names
    """
    return list(EMOTION_PRESETS.keys())


def list_emotions_by_category(category: str) -> list[str]:
    """Get emotions in a specific category

    Args:
        category: Category name (positive, calm, serious, etc.)

    Returns:
        List of emotion names in that category
    """
    return EMOTION_CATEGORIES.get(category.lower(), [])


def get_categories() -> list[str]:
    """Get list of all emotion categories

    Returns:
        List of category names
    """
    return list(EMOTION_CATEGORIES.keys())
