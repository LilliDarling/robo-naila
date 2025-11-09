#!/usr/bin/env python3
"""Test emotion/tone variations in TTS synthesis

This script demonstrates the emotion preset feature by synthesizing
the same text with different emotional tones.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tts import TTSService
from utils.emotion_presets import EMOTION_CATEGORIES, get_categories


async def test_emotions():
    """Test different emotion presets"""
    print("=" * 80)
    print("TTS EMOTION/TONE VARIATIONS TEST")
    print("=" * 80)
    print()

    # Initialize TTS service
    print("Loading TTS model...")
    tts_service = TTSService()

    if not await tts_service.load_model():
        print("Failed to load TTS model")
        return 1

    print("Model loaded successfully")
    print()

    # Create output directory
    output_dir = Path("output/emotion_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Test phrase
    test_text = "Hello! The robot has completed the navigation task successfully."

    # Get available emotions
    emotions = tts_service.get_available_emotions()
    print(f"Available emotions: {len(emotions)}")
    print()

    # Test emotions by category
    categories = get_categories()

    for category in categories:
        category_emotions = EMOTION_CATEGORIES[category]

        print(f"\n{'='*80}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*80}")

        for emotion in category_emotions:
            print(f"\nSynthesizing with emotion: {emotion}")

            try:
                # Synthesize with emotion
                audio_data = await tts_service.synthesize(
                    test_text,
                    emotion=emotion,
                    output_format="wav"
                )

                if audio_data.audio_bytes:
                    # Save to file
                    filename = f"{emotion}.wav"
                    filepath = output_dir / filename

                    with open(filepath, 'wb') as f:
                        f.write(audio_data.audio_bytes)

                    print(f"  Saved: {filename}")
                    print(f"  Duration: {audio_data.duration_ms}ms")
                    print(f"  Synthesis time: {audio_data.synthesis_time_ms}ms")
                else:
                    print(f"  Failed to synthesize")

            except Exception as e:
                print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Generated {len(emotions)} audio samples")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("USAGE EXAMPLES:")
    print()
    print("  # Use emotion preset")
    print('  audio = await tts_service.synthesize("Hello!", emotion="happy")')
    print()
    print("  # Different emotions for different contexts")
    print('  await tts_service.synthesize("Task completed!", emotion="cheerful")')
    print('  await tts_service.synthesize("Error occurred", emotion="calm")')
    print('  await tts_service.synthesize("Warning!", emotion="urgent")')
    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(test_emotions())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
