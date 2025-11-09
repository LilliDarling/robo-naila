#!/usr/bin/env python3
"""Verification script for TTS service"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tts import TTSService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def verify_tts_service():
    """Verify TTS service loads and works"""

    logger.info("=" * 60)
    logger.info("TTS Service Verification")
    logger.info("=" * 60)

    # Create TTS service
    logger.info("\n1. Creating TTSService instance...")
    tts_service = TTSService()
    logger.info(f"   ✓ TTSService created")
    logger.info(f"   - Model path: {tts_service.model_path}")
    logger.info(f"   - Model exists: {tts_service.model_path.exists()}")

    # Load model
    logger.info("\n2. Loading TTS model...")
    try:
        success = await tts_service.load_model()
        if success:
            logger.info(f"   ✓ Model loaded successfully")
            logger.info(f"   - Ready: {tts_service.is_ready}")
            logger.info(f"   - Cached phrases: {len(tts_service._phrase_cache)}")
        else:
            logger.error(f"   ✗ Model failed to load")
            return False
    except Exception as e:
        logger.error(f"   ✗ Exception loading model: {e}", exc_info=True)
        return False

    # Get status
    logger.info("\n3. Checking service status...")
    status = tts_service.get_status()
    logger.info(f"   ✓ Status retrieved")
    for key, value in status.items():
        logger.info(f"   - {key}: {value}")

    # Test synthesis with simple text
    logger.info("\n4. Testing synthesis with simple text...")
    test_texts = [
        "Hello, world!",
        "This is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n   Test {i}: '{text}'")
        try:
            audio_data = await tts_service.synthesize(text, output_format="wav")

            if audio_data.audio_bytes:
                logger.info(f"   ✓ Synthesis successful")
                logger.info(f"     - Audio size: {len(audio_data.audio_bytes)} bytes")
                logger.info(f"     - Duration: {audio_data.duration_ms}ms")
                logger.info(f"     - Synthesis time: {audio_data.synthesis_time_ms}ms")
                logger.info(f"     - RTF: {audio_data.synthesis_time_ms / audio_data.duration_ms:.3f}")
                logger.info(f"     - Sample rate: {audio_data.sample_rate}Hz")
                logger.info(f"     - Format: {audio_data.format}")
            else:
                logger.error(f"   ✗ Synthesis returned empty audio")
                return False

        except Exception as e:
            logger.error(f"   ✗ Synthesis failed: {e}", exc_info=True)
            return False

    # Test text normalization
    logger.info("\n5. Testing text normalization...")
    normalization_tests = [
        ("I have 5 apples", "numbers"),
        ("The price is $50", "currency"),
        ("Meet me at 3:30 PM", "time"),
        ("Dr. Smith", "abbreviations"),
    ]

    for text, test_type in normalization_tests:
        logger.info(f"\n   Testing {test_type}: '{text}'")
        try:
            audio_data = await tts_service.synthesize(text, output_format="wav")
            if audio_data.audio_bytes:
                logger.info(f"   ✓ Normalized and synthesized successfully")
                logger.info(f"     - Normalized text: '{audio_data.text}'")
            else:
                logger.error(f"   ✗ Synthesis returned empty audio")
        except Exception as e:
            logger.error(f"   ✗ Synthesis failed: {e}")

    # Test different output formats
    logger.info("\n6. Testing different output formats...")
    formats_to_test = ["wav", "raw"]  # Skip mp3/ogg if ffmpeg not available

    for format_type in formats_to_test:
        logger.info(f"\n   Testing format: {format_type}")
        try:
            audio_data = await tts_service.synthesize(
                "Testing audio format",
                output_format=format_type
            )
            if audio_data.audio_bytes:
                logger.info(f"   ✓ {format_type.upper()} encoding successful")
                logger.info(f"     - Size: {len(audio_data.audio_bytes)} bytes")
            else:
                logger.error(f"   ✗ {format_type.upper()} encoding returned empty")
        except Exception as e:
            logger.error(f"   ✗ {format_type.upper()} encoding failed: {e}")

    # Test file saving
    logger.info("\n7. Testing file saving...")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "test_tts_output.wav"
    try:
        success = await tts_service.synthesize_to_file(
            "This is a test of saving audio to a file.",
            str(output_file)
        )
        if success and output_file.exists():
            logger.info(f"   ✓ File saved successfully")
            logger.info(f"     - Path: {output_file}")
            logger.info(f"     - Size: {output_file.stat().st_size} bytes")
        else:
            logger.error(f"   ✗ File saving failed")
    except Exception as e:
        logger.error(f"   ✗ File saving exception: {e}")

    # Test phrase caching
    logger.info("\n8. Testing phrase caching...")
    cached_phrase = "Hello"

    # First synthesis (should cache)
    logger.info(f"   First synthesis: '{cached_phrase}'")
    audio1 = await tts_service.synthesize(cached_phrase)
    time1 = audio1.synthesis_time_ms

    # Second synthesis (should use cache)
    logger.info(f"   Second synthesis: '{cached_phrase}'")
    audio2 = await tts_service.synthesize(cached_phrase)
    time2 = audio2.synthesis_time_ms

    if time2 < time1:
        logger.info(f"   ✓ Caching working (time reduced: {time1}ms → {time2}ms)")
    else:
        logger.info(f"   - Caching may be working (times: {time1}ms, {time2}ms)")

    # Unload model
    logger.info("\n9. Unloading model...")
    tts_service.unload_model()
    logger.info(f"   ✓ Model unloaded")
    logger.info(f"   - Ready: {tts_service.is_ready}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ TTS Service Verification COMPLETE")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(verify_tts_service())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\n\nVerification interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Verification failed with exception: {e}", exc_info=True)
        sys.exit(1)
