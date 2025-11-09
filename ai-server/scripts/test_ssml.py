#!/usr/bin/env python3
"""Test SSML (Speech Synthesis Markup Language) support

Demonstrates various SSML features including:
- Prosody (rate, pitch, volume)
- Breaks/pauses
- Emphasis
- Emotion tags
- Voice selection (if multi-voice enabled)
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tts import TTSService


# SSML test cases
SSML_EXAMPLES = {
    "basic_text": {
        "ssml": "<speak>Hello, this is a basic SSML test.</speak>",
        "description": "Basic SSML with <speak> tag"
    },

    "break_pause": {
        "ssml": "<speak>Hello<break time='500ms'/>world<break time='1s'/>how are you?</speak>",
        "description": "Pauses/breaks with time attribute"
    },

    "prosody_rate": {
        "ssml": """<speak>
            <prosody rate='slow'>Speaking slowly.</prosody>
            <prosody rate='fast'>Speaking quickly!</prosody>
        </speak>""",
        "description": "Speaking rate control"
    },

    "prosody_pitch": {
        "ssml": """<speak>
            <prosody pitch='low'>Low pitch voice.</prosody>
            <prosody pitch='high'>High pitch voice!</prosody>
        </speak>""",
        "description": "Pitch variation"
    },

    "emphasis": {
        "ssml": """<speak>
            This is <emphasis level='strong'>very important</emphasis>.
            This is <emphasis level='moderate'>somewhat important</emphasis>.
            This is <emphasis level='reduced'>less important</emphasis>.
        </speak>""",
        "description": "Emphasis levels"
    },

    "emotion_tag": {
        "ssml": """<speak>
            <emotion name='happy'>Great news! The task is complete!</emotion>
            <break time='500ms'/>
            <emotion name='calm'>Please wait while I process your request.</emotion>
        </speak>""",
        "description": "Emotion presets via SSML"
    },

    "combined": {
        "ssml": """<speak>
            Welcome! <break time='500ms'/>
            <emotion name='cheerful'>I'm here to help you.</emotion>
            <break time='300ms'/>
            <prosody rate='slow'>
                <emphasis level='moderate'>Important:</emphasis>
                Your robot has completed all navigation tasks successfully.
            </prosody>
        </speak>""",
        "description": "Combined SSML features"
    },

    "announcement": {
        "ssml": """<speak>
            <emotion name='authoritative'>
                <emphasis level='strong'>Attention!</emphasis>
                <break time='500ms'/>
                System update will begin in 5 minutes.
                <break time='300ms'/>
                Please save your work.
            </emotion>
        </speak>""",
        "description": "Announcement style with emphasis"
    },

    "storytelling": {
        "ssml": """<speak>
            <emotion name='storytelling'>
                Once upon a time, <break time='400ms'/>
                there was a robot who loved to explore.
                <break time='600ms'/>
                <prosody rate='fast'>
                    <emotion name='excited'>
                        It discovered amazing new places every day!
                    </emotion>
                </prosody>
            </emotion>
        </speak>""",
        "description": "Storytelling with varied pacing and emotion"
    }
}


async def test_ssml():
    """Test SSML synthesis"""
    print("=" * 80)
    print("TTS SSML SUPPORT TEST")
    print("=" * 80)
    print()

    # Initialize TTS service
    print("Loading TTS model...")
    tts_service = TTSService()

    if not await tts_service.load_model():
        print("Failed to load TTS model")
        return 1

    print(f"Model loaded successfully")
    print(f"SSML enabled: {tts_service.ssml_enabled}")
    print()

    if not tts_service.ssml_enabled:
        print("⚠️  SSML is disabled in config")
        print("   Set TTS_ENABLE_SSML=true to enable SSML support")
        print()
        return 1

    # Create output directory
    output_dir = Path("output/ssml_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Test each SSML example
    for name, example in SSML_EXAMPLES.items():
        print(f"\n{'='*80}")
        print(f"TEST: {name}")
        print(f"{'='*80}")
        print(f"Description: {example['description']}")
        print()
        print("SSML:")
        print(example['ssml'])
        print()

        try:
            # Synthesize SSML
            audio_data = await tts_service.synthesize(
                example['ssml'],
                output_format="wav"
            )

            if audio_data.audio_bytes:
                # Save to file
                filename = f"{name}.wav"
                filepath = output_dir / filename

                with open(filepath, 'wb') as f:
                    f.write(audio_data.audio_bytes)

                print(f"✅ Saved: {filename}")
                print(f"   Duration: {audio_data.duration_ms}ms")
                print(f"   Synthesis time: {audio_data.synthesis_time_ms}ms")
                print(f"   Text: {audio_data.text[:60]}...")
            else:
                print("❌ Synthesis failed")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("SSML TEST COMPLETE")
    print("=" * 80)
    print()
    print(f"Generated {len(SSML_EXAMPLES)} SSML audio samples")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("SUPPORTED SSML TAGS:")
    print("  • <speak>         - Root element")
    print("  • <break>         - Pauses (time='500ms' or strength='medium')")
    print("  • <prosody>       - Rate, pitch, volume control")
    print("  • <emphasis>      - Emphasis levels (strong, moderate, reduced)")
    print("  • <emotion>       - Emotion presets (name='happy', 'calm', etc.)")
    print("  • <voice>         - Voice selection (multi-voice mode only)")
    print()
    print("USAGE EXAMPLE:")
    print('  ssml = """<speak>')
    print('      <emotion name="cheerful">Hello!</emotion>')
    print('      <break time="500ms"/>')
    print('      <prosody rate="slow">How can I help?</prosody>')
    print('  </speak>"""')
    print('  audio = await tts_service.synthesize(ssml)')
    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(test_ssml())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
