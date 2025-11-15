#!/usr/bin/env python3
"""TTS Parameter Fine-Tuning Research Template

This template allows you to research and test different TTS synthesis parameters
to find optimal settings for your specific use case.

CUSTOMIZATION GUIDE:
1. Edit TEST_PHRASES to match your robot's common responses
2. Edit PARAM_CONFIGS to test parameter combinations you're interested in
3. Run the script and listen to generated WAV files
4. Update config/tts.py with your preferred settings

PARAMETER REFERENCE:
- LENGTH_SCALE: Speaking rate
  * 0.5-0.8  = Very fast (rushing)
  * 0.85-0.95 = Fast (energetic)
  * 1.0      = Normal ✓
  * 1.1-1.3  = Slow (deliberate)
  * 1.4+     = Very slow (teaching)

- NOISE_SCALE: Pitch variation/expressiveness
  * 0.0-0.3  = Monotone (robot-like)
  * 0.4-0.6  = Moderate (natural) ✓
  * 0.7-0.9  = High (expressive)
  * 1.0      = Maximum expressiveness

- NOISE_W: Energy variation/dynamics
  * 0.0-0.4  = Flat (minimal dynamics)
  * 0.5-0.7  = Moderate (natural) ✓
  * 0.8-0.9  = High (dynamic)
  * 1.0      = Maximum dynamics
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tts import TTSService
from config import tts as tts_config


# ============================================================================
# CUSTOMIZE: Test Phrases
# ============================================================================
# Add phrases that represent your robot's typical responses
# Include variety: short/long, questions/statements, emotional/technical
TEST_PHRASES = [
    # TODO: Customize these phrases for your use case
    "Hello, how can I help you?",
    "What would you like me to do?",
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "I'm sorry, I didn't understand that. Could you please repeat?",
    "The temperature is 72 degrees, and it's 3:30 PM on January 15th, 2025.",
    "Your robot has completed the navigation task successfully with 95% accuracy."
]


# ============================================================================
# CUSTOMIZE: Parameter Configurations
# ============================================================================
# Add/remove/modify configurations to test different parameter combinations
# Each config needs: length_scale, noise_scale, noise_w, description
PARAM_CONFIGS = {
    # TODO: Customize these parameter configurations
    "default": {
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "description": "Default Piper settings"
    },
    "natural_balanced": {
        "length_scale": 1.0,
        "noise_scale": 0.5,
        "noise_w": 0.6,
        "description": "Balanced natural speech"
    },
    "expressive": {
        "length_scale": 0.95,
        "noise_scale": 0.8,
        "noise_w": 0.9,
        "description": "More expressive and dynamic"
    },
    "calm_clear": {
        "length_scale": 1.1,
        "noise_scale": 0.3,
        "noise_w": 0.4,
        "description": "Calm, clear, minimal variation"
    },
    "fast_energetic": {
        "length_scale": 0.85,
        "noise_scale": 0.75,
        "noise_w": 0.85,
        "description": "Fast and energetic"
    },
    "slow_deliberate": {
        "length_scale": 1.2,
        "noise_scale": 0.4,
        "noise_w": 0.5,
        "description": "Slow and deliberate"
    },
    "monotone_stable": {
        "length_scale": 1.0,
        "noise_scale": 0.2,
        "noise_w": 0.3,
        "description": "Monotone, stable (robot-like)"
    },
    "very_expressive": {
        "length_scale": 0.9,
        "noise_scale": 1.0,
        "noise_w": 1.0,
        "description": "Maximum expressiveness"
    }
    # Add your own configurations here:
    # "my_custom_config": {
    #     "length_scale": 1.0,
    #     "noise_scale": 0.5,
    #     "noise_w": 0.6,
    #     "description": "My custom configuration"
    # }
}


async def test_configuration(
    tts_service: TTSService,
    config_name: str,
    params: Dict,
    test_phrase: str,
    output_dir: Path
) -> Tuple[str, float, float, int]:
    """Test a specific parameter configuration

    Returns:
        Tuple of (config_name, synthesis_time, rtf, audio_duration)
    """
    try:
        # Synthesize with specific parameters
        audio_data = await tts_service.synthesize(
            test_phrase,
            length_scale=params["length_scale"],
            noise_scale=params["noise_scale"],
            noise_w=params["noise_w"],
            output_format="wav"  # Use WAV for quality testing
        )

        if not audio_data.audio_bytes:
            print(f"{config_name}: Synthesis failed")
            return (config_name, 0.0, 0.0, 0)

        # Save audio file
        safe_phrase = test_phrase[:30].replace(" ", "_").replace(".", "").replace(",", "")
        filename = f"{config_name}_{safe_phrase}.wav"
        filepath = output_dir / filename

        with open(filepath, 'wb') as f:
            f.write(audio_data.audio_bytes)

        # Calculate metrics
        synthesis_time_s = audio_data.synthesis_time_ms / 1000.0
        audio_duration_s = audio_data.duration_ms / 1000.0
        rtf = synthesis_time_s / audio_duration_s if audio_duration_s > 0 else 0.0

        return (config_name, synthesis_time_s, rtf, audio_data.duration_ms)

    except Exception as e:
        print(f"{config_name}: Error - {e}")
        return (config_name, 0.0, 0.0, 0)


async def run_fine_tuning():
    """Run fine-tuning tests on all parameter configurations"""
    print("=" * 80)
    print("TTS SYNTHESIS PARAMETER FINE-TUNING")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path("output/tts_finetuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    print()

    # Initialize TTS service
    print("🔧 Loading TTS model...")
    tts_service = TTSService()

    if not await tts_service.load_model():
        print("Failed to load TTS model")
        return 1

    print(f"Model loaded: {tts_config.MODEL_PATH}")
    print(f"Voice: {tts_config.VOICE}")
    print(f"Sample rate: {tts_config.SAMPLE_RATE}Hz")
    print()

    # Test each configuration
    results = {}

    for phrase_idx, test_phrase in enumerate(TEST_PHRASES, 1):
        print(f"\n{'='*80}")
        print(f"TEST PHRASE {phrase_idx}/{len(TEST_PHRASES)}")
        print(f"{'='*80}")
        print(f"Text: \"{test_phrase}\"")
        print()

        phrase_results = []

        for config_name, params in PARAM_CONFIGS.items():
            print(f"Testing '{config_name}': {params['description']}")
            print(f"  Parameters: length={params['length_scale']}, "
                  f"noise={params['noise_scale']}, noise_w={params['noise_w']}")

            result = await test_configuration(
                tts_service,
                config_name,
                params,
                test_phrase,
                output_dir
            )

            config_name, synth_time, rtf, duration_ms = result

            if duration_ms > 0:
                print(f"Synthesis: {synth_time:.3f}s | "
                      f"Audio: {duration_ms}ms | RTF: {rtf:.3f}")
                phrase_results.append({
                    "config": config_name,
                    "params": params,
                    "synthesis_time": synth_time,
                    "rtf": rtf,
                    "duration_ms": duration_ms
                })

            # Small delay between tests
            await asyncio.sleep(0.1)

        results[test_phrase] = phrase_results

    # Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Calculate average RTF for each config
    config_rtfs = {}
    for config_name in PARAM_CONFIGS.keys():
        rtfs = []
        for phrase_results in results.values():
            rtfs.extend(
                result["rtf"]
                for result in phrase_results
                if result["config"] == config_name
            )
        if rtfs:
            config_rtfs[config_name] = sum(rtfs) / len(rtfs)

    # Sort by RTF (performance)
    sorted_by_performance = sorted(config_rtfs.items(), key=lambda x: x[1])

    print("⚡ PERFORMANCE RANKING (by Real-Time Factor):")
    print()
    for rank, (config_name, avg_rtf) in enumerate(sorted_by_performance, 1):
        params = PARAM_CONFIGS[config_name]
        performance_rating = "Excellent" if avg_rtf < 0.3 else "Good" if avg_rtf < 0.5 else "Acceptable"
        print(f"{rank}. {config_name:20s} - RTF: {avg_rtf:.3f} {performance_rating}")
        print(f"   {params['description']}")
        print(f"   Parameters: length={params['length_scale']}, "
              f"noise={params['noise_scale']}, noise_w={params['noise_w']}")
        print()

    print("\nRECOMMENDATIONS:")
    print()
    print("1. BEST OVERALL (balanced quality & performance):")
    best_balanced = sorted_by_performance[len(sorted_by_performance)//2]  # Middle ground
    print(f"   → {best_balanced[0]}")
    print(f"   → {PARAM_CONFIGS[best_balanced[0]]['description']}")
    print()

    print("2. FASTEST (lowest latency):")
    fastest = sorted_by_performance[0]
    print(f"   → {fastest[0]} (RTF: {fastest[1]:.3f})")
    print(f"   → {PARAM_CONFIGS[fastest[0]]['description']}")
    print()

    print("3. FOR TESTING VOICE QUALITY:")
    print(f"   Listen to the WAV files in: {output_dir.absolute()}")
    print("    Compare expressiveness, naturalness, and clarity")
    print()

    print("4. SUGGESTED STARTING POINTS:")
    print()
    print("   • Natural conversation: 'natural_balanced'")
    print("     length_scale=1.0, noise_scale=0.5, noise_w=0.6")
    print()
    print("   • Robot assistant: 'calm_clear'")
    print("     length_scale=1.1, noise_scale=0.3, noise_w=0.4")
    print()
    print("   • Expressive responses: 'expressive'")
    print("     length_scale=0.95, noise_scale=0.8, noise_w=0.9")
    print()

    print("=" * 80)
    print("Fine-tuning complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run_fine_tuning())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
