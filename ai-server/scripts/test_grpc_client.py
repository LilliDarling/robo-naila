"""Simple gRPC test client for StreamConversation.

Simulates a voice utterance by sending synthetic PCM audio through
the full pipeline: gRPC -> STT -> LangGraph orchestration -> TTS -> gRPC.

Usage:
    uv run python scripts/test_grpc_client.py
"""

import asyncio
import struct
import math
import sys
from pathlib import Path

# Ensure ai-server root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc.aio

from rpc.generated import naila_pb2, naila_pb2_grpc


GRPC_ADDRESS = "localhost:50051"
SAMPLE_RATE = 16000
DURATION_S = 2.0


def generate_sine_pcm(freq_hz: float = 440.0, duration_s: float = DURATION_S, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate a sine wave as signed 16-bit PCM bytes."""
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * freq_hz * t))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


async def run_test():
    print(f"Connecting to gRPC server at {GRPC_ADDRESS}...")
    channel = grpc.aio.insecure_channel(GRPC_ADDRESS)
    stub = naila_pb2_grpc.NailaAIStub(channel)

    pcm_data = generate_sine_pcm()
    chunk_size = SAMPLE_RATE * 2  # 1 second of 16-bit mono = 32000 bytes
    chunks = [pcm_data[i : i + chunk_size] for i in range(0, len(pcm_data), chunk_size)]

    print(f"Generated {len(pcm_data)} bytes of test audio ({DURATION_S}s @ {SAMPLE_RATE}Hz)")
    print(f"Sending as {len(chunks)} chunk(s) with SPEECH_EVENT flow...\n")

    async def request_stream():
        # START
        yield naila_pb2.AudioInput(
            device_id="test-device",
            conversation_id="test-conv-001",
            audio_pcm=chunks[0],
            codec=naila_pb2.AUDIO_CODEC_PCM_S16LE,
            sample_rate=SAMPLE_RATE,
            event=naila_pb2.SPEECH_EVENT_START,
        )
        print("  -> Sent SPEECH_EVENT_START")

        # CONTINUE (remaining chunks)
        for i, chunk in enumerate(chunks[1:], 1):
            yield naila_pb2.AudioInput(
                device_id="test-device",
                conversation_id="test-conv-001",
                audio_pcm=chunk,
                codec=naila_pb2.AUDIO_CODEC_PCM_S16LE,
                sample_rate=SAMPLE_RATE,
                event=naila_pb2.SPEECH_EVENT_CONTINUE,
            )
            print(f"  -> Sent SPEECH_EVENT_CONTINUE (chunk {i})")

        # END
        yield naila_pb2.AudioInput(
            device_id="test-device",
            conversation_id="test-conv-001",
            event=naila_pb2.SPEECH_EVENT_END,
        )
        print("  -> Sent SPEECH_EVENT_END")
        print("\nWaiting for response...\n")

    # Call StreamConversation
    try:
        response_stream = stub.StreamConversation(request_stream())

        total_audio_bytes = 0
        chunk_count = 0
        async for response in response_stream:
            chunk_count += 1

            if response.error_code != naila_pb2.ERROR_NONE:
                print(f"  ERROR [{response.error_code}]: {response.error_message}")
                break

            audio_len = len(response.audio_pcm) if response.audio_pcm else 0
            total_audio_bytes += audio_len

            if response.final_stt:
                print(f"  STT transcription: \"{response.final_stt}\"")
            if audio_len > 0:
                print(f"  <- AudioOutput chunk #{response.sequence_num}: {audio_len} bytes @ {response.sample_rate}Hz (final={response.is_final})")
            elif response.is_final:
                print(f"  <- Final message (no audio)")

        print(f"\n--- Results ---")
        print(f"Total response chunks: {chunk_count}")
        print(f"Total audio bytes received: {total_audio_bytes}")
        if total_audio_bytes > 0 and chunk_count > 0:
            print(f"Approx TTS duration: {total_audio_bytes / (22050 * 2) * 1000:.0f}ms")

    except grpc.aio.AioRpcError as e:
        print(f"gRPC error: {e.code()} - {e.details()}")
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(run_test())
