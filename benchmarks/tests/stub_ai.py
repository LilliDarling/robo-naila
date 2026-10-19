"""Integration-only native model doubles; real services, graph and gRPC remain."""
import asyncio
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import grpc
import numpy as np
import telemetry as otel
from agents.input_processor import InputProcessor
from agents.orchestrator import NAILAOrchestrator
from memory.conversation import ConversationMemory
from rpc.generated import naila_pb2_grpc
from rpc.service import NailaAIServicer
from services.llm import LLMService
from services.stt import STTService
from services.tts import TTSService


class Whisper:
    def transcribe(self, audio, **options):
        assert len(audio) > 0
        segment = SimpleNamespace(start=0, end=1, text='Explain why leaves are green.', avg_logprob=-.01)
        return iter([segment]), SimpleNamespace(language='en')


class Llama:
    def __call__(self, prompt, **options):
        assert options['stream']
        yield {'choices': [{'text': 'Leaves contain chlorophyll. '}]}
        yield {'choices': [{'text': 'It absorbs light.'}]}


class Kokoro:
    def create(self, text, **options):
        return (np.sin(np.arange(12000) * 2 * np.pi * 400 / 24000) * .1).astype(np.float32), 24000


async def main(ready_path):
    # No transformer download; exercise the existing optional-model fallback.
    def unavailable(self):
        raise ImportError('integration fixture has no intent model')
    InputProcessor._load_transformer = unavailable
    provider = otel.init()
    stt, llm, tts = STTService(), LLMService(), TTSService()
    for service, model in ((stt, Whisper()), (llm, Llama()), (tts, Kokoro())):
        service.is_ready, service.model = True, model
    llm.system_prompt = await llm._load_system_prompt()
    memory = ConversationMemory(':memory:')
    servicer = NailaAIServicer()
    servicer.set_stt_service(stt)
    servicer.set_orchestrator(NAILAOrchestrator(memory, llm_service=llm, tts_service=tts))
    status = {'models_loaded': True, **{name: dict(ready=True, model_path='stub-' + name, hardware={'device_type': 'cpu'})
                                       for name in ('stt', 'llm', 'tts')}}
    servicer.set_ai_model_manager(SimpleNamespace(get_status=lambda: status))
    server = grpc.aio.server()
    naila_pb2_grpc.add_NailaAIServicer_to_server(servicer, server)
    port = server.add_insecure_port('127.0.0.1:0')
    await server.start()
    Path(ready_path).write_text(json.dumps({'port': port}))
    stopped = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stopped.set)
    try:
        await stopped.wait()
    finally:
        await server.stop(2)
        memory.close()
        assert await asyncio.to_thread(otel.shutdown, provider)


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1]))
