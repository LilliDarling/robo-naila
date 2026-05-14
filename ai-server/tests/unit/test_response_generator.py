"""Unit tests for ResponseGenerator agent"""

import asyncio

import pytest
from unittest.mock import AsyncMock, Mock, patch
from freezegun import freeze_time
from agents.response_generator import ResponseGenerator


class TestResponseGenerator:
    """Test cases for ResponseGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create ResponseGenerator instance"""
        return ResponseGenerator()

    @pytest.mark.asyncio
    async def test_basic_response_generation(self, generator, basic_state):
        """Test basic response generation"""
        result = await generator.process(basic_state)
        
        assert "response_text" in result
        assert "response_metadata" in result
        assert result["response_text"] is not None
        assert len(result["response_text"]) > 0

    @pytest.mark.asyncio
    async def test_intent_based_responses(self, generator):
        """Test responses vary by intent"""
        # ``time_query`` is action-handled before response_generator runs
        # (see agents.actions.time_handler), so it's no longer represented
        # in the pattern dict. ``weather_query`` will join it in chunk 4.
        intents = ["greeting", "weather_query", "question", "gratitude", "goodbye"]
        responses = []
        
        for intent in intents:
            state = {
                "intent": intent,
                "processed_text": f"test {intent}",
                "context": {},
                "conversation_history": [],
                "confidence": 0.9,
                "device_id": "test"
            }
            
            result = await generator.process(state)
            responses.append(result["response_text"])
        
        # All responses should be different
        assert len(set(responses)) == len(responses)

    @pytest.mark.asyncio
    async def test_low_confidence_handling(self, generator):
        """Test handling of low confidence inputs.

        ResponseGenerator triggers a clarification when confidence is strictly
        below 0.3 (see _generate_response). 0.2 is firmly in the "low" zone.
        """
        state = {
            "intent": "general",
            "processed_text": "unclear mumbling",
            "context": {},
            "conversation_history": [],
            "confidence": 0.2,
            "device_id": "test"
        }

        result = await generator.process(state)

        # Should generate clarification request
        assert "didn't catch that" in result["response_text"].lower() or \
               "confidence" in result["response_text"].lower()

    @pytest.mark.asyncio
    async def test_conversation_continuity(self, generator):
        """Test follow-up responses based on conversation history"""
        # Setup previous time query
        history = [{
            "user": "What time is it?",
            "assistant": "The current time is 10:30 AM", 
            "timestamp": "2025-01-15T10:30:00Z",
            "metadata": {"intent": "time_query"}
        }]
        
        state = {
            "intent": "question",
            "processed_text": "What time zone is that?",
            "context": {"history_count": 1},
            "conversation_history": history,
            "confidence": 0.9,
            "device_id": "test"
        }
        
        result = await generator.process(state)
        
        # Should reference timezone context
        assert "zone" in result["response_text"].lower()

    @pytest.mark.asyncio
    async def test_personalization_new_user(self, generator):
        """Test response personalization for new users"""
        state = {
            "intent": "greeting",
            "processed_text": "Hello",
            "context": {"history_count": 0},
            "conversation_history": [],
            "confidence": 0.9,
            "device_id": "new_user"
        }

        result = await generator.process(state)

        # Should be welcoming for new user
        assert "nice to meet you" in result["response_text"].lower() or \
               "hello there" in result["response_text"].lower()

    @pytest.mark.asyncio
    async def test_personalization_returning_user(self, generator):
        """Test response personalization for returning users"""
        state = {
            "intent": "greeting",
            "processed_text": "Hi again",
            "context": {"history_count": 5},
            "conversation_history": [{"user": "prev", "assistant": "prev"}],
            "confidence": 0.9,
            "device_id": "returning_user"
        }
        
        result = await generator.process(state)
        
        # Should acknowledge returning user
        assert "again" in result["response_text"].lower() or \
               "what else" in result["response_text"].lower()

    @pytest.mark.asyncio
    async def test_response_caching(self, generator):
        """Test response caching for identical inputs"""
        # Use ``greeting`` rather than ``time_query`` — the latter is now
        # action-handled and never reaches the pattern-cache codepath.
        # Deep-copy so the two calls don't share the conversation_history list
        # (the first ``process`` mutates it via append, which would shift
        # personalization behavior on the second call).
        import copy

        state1 = {
            "intent": "greeting",
            "processed_text": "hello there",
            "context": {},
            "conversation_history": [],
            "confidence": 0.9,
            "device_id": "test"
        }

        state2 = copy.deepcopy(state1)

        result1 = await generator.process(state1)
        result2 = await generator.process(state2)

        # Responses should be similar (may have timestamp differences)
        assert result1["response_text"].split()[0:3] == result2["response_text"].split()[0:3]

    # Removed ``test_time_query_accuracy`` — time formatting is now the time
    # handler's responsibility (see tests/unit/test_time_handler.py).

    @pytest.mark.asyncio
    async def test_conversation_history_update(self, generator):
        """Test conversation history is properly updated"""
        initial_history = [{"user": "old", "assistant": "old"}]
        
        state = {
            "intent": "greeting",
            "processed_text": "Hello",
            "context": {},
            "conversation_history": initial_history.copy(),
            "confidence": 0.9,
            "device_id": "test",
            "timestamp": "2025-01-15T10:30:00Z"
        }
        
        result = await generator.process(state)
        
        # History should be updated
        assert len(result["conversation_history"]) == 2
        new_exchange = result["conversation_history"][-1]
        assert new_exchange["user"] == "Hello"
        assert new_exchange["assistant"] == result["response_text"]
        assert new_exchange["timestamp"] == "2025-01-15T10:30:00Z"

    @pytest.mark.asyncio
    async def test_history_truncation(self, generator):
        """Test conversation history is truncated to limit"""
        # Create long history
        long_history = []
        for i in range(15):
            long_history.append({
                "user": f"message {i}",
                "assistant": f"response {i}",
                "timestamp": "2025-01-15T10:30:00Z"
            })
        
        state = {
            "intent": "general",
            "processed_text": "new message",
            "context": {},
            "conversation_history": long_history,
            "confidence": 0.9,
            "device_id": "test"
        }
        
        result = await generator.process(state)
        
        # Should keep only last 10 exchanges
        assert len(result["conversation_history"]) == 10

    @pytest.mark.asyncio
    async def test_response_metadata(self, generator, basic_state):
        """Test response metadata is generated"""
        result = await generator.process(basic_state)
        
        metadata = result["response_metadata"]
        assert "intent" in metadata
        assert "confidence" in metadata
        assert "generation_time_ms" in metadata
        assert "context_used" in metadata
        assert "device_id" in metadata
        
        assert isinstance(metadata["generation_time_ms"], int)
        assert metadata["generation_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_gratitude_contextual_response(self, generator):
        """Test gratitude responses reference recent help"""
        history = [{
            "user": "What time is it?",
            "assistant": "The current time is 10:30 AM",
            "timestamp": "2025-01-15T10:30:00Z"
        }]
        
        state = {
            "intent": "gratitude",
            "processed_text": "Thank you",
            "context": {},
            "conversation_history": history,
            "confidence": 0.9,
            "device_id": "test"
        }
        
        result = await generator.process(state)
        
        # Should acknowledge the help provided
        assert "welcome" in result["response_text"].lower()
        assert "help" in result["response_text"].lower() or \
               "anything else" in result["response_text"].lower()

    def test_cache_management(self, generator):
        """Test response cache management and limits"""
        # Fill cache beyond limit
        for i in range(250):
            cache_key = f"test:{i}"
            generator._cache_response(cache_key, f"response {i}")

        # Cache should be limited
        assert len(generator._response_cache) <= 200


class TestTransportAudio:
    """Test transport-specific TTS behavior in ResponseGenerator.process()."""

    @pytest.fixture
    def mock_tts(self):
        tts = Mock()
        tts.is_ready = True
        audio_data = Mock()
        audio_data.duration_ms = 100
        audio_data.format = "raw"
        audio_data.audio_bytes = b"\x00" * 4410
        audio_data.sample_rate = 22050
        tts.synthesize = AsyncMock(return_value=audio_data)
        return tts

    @pytest.fixture
    def base_state(self):
        return {
            "intent": "greeting",
            "processed_text": "Hello",
            "context": {},
            "conversation_history": [],
            "confidence": 0.9,
            "device_id": "test",
        }

    @pytest.mark.asyncio
    async def test_grpc_transport_synthesizes_raw_audio(self, mock_tts, base_state):
        """gRPC transport always synthesizes audio with output_format='raw'."""
        audio_delivery = AsyncMock()
        generator = ResponseGenerator(tts_service=mock_tts)
        config = {"configurable": {"transport": "grpc", "audio_delivery": audio_delivery}}

        result = await generator.process(base_state, config=config)

        mock_tts.synthesize.assert_called_once()
        _, kwargs = mock_tts.synthesize.call_args
        assert kwargs.get("output_format") == "raw"
        audio_delivery.assert_called_once()
        call_args = audio_delivery.call_args
        assert call_args[1].get("is_final") is True or call_args[0][2] is True
        assert "response_audio" in result

    @pytest.mark.asyncio
    async def test_mqtt_text_input_skips_tts(self, mock_tts, base_state):
        """MQTT with text input should not call TTS."""
        base_state["input_type"] = "text"
        generator = ResponseGenerator(tts_service=mock_tts)
        config = {"configurable": {"transport": "mqtt"}}

        result = await generator.process(base_state, config=config)

        mock_tts.synthesize.assert_not_called()
        assert "response_audio" not in result

    @pytest.mark.asyncio
    async def test_mqtt_audio_input_synthesizes_audio(self, mock_tts, base_state):
        """MQTT with audio input should call TTS without raw format."""
        base_state["input_type"] = "audio"
        generator = ResponseGenerator(tts_service=mock_tts)
        config = {"configurable": {"transport": "mqtt"}}

        result = await generator.process(base_state, config=config)

        mock_tts.synthesize.assert_called_once()
        _, kwargs = mock_tts.synthesize.call_args
        assert kwargs.get("output_format") is None
        assert "response_audio" in result

    @pytest.mark.asyncio
    async def test_tts_failure_still_returns_text(self, mock_tts, base_state):
        """TTS failure should not prevent text response from being returned."""
        mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS crashed"))
        generator = ResponseGenerator(tts_service=mock_tts)
        config = {"configurable": {"transport": "grpc"}}

        result = await generator.process(base_state, config=config)

        assert result["response_text"]
        assert "response_audio" not in result


class TestLLMTimeout:
    """Test LLM timeout and retry semantics in _generate_llm_response."""

    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.is_ready = True
        llm.build_chat_messages = Mock(return_value=[{"role": "user", "content": "hi"}])
        llm.generate_chat = AsyncMock(return_value="LLM response")
        return llm

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_no_retry(self, mock_llm):
        """TimeoutError should return '' immediately without retrying."""
        mock_llm.generate_chat = AsyncMock(side_effect=asyncio.TimeoutError)
        generator = ResponseGenerator(llm_service=mock_llm)

        result = await generator._generate_llm_response("hello", [])

        assert result == ""
        # Should be called exactly once — no retry after timeout
        mock_llm.generate_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_timeout_error_retries(self, mock_llm):
        """Non-timeout errors should be retried up to max_retries."""
        mock_llm.generate_chat = AsyncMock(side_effect=RuntimeError("OOM"))
        generator = ResponseGenerator(llm_service=mock_llm)

        result = await generator._generate_llm_response("hello", [], max_retries=2)

        assert result == ""
        assert mock_llm.generate_chat.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self, mock_llm):
        """Should return successful response if retry succeeds."""
        mock_llm.generate_chat = AsyncMock(
            side_effect=[RuntimeError("transient"), "recovered response"]
        )
        generator = ResponseGenerator(llm_service=mock_llm)

        result = await generator._generate_llm_response("hello", [], max_retries=2)

        assert result == "recovered response"
        assert mock_llm.generate_chat.call_count == 2

    @pytest.mark.asyncio
    async def test_llm_not_ready_returns_empty(self):
        """Should return '' when LLM service is not ready."""
        llm = Mock()
        llm.is_ready = False
        generator = ResponseGenerator(llm_service=llm)

        result = await generator._generate_llm_response("hello", [])

        assert result == ""

class TestResponseTextShortCircuit:
    """When a prior graph node (``dispatch_action``) has already populated
    ``state["response_text"]``, the response generator must skip both the
    streaming and LLM paths entirely. The doc warns this short-circuit must
    happen before any streaming setup or the audio/is_final callbacks fire
    incorrectly on text that has no LLM stream behind it.
    """

    @pytest.fixture
    def llm_service(self):
        llm = Mock()
        llm.is_ready = True
        llm.generate_chat = AsyncMock(return_value="LLM should NOT have been called")
        llm.generate_chat_stream = AsyncMock()
        llm.build_chat_messages = Mock(return_value=[])
        return llm

    @pytest.mark.asyncio
    async def test_short_circuits_when_response_text_set(self, llm_service):
        """LLM never invoked when the action layer already provided a response."""
        gen = ResponseGenerator(llm_service=llm_service)
        state = {
            "intent": "time_query",
            "processed_text": "what time is it",
            "response_text": "It's 3:47 PM.",
            "action_handled": True,
            "context": {},
            "conversation_history": [],
            "confidence": 0.9,
            "device_id": "test",
        }

        result = await gen.process(state)

        llm_service.generate_chat.assert_not_called()
        llm_service.generate_chat_stream.assert_not_called()
        assert result["response_text"] == "It's 3:47 PM."

    @pytest.mark.asyncio
    async def test_short_circuit_does_not_enter_streaming_path(self, llm_service):
        """The streaming path delivers audio sentence-by-sentence with
        ``is_final`` markers. If we entered it for action text, the callback
        would fire on text that has no underlying LLM stream — corrupting the
        ``is_final`` lookahead."""
        gen = ResponseGenerator(llm_service=llm_service)

        delivered = []

        async def audio_delivery(audio_data, text, is_final):
            delivered.append((text, is_final))

        config = {
            "configurable": {
                "transport": "grpc",
                "audio_delivery": audio_delivery,
            }
        }

        state = {
            "intent": "time_query",
            "processed_text": "what time is it",
            "response_text": "It's 3:47 PM.",
            "action_handled": True,
            "context": {},
            "conversation_history": [],
            "confidence": 0.9,
            "device_id": "test",
            "input_type": "audio",
        }

        # Stub TTS so the test doesn't pull a real model.
        gen.tts_service = Mock()
        gen.tts_service.is_ready = True
        fake_audio = Mock(duration_ms=500, format="raw")
        gen.tts_service.synthesize = AsyncMock(return_value=fake_audio)

        await gen.process(state, config=config)

        # Action text gets delivered exactly once with is_final=True — the
        # non-streaming TTS path. Streaming would have fired multiple chunks.
        assert len(delivered) == 1
        assert delivered[0] == ("It's 3:47 PM.", True)

    @pytest.mark.asyncio
    async def test_short_circuit_records_metadata(self, llm_service):
        """Even on the action path we want response_metadata so the orchestrator
        can persist it (intent, streamed=False, etc.)."""
        gen = ResponseGenerator(llm_service=llm_service)
        state = {
            "intent": "time_query",
            "processed_text": "what time is it",
            "response_text": "It's 3:47 PM.",
            "action_handled": True,
            "context": {},
            "conversation_history": [],
            "confidence": 0.9,
            "device_id": "test",
        }

        result = await gen.process(state)

        meta = result["response_metadata"]
        assert meta["intent"] == "time_query"
        assert meta["streamed"] is False

    @pytest.mark.asyncio
    async def test_no_short_circuit_when_response_text_unset(self, llm_service):
        """Sanity: the short-circuit must only fire when response_text is set.
        Normal LLM/pattern path runs otherwise."""
        gen = ResponseGenerator(llm_service=None)  # no LLM → falls to pattern
        state = {
            "intent": "greeting",
            "processed_text": "hi",
            "context": {},
            "conversation_history": [],
            "confidence": 0.9,
            "device_id": "test",
        }

        result = await gen.process(state)

        # Pattern path produced a non-empty response.
        assert result["response_text"]
        # And we explicitly did not short-circuit something pre-set.
        assert result["response_text"] != "It's 3:47 PM."
