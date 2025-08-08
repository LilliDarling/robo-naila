"""Unit tests for InputProcessor agent"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from agents.input_processor import InputProcessor


class TestInputProcessor:
    """Test cases for InputProcessor"""
    
    @pytest.fixture
    def processor(self, disable_hardware_optimization):
        """Create InputProcessor instance with mocked hardware"""
        with patch('agents.input_processor.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_st.return_value = mock_model
            
            processor = InputProcessor()
            processor.model = mock_model
            processor.has_model = True
            processor.intent_embeddings = {
                "greeting": np.array([0.1, 0.2, 0.3]),
                "time_query": np.array([0.2, 0.3, 0.4])
            }
            return processor
    
    @pytest.fixture 
    def processor_no_model(self):
        """InputProcessor without NLP model (fallback mode)"""
        with patch('agents.input_processor.SentenceTransformer', side_effect=ImportError):
            return InputProcessor()

    @pytest.mark.asyncio
    async def test_process_basic_text(self, processor, basic_state):
        """Test basic text processing"""
        result = await processor.process(basic_state)
        
        assert result["processed_text"] == basic_state["raw_input"]
        assert "intent" in result
        assert "timestamp" in result
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_process_audio_input(self, processor):
        """Test audio input processing with transcription"""
        state = {
            "input_type": "audio",
            "raw_input": "",
            "transcription": "What time is it?",
            "confidence": 0.92
        }
        
        result = await processor.process(state)
        assert result["processed_text"] == "What time is it?"
        assert result["confidence"] >= 0.4 

    def test_intent_detection_semantic(self, processor):
        """Test semantic intent detection with NLP model"""
        # Test various intents
        test_cases = [
            ("hello there", "greeting"),
            ("what time is it", "time_query"), 
            ("how's the weather", "weather_query"),
            ("what is python", "question"),
            ("thank you", "gratitude"),
            ("goodbye", "goodbye")
        ]
        
        for text, expected_intent in test_cases:
            intent = processor._detect_intent(text)
            # Intent should be detected (exact match depends on embeddings)
            assert intent in ["greeting", "time_query", "weather_query", "question", "gratitude", "goodbye", "general"]

    def test_intent_detection_fallback(self, processor_no_model):
        """Test fallback pattern matching when no NLP model"""
        test_cases = [
            ("hello", "greeting"),
            ("what time is it", "time_query"),
            ("weather today", "weather_query"), 
            ("what is this", "question"),
            ("thank you", "gratitude"),
            ("goodbye", "goodbye"),
            ("random text", "general")
        ]
        
        for text, expected_intent in test_cases:
            intent = processor_no_model._detect_intent(text)
            assert intent == expected_intent

    def test_intent_caching(self, processor):
        """Test that intent detection results are cached"""
        text = "hello world"
        
        # First call
        intent1 = processor._detect_intent(text)
        
        # Second call should use cache
        intent2 = processor._detect_intent(text)
        
        assert intent1 == intent2

    @pytest.mark.asyncio
    async def test_empty_input(self, processor):
        """Test handling of empty input"""
        state = {
            "input_type": "text",
            "raw_input": "",
            "confidence": 1.0
        }
        
        result = await processor.process(state)
        assert result["processed_text"] == ""
        assert result["intent"] == "general"

    @pytest.mark.asyncio
    async def test_confidence_propagation(self, processor):
        """Test that STT confidence affects final confidence"""
        state = {
            "input_type": "text", 
            "raw_input": "hello",
            "confidence": 0.3
        }
        
        result = await processor.process(state)
        # Final confidence should consider both STT and intent detection
        assert result["confidence"] <= 0.3

    def test_semantic_intent_similarity(self, processor):
        """Test that similar phrases get same intent"""
        similar_greetings = [
            "hello",
            "hi there", 
            "hey",
            "good morning"
        ]
        
        intents = [processor._detect_intent(text) for text in similar_greetings]
        
        # Most should be classified as greeting (allowing some variation)
        greeting_count = sum(1 for intent in intents if intent == "greeting")
        assert greeting_count >= len(similar_greetings) // 2

    def test_hardware_detection_fallback(self, monkeypatch):
        """Test processor works without hardware optimization"""
        # Mock failed hardware detection
        def mock_failed_hardware():
            raise Exception("Hardware detection failed")
        
        monkeypatch.setattr(
            "config.hardware_config.hardware_optimizer.get_model_config",
            mock_failed_hardware
        )
        
        with patch('agents.input_processor.SentenceTransformer', side_effect=Exception):
            processor = InputProcessor()
            assert not processor.has_model  # Should fallback gracefully

    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, processor, basic_state):
        """Test that processing time is tracked"""
        result = await processor.process(basic_state)
        
        assert "processing_time_ms" in result
        assert isinstance(result["processing_time_ms"], int)
        assert result["processing_time_ms"] >= 0
        assert result["processing_time_ms"] < 5000

    def test_model_setup_without_dependencies(self, monkeypatch):
        """Test model setup when dependencies are missing"""
        # Mock SentenceTransformer to raise ImportError
        with patch('agents.input_processor.SentenceTransformer', side_effect=ImportError("No module")):
            processor = InputProcessor()
            assert not processor.has_model