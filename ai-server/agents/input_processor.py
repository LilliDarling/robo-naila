"""Input processing agent with NLP libraries"""

import time
from typing import Any, Dict, Tuple
from functools import lru_cache
from datetime import datetime, timezone
from agents.base import BaseAgent

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class InputProcessor(BaseAgent):
    """Process raw inputs with NLP intent detection"""
    
    def __init__(self):
        super().__init__("input_processor")
        self.model = None
        self.intent_embeddings = {}
        self.has_model = False
        self._setup_model()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process input based on type"""
        import time
        start_time = time.time()
        
        self.log_state(state, "start")
        
        input_type = state.get("input_type", "text")
        raw_input = state.get("raw_input", "")
        
        # Process based on input type
        if input_type == "text":
            processed_text = raw_input.strip()
        elif input_type == "audio":
            # Placeholder for STT result
            processed_text = state.get("transcription", raw_input)
        else:
            processed_text = str(raw_input)
        
        # Basic intent detection
        intent = self._detect_intent(processed_text)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Update state
        state["processed_text"] = processed_text
        state["intent"] = intent
        state["timestamp"] = datetime.now(timezone.utc).isoformat()
        state["processing_time_ms"] = processing_time_ms
        
        self.logger.info(f"Processed: '{processed_text}' -> intent: {intent} ({processing_time_ms}ms)")
        return state
    
    def _setup_model(self):
        """Setup lightweight NLP model with hardware optimization"""
        try:
            self._load_transformer()
        except ImportError:
            self.logger.info("sentence-transformers not available, using fallback patterns")
            self.has_model = False
        except Exception as e:
            self.logger.warning(f"Failed to load model: {e}, using fallback")
            self.has_model = False

    def _load_transformer(self):
        from sentence_transformers import SentenceTransformer
        from config.hardware_config import hardware_optimizer

        # Get optimal hardware configuration
        model_config = hardware_optimizer.get_model_config("sentence_transformer")
        device = model_config["device"]

        # Log hardware info on first setup
        if not hasattr(self, '_hardware_logged'):
            hardware_optimizer.log_hardware_info()
            self._hardware_logged = True

        # Use small, fast model optimized for semantic similarity
        self.model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=device
        )

        self._setup_intent_embeddings()
        self.has_model = True
        self.logger.info(f"Loaded sentence transformer on {device} for intent detection")
    
    
    def _setup_intent_embeddings(self):
        """Pre-compute intent embeddings for fast comparison"""
        import numpy as np
        
        intent_examples = {
            "greeting": ["hello", "hi there", "hey", "good morning", "good afternoon"],
            "time_query": ["what time is it", "current time", "tell me the time", "what's the time"],
            "weather_query": ["weather today", "how's the weather", "weather forecast", "is it raining"],
            "question": ["what is this", "how does it work", "why is that", "when will", "where is"],
            "gratitude": ["thank you", "thanks", "appreciate it", "thanks a lot"],
            "goodbye": ["goodbye", "bye", "see you later", "talk to you soon"],
            "general": ["tell me about", "I want to know", "can you help", "please do"]
        }
        
        self.intent_embeddings = {}
        for intent, examples in intent_examples.items():
            embeddings = self.model.encode(examples)
            # Use mean embedding as intent representation
            self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
    
    @lru_cache(maxsize=256)
    def _detect_intent(self, text: str) -> str:
        """Fast intent detection with NLP model + fallback patterns"""
        if not text.strip():
            return "general"
        
        # Try NLP model first (cached for speed)
        if self.has_model:
            return self._detect_intent_semantic(text)
        
        # Fallback to fast pattern matching
        return self._detect_intent_patterns(text)
    
    def _detect_intent_semantic(self, text: str) -> str:
        """Use sentence similarity for intent detection"""
        import numpy as np
        
        # Get text embedding
        text_embedding = self.model.encode([text.lower()])[0]
        
        # Find most similar intent
        best_intent = "general"
        best_score = 0.4
        
        for intent, intent_embedding in self.intent_embeddings.items():
            # Compute cosine similarity
            similarity = np.dot(text_embedding, intent_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(intent_embedding)
            )
            
            if similarity > best_score:
                best_score = similarity
                best_intent = intent
        
        return best_intent
    
    def _detect_intent_patterns(self, text: str) -> str:
        """Fast pattern-based fallback"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Check for specific greeting words (as separate words to avoid false matches)
        if any(word in ["hello", "hi", "hey"] for word in words):
            return "greeting"
        elif "time" in text_lower:
            return "time_query"
        elif "weather" in text_lower:
            return "weather_query"
        elif any(q in text_lower for q in ["what", "how", "why", "when", "where"]):
            return "question"
        elif any(thanks in text_lower for thanks in ["thank", "thanks"]):
            return "gratitude"
        elif any(bye in text_lower for bye in ["bye", "goodbye", "see you"]):
            return "goodbye"
        else:
            return "general"