"""Context-aware response generation agent"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import OrderedDict
from agents.base import BaseAgent


class ResponseGenerator(BaseAgent):
    """Generate context-aware responses with conversation continuity"""

    def __init__(self, llm_service=None, tts_service=None):
        super().__init__("response_generator")
        # Use OrderedDict for LRU cache with O(1) operations
        self._response_cache = OrderedDict()
        self._cache_ttl = 600  # 10 minutes
        self._max_cache_size = 200
        self.llm_service = llm_service
        self.tts_service = tts_service
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate context-aware response"""
        start_time = time.time()
        self.log_state(state, "generating")

        intent = state.get("intent", "general")
        processed_text = state.get("processed_text", "")
        context = state.get("context", {})
        conversation_history = state.get("conversation_history", [])
        confidence = state.get("confidence", 1.0)
        device_id = state.get("device_id", "")
        visual_context = state.get("visual_context")

        # Check LLM readiness dynamically
        use_llm = self.llm_service is not None and self.llm_service.is_ready

        # Generate context-aware response
        response = await self._generate_response(
            intent, processed_text, context, conversation_history, confidence, use_llm=use_llm, visual_context=visual_context
        )
        
        # Add response metadata
        response_metadata = {
            "intent": intent,
            "confidence": confidence,
            "generation_time_ms": int((time.time() - start_time) * 1000),
            "context_used": bool(context.get("recent_exchanges")),
            "device_id": device_id
        }
        
        # Update conversation history
        history_entry = {
            "user": processed_text,
            "assistant": response,
            "timestamp": state.get("timestamp", ""),
            "metadata": response_metadata
        }
        
        conversation_history.append(history_entry)
        
        # Update state
        state["response_text"] = response
        state["conversation_history"] = conversation_history[-10:]  # Keep last 10
        state["response_metadata"] = response_metadata

        # Synthesize audio response if TTS is available
        if self.tts_service and self.tts_service.is_ready:
            try:
                audio_data = await self.tts_service.synthesize(response)
                state["response_audio"] = audio_data
                self.logger.debug(
                    "audio_synthesized",
                    duration_ms=audio_data.duration_ms,
                    format=audio_data.format
                )
            except Exception as e:
                self.logger.warning("tts_synthesis_failed", error=str(e), error_type=type(e).__name__)
                # Continue without audio - text response is still available

        self.logger.info(
            "response_generated",
            response=response,
            confidence=round(confidence, 2),
            generation_time_ms=response_metadata['generation_time_ms']
        )
        return state
    
    async def _generate_response(
        self,
        intent: str,
        text: str,
        context: Dict[str, Any],
        history: List[Dict],
        confidence: float,
        use_llm: bool = False,
        visual_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate context-aware response with conversation continuity"""

        # Handle visual query responses
        if visual_context:
            if result := visual_context.get("answer") or visual_context.get("description"):
                return result

        # Handle low confidence inputs
        if confidence < 0.6:
            return self._generate_clarification_response(text, confidence)

        # Try LLM generation first if available
        if use_llm:
            try:
                response = await self._generate_llm_response(text, history, visual_context=visual_context)
                if response:
                    self.logger.info("llm_response_used")
                    return response
            except Exception as e:
                self.logger.warning("llm_generation_failed", error=str(e), error_type=type(e).__name__, fallback="pattern-based")

        # Fallback to pattern-based responses
        self.logger.debug("pattern_response_used")

        # Check for conversation continuity
        if history:
            last_exchange = history[-1]
            last_intent = last_exchange.get("metadata", {}).get("intent", "")

            # Handle follow-up questions
            if intent == "question" and last_intent in ["time_query", "weather_query"]:
                return self._generate_followup_response(intent, text, last_intent, last_exchange)

        # Use cached response for repeated queries (speed optimization)
        cache_key = f"{intent}:{text.lower().strip()}"
        if cached_response := self._get_cached_response(cache_key):
            return self._personalize_response(cached_response, context)

        # Generate new response
        response = self._generate_base_response(intent, text, context, history)
        self._cache_response(cache_key, response)

        # Apply personalization to new responses too
        return self._personalize_response(response, context)

    async def _generate_llm_response(
        self,
        query: str,
        history: List[Dict],
        timeout: float = 10.0,
        max_retries: int = 3,
        visual_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response using LLM with timeout and retry logic"""
        if not self.llm_service or not self.llm_service.is_ready:
            return ""

        # Augment query with visual context if available
        augmented_query = query
        if visual_context:
            visual_info = f"Visual context: {visual_context.get('description', '')}"
            if visual_context.get("object_counts"):
                visual_info += f" Objects detected: {visual_context['object_counts']}"
            augmented_query = f"{visual_info}\n\nUser query: {query}"

        # Build chat messages with history
        messages = self.llm_service.build_chat_messages(augmented_query, history)

        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.llm_service.generate_chat(messages),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as e:
                last_exception = e
                self.logger.warning("llm_timeout", attempt=attempt, max_retries=max_retries)
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    "llm_error",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(last_exception),
                    error_type=type(last_exception).__name__
                )
        # If all retries fail, log the final failure
        self.logger.error("llm_failed_all_retries", max_retries=max_retries, error=str(last_exception), error_type=type(last_exception).__name__)
        return ""

    def _generate_clarification_response(self, text: str, confidence: float) -> str:
        """Generate response for low-confidence input"""
        return f"I didn't catch that clearly (confidence: {confidence:.1f}). Could you repeat that or rephrase?"
    
    def _generate_followup_response(self, intent: str, text: str, last_intent: str, last_exchange: Dict) -> str:
        """Generate contextual follow-up responses"""
        if last_intent == "time_query" and "zone" in text.lower():
            return "I'm using the local timezone. I don't have access to other time zones yet."
        elif last_intent == "weather_query" and any(w in text.lower() for w in ["tomorrow", "next", "later"]):
            return "I can't provide weather forecasts yet, only current conditions when available."
        else:
            return f"Following up on our previous topic, {text.lower()}. Let me help with that."
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response with TTL using LRU eviction

        Uses OrderedDict for O(1) get/set/delete operations.
        """
        if cache_key not in self._response_cache:
            return None

        cached_time, response = self._response_cache[cache_key]
        current_time = time.time()

        # Check if expired
        if current_time - cached_time >= self._cache_ttl:
            del self._response_cache[cache_key]
            return None

        # Move to end (mark as recently used)
        self._response_cache.move_to_end(cache_key)
        return response

    def _cache_response(self, cache_key: str, response: str):
        """Cache response with TTL and LRU eviction

        Uses OrderedDict for efficient LRU cache:
        - O(1) insertion
        - O(1) eviction of oldest item
        - No sorting required
        """
        current_time = time.time()

        # If key exists, update and move to end
        if cache_key in self._response_cache:
            self._response_cache[cache_key] = (current_time, response)
            self._response_cache.move_to_end(cache_key)
            return

        # Evict oldest items if cache is full (LRU eviction)
        if len(self._response_cache) >= self._max_cache_size:
            # Remove oldest 25% of items (scaled with max cache size) in O(1) per item
            eviction_count = max(1, self._max_cache_size // 4)
            # Ensure we don't try to evict more items than currently in the cache
            eviction_count = min(eviction_count, len(self._response_cache))

            for _ in range(eviction_count):
                # popitem(last=False) removes the least recently used (oldest) item in O(1)
                self._response_cache.popitem(last=False)

        # Add new entry
        self._response_cache[cache_key] = (current_time, response)
    
    def _personalize_response(self, base_response: str, context: Dict[str, Any]) -> str:
        """Add personalization based on context"""
        history_count = context.get("history_count", 0)
        
        if history_count > 5:
            # Long conversation - be more casual
            return base_response.replace("How can I help you", "What else can I do for you")
        elif history_count == 0:
            # First interaction - be welcoming
            return base_response.replace("Hello!", "Hello there! Nice to meet you.")
        
        return base_response
    
    def _generate_base_response(self, intent: str, text: str, context: Dict, history: List) -> str:
        """Generate base response for intent"""
        responses = {
            "greeting": self._generate_greeting_response(context, history),
            "time_query": f"The current time is {datetime.now().strftime('%I:%M %p')}",
            "weather_query": "I don't have weather data access yet, but I'm working on it!",
            "question": self._generate_question_response(text, context),
            "gratitude": self._generate_gratitude_response(history),
            "goodbye": self._generate_goodbye_response(context),
            "general": "I'm here to help. What would you like to know?"
        }
        return responses.get(intent, "I'm processing your request.")
    
    def _generate_greeting_response(self, context: Dict, history: List) -> str:
        """Context-aware greeting"""
        if history:
            return "Hello again! What can I help you with?"
        return "Hello! How can I help you today?"
    
    def _generate_question_response(self, text: str, context: Dict) -> str:
        """Context-aware question response"""
        if "you" in text.lower() and any(w in text.lower() for w in ["who", "what", "how"]):
            return "I'm NAILA, your desk AI assistant. I can help with questions, provide information, and chat with you."
        return f"That's an interesting question about '{text}'. Let me think about that."
    
    def _generate_gratitude_response(self, history: List) -> str:
        """Respond to thanks based on recent help"""
        if history and any(w in history[-1].get("assistant", "") for w in ["time", "weather", "help"]):
            return "You're welcome! Happy to help with anything else."
        return "You're welcome! Is there anything else I can do for you?"
    
    def _generate_goodbye_response(self, context: Dict) -> str:
        """Contextual goodbye"""
        return "Goodbye! It was nice talking with you. Feel free to chat anytime!"