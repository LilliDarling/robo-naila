"""Context-aware response generation agent"""

import time
from typing import Any, Dict, List, Optional
from functools import lru_cache
from datetime import datetime
from agents.base import BaseAgent


class ResponseGenerator(BaseAgent):
    """Generate context-aware responses with conversation continuity"""
    
    def __init__(self):
        super().__init__("response_generator")
        self._response_cache = {}
        self._cache_ttl = 600  # 10 minutes
    
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
        
        # Generate context-aware response
        response = self._generate_response(
            intent, processed_text, context, conversation_history, confidence
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
        
        self.logger.info(f"Generated: '{response}' (conf: {confidence:.2f}, {response_metadata['generation_time_ms']}ms)")
        return state
    
    def _generate_response(
        self, 
        intent: str, 
        text: str, 
        context: Dict[str, Any], 
        history: List[Dict], 
        confidence: float
    ) -> str:
        """Generate context-aware response with conversation continuity"""
        
        # Handle low confidence inputs
        if confidence < 0.6:
            return self._generate_clarification_response(text, confidence)
        
        # Check for conversation continuity
        if history:
            last_exchange = history[-1]
            last_intent = last_exchange.get("metadata", {}).get("intent", "")
            
            # Handle follow-up questions
            if intent == "question" and last_intent in ["time_query", "weather_query"]:
                return self._generate_followup_response(intent, text, last_intent, last_exchange)
        
        # Use cached response for repeated queries (speed optimization)
        cache_key = f"{intent}:{text.lower().strip()}"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return self._personalize_response(cached_response, context)
        
        # Generate new response
        response = self._generate_base_response(intent, text, context, history)
        self._cache_response(cache_key, response)
        
        # Apply personalization to new responses too
        return self._personalize_response(response, context)
    
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
    
    @lru_cache(maxsize=128)
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response with TTL"""
        if cache_key in self._response_cache:
            cached_time, response = self._response_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return response
            else:
                del self._response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache response with TTL"""
        # Limit cache size
        if len(self._response_cache) > 200:
            oldest_items = sorted(self._response_cache.items(), key=lambda x: x[1][0])[:50]
            for key, _ in oldest_items:
                del self._response_cache[key]
        
        self._response_cache[cache_key] = (time.time(), response)
    
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