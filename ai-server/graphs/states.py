"""State definitions for LangGraph workflows"""

from typing import TypedDict, List, Dict, Optional, Any


class NAILAState(TypedDict):
    """Core state for NAILA orchestration graph"""
    
    # Input data
    device_id: str
    task_id: str
    input_type: str  # "text", "audio", "vision"
    raw_input: Any
    
    # Processing
    processed_text: Optional[str]
    intent: Optional[str]
    confidence: float
    context: Dict[str, Any]
    
    # Response
    response_text: Optional[str]
    
    # Memory
    conversation_history: List[Dict[str, str]]
    
    # Metadata
    timestamp: str
    errors: List[str]