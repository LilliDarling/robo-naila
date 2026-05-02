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

    # Vision processing
    image_data: Optional[bytes]  # Raw image data if provided
    visual_context: Optional[Dict[str, Any]]  # Vision analysis results

    # Response
    response_text: Optional[str]
    response_metadata: Dict[str, Any]

    # Memory
    conversation_history: List[Dict[str, Any]]

    # Metadata
    timestamp: str
    errors: List[str]