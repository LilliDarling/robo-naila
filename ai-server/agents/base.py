"""Base agent class for LangGraph agents"""

import logging
from typing import Any, Dict
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all NAILA agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agents.{name}")
    
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state and return updated state"""
        pass
    
    def log_state(self, state: Dict[str, Any], phase: str = "processing"):
        """Log current state for debugging"""
        self.logger.debug(f"{self.name} {phase}: device={state.get('device_id')}, task={state.get('task_id')}")