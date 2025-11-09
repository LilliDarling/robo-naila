"""Base agent class for LangGraph agents"""

from typing import Any, Dict
from abc import ABC, abstractmethod
from utils import get_logger


class BaseAgent(ABC):
    """Base class for all NAILA agents"""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger("agents.{}".format(name))

    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state and return updated state"""
        pass

    def log_state(self, state: Dict[str, Any], phase: str = "processing"):
        """Log current state for debugging"""
        self.logger.debug(
            "agent_state",
            agent=self.name,
            phase=phase,
            device_id=state.get('device_id'),
            task_id=state.get('task_id')
        )