"""High-performance conversation memory with background cleanup"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
from utils import get_logger


logger = get_logger(__name__)



class ConversationMemory:
    """High-performance conversation memory with background cleanup"""
    
    def __init__(self, max_history: int = 20, ttl_hours: int = 24):
        # Use deque for O(1) append/pop operations
        self.conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.device_metadata: Dict[str, Dict] = {}  # Track per-device stats
        self.max_history = max_history
        self.ttl = timedelta(hours=ttl_hours)
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # Cleanup every hour
        self._cleanup_task: Optional[asyncio.Task] = None
        self._active_devices: Set[str] = set()  # Track recently active devices
        
        # Thread safety for concurrent access
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        
        # Start background cleanup
        self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background cleanup task"""
        try:
            # Try to get running loop first (modern approach)
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._background_cleanup())
        except RuntimeError:
            # No running loop - cleanup will be manual
            # This is expected in test scenarios or non-async contexts
            logger.debug("background_cleanup_disabled", reason="no_running_event_loop")
            self._cleanup_task = None
        except Exception as e:
            # Other errors - cleanup will be manual
            logger.info("background_cleanup_disabled", error=str(e), error_type=type(e).__name__)
            self._cleanup_task = None
    
    async def _background_cleanup(self):
        """Background task for memory cleanup"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                self.cleanup_old_conversations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("background_cleanup_error", error=str(e), error_type=type(e).__name__)
    
    def add_exchange(self, device_id: str, user_msg: str, assistant_msg: str, metadata: Dict):
        """Add a conversation exchange with thread safety"""
        current_time = time.time()
        
        exchange = {
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": datetime.now().isoformat(),
            "timestamp_unix": current_time,  # For fast comparisons
            "metadata": metadata or {}
        }
        
        with self._lock:
            # Use deque for O(1) append - automatically handles max length
            self.conversations[device_id].append(exchange)
            
            # Track device activity and metadata
            self._active_devices.add(device_id)
            self._update_device_metadata(device_id, current_time)

        logger.debug("exchange_added", device_id=device_id, history_size=len(self.conversations[device_id]))
    
    def _update_device_metadata(self, device_id: str, current_time: float):
        """Update device metadata for analytics"""
        if device_id not in self.device_metadata:
            self.device_metadata[device_id] = {
                "first_seen": current_time,
                "last_active": current_time,
                "total_exchanges": 0,
                "total_sessions": 1
            }
        else:
            metadata = self.device_metadata[device_id]
            
            # Detect new session (gap > 30 minutes)
            if current_time - metadata["last_active"] > 1800:
                metadata["total_sessions"] += 1
            
            metadata["last_active"] = current_time
        
        self.device_metadata[device_id]["total_exchanges"] += 1
    
    def get_history(self, device_id: str, limit: int) -> List[Dict]:
        """Get conversation history for a device - O(1) or O(limit) performance"""
        if conversation := self.conversations.get(device_id):
            return (
                list(conversation)[-limit:]
                if limit and limit < len(conversation)
                else list(conversation)
            )
        else:
            return []
    
    def get_context(self, device_id: str) -> Dict:
        """Get optimized conversation context for a device with thread safety"""
        with self._lock:
            conversation = self.conversations.get(device_id)
            device_meta = self.device_metadata.get(device_id, {}).copy()  # Copy to avoid mutations

            context = {
                "device_id": device_id,
                "history_count": len(conversation) if conversation else 0,
                "total_exchanges": device_meta.get("total_exchanges", 0),
                "total_sessions": device_meta.get("total_sessions", 0),
                "is_returning_user": device_meta.get("total_exchanges", 0) > 1
            }
            
            # Only get recent exchanges if needed (lazy loading)
            if conversation and len(conversation) > 0:
                recent = list(conversation)[-5:] if len(conversation) >= 5 else list(conversation)
                context["recent_exchanges"] = recent
                
                # Extract last intent efficiently
                if recent:
                    last_meta = recent[-1].get("metadata", {})
                    if "intent" in last_meta:
                        context["last_intent"] = last_meta["intent"]
            else:
                context["recent_exchanges"] = []
        
        return context
    
    def clear_device(self, device_id: str):
        """Clear history for a specific device"""
        if device_id in self.conversations:
            self.cleared(device_id)
            logger.info("conversation_cleared", device_id=device_id)
    
    def cleanup_old_conversations(self):
        """Efficient cleanup of old conversations with thread safety"""
        current_time = time.time()

        # Only cleanup if enough time has passed
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            ttl_seconds = self.ttl.total_seconds()
            cutoff_time = current_time - ttl_seconds
            removed_devices = []

            # Process only active devices first (hot path)
            for device_id in list(self._active_devices):
                conversation = self.conversations.get(device_id)
                if not conversation:
                    self._active_devices.discard(device_id)
                    continue

                # Use fast deque filtering
                original_len = len(conversation)

                # Remove old exchanges efficiently
                while conversation and conversation[0].get("timestamp_unix", 0) < cutoff_time:
                    conversation.popleft()

                # Remove empty conversations
                if not conversation:
                    self.cleared(device_id)
                    removed_devices.append(device_id)

                if original_len != len(conversation):
                    logger.debug("conversation_cleaned", device_id=device_id, original_size=original_len, new_size=len(conversation))

            # Clean up inactive devices (cold path)
            inactive_devices = set(self.conversations.keys()) - self._active_devices
            for device_id in inactive_devices:
                metadata = self.device_metadata.get(device_id, {})
                if metadata.get("last_active", 0) < cutoff_time:
                    del self.conversations[device_id]
                    if device_id in self.device_metadata:
                        del self.device_metadata[device_id]
                    removed_devices.append(device_id)

            self.last_cleanup = current_time

        if removed_devices:
            logger.info("cleanup_completed", removed_count=len(removed_devices), remaining_devices=len(self.conversations))

    def cleared(self, device_id):
        del self.conversations[device_id]
        self._active_devices.discard(device_id)
        if device_id in self.device_metadata:
            del self.device_metadata[device_id]
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        total_exchanges = sum(len(conv) for conv in self.conversations.values())
        active_devices = len(self._active_devices)
        total_devices = len(self.conversations)
        
        return {
            "total_devices": total_devices,
            "active_devices": active_devices,
            "total_exchanges": total_exchanges,
            "average_exchanges_per_device": total_exchanges / max(1, total_devices),
            "memory_cleanup_interval": self.cleanup_interval,
            "background_cleanup_active": self._cleanup_task is not None and not self._cleanup_task.done()
        }
    
    def shutdown(self):
        """Cleanup resources on shutdown"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("cleanup_task_cancelled")


# Global instance
memory_manager = ConversationMemory()