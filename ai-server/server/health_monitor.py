import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class HealthMonitor:
    """System health monitoring and metrics collection"""
    
    def __init__(self, mqtt_service, protocol_handlers):
        self.mqtt_service = mqtt_service
        self.protocol_handlers = protocol_handlers
        self.start_time = datetime.now(timezone.utc)
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self, interval: int = 30):
        """Start periodic health monitoring"""
        self._monitoring_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Health monitoring started (interval: {interval}s)")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self, interval: int):
        """Periodic health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(interval)
                await self._publish_health_status()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _publish_health_status(self):
        """Publish comprehensive system health status"""
        try:
            mqtt_stats = self.mqtt_service.get_stats()
            handler_stats = await self.protocol_handlers.get_performance_stats()
            
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            
            health_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": {
                    "uptime_seconds": uptime,
                    "status": "healthy",
                    "memory_usage_mb": self._get_memory_usage(),
                    "python_version": sys.version.split()[0],
                    "platform": sys.platform
                },
                "mqtt_service": {
                    "status": "healthy" if mqtt_stats["connected"] else "unhealthy",
                    "message_count": mqtt_stats["message_count"],
                    "error_count": mqtt_stats["error_count"],
                    "error_rate": mqtt_stats["error_rate"],
                    "handlers_registered": mqtt_stats["handlers_registered"],
                    "cache_efficiency": self._calculate_cache_efficiency(mqtt_stats)
                },
                "protocol_handlers": {
                    "status": "healthy",
                    "active_devices": handler_stats["active_devices"],
                    "device_states": handler_stats["device_states"],
                    "conversation_contexts": handler_stats["conversation_contexts"],
                    "task_queue_utilization": handler_stats.get("task_queue_size", 0) / max(handler_stats.get("task_queue_max", 1), 1)
                }
            }
            
            # Only publish if connected
            if self.mqtt_service.is_connected():
                self.mqtt_service.publish_system_message("health", "services", health_data, qos=1)
            
        except Exception as e:
            logger.error(f"Failed to publish health status: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB with cross-platform support"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback for systems without psutil
            logger.debug("psutil not available, using basic memory estimation")
            return 0.0
        except Exception as e:
            logger.debug(f"Memory usage calculation failed: {e}")
            return 0.0
    
    def _calculate_cache_efficiency(self, stats: Dict[str, Any]) -> float:
        """Calculate cache hit efficiency"""
        cache_size = stats.get("cache_size", 0)
        handlers_registered = stats.get("handlers_registered", 1)
        return min(cache_size / max(handlers_registered, 1), 1.0)