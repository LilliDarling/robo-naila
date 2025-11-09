import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class HealthMonitor:
    """System health monitoring and metrics collection"""

    def __init__(self, mqtt_service, protocol_handlers, ai_model_manager=None):
        self.mqtt_service = mqtt_service
        self.protocol_handlers = protocol_handlers
        self.ai_model_manager = ai_model_manager
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
                },
                "ai_services": self._get_ai_services_status()
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

    def _get_ai_services_status(self) -> Dict[str, Any]:
        """Get AI services (LLM, STT, TTS) health status"""
        if not self.ai_model_manager:
            return {"status": "not_configured"}

        try:
            ai_status = self.ai_model_manager.get_status()

            # Build comprehensive status
            services_status = {
                "status": "healthy" if ai_status.get("models_loaded") else "unhealthy",
                "models_loaded": ai_status.get("models_loaded", False),
                "services": {}
            }

            # LLM status
            if ai_status.get("llm"):
                llm_status = ai_status["llm"]
                services_status["services"]["llm"] = {
                    "status": "ready" if llm_status.get("ready") else "not_ready",
                    "model": llm_status.get("model_path", "").split("/")[-1] if llm_status.get("model_path") else "none",
                    "hardware": llm_status.get("hardware", {}).get("device_type", "unknown"),
                }

            # STT status
            if ai_status.get("stt"):
                stt_status = ai_status["stt"]
                services_status["services"]["stt"] = {
                    "status": "ready" if stt_status.get("ready") else "not_ready",
                    "model": stt_status.get("model_path", "").split("/")[-1] if stt_status.get("model_path") else "none",
                    "hardware": stt_status.get("hardware", {}).get("device_type", "unknown"),
                }

            # TTS status
            if ai_status.get("tts"):
                tts_status = ai_status["tts"]
                services_status["services"]["tts"] = {
                    "status": "ready" if tts_status.get("ready") else "not_ready",
                    "model": tts_status.get("model_path", "").split("/")[-1] if tts_status.get("model_path") else "none",
                    "voice": tts_status.get("voice", "unknown"),
                    "sample_rate": tts_status.get("sample_rate", 0),
                    "output_format": tts_status.get("output_format", "unknown"),
                    "cached_phrases": tts_status.get("cached_phrases", 0),
                }

            return services_status

        except Exception as e:
            logger.error(f"Failed to get AI services status: {e}")
            return {"status": "error", "error": str(e)}