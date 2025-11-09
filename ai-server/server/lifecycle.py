import asyncio
import contextlib
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from .health_monitor import HealthMonitor
from .platform_utils import get_platform_info
from services.ai_model_manager import AIModelManager


logger = logging.getLogger(__name__)


class StartupStage(Enum):
    """Descriptive startup stages for server initialization"""
    LOAD_AI_MODELS = "Loading AI models"
    REGISTER_HANDLERS = "Registering protocol handlers"
    START_MQTT = "Starting MQTT service"
    START_HEALTH_MONITORING = "Starting health monitoring"
    PUBLISH_STATUS = "Publishing initial system status"


class ShutdownStage(Enum):
    """Descriptive shutdown stages for graceful termination"""
    STOP_HEALTH_MONITORING = "Stopping health monitoring"
    STOP_MQTT = "Stopping MQTT service"
    UNLOAD_AI_MODELS = "Unloading AI models"


class ServerLifecycleManager:
    """Manages server startup, shutdown, and lifecycle events"""

    def __init__(self, mqtt_service, protocol_handlers, llm_service=None, stt_service=None, tts_service=None):
        self.mqtt_service = mqtt_service
        self.protocol_handlers = protocol_handlers
        self.ai_model_manager = AIModelManager(llm_service, stt_service, tts_service)
        self.health_monitor = HealthMonitor(mqtt_service, protocol_handlers, self.ai_model_manager)

        # Server state
        self._running = False
        self._shutdown_requested = False
        self._shutdown_event = asyncio.Event()
        self._start_time: Optional[datetime] = None

        # Performance tracking
        self._heartbeat_count = 0
    
    async def start_server(self):
        """Start the server with comprehensive initialization"""
        if self._running:
            logger.warning("Server already running")
            return

        self._start_time = datetime.now(timezone.utc)
        platform_info = get_platform_info()

        logger.info("=" * 60)
        logger.info("Starting NAILA AI Server")
        logger.info(f"Platform: {platform_info['system']} {platform_info['architecture']}")
        logger.info(f"Python: {platform_info['python_version']} ({platform_info['python_implementation']})")
        logger.info(f"MQTT Broker: {self.mqtt_service.config.broker_host}:{self.mqtt_service.config.broker_port}")
        logger.info("=" * 60)

        try:
            # Stage: Load AI models
            logger.info(f"{StartupStage.LOAD_AI_MODELS.value}...")
            await self.ai_model_manager.load_models()

            if llm_service := self.ai_model_manager.get_llm_service():
                self.protocol_handlers.set_llm_service(llm_service)

            if stt_service := self.ai_model_manager.get_stt_service():
                self.protocol_handlers.set_stt_service(stt_service)

            if tts_service := self.ai_model_manager.get_tts_service():
                self.protocol_handlers.set_tts_service(tts_service)

            # Stage: Register protocol handlers
            logger.info(f"{StartupStage.REGISTER_HANDLERS.value}...")
            self.protocol_handlers.register_all_handlers()
            logger.info(f"Registered handlers for {len(self.mqtt_service.event_handlers)} topics")

            # Stage: Start MQTT service
            logger.info(f"{StartupStage.START_MQTT.value}...")
            await self.mqtt_service.start()
            logger.info("MQTT service connected and subscribed")

            # Stage: Start health monitoring
            logger.info(f"{StartupStage.START_HEALTH_MONITORING.value}...")
            await self.health_monitor.start_monitoring(interval=30)
            logger.info("Health monitoring active")

            # Stage: Publish initial status
            logger.info(f"{StartupStage.PUBLISH_STATUS.value}...")
            await self._publish_startup_status()
            logger.info("Initial status published")

            self._running = True

            # Success banner
            logger.info("=" * 60)
            logger.info("NAILA AI Server ONLINE")
            logger.info(f"MQTT: Connected to {self.mqtt_service.config.broker_host}:{self.mqtt_service.config.broker_port}")
            logger.info(f"Handlers: {len(self.mqtt_service.event_handlers)} topics registered")
            logger.info("Ready for robot connections")
            logger.info("=" * 60)

            # Main server loop
            await self._main_loop()

        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            await self._emergency_shutdown()
            raise
    
    async def _main_loop(self):
        """Main server event loop with health monitoring"""
        heartbeat_interval = 5.0  # Heartbeat every 5 seconds
        last_heartbeat = 0

        while self._running and not self._shutdown_requested:
            try:
                # Non-blocking wait with timeout
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=heartbeat_interval)
                    break  # Shutdown requested
                # Periodic heartbeat and health checks
                current_time = datetime.now(timezone.utc).timestamp()
                if current_time - last_heartbeat > heartbeat_interval:
                    await self._heartbeat_check()
                    last_heartbeat = current_time

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(1)


    async def _heartbeat_check(self):
        """Perform periodic health checks"""
        try:
            if not self.mqtt_service.is_connected():
                logger.warning("MQTT connection lost, attempting reconnection...")
            
            # Log performance metrics periodically (every 10 heartbeats = ~50 seconds)
            self._heartbeat_count += 1
            
            if self._heartbeat_count % 10 == 0:
                stats = self.mqtt_service.get_stats()
                logger.info(
                    f"Performance: {stats['message_count']} msgs, "
                    f"{stats['error_rate']:.1%} error rate, "
                    f"{stats['uptime_seconds']:.0f}s uptime"
                )
        
        except Exception as e:
            logger.error(f"Heartbeat check failed: {e}")
    
    async def _publish_startup_status(self):
        """Publish comprehensive startup status"""
        platform_info = get_platform_info()
        
        startup_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "server_startup",
            "server_version": "1.0.0",
            "config": {
                "mqtt_broker": f"{self.mqtt_service.config.broker_host}:{self.mqtt_service.config.broker_port}",
                "mqtt_qos": self.mqtt_service.config.qos,
                "client_id": self.mqtt_service.config.client_id
            },
            "capabilities": {
                "protocol_version": "1.0",
                "supported_devices": ["naila_robot"],
                "ai_services": ["stt", "vision", "orchestration"],
                "system_services": ["health", "security", "updates"]
            },
            "system_info": {
                "python_version": platform_info["python_version"],
                "platform": platform_info["platform"],
                "system": platform_info["system"],
                "architecture": platform_info["architecture"],
                "pid": os.getpid()
            }
        }
        
        self.mqtt_service.publish_system_message("health", "startup", startup_data, qos=1)
    
    async def stop_server(self):
        """Graceful server shutdown with proper cleanup"""
        if not self._running:
            logger.info("Server already stopped")
            return
        
        logger.info("Initiating graceful shutdown...")
        self._shutdown_requested = True
        self._shutdown_event.set()
        
        try:
            # Publish shutdown notification
            if self.mqtt_service.is_connected():
                shutdown_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "server_shutdown",
                    "reason": "graceful_shutdown",
                    "uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds() if self._start_time else 0
                }
                self.mqtt_service.publish_system_message("health", "shutdown", shutdown_data, qos=1)
                await asyncio.sleep(0.5)

            # Stage: Stop health monitoring
            logger.info(f"{ShutdownStage.STOP_HEALTH_MONITORING.value}...")
            await self.health_monitor.stop_monitoring()

            # Stage: Stop MQTT service
            logger.info(f"{ShutdownStage.STOP_MQTT.value}...")
            await self.mqtt_service.stop()

            # Stage: Unload AI models
            logger.info(f"{ShutdownStage.UNLOAD_AI_MODELS.value}...")
            self.ai_model_manager.unload_models()

            self._running = False
            
            # Final stats
            if self._start_time:
                uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
                logger.info(f"Server stopped gracefully (uptime: {uptime:.1f}s)")
            else:
                logger.info("Server stopped gracefully")
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("NAILA AI Server shutdown complete")
    
    async def _emergency_shutdown(self):
        """Emergency shutdown for critical errors"""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")

        try:
            if self.mqtt_service.is_connected():
                emergency_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": "critical_error"
                }
                self.mqtt_service.publish_system_message("health", "emergency", emergency_data, qos=2)
        except Exception as exc:
            logger.error("Exception during emergency shutdown: %s", exc, exc_info=True)
                    "event": "emergency_shutdown",
                    "reason": "critical_error"
                }
                self.mqtt_service.publish_system_message("health", "emergency", emergency_data, qos=2)
        # Force stop everything
        self._running = False
        with contextlib.suppress(Exception):
            await self.mqtt_service.stop()
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running and not self._shutdown_requested