import asyncio
import contextlib
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from .health_monitor import HealthMonitor
from utils.platform import get_platform_info
from managers.ai_model import AIModelManager
from utils import get_logger


logger = get_logger(__name__)


class StartupStage(Enum):
    """Descriptive startup stages for server initialization"""
    LOAD_AI_MODELS = "Loading AI models"
    REGISTER_HANDLERS = "Registering protocol handlers"
    START_MQTT = "Starting MQTT service"
    START_GRPC = "Starting gRPC server"
    START_HEALTH_MONITORING = "Starting health monitoring"
    PUBLISH_STATUS = "Publishing initial system status"


class ShutdownStage(Enum):
    """Descriptive shutdown stages for graceful termination"""
    STOP_HEALTH_MONITORING = "Stopping health monitoring"
    STOP_GRPC = "Stopping gRPC server"
    STOP_MQTT = "Stopping MQTT service"
    UNLOAD_AI_MODELS = "Unloading AI models"


class ServerLifecycleManager:
    """Manages server startup, shutdown, and lifecycle events"""

    def __init__(self, mqtt_service, protocol_handlers, llm_service=None, stt_service=None, tts_service=None, vision_service=None, grpc_server=None, grpc_servicer=None):
        self.mqtt_service = mqtt_service
        self.protocol_handlers = protocol_handlers
        self.grpc_server = grpc_server
        self.grpc_servicer = grpc_servicer
        self.ai_model_manager = AIModelManager(llm_service, stt_service, tts_service, vision_service)
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

        logger.info("server_startup_banner", separator="=" * 60)
        logger.info("server_starting", server_name="NAILA AI Server")
        logger.info("platform_info", system=platform_info['system'], architecture=platform_info['architecture'])
        logger.info("python_info", version=platform_info['python_version'], implementation=platform_info['python_implementation'])
        logger.info("mqtt_broker_info", broker_host=self.mqtt_service.config.broker_host, broker_port=self.mqtt_service.config.broker_port)
        logger.info("server_startup_banner_end", separator="=" * 60)

        try:
            # Stage: Load AI models
            logger.info("startup_stage", stage=StartupStage.LOAD_AI_MODELS.value)
            await self.ai_model_manager.load_models()

            if llm_service := self.ai_model_manager.get_llm_service():
                self.protocol_handlers.set_llm_service(llm_service)
                if self.grpc_servicer:
                    self.grpc_servicer.set_llm_service(llm_service)

            if stt_service := self.ai_model_manager.get_stt_service():
                self.protocol_handlers.set_stt_service(stt_service)
                if self.grpc_servicer:
                    self.grpc_servicer.set_stt_service(stt_service)

            if tts_service := self.ai_model_manager.get_tts_service():
                self.protocol_handlers.set_tts_service(tts_service)
                if self.grpc_servicer:
                    self.grpc_servicer.set_tts_service(tts_service)

            if vision_service := self.ai_model_manager.get_vision_service():
                self.protocol_handlers.set_vision_service(vision_service)

            # Stage: Register protocol handlers
            logger.info("startup_stage", stage=StartupStage.REGISTER_HANDLERS.value)
            self.protocol_handlers.register_all_handlers()
            logger.info("handlers_registered", topic_count=len(self.mqtt_service.event_handlers))

            # Stage: Start MQTT service
            logger.info("startup_stage", stage=StartupStage.START_MQTT.value)
            await self.mqtt_service.start()
            logger.info("mqtt_service_ready")

            # Stage: Start gRPC server
            if self.grpc_server:
                logger.info("startup_stage", stage=StartupStage.START_GRPC.value)
                await self.grpc_server.start()
                logger.info("grpc_server_ready", address=self.grpc_server.config.address)

            # Stage: Start health monitoring
            logger.info("startup_stage", stage=StartupStage.START_HEALTH_MONITORING.value)
            await self.health_monitor.start_monitoring(interval=30)
            logger.info("health_monitoring_active")

            # Stage: Publish initial status
            logger.info("startup_stage", stage=StartupStage.PUBLISH_STATUS.value)
            await self._publish_startup_status()
            logger.info("initial_status_published")

            self._running = True

            # Success banner
            logger.info("server_online_banner", separator="=" * 60)
            logger.info("server_online")
            logger.info("mqtt_connected", broker_host=self.mqtt_service.config.broker_host, broker_port=self.mqtt_service.config.broker_port)
            if self.grpc_server and self.grpc_server.is_running():
                logger.info("grpc_listening", address=self.grpc_server.config.address)
            logger.info("handlers_ready", topic_count=len(self.mqtt_service.event_handlers))
            logger.info("ready_for_connections")
            logger.info("server_online_banner_end", separator="=" * 60)

            # Main server loop
            await self._main_loop()

        except Exception as e:
            logger.error("server_startup_failed", error=str(e), error_type=type(e).__name__)
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
                logger.info("main_loop_cancelled")
                break
            except Exception as e:
                logger.error("main_loop_error", error=str(e), error_type=type(e).__name__)
                await asyncio.sleep(1)


    async def _heartbeat_check(self):
        """Perform periodic health checks"""
        try:
            if not self.mqtt_service.is_connected():
                logger.warning("mqtt_connection_lost")

            # Log performance metrics periodically (every 10 heartbeats = ~50 seconds)
            self._heartbeat_count += 1

            if self._heartbeat_count % 10 == 0:
                stats = self.mqtt_service.get_stats()
                logger.info(
                    "performance_metrics",
                    message_count=stats['message_count'],
                    error_rate=round(stats['error_rate'] * 100, 1),
                    uptime_seconds=int(stats['uptime_seconds'])
                )

        except Exception as e:
            logger.error("heartbeat_check_failed", error=str(e), error_type=type(e).__name__)
    
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
            logger.info("shutdown_stage", stage=ShutdownStage.STOP_HEALTH_MONITORING.value)
            await self.health_monitor.stop_monitoring()

            # Stage: Stop gRPC server
            if self.grpc_server and self.grpc_server.is_running():
                logger.info("shutdown_stage", stage=ShutdownStage.STOP_GRPC.value)
                await self.grpc_server.stop()

            # Stage: Stop MQTT service
            logger.info("shutdown_stage", stage=ShutdownStage.STOP_MQTT.value)
            await self.mqtt_service.stop()

            # Stage: Unload AI models
            logger.info("shutdown_stage", stage=ShutdownStage.UNLOAD_AI_MODELS.value)
            self.ai_model_manager.unload_models()

            self._running = False

            # Final stats
            if self._start_time:
                uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
                logger.info("server_stopped_gracefully", uptime_seconds=round(uptime, 1))
            else:
                logger.info("server_stopped_gracefully")

        except Exception as e:
            logger.error("shutdown_error", error=str(e), error_type=type(e).__name__)
        
        logger.info("shutdown_complete", server="NAILA AI Server")

    async def _emergency_shutdown(self):
        """Emergency shutdown for critical errors"""
        logger.critical("emergency_shutdown_initiated")

        try:
            if self.mqtt_service.is_connected():
                emergency_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "emergency_shutdown",
                    "reason": "critical_error"
                }
                self.mqtt_service.publish_system_message("health", "emergency", emergency_data, qos=2)
        except Exception as exc:
            logger.error("emergency_shutdown_exception", error=str(exc), error_type=type(exc).__name__)

        # Force stop everything
        self._running = False
        if self.grpc_server:
            with contextlib.suppress(Exception):
                await self.grpc_server.stop()
        with contextlib.suppress(Exception):
            await self.mqtt_service.stop()
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running and not self._shutdown_requested