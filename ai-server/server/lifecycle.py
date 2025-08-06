import asyncio
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Optional
from .health_monitor import HealthMonitor
from .platform_utils import get_platform_info


logger = logging.getLogger(__name__)


class ServerLifecycleManager:
    """Manages server startup, shutdown, and lifecycle events"""
    
    def __init__(self, mqtt_service, protocol_handlers):
        self.mqtt_service = mqtt_service
        self.protocol_handlers = protocol_handlers
        self.health_monitor = HealthMonitor(mqtt_service, protocol_handlers)
        
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
            # Phase 1: Register protocol handlers
            logger.info("Phase 1: Registering protocol handlers...")
            self.protocol_handlers.register_all_handlers()
            logger.info(f"Registered handlers for {len(self.mqtt_service.event_handlers)} topics")
            
            # Phase 2: Start MQTT service
            logger.info("Phase 2: Starting MQTT service...")
            await self.mqtt_service.start()
            logger.info("MQTT service connected and subscribed")
            
            # Phase 3: Start health monitoring
            logger.info("Phase 3: Starting health monitoring...")
            await self.health_monitor.start_monitoring(interval=30)
            logger.info("Health monitoring active")
            
            # Phase 4: Publish initial status
            logger.info("Phase 4: Publishing initial system status...")
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
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=heartbeat_interval)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop
                
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
            # Phase 1: Publish shutdown notification
            if self.mqtt_service.is_connected():
                shutdown_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "server_shutdown",
                    "reason": "graceful_shutdown",
                    "uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds() if self._start_time else 0
                }
                self.mqtt_service.publish_system_message("health", "shutdown", shutdown_data, qos=1)
                await asyncio.sleep(0.5)
            
            # Phase 2: Stop health monitoring
            logger.info("Phase 1: Stopping health monitoring...")
            await self.health_monitor.stop_monitoring()
            
            # Phase 3: Stop MQTT service
            logger.info("Phase 2: Stopping MQTT service...")
            await self.mqtt_service.stop()
            
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
                    "event": "emergency_shutdown",
                    "reason": "critical_error"
                }
                self.mqtt_service.publish_system_message("health", "emergency", emergency_data, qos=2)
        except:
            pass  # Don't let emergency shutdown fail
        
        # Force stop everything
        self._running = False
        try:
            await self.mqtt_service.stop()
        except:
            pass
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running and not self._shutdown_requested