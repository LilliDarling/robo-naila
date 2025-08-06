from mqtt.core.models import MQTTMessage
from .base import BaseHandler


class SystemHandlers(BaseHandler):
    """Handlers for system monitoring and management messages"""
    
    def register_handlers(self):
        """Register all system-related handlers"""
        handlers = {
            "naila/system/health/+": self.handle_system_health,
            "naila/system/security/alert": self.handle_security_alert,
            "naila/system/errors/+": self.handle_system_error,
        }
        
        for topic, handler in handlers.items():
            self.mqtt_service.register_handler([topic], handler)
    
    async def handle_system_health(self, message: MQTTMessage):
        """Handle system health monitoring - only log issues"""
        if "services" in message.payload:
            services = message.payload["services"]
            for service, health in services.items():
                if health.get("status") != "healthy":
                    self.logger.warning(f"Service {service} unhealthy: {health}")
    
    async def handle_security_alert(self, message: MQTTMessage):
        """Handle security alerts - immediate logging"""
        severity = message.payload.get("severity", "unknown")
        event_type = message.payload.get("event_type", "unknown")
        alert_id = message.payload.get("alert_id", "unknown")
        
        if severity in ["high", "critical"]:
            self.logger.critical(f"SECURITY ALERT {alert_id}: {event_type}")
        else:
            self.logger.warning(f"Security alert {alert_id}: {event_type}")
    
    async def handle_system_error(self, message: MQTTMessage):
        """Handle system errors - fast logging"""
        service_name = message.subtype or "unknown"
        error_code = message.payload.get("error_code", "UNKNOWN")
        error_message = message.payload.get("error_message", "No details")
        
        self.logger.error(f"System error [{service_name}]: {error_code} - {error_message}")