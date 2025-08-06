import asyncio
import logging
import sys
from server.naila_server import NailaAIServer
from server.platform_utils import CrossPlatformSignalHandler, check_python_version


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Reduce noise from external libraries
logging.getLogger('paho').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# Suppress the old MQTT client's print statements by redirecting them to our logger
import mqtt.client as mqtt_client_module

logger = logging.getLogger(__name__)


async def main():
    """Main entry point with production-grade error handling"""
    server = None
    
    try:
        server = NailaAIServer()
        
        # Setup cross-platform signal handling
        signal_handler = CrossPlatformSignalHandler(server.stop)
        signal_handler.setup_signals()
        
        # Start server
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Critical server error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure cleanup
        if server and server.is_running():
            try:
                await server.stop()
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")


if __name__ == "__main__":
    try:
        if not check_python_version((3, 8)):
            sys.exit(1)
        
        # Run the server
        asyncio.run(main())
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        sys.exit(1)