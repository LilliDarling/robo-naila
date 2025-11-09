import asyncio
import sys
from server.naila_server import NailaAIServer
from utils.platform import CrossPlatformSignalHandler, check_python_version
from utils import get_logger, setup_logging, silence_module


# Setup centralized logging
setup_logging(level="INFO", colored=True)

# Silence noisy modules
silence_module('paho')
silence_module('asyncio')

logger = get_logger(__name__)


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
        logger.info("keyboard_interrupt")
    except Exception as e:
        logger.error("critical_server_error", error=str(e), error_type=type(e).__name__)
        sys.exit(1)
    finally:
        # Ensure cleanup
        if server and server.is_running():
            try:
                await server.stop()
            except Exception as e:
                logger.error("cleanup_error", error=str(e), error_type=type(e).__name__)


if __name__ == "__main__":
    try:
        if not check_python_version((3, 8)):
            sys.exit(1)

        # Run the server
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("application_interrupted")
    except Exception as e:
        logger.critical("startup_failed", error=str(e), error_type=type(e).__name__)
        sys.exit(1)