"""gRPC server lifecycle management.

Wraps grpc.aio.server() with start/stop lifecycle, matching the pattern
used by NailaMQTTService for MQTT.
"""

import grpc.aio

from config.grpc import GRPCConfig
from rpc.generated import naila_pb2_grpc
from rpc.service import NailaAIServicer
from utils import get_logger


logger = get_logger(__name__)

# Grace period for in-flight RPCs during shutdown (seconds)
SHUTDOWN_GRACE_SECONDS = 5


class GRPCServer:
    """Manages the async gRPC server lifecycle."""

    def __init__(self, config: GRPCConfig, servicer: NailaAIServicer):
        self.config = config
        self.servicer = servicer
        self._server: grpc.aio.Server | None = None
        self._running = False

    async def start(self):
        """Create and start the gRPC server."""
        if self._running:
            logger.warning("grpc_server_already_running")
            return

        self._server = grpc.aio.server()
        naila_pb2_grpc.add_NailaAIServicer_to_server(self.servicer, self._server)
        self._server.add_insecure_port(self.config.address)

        await self._server.start()
        self._running = True
        logger.info("grpc_server_started", address=self.config.address)

    async def stop(self):
        """Gracefully stop the gRPC server."""
        if not self._running or self._server is None:
            return

        logger.info("grpc_server_stopping", grace_seconds=SHUTDOWN_GRACE_SECONDS)
        await self._server.stop(grace=SHUTDOWN_GRACE_SECONDS)
        self._running = False
        logger.info("grpc_server_stopped")

    def is_running(self) -> bool:
        return self._running
