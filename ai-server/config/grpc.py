"""gRPC Server Configuration"""

import os
from dataclasses import dataclass


@dataclass
class GRPCConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 4

    @classmethod
    def from_env(cls):
        kwargs = {}

        if host := os.getenv("GRPC_HOST"):
            kwargs["host"] = host
        if port := os.getenv("GRPC_PORT"):
            kwargs["port"] = int(port)
        if max_workers := os.getenv("GRPC_MAX_WORKERS"):
            kwargs["max_workers"] = int(max_workers)

        return cls(**kwargs)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"
