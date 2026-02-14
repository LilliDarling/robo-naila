#!/usr/bin/env bash
# Regenerate Python gRPC stubs from proto/naila.proto
#
# Run from anywhere — the script finds its own location.
# Requires: grpcio-tools (uv add grpcio-tools)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AI_SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROTO_DIR="$(cd "$AI_SERVER_DIR/../proto" && pwd)"
OUT_DIR="$AI_SERVER_DIR/rpc/generated"

mkdir -p "$OUT_DIR"

echo "Generating Python stubs from $PROTO_DIR/naila.proto -> $OUT_DIR/"

python -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/naila.proto"

# Fix generated import: naila_pb2_grpc.py imports naila_pb2 absolutely,
# but we need a relative import since it's inside a package.
sed -i 's/^import naila_pb2 as naila__pb2$/from . import naila_pb2 as naila__pb2/' "$OUT_DIR/naila_pb2_grpc.py"

echo "Done. Generated files:"
ls -la "$OUT_DIR"/naila_pb2*.py
