#!/usr/bin/env bash
# Generate Python gRPC stubs from policy.proto. The generated *_pb2.py / *_pb2_grpc.py are
# gitignored; run this after cloning (and after editing the proto).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m grpc_tools.protoc \
  -I "${HERE}/proto" \
  --python_out="${HERE}" \
  --grpc_python_out="${HERE}" \
  "${HERE}/proto/policy.proto"

echo "[gen_proto] wrote policy_pb2.py and policy_pb2_grpc.py in ${HERE}"
