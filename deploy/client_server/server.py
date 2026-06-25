"""gRPC policy server — runs on the workstation (Titan X), holds the SmolVLA policy.

    # 1. generate stubs once
    bash deploy/client_server/gen_proto.sh
    # 2. serve
    python deploy/client_server/server.py \
        --policy-path outputs/train/smolvla_so101/checkpoints/last \
        --precision fp16 --port 50051

The Xavier NX connects with client.py and streams observations; the server returns action
chunks. This is benchmark tier 3 (NX-client / workstation-server).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent import futures
from pathlib import Path

# Allow importing the package without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    import policy_pb2  # type: ignore
    import policy_pb2_grpc  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "gRPC stubs not found. Run: bash deploy/client_server/gen_proto.sh"
    ) from e

import grpc  # noqa: E402

from smolvla_edge.common import load_policy  # noqa: E402


class PolicyServicer(policy_pb2_grpc.PolicyServicer):
    def __init__(self, policy_path: str, device: str, precision: str):
        self.policy_path = policy_path
        self.precision = precision
        self.policy, self.device = load_policy(policy_path, device)
        if precision == "fp16":
            self.policy = self.policy.half()
        self.policy.reset()
        print(f"[server] loaded {policy_path} on {self.device} ({precision})")

    def _to_batch(self, obs):
        import numpy as np
        import torch

        batch = {}
        for img in obs.images:
            arr = np.frombuffer(img.data, dtype=np.uint8).reshape(tuple(img.shape))
            t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
            batch[img.key] = t.unsqueeze(0).to(self.device)
        for tn in obs.tensors:
            t = torch.tensor(tn.data, dtype=torch.float32).reshape(tuple(tn.shape))
            batch[tn.key] = t.unsqueeze(0).to(self.device)
        if obs.task:
            batch["task"] = obs.task
        return batch

    def SelectAction(self, request, context):
        import torch

        recv = time.time()
        batch = self._to_batch(request)
        with torch.no_grad():
            action = self.policy.select_action(batch)
        action = action.squeeze(0).float().cpu().numpy()
        return policy_pb2.ActionChunk(
            data=action.reshape(-1).tolist(),
            shape=list(action.shape),
            server_recv_ts=recv,
            server_send_ts=time.time(),
        )

    def Reset(self, request, context):
        self.policy.reset()
        return policy_pb2.ResetReply(ok=True)

    def Health(self, request, context):
        return policy_pb2.HealthReply(
            ok=True, device=self.device, precision=self.precision, policy_path=self.policy_path
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="SmolVLA gRPC policy server.")
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    policy_pb2_grpc.add_PolicyServicer_to_server(
        PolicyServicer(args.policy_path, args.device, args.precision), server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"[server] listening on :{args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
