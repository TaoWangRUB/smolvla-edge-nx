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
    def __init__(self, policy_path: str, device: str, precision: str,
                 flow_steps: int | None = None, noise_seed: int | None = None):
        self.policy_path = policy_path
        self.precision = precision
        self.policy, self.device = load_policy(policy_path, device)
        if precision == "fp16":
            self.policy = self.policy.half()
        if flow_steps is not None:
            self.policy.config.num_steps = flow_steps
        self.noise_seed = noise_seed
        self.policy.reset()
        # chunk predictors for the async stack, built lazily per task string so
        # the observation->policy mapping matches the local eval path exactly
        self._chunk_predictors: dict[str, object] = {}
        print(f"[server] loaded {policy_path} on {self.device} ({precision})")

    def _chunk_predictor(self, task: str):
        if task not in self._chunk_predictors:
            from smolvla_edge.async_infer import make_chunk_predictor

            self._chunk_predictors[task] = make_chunk_predictor(
                self.policy, self.policy_path, self.device, task or None,
                precision="fp32",  # server-side autocast measured slower; keep native
            )
        return self._chunk_predictors[task]

    @staticmethod
    def _to_gym_obs(request):
        """Rebuild the gym-aloha style obs dict the chunk predictor expects."""
        import numpy as np

        obs: dict = {"pixels": {}}
        for img in request.images:
            arr = np.frombuffer(img.data, dtype=np.uint8).reshape(tuple(img.shape))
            key = img.key.split(".")[-1]  # "observation.images.top" / "pixels.top" -> "top"
            obs["pixels"][key] = arr
        for tn in request.tensors:
            key = tn.key.split(".")[-1]  # "observation.state" -> "state"; "agent_pos" kept
            key = "agent_pos" if key in ("state", "agent_pos") else key
            obs[key] = np.asarray(tn.data, dtype=np.float32).reshape(tuple(tn.shape))
        return obs

    def PredictChunk(self, request, context):
        recv = time.time()
        if self.noise_seed is not None:
            # fixed flow-matching noise: consecutive chunks from nearby states
            # become near-identical trajectories, so mid-chunk splices stay
            # coherent (tests the multimodality-splice failure mode)
            import torch

            torch.manual_seed(self.noise_seed)
        chunk = self._chunk_predictor(request.task)(self._to_gym_obs(request))
        return policy_pb2.ActionChunk(
            data=chunk.reshape(-1).tolist(),
            shape=list(chunk.shape),
            server_recv_ts=recv,
            server_send_ts=time.time(),
        )

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
    ap.add_argument("--flow-steps", type=int, default=None,
                    help="(SmolVLA) override flow-matching denoising steps")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="fix flow-matching noise per chunk (coherent splices)")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    policy_pb2_grpc.add_PolicyServicer_to_server(
        PolicyServicer(args.policy_path, args.device, args.precision, args.flow_steps,
                       args.noise_seed),
        server,
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"[server] listening on :{args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
