"""gRPC policy client — runs on the Xavier NX, streams observations to the workstation.

Replays held-out dataset frames as the observation source (no physical robot needed) and
measures end-to-end round-trip latency, so this doubles as benchmark tier 3.

    python deploy/client_server/client.py \
        --server <workstation-ip>:50051 \
        --dataset-repo-id lerobot/svla_so101_pickplace \
        --steps 200 --out benchmarks/results/raw/client_server.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    import policy_pb2  # type: ignore
    import policy_pb2_grpc  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "gRPC stubs not found. Run: bash deploy/client_server/gen_proto.sh"
    ) from e

import grpc  # noqa: E402

from smolvla_edge.common import load_dataset  # noqa: E402


def frame_to_observation(frame, task: str):
    import numpy as np

    obs = policy_pb2.Observation(task=task, client_send_ts=time.time())
    for k, v in frame.items():
        if hasattr(v, "ndim") and v.ndim == 3 and "image" in k.lower():
            arr = v.detach().cpu().numpy()
            if arr.shape[0] in (1, 3):  # CHW -> HWC
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != np.uint8:
                arr = (arr.clip(0, 1) * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
            obs.images.append(
                policy_pb2.Image(
                    key=k, data=arr.tobytes(), shape=list(arr.shape), encoding="raw_uint8"
                )
            )
        elif "state" in k.lower() and hasattr(v, "ndim"):
            arr = v.detach().cpu().numpy().astype("float32").reshape(-1)
            obs.tensors.append(policy_pb2.Tensor(key=k, data=arr.tolist(), shape=[arr.size]))
    return obs


def percentile(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser(description="SmolVLA gRPC client / tier-3 benchmark.")
    ap.add_argument("--server", default="localhost:50051")
    ap.add_argument("--dataset-repo-id", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--task", default="pick up the cube and place it in the box")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = load_dataset(args.dataset_repo_id, episodes=[0])

    with grpc.insecure_channel(args.server) as channel:
        stub = policy_pb2_grpc.PolicyStub(channel)
        h = stub.Health(policy_pb2.HealthRequest())
        print(f"[client] server: device={h.device} precision={h.precision}")
        stub.Reset(policy_pb2.ResetRequest())

        rtt_ms, server_ms = [], []
        n = min(len(ds), args.steps + args.warmup)
        for i in range(n):
            obs = frame_to_observation(ds[i], args.task)
            t0 = time.time()
            resp = stub.SelectAction(obs)
            t1 = time.time()
            if i < args.warmup:
                continue
            rtt_ms.append((t1 - t0) * 1e3)
            server_ms.append((resp.server_send_ts - resp.server_recv_ts) * 1e3)

        result = {
            "tag": "nx-client_workstation-server",
            "server": args.server,
            "steps_measured": len(rtt_ms),
            "rtt_mean_ms": round(sum(rtt_ms) / len(rtt_ms), 3),
            "rtt_p50_ms": round(percentile(rtt_ms, 0.50), 3),
            "rtt_p95_ms": round(percentile(rtt_ms, 0.95), 3),
            "server_compute_mean_ms": round(sum(server_ms) / len(server_ms), 3),
            "network_overhead_mean_ms": round(
                (sum(rtt_ms) - sum(server_ms)) / len(rtt_ms), 3
            ),
            "throughput_hz": round(1000.0 / (sum(rtt_ms) / len(rtt_ms)), 3),
        }
        print("[client] result:")
        print(json.dumps(result, indent=2))
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
            print(f"[client] wrote {out}")


if __name__ == "__main__":
    main()
