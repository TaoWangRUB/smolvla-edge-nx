"""Remote chunk predictor: the RobotClient half of the paper's async stack.

Returns a callable with the same signature as ``make_chunk_predictor`` (gym obs
-> (chunk_len, action_dim) array), but the chunk is computed by a gRPC
PolicyServer (``deploy/client_server/server.py``) — a separate process or host,
which is also what removes the in-process GIL/CPU contention between inference
and mujoco stepping measured on the local tier.

Latency split per call (paper §3.3): rtt = t_C→S + l_S + t_S→C. Server compute
l_S = server_send_ts - server_recv_ts uses only the server's clock, so it is
clock-skew free; network overhead = rtt - l_S.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# generated stubs live next to the server, not in the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "client_server"))


def make_remote_chunk_predictor(server: str, task: str | None):
    """Connect to ``host:port`` and return predict(obs) plus a stats list.

    The returned callable carries a ``.calls`` list of per-call dicts
    (rtt_s, server_s, network_s) for the split-latency report.
    """
    import grpc
    import policy_pb2
    import policy_pb2_grpc

    channel = grpc.insecure_channel(server)
    grpc.channel_ready_future(channel).result(timeout=20)
    stub = policy_pb2_grpc.PolicyStub(channel)

    def predict(obs):
        cams = obs.get("pixels", obs.get("images"))
        top = cams.get("top") if isinstance(cams, dict) else cams
        if isinstance(cams, dict) and "top" not in cams:
            top = next(iter(cams.values()))
        top = np.ascontiguousarray(top, dtype=np.uint8)
        state = np.asarray(obs.get("agent_pos", obs.get("qpos")), dtype=np.float32)

        req = policy_pb2.Observation(
            images=[policy_pb2.Image(key="pixels.top", data=top.tobytes(),
                                     shape=list(top.shape), encoding="raw_uint8")],
            tensors=[policy_pb2.Tensor(key="agent_pos", data=state.ravel().tolist(),
                                       shape=list(state.shape))],
            task=task or "",
            client_send_ts=time.time(),
        )
        t0 = time.perf_counter()
        reply = stub.PredictChunk(req)
        rtt = time.perf_counter() - t0
        server_s = reply.server_send_ts - reply.server_recv_ts
        predict.calls.append(
            {"rtt_s": rtt, "server_s": server_s, "network_s": rtt - server_s})
        return np.asarray(reply.data, dtype=np.float32).reshape(tuple(reply.shape))

    predict.calls = []
    return predict


def split_latency_summary(calls: list[dict]) -> dict:
    import statistics

    if not calls:
        return {}
    med = lambda k: statistics.median(c[k] for c in calls)  # noqa: E731
    return {
        "calls": len(calls),
        "median_rtt_s": med("rtt_s"),
        "median_server_s": med("server_s"),
        "median_network_s": med("network_s"),
    }
