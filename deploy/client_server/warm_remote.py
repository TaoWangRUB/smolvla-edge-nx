"""Warm a remote PolicyServer and print the split-latency stats (rtt / server / network)."""
import sys
sys.path.insert(0, "src")
import numpy as np
from smolvla_edge.remote import make_remote_chunk_predictor

server = sys.argv[1] if len(sys.argv) > 1 else "10.42.0.2:50051"
TASK = "Pick up the cube with the right arm and transfer it to the left arm."
pred = make_remote_chunk_predictor(server, TASK)
calls = pred.calls
rng = np.random.default_rng(0)
obs = {"pixels": {"top": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)},
       "agent_pos": rng.standard_normal(14).astype(np.float32)}
for i in range(6):
    chunk = pred(obs)
    c = calls[-1]
    print(f"call {i}: rtt={c['rtt_s']*1e3:.0f}ms server={c['server_s']*1e3:.0f}ms "
          f"network={c['network_s']*1e3:.0f}ms chunk={chunk.shape}", flush=True)
