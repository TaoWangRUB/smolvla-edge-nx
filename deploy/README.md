# Deployment (Phase 2 — the differentiator)

Two deployment modes for the same fine-tuned checkpoint. This is where the real engineering —
and the portfolio value — lives.

## Mode A — On-device (Xavier NX, 8 GB)

Goal: run the policy entirely on the NX under real-time constraints. Levers, in order of
payoff:

1. **Action chunking + decoupled execution.** SmolVLA separates action *prediction* from
   *execution*, cutting task time ~30% on average. Predict a chunk, execute it open-loop while
   the next prediction computes.
2. **Low-Hz VLM stage.** The SmolVLM-2 backbone is the expensive part — run it at a lower rate
   than the action expert / control loop.
3. **Precision.** FP16 first (free-ish, big memory win on 8 GB). INT8 only via real
   quantization/TensorRT on the subgraphs that convert — **not** a naive cast (the benchmark
   harness refuses naive INT8 on purpose).

### Honest framing for the writeup

Full TensorRT conversion of a SmolVLM-2 + flow-matching-expert VLA is non-trivial. Don't
claim a clean end-to-end TRT engine. The credible, valuable deliverable is:

- **What converts** (e.g. vision encoder / static subgraphs) → engine + speedup.
- **What doesn't** (dynamic control flow, the flow-matching sampler loop) → why, and the
  fallback.
- **The latency budget you hit anyway**, with the per-stage breakdown.

Capture conversion attempts and the per-stage budget in
[ondevice/conversion_notes.md](ondevice/conversion_notes.md).

## Mode B — Client / server

Policy on the Titan X workstation, NX as a thin gRPC control client. Mirrors how real customer
robots offload heavy inference; gives the second benchmark point (latency now includes the
network).

```bash
# once, on both machines (after editing the proto):
bash deploy/client_server/gen_proto.sh

# workstation (Titan X):
python deploy/client_server/server.py --policy-path <checkpoint> --precision fp16

# edge (Xavier NX):
python deploy/client_server/client.py --server <workstation-ip>:50051 \
    --out benchmarks/results/raw/client_server.json
```

The client replays held-out dataset frames as observations (no robot needed) and reports
round-trip latency split into server-compute vs. network overhead.
