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

### Xavier NX — pure-Python ONNX Runtime GPU (measured)

The first on-device milestone serves the Stage-2a monolithic ONNX graph via **pure-Python
onnxruntime-gpu** — no C++ server, no torch, no lerobot. The graph bakes in the instruction
tokens and all normalization (`models/onnx/*.meta.json`), so inference needs only `numpy` +
`onnxruntime-gpu`. Measured on a **Jetson Xavier NX 8 GB Dev Kit** (JetPack 5.1 / L4T R35.4.1,
CUDA 11.4, 20 W/6-core): **~610 ms mean per chunk** on the CUDA EP (fp32), vs ~300 ms PyTorch on
the RTX 2000 Ada dev box.

```bash
# on the Jetson, repo at ~/workspace/smolvla-edge-nx
# one-time build (Jetson's compose lacks buildx >=0.17, so use the legacy builder):
DOCKER_BUILDKIT=0 docker build -f docker/jetson_infer.Dockerfile -t smolvla-edge:jetson .
docker compose -f docker-compose.jetson.yml run --rm bench    # CUDA EP latency
ORT_PROVIDER=tensorrt docker compose -f docker-compose.jetson.yml run --rm bench   # try TRT EP
```

Three things this required, all captured in the Jetson files:

- **ORT wheel.** The only cp38 / CUDA-11.4 build for JetPack 5 is `onnxruntime-gpu 1.15.1`
  (vendored under [onnx/wheels/](onnx/wheels/); the jetson-ai-lab pip index has unreliable DNS).
- **Base + CUDA libs.** [docker/jetson_infer.Dockerfile](../docker/jetson_infer.Dockerfile) uses
  the lean `nvcr.io/nvidia/l4t-base:r35.2.1` (l4t-jetpack:r35.4.1 is 5.3 GB — too big for the
  NX's ~14 GB free disk). That base ships no CUDA toolkit, so
  [docker-compose.jetson.yml](../docker-compose.jetson.yml) bind-mounts the host `/usr/local/cuda`
  + cuDNN/TensorRT and sets `LD_LIBRARY_PATH` — the same pattern the ackermann_rover Jetson stack
  uses. Do **not** base on the rover image itself: it's Ubuntu 24.04 / Python 3.12, which has no
  cp38 ORT wheel.
- **Graph shims for ORT 1.15.** [onnx/patch_for_ort115.py](onnx/patch_for_ort115.py) downgrades
  the graph IR 10→9 and inserts 24 int64→int32 Casts before `ArgMin`/`ArgMax` (ORT 1.15 has no
  int64 kernel). Run it in the dev **sim container** (32 GB RAM) — the 1.8 GB load/save OOMs the
  6.7 GB Jetson. Output: `models/onnx/smolvla_transfer_cube_ort115.onnx`.

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
