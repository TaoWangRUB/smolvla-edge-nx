# On-device conversion log (Xavier NX, 8 GB)

Honest log of every optimization attempted on the Jetson Xavier NX (JetPack 5.1, Volta
sm_72, 20 W/6-core Carmel). All latencies are **per 50-action chunk** (SmolVLA 450M
fine-tune, 3 flow steps, 512×512 single camera) unless noted. Full debug narrative:
[../jetson-native-torch/README.md](../jetson-native-torch/README.md).

## Baseline

| Config | Latency p50 (ms) | Chunk rate (Hz) | Peak GPU mem (GB) | Notes |
|--------|------------------|-----------------|-------------------|-------|
| ORT CUDA EP, fp32 (pure-Python, torch-free) | ~610 | 1.6 | ~1.8 | IR-9 + int64-ArgMin patches for ORT 1.15.1 |
| native torch eager, bf16 | 1795 | 0.6 | — | ❌ Volta has NO hw bf16 — emulated |
| native torch eager, fp32 | 817 | 1.2 | ~2.0 | no tensor cores |
| native torch eager, fp16 | ~608–650 | 1.6 | 1.02 | Volta tensor cores; **launch-bound** (GPU ~80% idle) |
| **native torch fp16 + CUDA Graph replay** | **233** | **4.3** | 1.18 | ✅ the shipped config (`precision="fp16-graph"`) |

Chunk rate 4.3 Hz × 50 actions = **215 action-steps/s** of motion capacity — 4.3× a 50 Hz
control loop, 21× a 10 Hz rover base.

## Per-stage latency budget (measured, eager fp16)

| Stage | ms | Notes |
|-------|----|-------|
| Image preprocess + normalization | 7 | not the bottleneck |
| model forward (VLM prefix + 3× flow expert) | 632 | 99% of wall; GR3D 15–26% ⇒ launch-bound |
| Postprocess / unnormalize | 0.8 | |
| **End-to-end eager** | **~640** | |
| **End-to-end graphed** (copies + replay + pre/post) | **233** | replay floor 211 ms = true GPU work |

The ~400 ms gap between eager and replay **is** the per-op CPU dispatch overhead — the
Carmel CPU cannot feed the GPU thousands of small kernels fast enough.

## What converts / what doesn't (TensorRT / quantization / graphs)

| Subgraph | Target | Result | Notes |
|----------|--------|--------|-------|
| Full model | TRT (via ONNX) | ❌ | 8 GB builder OOM + parser rejects the flow-matching dynamo graph |
| Vision encoder only | TRT | ◐ untried | superseded — capture already fuses the launch cost away |
| Full model | ORT CUDA EP | ✅ | fp32 only practical (~610 ms); ORT CUDA Graphs blocked by KV-cache sequence ops |
| Full model | INT8 | ❌ pointless | attacks per-op compute; workload was launch-bound (see 256M row below) |
| **Full forward (SigLIP + VLM prefix + 3 denoise steps)** | **manual `torch.cuda.CUDAGraph`** | ✅✅ | one graph, bitwise-identical actions; 3 constant-H2D blockers folded (torch.tensor constants, HF patch mask, NaViT pos-ids) |
| Full forward | `torch.compile(cudagraphs)` | ❌ | dynamo breaks on the flow loop → ~4% only |
| Full forward | `torch.compile` inductor | ❌ | no triton wheel for CUDA-11.8 aarch64 |

## Knobs tried

- [x] Jetson power mode / `jetson_clocks` (GPU 114→1109 MHz) — **no change** ⇒ not clock-bound
- [x] FP16 cast — 817 → ~630 ms (tensor cores); bf16 is a trap (1795 ms, emulated)
- [x] **CUDA graph capture for the static path — THE lever: 608 → 233 ms, bitwise-exact**
- [x] 256M backbone (SmolVLM2-256M, width): eager 577 ms (**no help** — launch-bound);
      graphed **196 ms** (replay 176 ms, 0.61 GB) — width only pays once dispatch is gone.
      Would need a 256M fine-tune to use.
- [x] Depth (fewer VLM layers): 16/8/4 = 613/441/287 ms eager — linear, needs retrain
- [x] INT8 — not pursued on GPU: launch-bound ⇒ can't help; TRT-INT8 calibration OOMs anyway
- [x] VLM-stage decimation — subsumed by chunking: the whole net (VLM included) already runs
      once per 50-action chunk (see async notes below)
- [ ] Action chunk horizon sweep — not needed at 233 ms

## Takeaways

The Xavier NX was never compute-starved for SmolVLA — it was **dispatch-starved**: eager
fp16 leaves the GPU idle ~80% while the weak CPU issues thousands of small kernel launches,
which is why precision, width (256M), INT8, and clocks all did nothing. Recording the whole
static-shape forward as **one CUDA graph** removed the dispatch entirely and hit **233
ms/chunk with bitwise-identical actions** — laptop-class latency (the original 230 ms
target) with no retrain and no quantization. Once dispatch is gone the board becomes
compute-bound again, so model-size levers reactivate (256M → 196 ms, 4-layer → ~150 ms
territory with retrain). Served over gRPC from the NX, the chunk arrives in ~250 ms RTT
(~15 ms network), well inside the async stack's operating window.
