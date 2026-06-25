# On-device conversion log (Xavier NX, 8 GB)

Living log of optimization attempts. Fill in as you go — this *is* the differentiating
content for the writeup. Be specific and honest.

## Baseline (fill in)

| Config | Latency mean (ms) | p95 (ms) | Throughput (Hz) | Peak mem (MB) |
|--------|-------------------|----------|-----------------|---------------|
| FP32, no chunking |  |  |  |  |
| FP16, no chunking |  |  |  |  |
| FP16, chunking    |  |  |  |  |

## Per-stage latency budget (fill in)

| Stage | ms | Notes |
|-------|----|-------|
| Image preprocess |  |  |
| VLM (SmolVLM-2) backbone |  | run at low Hz |
| Action expert / flow-matching sampler |  | N sampling steps |
| Postprocess / queue |  |  |
| **End-to-end** |  |  |

## What converts / what doesn't (TensorRT / quantization)

| Subgraph | Target | Result | Notes |
|----------|--------|--------|-------|
| Vision encoder | TRT FP16/INT8 | ☐ | static shapes — most likely to convert |
| LLM/connector | TRT | ☐ | dynamic shapes / KV cache — expect friction |
| Flow-matching sampler loop | TRT | ☐ | iterative control flow — likely stays in PyTorch |

## Knobs tried

- [ ] Jetson power mode / `jetson_clocks` (record `nvpmodel -q`)
- [ ] FP16 cast
- [ ] Action chunk horizon sweep
- [ ] VLM-stage decimation (run every k control steps)
- [ ] INT8 quantization (only on convertible subgraphs)
- [ ] CUDA graph capture for the static path

## Takeaways

> Write the 3–5 sentence summary here once numbers are in: the budget you hit, the biggest
> lever, and the honest limitation.
