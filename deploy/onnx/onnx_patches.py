"""Export-time model patches so the ONNX graph is all-CUDA (ros2-cpp-async-deployment, D6).

lerobot's SmolVLA `apply_rope` uses `torch.split` (traces to the CPU-only `SplitToSequence`
op) and an in-place index-assign (traces to `ScatterND`). Either keeps the graph off a single
execution provider, which blocks ONNX Runtime's CUDA Graphs (it requires every node on the CUDA
EP). CUDA Graphs is the fix for this launch-overhead-bound graph (~30k tiny nodes) — capturing
the kernel sequence once and replaying it removes the ~600 ms/inference CPU dispatch cost.

`apply_rope` rewritten with slicing + `torch.cat` is numerically identical (verified by the
parity gate) but emits only `Slice`/`Concat` (both CUDA-supported).
"""

from __future__ import annotations


def patch_rope_no_sequence_ops() -> None:
    import torch
    import lerobot.policies.smolvla.smolvlm_with_expert as swe

    def apply_rope(x, positions, max_wavelength: int = 10_000):
        d_half = x.shape[-1] // 2
        device = x.device
        dtype = x.dtype
        x = x.to(torch.float32)

        freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32,
                                                            device=device)
        timescale = max_wavelength ** freq_exponents
        radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
        radians = radians[..., None, :]
        sin = torch.sin(radians)
        cos = torch.cos(radians)

        x1 = x[..., :d_half]          # slice, not split -> Slice (CUDA), not SplitToSequence (CPU)
        x2 = x[..., d_half:]
        res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)  # cat, not in-place
        return res.to(dtype)

    swe.apply_rope = apply_rope
