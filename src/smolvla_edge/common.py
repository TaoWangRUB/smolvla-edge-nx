"""Shared helpers for inference, eval, and benchmarking.

These wrap LeRobot so the entrypoints stay thin. They are written defensively: LeRobot's
exact import paths have shifted across releases, so loading is centralized here and pinned
against v0.5.0 (see requirements.txt).
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Iterator


def select_device(requested: str = "auto") -> str:
    """Resolve a torch device string. 'auto' prefers CUDA, then MPS, then CPU."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_policy(policy_path: str, device: str = "auto"):
    """Load a SmolVLA policy from a local checkpoint dir or a HF hub repo id.

    Args:
        policy_path: e.g. "lerobot/smolvla_base" or "outputs/train/.../checkpoints/last".
        device: torch device string or "auto".
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = select_device(device)
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.to(dev)
    policy.eval()
    return policy, dev


def load_dataset(repo_id: str, episodes: list[int] | None = None):
    """Load a LeRobot dataset from the hub (cached locally on first use)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, episodes=episodes)


@dataclass
class Timer:
    """Accumulating wall-clock timer with CUDA-sync awareness.

    Usage:
        t = Timer(device="cuda")
        with t.section("forward"):
            ...
        print(t.summary())
    """

    device: str = "cpu"
    samples: dict[str, list[float]] = field(default_factory=dict)

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.samples.setdefault(name, []).append(time.perf_counter() - start)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return per-section count/mean_ms/p50_ms/p95_ms."""
        out: dict[str, dict[str, float]] = {}
        for name, xs in self.samples.items():
            xs_sorted = sorted(xs)
            n = len(xs_sorted)
            mean_ms = 1e3 * sum(xs_sorted) / n
            p50 = 1e3 * xs_sorted[int(0.50 * (n - 1))]
            p95 = 1e3 * xs_sorted[int(0.95 * (n - 1))]
            out[name] = {"count": n, "mean_ms": mean_ms, "p50_ms": p50, "p95_ms": p95}
        return out


def peak_gpu_memory_mb(device: str) -> float | None:
    """Peak allocated CUDA memory since last reset, in MiB (None on non-CUDA)."""
    if not device.startswith("cuda"):
        return None
    import torch

    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_gpu_memory_stats(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()
