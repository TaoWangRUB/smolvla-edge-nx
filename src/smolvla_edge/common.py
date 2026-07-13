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


# Fallback map of LeRobot policy `type` -> (module, class), used only if the official
# factory import fails. Both are version-sensitive; the `type` field lives in a checkpoint's
# config.json and is stable across LeRobot releases.
_POLICY_CLASS_BY_TYPE = {
    "smolvla": ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
    "act": ("lerobot.policies.act.modeling_act", "ACTPolicy"),
    "diffusion": ("lerobot.policies.diffusion.modeling_diffusion", "DiffusionPolicy"),
    "pi0": ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
    "vqbet": ("lerobot.policies.vqbet.modeling_vqbet", "VQBeTPolicy"),
    "tdmpc": ("lerobot.policies.tdmpc.modeling_tdmpc", "TDMPCPolicy"),
}


def _detect_policy_type(policy_path: str) -> str | None:
    """Read the policy `type` from a checkpoint dir or HF repo `config.json`."""
    import json
    from pathlib import Path

    local = Path(policy_path) / "config.json"
    if local.exists():
        return json.loads(local.read_text()).get("type")
    try:  # hub repo id -> pull just the config
        from huggingface_hub import hf_hub_download

        cfg = hf_hub_download(policy_path, "config.json")
        return json.loads(Path(cfg).read_text()).get("type")
    except Exception:
        return None


def _get_policy_class(policy_type: str):
    """Resolve a LeRobot policy class, preferring the official factory."""
    try:
        from lerobot.policies.factory import get_policy_class

        return get_policy_class(policy_type)
    except Exception:
        import importlib

        if policy_type not in _POLICY_CLASS_BY_TYPE:
            raise SystemExit(
                f"unknown policy type {policy_type!r}; known: {sorted(_POLICY_CLASS_BY_TYPE)}"
            )
        mod, cls = _POLICY_CLASS_BY_TYPE[policy_type]
        return getattr(importlib.import_module(mod), cls)


def load_policy(policy_path: str, device: str = "auto", policy_type: str = "auto"):
    """Load ANY LeRobot policy from a local checkpoint dir or a HF hub repo id.

    Auto-detects the policy class from the checkpoint's config (`type`), so you can load a
    pretrained SmolVLA (`lerobot/smolvla_base`) OR a pretrained ACT/diffusion checkpoint trained
    on a sim env (e.g. `lerobot/act_aloha_sim_insertion_human`) to verify the sim/eval harness
    with no fine-tuning.

    Args:
        policy_path: e.g. "lerobot/smolvla_base", "lerobot/act_aloha_sim_insertion_human", or a
            local "outputs/train/.../checkpoints/last".
        device: torch device string or "auto".
        policy_type: "auto" to read it from the checkpoint config, or force one of
            smolvla/act/diffusion/pi0/vqbet/tdmpc.
    """
    dev = select_device(device)
    ptype = policy_type if policy_type != "auto" else (_detect_policy_type(policy_path) or "smolvla")
    policy = _get_policy_class(ptype).from_pretrained(policy_path)
    policy.to(dev)
    policy.eval()
    print(f"[load_policy] loaded '{policy_path}' as policy type '{ptype}'")
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
