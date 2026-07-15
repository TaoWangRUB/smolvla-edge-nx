"""Reproduce SmolVLA paper Figure 3: action-queue size |A_t| over control ticks.

Feed it one eval JSON per g value (produced by
`python -m smolvla_edge.eval --mode sim --inference async --save-traces --out <json>`);
it plots the per-tick queue trace of one episode per run, one line per g.

    python benchmarks/plot_async_queue.py \
        benchmarks/results/raw/async_g0.json \
        benchmarks/results/raw/async_g07.json \
        benchmarks/results/raw/async_g10.json \
        --episode 0 --out benchmarks/results/async_queue_trace.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the skill reference palette (validated, fixed order).
SERIES = ["#2a78d6", "#1baf7a", "#eda100"]
INK = "#3d3d3a"  # text/axis ink: neutral, never a series hue
GRID = "#e6e5e0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="eval --out JSON files (with --save-traces)")
    ap.add_argument("--episode", type=int, default=0, help="episode index to trace")
    ap.add_argument("--out", default="benchmarks/results/async_queue_trace.png")
    args = ap.parse_args()

    if len(args.runs) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs (one categorical slot each)")

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    for path, color in zip(args.runs, SERIES):
        with open(path) as f:
            run = json.load(f)
        ep = run["per_episode"][args.episode]
        trace = ep.get("trace")
        if not trace:
            raise SystemExit(f"{path}: no trace — re-run eval with --save-traces")
        ticks = [ev["tick"] for ev in trace]
        qsize = [ev["queue_after"] for ev in trace]
        g = run["inference"]["g"]
        ax.plot(ticks, qsize, color=color, linewidth=1.6, label=f"g = {g:g}")
        # direct label at the line's end, in the series hue next to neutral text
        ax.annotate(f"g = {g:g}", (ticks[-1], qsize[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=color, va="center")

    n = run["per_episode"][args.episode]["chunk_size"]
    ax.set_ylim(0, n * 1.08)
    ax.set_xlabel("control tick (Δt = 1/fps)", color=INK)
    ax.set_ylabel("action queue size |Aₜ|", color=INK)
    ax.set_title("Async inference: queue evolution vs. threshold g (paper Fig. 3)",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
