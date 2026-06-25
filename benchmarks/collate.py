"""Collate raw benchmark JSON files into a CSV + a markdown table.

    python benchmarks/collate.py
    # reads benchmarks/results/raw/*.json
    # writes benchmarks/results/summary.csv and prints a markdown table
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"
OUT_CSV = HERE / "results" / "summary.csv"

# Union of fields emitted by bench.py and the gRPC client; missing -> blank.
FIELDS = [
    "tag",
    "device",
    "precision",
    "chunking",
    "steps_measured",
    "latency_mean_ms",
    "latency_p95_ms",
    "rtt_mean_ms",
    "rtt_p95_ms",
    "server_compute_mean_ms",
    "network_overhead_mean_ms",
    "throughput_hz",
    "peak_gpu_mem_mb",
]


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"no raw results dir: {RAW} (run some benchmarks first)")
    rows = []
    for p in sorted(RAW.glob("*.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        raise SystemExit(f"no *.json in {RAW}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[collate] wrote {OUT_CSV} ({len(rows)} rows)")

    # Markdown table to stdout (paste into benchmarks/README.md).
    print("\n| " + " | ".join(FIELDS) + " |")
    print("|" + "|".join(["---"] * len(FIELDS)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(k, "")) for k in FIELDS) + " |")


if __name__ == "__main__":
    main()
