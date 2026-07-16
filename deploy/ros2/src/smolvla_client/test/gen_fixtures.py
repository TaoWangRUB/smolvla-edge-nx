"""Export aggregation fixtures from the Python reference implementation (task 3.5).

Runs INSIDE the sim container (the reference lives there) and writes
aggregate_fixtures.txt, which the C++ gtest replays against smolvla::aggregate_chunks
and smolvla::ramp_in. Regenerate after any change to async_infer.aggregate_chunks:

    docker compose run --rm shell python deploy/ros2/src/smolvla_client/test/gen_fixtures.py

Format (whitespace-separated):
    n_cases
    per case:  how old_rows new_rows dim ramp_in has_last(0|1)
               old_rows lines of dim floats, new_rows lines, [last_action line],
               new_rows lines of expected output
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/src")
from smolvla_edge.async_infer import aggregate_chunks  # noqa: E402


def expected(old, new, how, ramp_in, last_action):
    merged = aggregate_chunks(old, new, how)
    if ramp_in > 0 and last_action is not None:  # mirror AsyncRunner._merge
        k = min(ramp_in, len(merged))
        w = np.linspace(1.0 / (k + 1), k / (k + 1.0), k)[:, None]
        merged = merged.copy()
        merged[:k] = (1.0 - w) * np.asarray(last_action) + w * merged[:k]
    return merged


def main() -> None:
    rng = np.random.default_rng(42)
    cases = []
    for how in ("new_wins", "blend"):
        for old_rows, new_rows, dim in [(0, 5, 3), (1, 4, 3), (4, 4, 2), (3, 8, 14), (8, 8, 14)]:
            for ramp_in in (0, 3):
                old = rng.normal(size=(old_rows, dim)).astype(np.float32)
                new = rng.normal(size=(new_rows, dim)).astype(np.float32)
                last = rng.normal(size=dim).astype(np.float32) if ramp_in else None
                cases.append((how, old, new, ramp_in, last,
                              expected(old, new, how, ramp_in, last)))

    out = [str(len(cases))]
    for how, old, new, ramp_in, last, exp in cases:
        out.append(f"{how} {len(old)} {len(new)} {new.shape[1]} {ramp_in} {int(last is not None)}")
        for arr in (old, new):
            out += [" ".join(f"{v:.9g}" for v in row) for row in arr]
        if last is not None:
            out.append(" ".join(f"{v:.9g}" for v in last))
        out += [" ".join(f"{v:.9g}" for v in row) for row in exp]

    path = Path(__file__).parent / "aggregate_fixtures.txt"
    path.write_text("\n".join(out) + "\n")
    print(f"[gen_fixtures] wrote {len(cases)} cases -> {path}")


if __name__ == "__main__":
    main()
