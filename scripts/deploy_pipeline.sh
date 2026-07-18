#!/usr/bin/env bash
# Deployment pipeline (openspec ros2-cpp-async-deployment §6): chain the deployment gates
# for one checkpoint, fail-fast with the failing gate named, artifacts + manifest under
# benchmarks/results/pipeline/<ts>/.
#
#   scripts/deploy_pipeline.sh <checkpoint-dir> [--episodes N]
#
# Gates, in order:
#   1 export      ONNX export (skipped when REUSE_ONNX=1 and the model already exists —
#                 the export is deterministic per checkpoint; meta.json records the hash)
#   2 parity      enforced ONNX-vs-torch gate (exit code; max-abs-diff <= 1e-4, cos >= 0.9999)
#   3 graph-smoke the shipped serving path: precision="fp16-graph" lazy CUDA-graph capture
#                 (validates capture + latency on this host's GPU)
#   4 closed-loop ROS2 stack vs the fp16-graph policy server, N episodes (default 5),
#                 gate: >= 60% success
#   5 collate     regenerate benchmarks/results/summary.csv
#
# Knobs: REUSE_ONNX=1 (skip export if present), EPISODES, PARITY_OBS (default 20 for the
# pipeline; the standalone gate default of 100 stays in parity.py).
set -uo pipefail

CKPT="${1:?usage: deploy_pipeline.sh <checkpoint-dir> [--episodes N]}"
shift || true
EPISODES="${EPISODES:-5}"
[ "${1:-}" = "--episodes" ] && EPISODES="$2"
PARITY_OBS="${PARITY_OBS:-20}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="benchmarks/results/pipeline/${TS}"
mkdir -p "$OUT"
ONNX="models/onnx/smolvla_transfer_cube.onnx"
TASK="Pick up the cube with the right arm and transfer it to the left arm."

MANIFEST="$OUT/manifest.json"
declare -A STATUS
write_manifest() {
    python3 - "$MANIFEST" "$CKPT" "$ONNX" <<'EOF'
import hashlib, json, os, sys
manifest, ckpt, onnx = sys.argv[1], sys.argv[2], sys.argv[3]
gates = json.loads(os.environ.get("GATES_JSON", "{}"))
def sha(p, n=1 << 22):
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:  # first 4 MB is enough to fingerprint the 1.8 GB graph
        h.update(f.read(n))
    return h.hexdigest()[:16]
model_file = os.path.join(ckpt, "pretrained_model", "model.safetensors")
json.dump({"checkpoint": ckpt, "checkpoint_sha16": sha(model_file),
           "onnx": onnx, "onnx_sha16_head": sha(onnx), "gates": gates},
          open(manifest, "w"), indent=2)
EOF
}
finish() {
    GATES_JSON="$(python3 -c "import json,sys; print(json.dumps(dict(kv.split('=',1) for kv in sys.argv[1:])))" \
        $(for k in "${!STATUS[@]}"; do echo "$k=${STATUS[$k]}"; done))" write_manifest
    echo "[pipeline] manifest: $MANIFEST"
}
trap finish EXIT

fail() { STATUS[$1]="FAIL"; echo "[pipeline] GATE FAILED: $1 (log: $OUT/$1.log)"; exit 1; }
run_gate() {  # run_gate <name> <cmd...>
    local name="$1"; shift
    echo "[pipeline] gate: $name"
    if "$@" > "$OUT/$name.log" 2>&1; then STATUS[$name]="PASS"; else fail "$name"; fi
}

# 1 — export (fs3 = the deployed config)
META="${ONNX%.onnx}.meta.json"
if [ "${REUSE_ONNX:-0}" = "1" ] && [ -f "$ONNX" ] && [ -f "$META" ]; then
    echo "[pipeline] gate: export — REUSED existing $ONNX"
    STATUS[export]="REUSED"
else
    run_gate export docker compose run --rm shell \
        python3 deploy/onnx/export_smolvla.py --checkpoint "$CKPT" --out "$ONNX" \
        --flow-steps 3 --task "$TASK"
fi

# 2 — parity (enforced gate: exit code). PARITY_FLOW_STEPS != 3 deliberately breaks the
# torch reference vs the fs3 graph — used to demonstrate fail-fast (task 6.3).
run_gate parity docker compose run --rm shell \
    python3 deploy/onnx/parity.py --checkpoint "$CKPT" --onnx "$ONNX" \
    --observations "$PARITY_OBS" --flow-steps "${PARITY_FLOW_STEPS:-3}" --task "$TASK" \
    --report "$OUT/onnx_parity.json"

# 3 — fp16-graph capture smoke (the shipped serving path)
run_gate graph-smoke docker compose run --rm -e CKPT="$CKPT" shell \
    python3 deploy/jetson-native-torch/smoke_fp16_graph.py

# 4 — closed-loop regression (ROS2 stack vs fp16-graph server)
echo "[pipeline] gate: closed-loop ($EPISODES eps)"
POLICY_PATH="$CKPT" POLICY_SERVER_ARGS="--flow-steps 3 --precision fp16-graph" \
    docker compose up -d sim-server policy-server >> "$OUT/closed-loop.log" 2>&1
for i in $(seq 1 36); do
    docker logs smolvla-edge-nx-policy-server-1 2>&1 | grep -q listening && break
    sleep 5
done
if docker compose run --rm ros2 bash -lc "source /opt/ros/jazzy/setup.bash 2>/dev/null; \
    source deploy/ros2/install/setup.bash && ros2 launch smolvla_bridge stage1.launch.py \
    episodes:=$EPISODES g:=0.5 ramp_in:=5 max_steps:=400 task:=\"$TASK\" \
    results_path:=/workspace/$OUT/closed_loop.json \
    events_path:=/workspace/$OUT/closed_loop_events.jsonl" >> "$OUT/closed-loop.log" 2>&1 \
   && python3 -c "
import json, sys
d = json.load(open('$OUT/closed_loop.json'))
rate = d['success_rate']
print(f'closed-loop success: {d[\"successes\"]}/{d[\"episodes\"]} = {rate:.0%}')
sys.exit(0 if rate >= 0.6 else 1)"; then
    STATUS[closed-loop]="PASS"
else
    docker compose stop sim-server policy-server >/dev/null 2>&1
    fail closed-loop
fi
docker compose stop sim-server policy-server >/dev/null 2>&1

# 5 — collate
run_gate collate python3 benchmarks/collate.py

echo "[pipeline] ALL GATES PASS — artifacts in $OUT"
