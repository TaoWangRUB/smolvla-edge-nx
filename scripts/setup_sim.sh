#!/usr/bin/env bash
# Verified simulation setup for the no-robot correctness path (ALOHA sim / gym-aloha).
#
# Why this isn't just `pip install -r requirements.txt`:
#   gym-aloha 0.1.1 pins `mujoco<3.0.0` -> resolves to mujoco 2.3.7, which has NO Python 3.12
#   wheel and won't build from source (needs the MuJoCo C SDK). On Python 3.12 we instead install
#   a modern mujoco 3.x + a matching dm_control, then add gym-aloha with --no-deps to bypass its
#   stale pin. gym-aloha is a thin dm_control wrapper, so mujoco 3.x works at runtime.
#
# On Python 3.10 / 3.11 the plain `pip install gym-aloha==0.1.1` works (mujoco 2.3.7 has wheels).
#
# This was verified on: Python 3.12, torch 2.10.0+cu128, mujoco 3.10.0, dm_control 1.0.43,
# lerobot 0.5.0, gym-aloha 0.1.1 — WSL2 + RTX 2000 Ada, headless via EGL.
set -euo pipefail

PIP="${PIP:-python3 -m pip}"
FLAGS="${PIP_FLAGS:-}"   # e.g. "--user --break-system-packages" on a PEP-668 system python

echo "[setup_sim] installing lerobot + a matched mujoco 3.x / dm_control ..."
$PIP install $FLAGS "lerobot==0.5.0" "mujoco" "dm-control" huggingface_hub

echo "[setup_sim] adding gym-aloha (no-deps, to skip its stale mujoco<3 pin) ..."
$PIP install $FLAGS --no-deps "gym-aloha==0.1.1"

echo "[setup_sim] smoke-test the env (headless EGL) ..."
MUJOCO_GL=egl python3 - <<'PY'
import gymnasium as gym, gym_aloha
env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
obs, _ = env.reset(seed=0); env.step(env.action_space.sample()); env.close()
print("[setup_sim] OK: gym_aloha env resets + steps.")
PY

cat <<'NOTE'

[setup_sim] done. Next:
  # verify-first: run a pretrained ALOHA policy through the harness (no fine-tune)
  MUJOCO_GL=egl python -m smolvla_edge.eval --mode sim \
      --policy-path lerobot/act_aloha_sim_insertion_human \
      --env-id gym_aloha/AlohaInsertion-v0 --episodes 5 --task ""

Always export MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (CPU) for headless rendering.
NOTE
