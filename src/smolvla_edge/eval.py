"""Phase 1 deliverable: evaluate a checkpoint and report a success-rate number.

Two eval styles are supported, in order of fidelity:

1. **Sim rollout** (default; the number you quote): the policy drives a gym-aloha MuJoCo env and
   success is the env's own success flag (reward >= 4). No robot required. Needs `gym-aloha`.
2. **Open-loop replay** (fallback, no sim install): step through held-out episodes and measure
   action-prediction agreement (MSE / threshold accuracy) against logged actions. Cheap, but a
   proxy — label it as such in the writeup.

    # VERIFY-FIRST (no fine-tune): run a pretrained policy already trained on ALOHA sim through
    # the harness to confirm env + rollout + obs-mapping work and get a baseline success rate.
    python -m smolvla_edge.eval \
        --policy-path lerobot/act_aloha_sim_insertion_human \
        --mode sim --env-id gym_aloha/AlohaInsertion-v0 --episodes 20 --task ""

    # closed-loop success rate for YOUR fine-tuned SmolVLA (the deliverable):
    python -m smolvla_edge.eval \
        --policy-path outputs/train/smolvla_aloha/checkpoints/last \
        --mode sim --env-id gym_aloha/AlohaInsertion-v0 --episodes 20

    # open-loop proxy (no sim):
    python -m smolvla_edge.eval --policy-path <ckpt> --mode replay \
        --dataset-repo-id lerobot/aloha_sim_insertion_human
"""

from __future__ import annotations

import argparse

from .common import load_dataset, load_policy


def eval_replay(policy, device, ds, threshold: float, max_frames: int) -> dict:
    """Open-loop replay proxy: compare predicted vs. logged actions on held-out frames."""
    import torch

    abs_errors: list[float] = []
    within: list[float] = []
    policy.reset()
    n = min(len(ds), max_frames)
    for i in range(n):
        frame = ds[i]
        batch = {
            k: (v.to(device).unsqueeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in frame.items()
        }
        gt = frame.get("action")
        if gt is None:
            continue
        with torch.no_grad():
            pred = policy.select_action(batch).squeeze(0).cpu()
        err = (pred - gt.cpu()).abs()
        abs_errors.append(err.mean().item())
        within.append((err < threshold).float().mean().item())

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    acc = sum(within) / len(within) if within else float("nan")
    return {
        "mode": "replay",
        "frames": len(abs_errors),
        "action_mae": mae,
        f"within_{threshold}_acc": acc,
        "note": "open-loop proxy, NOT a true task success rate",
    }


# Observation-key mapping from gym-aloha -> the SmolVLA policy batch.
#
# VERIFY THESE against your checkpoint on the dev box: the exact keys must match the
# checkpoint's `policy.config.input_features`. For LeRobot ALOHA sim the convention is a top
# camera at `observation.images.top` and the 14-D joint state at `observation.state`. gym-aloha
# returns cameras under `pixels` (dict) or `images`, and state under `agent_pos` (obs_type
# "pixels_agent_pos") or `qpos`. This helper is defensive about both.
IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"


def _aloha_obs_to_batch(obs: dict, device: str, task: str | None = None) -> dict:
    """Map one gym-aloha observation dict to a batched policy input dict.

    `task` is included only for language-conditioned policies (SmolVLA). Pass task=None/"" for
    ACT/diffusion checkpoints, which have no language input.
    """
    import numpy as np
    import torch

    cams = obs.get("pixels", obs.get("images"))
    top = cams.get("top") if isinstance(cams, dict) else cams  # HxWx3 uint8
    if isinstance(cams, dict) and "top" not in cams:
        top = next(iter(cams.values()))
    img = torch.from_numpy(np.asarray(top)).permute(2, 0, 1).float() / 255.0  # CHW in [0,1]

    state_np = obs.get("agent_pos", obs.get("qpos"))
    state = torch.from_numpy(np.asarray(state_np, dtype=np.float32))

    batch = {
        IMAGE_KEY: img.unsqueeze(0).to(device),
        STATE_KEY: state.unsqueeze(0).to(device),
    }
    if task:
        batch["task"] = [task]  # one instruction per batch element
    return batch


def _load_normalizers(policy, policy_path: str, device: str):
    """Return (normalize_obs, unnormalize_action) callables.

    LeRobot 0.5.0 moved input/output normalization OUT of the policy model into a separate
    processor pipeline, so calling `select_action` on raw inputs yields garbage. We recover the
    normalization two ways:
      1. New-format checkpoints (saved by LeRobot >=0.5.0) ship processor configs -> use the
         official `make_pre_post_processors`.
      2. Old-format checkpoints bake the stats into `model.safetensors` under
         `normalize_inputs.*` / `unnormalize_outputs.*` -> apply them manually.
    Falls back to identity (with a warning) if neither is found.
    """
    import torch

    # 0) policy normalizes internally (lerobot <=0.4.x): its state_dict carries loaded
    #    normalize/unnormalize buffers, and select_action applies them itself. Applying stats
    #    again here would DOUBLE-normalize — so do nothing.
    norm_keys = [k for k in policy.state_dict() if "normalize" in k.lower()]
    if norm_keys:
        print(f"[eval] normalization: internal to the policy ({len(norm_keys)} buffers) — no-op")
        return (lambda b: b), (lambda a: a)

    # 1) official processor pipeline (preferred; what a fine-tuned 0.5.0 checkpoint has)
    try:
        from lerobot.policies.factory import make_pre_post_processors

        pre, post = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        print("[eval] normalization: LeRobot processor pipeline")
        return (lambda b: pre(b)), (lambda a: post(a))
    except Exception as e:
        print(f"[eval] no processor pipeline ({type(e).__name__}); trying baked-in stats")

    # 2) stats baked into an old-format checkpoint's model.safetensors
    try:
        import os

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        st = (
            os.path.join(policy_path, "model.safetensors")
            if os.path.isdir(policy_path)
            else hf_hub_download(policy_path, "model.safetensors")
        )
        sd = load_file(st)
        need = [
            "normalize_inputs.buffer_observation_images_top.mean",
            "normalize_inputs.buffer_observation_images_top.std",
            "normalize_inputs.buffer_observation_state.mean",
            "normalize_inputs.buffer_observation_state.std",
            "unnormalize_outputs.buffer_action.mean",
            "unnormalize_outputs.buffer_action.std",
        ]
        if all(k in sd for k in need):
            g = {k: sd[k].to(device) for k in need}
            im = g["normalize_inputs.buffer_observation_images_top.mean"]
            istd = g["normalize_inputs.buffer_observation_images_top.std"]
            sm = g["normalize_inputs.buffer_observation_state.mean"]
            sstd = g["normalize_inputs.buffer_observation_state.std"]
            am = g["unnormalize_outputs.buffer_action.mean"]
            astd = g["unnormalize_outputs.buffer_action.std"]

            def norm(b):
                b = dict(b)
                b[IMAGE_KEY] = (b[IMAGE_KEY] - im) / istd
                b[STATE_KEY] = (b[STATE_KEY] - sm) / sstd
                return b

            print("[eval] normalization: baked-in checkpoint stats (old format)")
            return norm, (lambda a: a * astd + am)
    except Exception as e:
        print(f"[eval] baked-in stats unavailable ({type(e).__name__})")

    print("[eval] WARNING: no normalization found — actions will be UN-normalized (garbage).")
    return (lambda b: b), (lambda a: a)


def _batchify_obs(obs: dict):
    """Add a leading batch axis to a single gym observation (mimics a 1-env vector env)."""
    import numpy as np

    out = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            out[k] = {kk: np.asarray(vv)[None] for kk, vv in v.items()}
        else:
            out[k] = np.asarray(v)[None]
    return out


def make_sim_stepper(policy, policy_path: str, device: str, task: str | None):
    """Return step_fn(obs) -> np action, handling checkpoint-format differences in one place.

    1. **Pipeline mode** (fresh lerobot 0.4.x/0.5.x checkpoints, e.g. your fine-tuned SmolVLA):
       the checkpoint ships a saved processor pipeline that does key-renaming, batching, task
       tokenization, device placement and normalization. We mirror lerobot_eval exactly:
       preprocess_observation -> pre() -> select_action -> post().
    2. **Manual mode** (old hub checkpoints without processor files, e.g. the pretrained ACT
       baselines): hand-built obs mapping + stats recovered by `_load_normalizers`, plus manual
       language tokenization for language-conditioned policies.
    """
    import torch

    from .common import resolve_policy_path

    policy_path = resolve_policy_path(policy_path)

    try:
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_pre_post_processors

        pre, post = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        def step_pipeline(obs):
            o = preprocess_observation(_batchify_obs(obs))
            if task:
                o["task"] = [task]
            o = pre(o)
            with torch.no_grad():
                action = policy.select_action(o)
            return post(action).squeeze(0).float().cpu().numpy()

        print("[eval] stepper: saved processor pipeline (rename/tokenize/normalize)")
        return step_pipeline
    except Exception as e:
        print(f"[eval] no processor pipeline ({type(e).__name__}); using manual obs mapping")

    normalize_obs, unnormalize_action = _load_normalizers(policy, policy_path, device)
    from .common import make_language_tokenizer

    tokenize = make_language_tokenizer(policy, device)

    def step_manual(obs):
        batch = normalize_obs(_aloha_obs_to_batch(obs, device, task))
        if tokenize is not None:
            batch.update(tokenize(task or "do the task"))
        with torch.no_grad():
            action = unnormalize_action(policy.select_action(batch))
        return action.squeeze(0).float().cpu().numpy()

    return step_manual


def eval_sim(
    policy,
    policy_path: str,
    device,
    env_id: str,
    task: str,
    n_episodes: int,
    max_steps: int,
    obs_type: str = "pixels_agent_pos",
) -> dict:
    """Closed-loop rollout in gym-aloha; the env's success flag is the quoted success rate.

    gym-aloha gives a max reward of 4 on task success (peg insertion / cube transfer), so an
    episode is a success iff it reaches reward >= 4. This is the true (closed-loop) success rate,
    unlike the open-loop `eval_replay` proxy.

    Requires `pip install gym-aloha`. The obs->batch key mapping (see `_aloha_obs_to_batch`) is a
    best-effort default; verify it against `policy.config.input_features` on the dev box.
    """
    import gymnasium as gym
    import gym_aloha  # noqa: F401  # registers the gym_aloha/* env ids

    stepper = make_sim_stepper(policy, policy_path, device, task or None)
    env = gym.make(env_id, obs_type=obs_type, render_mode="rgb_array")
    successes = 0
    max_rewards: list[float] = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            policy.reset()
            ep_max_r = 0.0
            for _ in range(max_steps):
                act_np = stepper(obs)
                obs, reward, terminated, truncated, _ = env.step(act_np)
                ep_max_r = max(ep_max_r, float(reward))
                if terminated or truncated:
                    break
            successes += int(ep_max_r >= 4.0)
            max_rewards.append(ep_max_r)
            print(f"[eval] episode {ep + 1}/{n_episodes}: max_reward={ep_max_r:.1f} "
                  f"success={ep_max_r >= 4.0}")
    finally:
        env.close()

    return {
        "mode": "sim",
        "env_id": env_id,
        "episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes if n_episodes else float("nan"),
        "mean_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else float("nan"),
        "note": "closed-loop gym-aloha success rate (reward>=4)",
    }


def eval_sim_async(
    policy,
    policy_path: str,
    device,
    env_id: str,
    task: str,
    n_episodes: int,
    max_steps: int,
    obs_type: str = "pixels_agent_pos",
    g: float = 0.7,
    epsilon: float = 0.0,
    fps: float = 50.0,
    aggregate: str = "new_wins",
    idle: str = "hold",
    save_traces: bool = False,
    precision: str = "fp32",
    flow_steps: int | None = None,
    server: str | None = None,
    ramp_in: int = 0,
) -> dict:
    """Closed-loop rollout under the async inference stack (SmolVLA §3.3, Algorithm 1).

    Same env/success protocol as `eval_sim` (identical seeds, reward>=4 == success), but the
    policy serves full action chunks from a background worker and the client loop pops one
    action per control tick. Idle ticks (queue empty before the next chunk lands, in virtual
    time at `fps`) execute a hold action — the sim analogue of the robot standing still.
    `--g 0` is the paper's sequential (sync) limit of the same loop.
    """
    import gymnasium as gym
    import gym_aloha  # noqa: F401  # registers the gym_aloha/* env ids

    from .async_infer import AsyncRunner, make_chunk_predictor

    if server:
        from .remote import make_remote_chunk_predictor, split_latency_summary

        predict = make_remote_chunk_predictor(server, task or None)
        print(f"[eval] chunk predictor: remote PolicyServer at {server}")
    else:
        if flow_steps is not None:
            if not hasattr(policy.config, "num_steps"):
                raise SystemExit("--flow-steps only applies to flow-matching policies (SmolVLA)")
            policy.config.num_steps = flow_steps
        predict = make_chunk_predictor(policy, policy_path, device, task or None,
                                       precision=precision)
    runner = AsyncRunner(predict, g=g, epsilon=epsilon, dt=1.0 / fps,
                         aggregate=aggregate, idle=idle, ramp_in=ramp_in)
    # override the env's registered 400-step TimeLimit: under the fixed-time
    # protocol idle ticks eat into the budget, so the cap must be ours
    env = gym.make(env_id, obs_type=obs_type, render_mode="rgb_array",
                   max_episode_steps=max_steps)
    episodes: list[dict] = []
    try:
        # warmup: the first CUDA call pays compile/alloc cost (~3x a warm call);
        # keep it out of episode latency stats and out of episode 0's queue timing
        warm_obs, _ = env.reset(seed=0)
        predict(warm_obs)
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            if policy is not None:
                policy.reset()
            runner.start_episode(obs)
            last_action = None
            ep_max_r, success_tick = 0.0, None
            for _ in range(max_steps):
                action, _ev = runner.act(obs)
                if action is None:  # idle tick: hold the last commanded pose
                    action = last_action
                last_action = action
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_max_r = max(ep_max_r, float(reward))
                if success_tick is None and float(reward) >= 4.0:
                    success_tick = runner.tick
                if terminated or truncated:
                    break
            stats = runner.episode_stats()
            stats.update(episode=ep, max_reward=ep_max_r,
                         success=ep_max_r >= 4.0, success_tick=success_tick)
            if save_traces:
                stats["trace"] = runner.trace
            episodes.append(stats)
            print(f"[eval] episode {ep + 1}/{n_episodes}: max_reward={ep_max_r:.1f} "
                  f"success={ep_max_r >= 4.0} idle={stats['idle_ticks']} "
                  f"sent={stats['obs_sent']} success_tick={success_tick}")
    finally:
        env.close()
        runner.close()

    n = len(episodes)
    successes = sum(e["success"] for e in episodes)
    ticks_to_success = [e["success_tick"] for e in episodes if e["success_tick"] is not None]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "mode": "sim-async",
        "env_id": env_id,
        "inference": {"g": g, "epsilon": epsilon, "fps": fps,
                      "aggregate": aggregate, "idle": idle, "precision": precision,
                      "flow_steps": flow_steps},
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n if n else float("nan"),
        "mean_max_reward": mean([e["max_reward"] for e in episodes]),
        "mean_ticks_to_success": mean(ticks_to_success),
        "mean_idle_ticks": mean([e["idle_ticks"] for e in episodes]),
        "mean_obs_sent": mean([e["obs_sent"] for e in episodes]),
        "mean_obs_filtered": mean([e["obs_filtered"] for e in episodes]),
        "mean_latency_s": mean([e["mean_latency_s"] for e in episodes]),
        "median_latency_s": mean([e["median_latency_s"] for e in episodes]),
        "per_episode": episodes,
        **({"remote": split_latency_summary(predict.calls)} if server else {}),
        "note": "async inference stack (Algorithm 1); success = gym-aloha reward>=4",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a SmolVLA checkpoint.")
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset-repo-id", default="lerobot/aloha_sim_insertion_human")
    ap.add_argument("--mode", choices=["sim", "replay"], default="sim")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=2000, help="(replay) frame cap")
    ap.add_argument("--max-steps", type=int, default=400, help="(sim) steps per episode")
    ap.add_argument("--threshold", type=float, default=0.05, help="(replay) per-dim tol")
    ap.add_argument("--env-id", default="gym_aloha/AlohaInsertion-v0", help="(sim) gym env id")
    ap.add_argument("--task", default="insert the peg into the socket", help="(sim) instruction")
    ap.add_argument("--obs-type", default="pixels_agent_pos", help="(sim) gym-aloha obs_type")
    ap.add_argument("--inference", choices=["sync", "async"], default="sync",
                    help="(sim) sync = legacy select_action loop; async = Algorithm 1 stack")
    ap.add_argument("--g", type=float, default=0.7,
                    help="(async) queue threshold in [0,1]; 0 = sequential/sync limit")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="(async) joint-space similarity filter; 0 disables")
    ap.add_argument("--fps", type=float, default=50.0,
                    help="(async) control rate defining the virtual-time tick (ALOHA: 50)")
    ap.add_argument("--aggregate", choices=["new_wins", "blend"], default="new_wins",
                    help="(async) chunk aggregation f on overlapping timesteps")
    ap.add_argument("--idle", choices=["hold", "freeze"], default="hold",
                    help="(async) hold = emulate idle ticks; freeze = legacy frozen-env")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="(async) fp16 = autocast around chunk prediction (cuts l_S)")
    ap.add_argument("--flow-steps", type=int, default=None,
                    help="(async, SmolVLA) override flow-matching denoising steps; "
                    "the expert dominates l_S (~29 ms/step measured), not the VLM")
    ap.add_argument("--server", default=None,
                    help="(async) host:port of a gRPC PolicyServer; chunks are "
                    "computed remotely (separate process/host — no GIL contention)")
    ap.add_argument("--ramp-in", type=int, default=0,
                    help="(async) blend this many post-merge actions from the last "
                    "executed action, smoothing splice discontinuities")
    ap.add_argument("--out", default=None,
                    help="write the full result (incl. per-episode stats) to this JSON file")
    ap.add_argument("--save-traces", action="store_true",
                    help="(async) include per-tick queue traces in --out (Figure-3 data)")
    ap.add_argument(
        "--policy-type",
        default="auto",
        help="auto-detect from the checkpoint, or force smolvla/act/diffusion/... "
        "(use act/diffusion to verify the harness with a pretrained ALOHA policy)",
    )
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.mode == "sim" and args.inference == "async" and args.server:
        policy, device = None, "remote"  # thin client: the server owns the policy
    else:
        policy, device = load_policy(args.policy_path, args.device, args.policy_type)

    if args.mode == "sim" and args.inference == "async":
        result = eval_sim_async(
            policy, args.policy_path, device, args.env_id, args.task,
            args.episodes, args.max_steps, args.obs_type,
            g=args.g, epsilon=args.epsilon, fps=args.fps,
            aggregate=args.aggregate, idle=args.idle, save_traces=args.save_traces,
            precision=args.precision, flow_steps=args.flow_steps, server=args.server,
            ramp_in=args.ramp_in,
        )
    elif args.mode == "sim":
        result = eval_sim(
            policy, args.policy_path, device, args.env_id, args.task,
            args.episodes, args.max_steps, args.obs_type,
        )
    else:
        ds = load_dataset(args.dataset_repo_id, episodes=list(range(args.episodes)))
        result = eval_replay(policy, device, ds, args.threshold, args.max_frames)

    if args.out:
        import json
        import os

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] wrote {args.out}")

    print("[eval] result:")
    for k, v in result.items():
        if k == "per_episode":
            continue  # bulky; in --out JSON
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
