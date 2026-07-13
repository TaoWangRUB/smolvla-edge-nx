"""Assemble a demo GIF — either a live policy ROLLOUT in gym-aloha, or a dataset replay.

**rollout** (default; the real demo): run a policy closed-loop in the sim via the same
machinery as `smolvla_edge.eval`, render every step, and keep the first SUCCESSFUL episode
(retrying seeds). This shows the policy actually doing the task, not a recorded human demo.

    python scripts/make_demo_gif.py --mode rollout \
        --policy-path lerobot/act_aloha_sim_transfer_cube_human \
        --env-id gym_aloha/AlohaTransferCube-v0 --out benchmarks/results/demo.gif

**replay** (fallback; no policy needed): read frames straight out of a LeRobot dataset.
Use --max-frames large enough to cover the whole episode (ALOHA sim episodes are ~500
frames @ 50 fps), and --stride to keep the GIF small and ~real-time.

    python scripts/make_demo_gif.py --mode replay \
        --dataset-repo-id lerobot/aloha_sim_insertion_human --max-frames 500 --stride 3
"""

from __future__ import annotations

import argparse
from pathlib import Path


def find_image_key(frame: dict) -> str:
    for k, v in frame.items():
        if "image" in k.lower() and hasattr(v, "ndim") and v.ndim == 3:
            return k
    raise SystemExit(f"No image-like key found in frame keys: {list(frame)}")


def to_uint8_hwc(t):
    import numpy as np

    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    if arr.shape[0] in (1, 3) and arr.ndim == 3:  # CHW -> HWC
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != "uint8":
        arr = (arr.clip(0, 1) * 255).astype("uint8") if arr.max() <= 1.0 else arr.astype("uint8")
    return arr


def frames_replay(args) -> list:
    """Read frames straight from the dataset (human demo replay)."""
    from smolvla_edge.common import load_dataset

    ds = load_dataset(args.dataset_repo_id, episodes=list(range(args.episodes)))
    img_key = find_image_key(ds[0])
    n = min(len(ds), args.max_frames)
    print(f"[gif] replay: {n}/{len(ds)} frames from {args.dataset_repo_id} (key={img_key})")
    return [to_uint8_hwc(ds[i][img_key]) for i in range(0, n, args.stride)]


CONTROL_HZ = 50  # gym-aloha control/render rate (dt = 20 ms)


def _overlay(frame_np, step: int, reward: float, success: bool):
    """Stamp sim time / control rate / reward status onto a frame (returns np.uint8 HWC)."""
    import numpy as np
    from PIL import Image, ImageDraw

    try:  # DejaVu ships with matplotlib; fall back to PIL's bitmap font
        import matplotlib
        from PIL import ImageFont
        from pathlib import Path

        ttf = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(str(ttf), 18)
    except Exception:
        font = None

    img = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(img, "RGBA")
    t_sim = step / CONTROL_HZ
    status = "SUCCESS" if success else f"reward {reward:.0f}/4"
    text = f"sim t = {t_sim:5.2f} s   |   {CONTROL_HZ} Hz control   |   {status}"
    draw.rectangle([0, 0, img.width, 30], fill=(0, 0, 0, 160))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def frames_rollout(args) -> list:
    """Run the policy closed-loop in gym-aloha and render; keep the first successful episode.

    The env terminates the instant reward hits 4, which would cut the GIF at the moment of
    success — so after success we keep stepping the physics for --post-steps (the "epilogue":
    the arms holding the handover) and then hold the final frame for --hold seconds.
    """
    import gymnasium as gym
    import gym_aloha  # noqa: F401
    import torch

    from smolvla_edge.common import load_policy
    from smolvla_edge.eval import _aloha_obs_to_batch, _load_normalizers

    policy, device = load_policy(args.policy_path, args.device)
    normalize_obs, unnormalize_action = _load_normalizers(policy, args.policy_path, device)
    env = gym.make(args.env_id, obs_type="pixels_agent_pos", render_mode="rgb_array")

    best: list = []  # (frame, step, reward, success_so_far)
    best_reward = -1.0
    try:
        for seed in range(args.max_tries):
            obs, _ = env.reset(seed=seed)
            policy.reset()
            recs, ep_max_r, post = [], 0.0, None
            for step in range(args.max_steps + args.post_steps):
                recs.append((env.render(), step, ep_max_r, ep_max_r >= 4.0))
                batch = normalize_obs(_aloha_obs_to_batch(obs, device, args.task or None))
                with torch.no_grad():
                    action = unnormalize_action(policy.select_action(batch))
                obs, reward, terminated, truncated, _ = env.step(
                    action.squeeze(0).float().cpu().numpy()
                )
                ep_max_r = max(ep_max_r, float(reward))
                # On success, don't stop — run the epilogue so the GIF doesn't cut mid-handover.
                if post is None and (terminated or truncated or ep_max_r >= 4.0):
                    post = args.post_steps
                elif post is not None:
                    post -= 1
                    if post <= 0:
                        break
            print(f"[gif] rollout seed {seed}: max_reward={ep_max_r} "
                  f"({'SUCCESS' if ep_max_r >= 4.0 else 'no success'})")
            if ep_max_r > best_reward:
                best_reward, best = ep_max_r, recs
            if ep_max_r >= 4.0:
                break
    finally:
        env.close()

    if best_reward < 4.0:
        print(f"[gif] WARNING: no successful episode in {args.max_tries} tries; "
              f"using the best one (max_reward={best_reward})")

    kept = best[:: args.stride]
    frames = [
        _overlay(to_uint8_hwc(f), step, r, ok) if args.overlay else to_uint8_hwc(f)
        for f, step, r, ok in kept
    ]
    # Freeze the final frame so the end state is readable before the GIF loops.
    if frames and args.hold > 0:
        frames += [frames[-1]] * int(args.hold * args.fps)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a demo GIF (policy rollout or dataset replay).")
    ap.add_argument("--mode", choices=["rollout", "replay"], default="rollout")
    # rollout args
    ap.add_argument("--policy-path", default="lerobot/act_aloha_sim_transfer_cube_human")
    ap.add_argument("--env-id", default="gym_aloha/AlohaTransferCube-v0")
    ap.add_argument("--task", default="", help="language instruction (SmolVLA); empty for ACT")
    ap.add_argument("--max-steps", type=int, default=400, help="(rollout) steps per episode")
    ap.add_argument("--max-tries", type=int, default=5, help="(rollout) seeds to try for a success")
    ap.add_argument("--post-steps", type=int, default=75,
                    help="(rollout) extra sim steps after success so the GIF doesn't cut at the handover")
    ap.add_argument("--hold", type=float, default=1.5, help="(rollout) freeze final frame this many s")
    ap.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
                    help="stamp sim time / control Hz / reward onto rollout frames")
    ap.add_argument("--device", default="auto")
    # replay args
    ap.add_argument("--dataset-repo-id", default="lerobot/aloha_sim_insertion_human")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=500)
    # shared
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth sim frame (size/speed)")
    ap.add_argument("--fps", type=int, default=17,
                    help="GIF PLAYBACK fps (display rate, not the sim rate): the sim runs at "
                    "50 Hz control, so playback fps = 50/stride plays back at ~1x real time")
    ap.add_argument("--out", default="benchmarks/results/demo.gif")
    args = ap.parse_args()

    import imageio.v2 as imageio

    frames = frames_rollout(args) if args.mode == "rollout" else frames_replay(args)
    if not frames:
        raise SystemExit("[gif] no frames produced")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"[gif] wrote {out} ({len(frames)} frames @ {args.fps}fps, mode={args.mode})")


if __name__ == "__main__":
    main()
