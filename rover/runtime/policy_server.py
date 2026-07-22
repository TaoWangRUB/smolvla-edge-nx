#!/usr/bin/env python3
"""Waypoint policy server (task 2.6, torch side).

Runs in the smolvla-edge:sim image on the Titan X. Serves the trained
SmolVLA rover checkpoint over a stdlib TCP socket (length-prefixed JSON
header + JPEG payload; see chunk_client_node.py). One request at a time —
the client's sequential loop is the async policy cadence.

  docker run --rm --runtime nvidia --shm-size=2g \
    -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=1 \
    -e HF_HOME=/work/.hf_cache -e PYTHONPATH=/work/src \
    -v "$PWD":/work -w /work --network host smolvla-edge:sim \
    bash -c "sed -i 's/torch_dtype=\"bfloat16\"/torch_dtype=\"float32\"/' \
      /opt/conda/lib/python3.11/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py && \
      python rover/runtime/policy_server.py --checkpoint <ckpt_dir>"
"""

import argparse
import json
import socket
import struct
import time

import cv2
import numpy as np
import torch


def recv_exact(conn, n):
    buf = b''
    while len(buf) < n:
        d = conn.recv(n - len(buf))
        if not d:
            raise ConnectionError('short read')
        buf += d
    return buf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True,
                    help='.../checkpoints/last/pretrained_model')
    ap.add_argument('--port', type=int, default=8790)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--no-arrival-assist', dest='arrival_assist',
                    action='store_false',
                    help='disable the D10 approach/recovery chunk substitution '
                         '(default ON: goal-mode arrival needs the executor to '
                         'carry the last ~0.15 m into the ring, since the '
                         'expert demos -- and thus the policy -- stop at the '
                         'ring edge ~0.56 m, and the tracker parks ~0.15 m '
                         'short of any chunk end)')
    ap.set_defaults(arrival_assist=True)
    args = ap.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from smolvla_edge.common import make_language_tokenizer

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.eval().to(args.device)
    tokenize = make_language_tokenizer(policy, args.device)
    tok_cache = {}

    # lerobot 0.4.4 externalizes normalization to processor pipelines saved
    # BESIDE the checkpoint (no normalize buffers in model.safetensors, and
    # select_action does not normalize). Serving raw tensors is therefore a
    # train/serve mismatch: actions come back in normalized units and the
    # state (incl. the ~7 m goal dims) lands far outside the trained
    # distribution — measured as zero goal-side sensitivity until this fix.
    import os
    pre = post = None
    if os.path.exists(os.path.join(args.checkpoint, 'policy_preprocessor.json')):
        from lerobot.policies.factory import make_pre_post_processors
        pre, post = make_pre_post_processors(policy.config,
                                             pretrained_path=args.checkpoint)
        print('processor pipeline loaded (normalized serving)', flush=True)
    else:
        print('no processor files — legacy raw serving', flush=True)

    # 3 for the classic checkpoints, 7 for goal-state ones (task 2.11). The
    # config's input_features shape is inherited from the BASE model (stale:
    # smolvla_base says [6]); the fitted normalizer stats are the truth.
    state_dim = 3
    if pre is not None:
        try:
            for step in pre.steps:
                t = getattr(step, 'stats', None) or {}
                if 'observation.state' in t and 'mean' in t['observation.state']:
                    state_dim = int(t['observation.state']['mean'].shape[-1])
        except Exception:
            pass
    print(f'state_dim={state_dim}', flush=True)

    srv = socket.create_server(('0.0.0.0', args.port))
    print(f'policy server on :{args.port} (device={args.device})', flush=True)

    while True:
        conn, _ = srv.accept()
        try:
            hlen, blen = struct.unpack('!II', recv_exact(conn, 8))
            header = json.loads(recv_exact(conn, hlen).decode())
            jpg = recv_exact(conn, blen)

            t0 = time.monotonic()
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
            task = header['instruction']
            if pre is None and task not in tok_cache:
                tok_cache[task] = tokenize(task)
            state = list(header['state'])
            gb = None
            if state_dim in (7, 8) and len(state) == 3:
                # Goal-state checkpoint (task 2.11): append [gx, gy, cos, sin]
                # from the client's optional goal_body (sent under
                # --send-goal / by goal_memory_node). Absent goal -> the
                # reserved all-zero no-goal value the policy was trained on.
                gb = header.get('goal_body')
                if gb:
                    import math as _m
                    rng = _m.hypot(gb[0], gb[1])
                    # Carrot clamp: cap the commanded range at 2.5 m so far
                    # goals stay inside the trained distribution (goal_x
                    # q99 = 2.9 m -- episodes spend most frames near the
                    # goal). Same trick as OmniVLA's thres_dist clamp; the
                    # D3 mission layer would emit exactly this.
                    sgb = ([gb[0] * 2.5 / rng, gb[1] * 2.5 / rng]
                           if rng > 2.5 else gb)
                    psi = _m.atan2(sgb[1], sgb[0])
                    state += [sgb[0], sgb[1], _m.cos(psi), _m.sin(psi)]
                    if state_dim == 8:
                        # 1/range from the CLAMPED goal -- the carrot is a
                        # synthetic goal at <=2.5 m and all its dims must
                        # describe the same point.
                        state += [1.0 / max(_m.hypot(sgb[0], sgb[1]), 0.3)]
                else:
                    state += [0.0, 0.0, 0.0, 0.0] + ([0.0] if state_dim == 8 else [])
            if pre is not None:
                # Processor path: UNBATCHED cpu tensors + raw task string; the
                # pipeline renames, batches, tokenizes, moves, normalizes.
                batch = pre({
                    'observation.image': im,
                    'observation.state': torch.tensor(state, dtype=torch.float32),
                    'task': task,
                })
                with torch.no_grad():
                    policy.reset()
                    a = post(policy.select_action(batch))
            else:
                batch = {
                    'observation.images.camera1': im.unsqueeze(0).to(args.device),
                    'observation.state': torch.tensor(
                        state, dtype=torch.float32).unsqueeze(0).to(args.device),
                }
                batch.update(tok_cache[task])
                with torch.no_grad():
                    # reset() clears the internal action queue: with
                    # n_action_steps=1 each call is one fresh chunk prediction.
                    policy.reset()
                    a = policy.select_action(batch)
            chunk = a.squeeze().reshape(args.k, 3).cpu().tolist()

            # Arrival + anti-stall contract (design D10), goal-mode only --
            # the same executor treatment the OmniVLA reference serving got,
            # so the bake-off stays symmetric in kind.
            note = ''
            if gb and args.arrival_assist:
                import math as _m
                from omnivla_server import approach_chunk, arc_chunk
                rng = _m.hypot(gb[0], gb[1])
                brg = _m.atan2(gb[1], gb[0])
                endd = _m.hypot(chunk[-1][0], chunk[-1][1])
                if rng < 1.0 and abs(brg) < 1.05:
                    # Ring-stop: the learned stop fires ~30 cm early and the
                    # tracker parks; impose the arrival envelope instead.
                    chunk = [list(c) for c in
                             approach_chunk(gb[0], gb[1], args.k, 0.25)]
                    note = 'approach'
                elif rng > 1.5 and endd < 0.35:
                    # Passing-stop: policy predicts a stop far from the goal
                    # (prop looming en route) -- frozen rover would reproduce
                    # it forever. Substitute a slow arc toward the goal.
                    chunk = [list(c) for c in
                             arc_chunk(brg, args.k, 0.25, v=0.25)]
                    note = f'recovery-arc {_m.degrees(brg):+.0f}deg'

            ms = (time.monotonic() - t0) * 1000.0
            rep = json.dumps({'chunk': chunk, 'capture_t': header['capture_t'],
                              'infer_ms': round(ms, 1)}).encode()
            conn.sendall(struct.pack('!I', len(rep)) + rep)
            print(f'chunk served in {ms:.0f} ms {note}', flush=True)
        except (ConnectionError, OSError) as e:
            print(f'conn error: {e}', flush=True)
        finally:
            conn.close()


if __name__ == '__main__':
    main()
