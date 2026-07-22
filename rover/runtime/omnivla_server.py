#!/usr/bin/env python3
"""OmniVLA-edge waypoint server — drop-in for runtime/policy_server.py.

Speaks the identical wire protocol (see chunk_client_node.py), so run_eval.py
drives it unchanged; only --server-port differs. This lets the released
OmniVLA-edge checkpoint be evaluated on the same eval seeds, through the same
tracker and the same referee, as the SmolVLA checkpoints.

Two differences from SmolVLA are absorbed here:

  frame history   OmniVLA-edge consumes 6 frames (5 context + current). The
                  client sends one frame per request, so the server keeps a
                  deque and resets it when capture_t jumps (new episode).
  action units    OmniVLA emits 8 x (front, left, cos, sin) in units of
                  S = 0.1 m with no time base. We treat that as a *path*,
                  and impose this project's speed profile: resample to
                  K x (x, y, v) at DT with a fixed cruise speed, matching
                  datagen/relabel.py's chunk contract.

Modalities (ids read off OmniVLA/inference/run_omnivla_edge.py):
  language (7)  instruction only — the fair comparison against SmolVLA
  pose     (4)  goal pose only — privileged upper bound; needs --goal-source
  both     (8)  goal pose + instruction

Run (torch side):
  docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=1 -v "$PWD":/vla -v <omnivla-ref>:/ref \
    -w /vla --network host smolvla-edge:sim \
    python rover/runtime/omnivla_server.py --modality language
"""

import argparse
import collections
import json
import math
import os
import socket
import struct
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

S = 0.1              # OmniVLA metric_waypoint_spacing
CONTEXT = 5          # + current = 6 frames
IMGSIZE = (96, 96)
IMGSIZE_CLIP = (224, 224)
MODALITY = {'pose': 4, 'language': 7, 'both': 8}

MODEL_PARAMS = {
    'model_type': 'omnivla-edge', 'len_traj_pred': 8, 'learn_angle': True,
    'context_size': CONTEXT, 'obs_encoder': 'efficientnet-b0',
    'encoding_size': 256, 'obs_encoding_size': 1024, 'goal_encoding_size': 1024,
    'late_fusion': False, 'mha_num_attention_heads': 4,
    'mha_num_attention_layers': 4, 'mha_ff_dim_factor': 4, 'clip_type': 'ViT-B/32',
}


def recv_exact(conn, n):
    buf = b''
    while len(buf) < n:
        d = conn.recv(n - len(buf))
        if not d:
            raise ConnectionError('short read')
        buf += d
    return buf


def resample(path_m, k, dt, cruise):
    """OmniVLA path (list of (front,left) in metres, from the origin) ->
    K x (x, y, v) at fixed dt and cruise speed.

    OmniVLA's 8 points span ~1.0-1.1 m, short of the 1.25 m a 2.5 s chunk needs
    at cruise. Its output is a heading intent rather than a terminal goal, so
    the tail is extrapolated along the final segment instead of clamped to a
    stop -- clamping makes the tracker creep toward a stationary point. Only
    the first waypoints are ever executed (the chunk is replaced at ~3 Hz), so
    the extrapolated tail acts as a jitter buffer.
    """
    pts = [(0.0, 0.0)] + list(path_m)
    seg = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
           for i in range(len(pts) - 1)]
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)
    total = cum[-1]
    if total < 0.35:
        # Degenerate/stop-intent path. NEVER emit a zero chunk: all points
        # land within the tracker's 0.45 m lookahead with the last inside
        # GOAL_TOL, which latches at_end and parks the rover permanently
        # (frozen rover -> identical frames -> identical prediction). Seen
        # closed-loop as seeds 9001/9007/9009 freezing. The caller substitutes
        # a recovery arc via arc_chunk(); returning None signals that.
        return None
    # unit direction of the final segment, for extrapolation past the path end
    ex, ey = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
    en = math.hypot(ex, ey) or 1.0
    ex, ey = ex / en, ey / en
    out = []
    for i in range(1, k + 1):
        s = cruise * i * dt
        if s >= total:
            d = s - total
            out.append((pts[-1][0] + ex * d, pts[-1][1] + ey * d, cruise))
            continue
        j = max(0, min(len(seg) - 1, next(x for x in range(len(cum) - 1) if cum[x + 1] > s)))
        a = (s - cum[j]) / seg[j] if seg[j] > 1e-9 else 0.0
        out.append((pts[j][0] + a * (pts[j + 1][0] - pts[j][0]),
                    pts[j][1] + a * (pts[j + 1][1] - pts[j][1]),
                    cruise))
    return out


def arc_chunk(bearing_rad, k, dt, v=0.25, r_min=0.36):
    """Ackermann-feasible recovery chunk: a constant-curvature arc turning
    toward `bearing_rad` (0 = straight ahead). Used when the model's path is
    degenerate -- the vehicle cannot rotate in place (R_MIN 0.341 m), so
    "turn toward the goal" must be driven as a tight arc. Curvature is capped
    at 1/r_min and the arc is capped at 90 deg of turn per chunk."""
    if abs(bearing_rad) < 1e-3:
        return [((i + 1) * v * dt, 0.0, v) for i in range(k)]
    kappa = math.copysign(min(1.0 / r_min, abs(bearing_rad) / max(v * dt * k, 1e-6)),
                          bearing_rad)
    out = []
    for i in range(1, k + 1):
        s = min(v * i * dt, abs(math.pi / 2 / kappa))   # cap at 90 deg of arc
        th = kappa * s
        out.append((math.sin(th) / kappa, (1.0 - math.cos(th)) / kappa, v))
    return out


def approach_chunk(gf, gl, k, dt, ring=0.40, v_max=0.3):
    """Pose-mode arrival: drive straight at the goal bearing and stop INSIDE
    the referee ring (0.6 m) rather than orbiting it -- OmniVLA has no notion
    of the ring, so the executor imposes the stop. Only used when the goal is
    near (< ~1 m) and roughly ahead.

    ring=0.40, not 0.6: the tracker parks when the rover is within its
    GOAL_TOL (0.15 m) of the chunk's stop point, so the realised stop is
    ring..ring+0.15 = 0.40-0.55 m -- inside the referee's 0.6 m ring, with
    clearance still positive for the widest prop (crate radius 0.29 m).
    Measured with ring=0.55: eight episodes parked at 0.69-0.70 m, 10 cm
    OUTSIDE the ring -> uniform timeout."""
    rng = math.hypot(gf, gl)
    stop_d = max(0.0, rng - ring)
    ux, uy = (gf / rng, gl / rng) if rng > 1e-6 else (1.0, 0.0)
    out = []
    for i in range(1, k + 1):
        s = min(v_max * i * dt, stop_d)
        v = v_max if s < stop_d - 1e-6 else 0.0
        out.append((ux * s, uy * s, v))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--omnivla-root', default='/ref/OmniVLA')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--modality', default='language', choices=list(MODALITY))
    ap.add_argument('--port', type=int, default=8791)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--dt', type=float, default=0.25)
    ap.add_argument('--cruise', type=float, default=0.5, help='m/s imposed on the path')
    ap.add_argument('--reset-gap', type=float, default=2.0,
                    help='capture_t jump (s) that means a new episode')
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.omnivla_root, 'inference'))
    from utils_policy import load_model, transform_images_PIL_mask, transform_images_map
    import clip

    device = torch.device(args.device)
    ckpt = args.ckpt or os.path.join(args.omnivla_root, 'omnivla-edge', 'omnivla-edge.pth')
    model, text_encoder, _ = load_model(ckpt, MODEL_PARAMS, device)
    model, text_encoder = model.to(device).eval(), text_encoder.to(device).eval()
    mask96 = np.ones((96, 96, 3), dtype=np.float32)
    mask224 = np.ones((224, 224, 3), dtype=np.float32)
    mid = torch.tensor([MODALITY[args.modality]], device=device)
    black352 = Image.new('RGB', (352, 352), color=(0, 0, 0))
    black96 = Image.new('RGB', IMGSIZE, color=(0, 0, 0))

    ctx = collections.deque(maxlen=CONTEXT + 1)
    tok_cache = {}
    last_t = [None]

    srv = socket.create_server(('0.0.0.0', args.port))
    print(f'omnivla-edge server on :{args.port} '
          f'(modality={args.modality}, device={device})', flush=True)

    while True:
        conn, _ = srv.accept()
        header = {}          # defined before try: the except path reads it
        try:
            hlen, blen = struct.unpack('!II', recv_exact(conn, 8))
            header = json.loads(recv_exact(conn, hlen).decode())
            jpg = recv_exact(conn, blen)

            t0 = time.monotonic()
            capture_t = header['capture_t']
            if last_t[0] is None or abs(capture_t - last_t[0]) > args.reset_gap:
                ctx.clear()
                print(f'context reset at t={capture_t:.2f}', flush=True)
            last_t[0] = capture_t

            bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ctx.append(pil.resize(IMGSIZE))
            while len(ctx) < CONTEXT + 1:        # episode start: pad with the first frame
                ctx.appendleft(ctx[0])
            cur224 = pil.resize(IMGSIZE_CLIP)

            obs = transform_images_PIL_mask(list(ctx), mask96).to(device)
            obs_cur = torch.split(obs, 3, dim=1)[-1]
            cur_large = transform_images_PIL_mask(cur224, mask224).to(device)
            map_images = torch.cat((transform_images_map(black352).to(device),
                                    transform_images_map(black352).to(device),
                                    obs_cur), axis=1)
            goal_image = transform_images_PIL_mask(black96, mask96).to(device)

            # Goal pose: supplied by the client for pose/both, else zeros.
            gb = header.get('goal_body')
            if gb:
                dyaw = math.atan2(gb[1], gb[0])
                gp = [gb[0] / S, gb[1] / S, math.cos(dyaw), math.sin(dyaw)]
            else:
                gp = [0.0, 0.0, 1.0, 0.0]
            goal_pose = torch.tensor([gp], dtype=torch.float32, device=device)

            task = header['instruction']
            if task not in tok_cache:
                with torch.no_grad():
                    tok_cache[task] = text_encoder.encode_text(
                        clip.tokenize(task, truncate=True).to(device)).float()

            with torch.no_grad():
                actions, _, _ = model(obs, goal_pose, map_images, goal_image,
                                      mid, tok_cache[task], cur_large)
            wp = actions[0].float().cpu().numpy()
            path_m = [(float(p[0]) * S, float(p[1]) * S) for p in wp]

            note = ''
            grange = math.hypot(gb[0], gb[1]) if gb else None
            if gb and grange < 1.0 and abs(math.atan2(gb[1], gb[0])) < 1.05:
                # Near-goal arrival (pose mode): stop ON the referee ring
                # instead of orbiting it.
                chunk = approach_chunk(gb[0], gb[1], args.k, args.dt)
                note = 'approach'
            else:
                chunk = resample(path_m, args.k, args.dt, args.cruise)
                if chunk is None:
                    # Stop-intent/degenerate path: recovery arc toward the
                    # goal (pose mode) or the model's own heading channel
                    # (waypoint 4 -- the one the released controller executes).
                    if gb:
                        bearing = math.atan2(gb[1], gb[0])
                    else:
                        bearing = math.atan2(float(wp[4][3]), float(wp[4][2]))
                    chunk = arc_chunk(bearing, args.k, args.dt)
                    note = f'recovery-arc {math.degrees(bearing):+.0f}deg'
            chunk = [list(c) for c in chunk]

            ms = (time.monotonic() - t0) * 1000.0
            rep = json.dumps({'chunk': chunk, 'capture_t': capture_t,
                              'infer_ms': round(ms, 1)}).encode()
            conn.sendall(struct.pack('!I', len(rep)) + rep)
            print(f'chunk served in {ms:.0f} ms  end=({chunk[-1][0]:+.2f},'
                  f'{chunk[-1][1]:+.2f}) m {note}', flush=True)
        except (ConnectionError, OSError) as e:
            print(f'conn error: {e}', flush=True)
        except Exception as e:
            # NEVER die silently: an unanswered request starves the tracker
            # (the client just warns and retries), which closed-loop looks
            # like a frozen rover. Reply with a slow straight chunk so the
            # loop keeps breathing while the error is visible in the log.
            import traceback
            traceback.print_exc()
            try:
                fallback = [[0.1 * (i + 1) * 0.25, 0.0, 0.1] for i in range(args.k)]
                rep = json.dumps({'chunk': fallback,
                                  'capture_t': header.get('capture_t', 0.0),
                                  'error': f'{type(e).__name__}: {e}'}).encode()
                conn.sendall(struct.pack('!I', len(rep)) + rep)
            except OSError:
                pass
        finally:
            conn.close()


if __name__ == '__main__':
    main()
