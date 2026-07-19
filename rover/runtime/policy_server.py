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
    args = ap.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from smolvla_edge.common import make_language_tokenizer

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.eval().to(args.device)
    tokenize = make_language_tokenizer(policy, args.device)
    tok_cache = {}

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
            if task not in tok_cache:
                tok_cache[task] = tokenize(task)
            batch = {
                'observation.images.camera1': im.unsqueeze(0).to(args.device),
                'observation.state': torch.tensor(
                    header['state'], dtype=torch.float32).unsqueeze(0).to(args.device),
            }
            batch.update(tok_cache[task])
            with torch.no_grad():
                # reset() clears the internal action queue: with
                # n_action_steps=1 each call is one fresh chunk prediction.
                policy.reset()
                a = policy.select_action(batch)
            chunk = a.squeeze().reshape(args.k, 3).cpu().tolist()
            ms = (time.monotonic() - t0) * 1000.0
            rep = json.dumps({'chunk': chunk, 'capture_t': header['capture_t'],
                              'infer_ms': round(ms, 1)}).encode()
            conn.sendall(struct.pack('!I', len(rep)) + rep)
            print(f'chunk served in {ms:.0f} ms', flush=True)
        except (ConnectionError, OSError) as e:
            print(f'conn error: {e}', flush=True)
        finally:
            conn.close()


if __name__ == '__main__':
    main()
