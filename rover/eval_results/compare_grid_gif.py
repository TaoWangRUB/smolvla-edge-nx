#!/usr/bin/env python3
"""2x2 comparison GIF: expert / SmolVLA / OmniVLA-edge (language) / OmniVLA-edge (pose).

The existing compare_seed*.gif are two-panel EXPERT|POLICY strips produced by
side_by_side_gif.py. Rather than re-running the expert, this splits that strip
back into its two 760x560 panels and adds the two OmniVLA-edge overview GIFs,
so all four runs of the same seed sit in one frame-synchronised grid.

Runs have different lengths; each cell holds its last frame once it ends, and
every run starts at t=0, so the grid is time-aligned throughout.

  python compare_grid_gif.py --compare rover/compare_seed9000.gif \
      --lang ov_lang_9000.gif --pose ov_pose_9000.gif --out grid_9000.gif
"""

import argparse

from PIL import Image, ImageDraw, ImageSequence

BAR = 26          # label strip height, matches side_by_side_gif.py
CELL_W, CELL_H = 760, 560
GAP = 6


def frames_of(path):
    im = Image.open(path)
    return [f.convert('RGB').copy() for f in ImageSequence.Iterator(im)]


def split_compare(path):
    """compare_seed*.gif -> (expert frames, policy frames), label bar removed."""
    fs = frames_of(path)
    left = [f.crop((0, BAR, CELL_W, BAR + CELL_H)) for f in fs]
    right = [f.crop((CELL_W + GAP, BAR, CELL_W + GAP + CELL_W, BAR + CELL_H)) for f in fs]
    return left, right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', required=True, help='existing 2-panel compare_seed*.gif')
    ap.add_argument('--lang', required=True, help='OmniVLA-edge language-conditioned overview')
    ap.add_argument('--pose', required=True, help='OmniVLA-edge pose-conditioned overview')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=10)
    a = ap.parse_args()

    exp, smol = split_compare(a.compare)
    lang, pose = frames_of(a.lang), frames_of(a.pose)
    cells = [
        (exp,  'EXPERT (privileged A*)',            (120, 240, 120)),
        (smol, 'SMOLVLA stage1c_v3',                (120, 190, 255)),
        (lang, 'OMNIVLA-EDGE  language',            (255, 200, 100)),
        (pose, 'OMNIVLA-EDGE  goal pose (privileged)', (240, 150, 240)),
    ]
    n = max(len(c[0]) for c in cells)
    W = CELL_W * 2 + GAP
    H = (CELL_H + BAR) * 2 + GAP

    out = []
    for i in range(n):
        canvas = Image.new('RGB', (W, H), (18, 18, 18))
        d = ImageDraw.Draw(canvas)
        for j, (fr, label, col) in enumerate(cells):
            cx = (j % 2) * (CELL_W + GAP)
            cy = (j // 2) * (CELL_H + BAR + GAP)
            canvas.paste(fr[min(i, len(fr) - 1)], (cx, cy + BAR))
            d.text((cx + 10, cy + 7), f'{label}  ({len(fr)} frames)', fill=col)
        out.append(canvas)

    out[0].save(a.out, save_all=True, append_images=out[1:],
                duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f'wrote {a.out}: {len(out)} frames, {W}x{H}')


if __name__ == '__main__':
    main()
