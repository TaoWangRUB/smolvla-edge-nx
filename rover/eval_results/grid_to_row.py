#!/usr/bin/env python3
"""Re-slice a 2x2 comparison grid (compare_grid_gif.py output) into a 1x4 row.

The four panels of full4_*.gif / compare4_*.gif already carry their baked-in
label bars, so we crop each cell block (label strip + frame) whole and re-stack
them horizontally. Cell geometry matches compare_grid_gif.py exactly.

  python grid_to_row.py --grid rover/gifs/full4_9006.gif --out rover/gifs/row4_9006.gif
"""

import argparse

from PIL import Image, ImageSequence

BAR = 26
CELL_W, CELL_H = 760, 560
GAP = 6
BLOCK_H = BAR + CELL_H  # 586: one cell's label strip + frame


def frames_of(path):
    im = Image.open(path)
    return [f.convert('RGB').copy() for f in ImageSequence.Iterator(im)]


def cell(frame, j):
    """Crop cell j (0..3) of the 2x2 grid, label bar included."""
    cx = (j % 2) * (CELL_W + GAP)
    cy = (j // 2) * (BLOCK_H + GAP)
    return frame.crop((cx, cy, cx + CELL_W, cy + BLOCK_H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True, help='2x2 grid gif (full4_*/compare4_*)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=10)
    a = ap.parse_args()

    grid = frames_of(a.grid)
    W = CELL_W * 4 + GAP * 3
    H = BLOCK_H
    out = []
    for f in grid:
        canvas = Image.new('RGB', (W, H), (18, 18, 18))
        for j in range(4):
            canvas.paste(cell(f, j), (j * (CELL_W + GAP), 0))
        out.append(canvas)

    out[0].save(a.out, save_all=True, append_images=out[1:],
                duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f'wrote {a.out}: {len(out)} frames, {W}x{H}')


if __name__ == '__main__':
    main()
