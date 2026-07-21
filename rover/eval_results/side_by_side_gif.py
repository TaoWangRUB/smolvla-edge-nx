#!/usr/bin/env python3
"""Combine two overview GIFs into one side-by-side comparison GIF.

Frame counts differ between runs, so the shorter clip holds its last frame
until the longer one finishes (both start together, so the comparison is
time-aligned from t=0).

  python side_by_side_gif.py --left expert.gif --right policy.gif \
      --left-label EXPERT --right-label POLICY --out cmp.gif
"""

import argparse

from PIL import Image, ImageDraw, ImageSequence

BAR = 26  # header strip height


def frames_of(path):
    im = Image.open(path)
    return [f.convert('RGB').copy() for f in ImageSequence.Iterator(im)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--left', required=True)
    ap.add_argument('--right', required=True)
    ap.add_argument('--left-label', default='EXPERT')
    ap.add_argument('--right-label', default='POLICY')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=10)
    a = ap.parse_args()

    L, R = frames_of(a.left), frames_of(a.right)
    n = max(len(L), len(R))
    w, h = L[0].size
    out = []
    for i in range(n):
        l = L[min(i, len(L) - 1)]
        r = R[min(i, len(R) - 1)]
        canvas = Image.new('RGB', (w * 2 + 6, h + BAR), (18, 18, 18))
        canvas.paste(l, (0, BAR))
        canvas.paste(r, (w + 6, BAR))
        d = ImageDraw.Draw(canvas)
        d.text((10, 7), f'{a.left_label}  ({len(L)} frames)', fill=(120, 240, 120))
        d.text((w + 16, 7), f'{a.right_label}  ({len(R)} frames)', fill=(120, 190, 255))
        out.append(canvas)

    out[0].save(a.out, save_all=True, append_images=out[1:],
                duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f'wrote {a.out}: {len(out)} frames')


if __name__ == '__main__':
    main()
