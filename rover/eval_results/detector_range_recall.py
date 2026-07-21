"""Detection recall vs range, for the DEPLOYABLE detector (design D9 step c).

If patch32's recall rises as the rover closes in, an approach-and-lock search
recovers the goal and NanoOWL stays viable on the NX. If it stays flat, the
deployable model cannot acquire and OWLv2 must be made to fit instead.
"""
import bisect, glob, json, math, os, sys, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

M = sys.argv[1] if len(sys.argv) > 1 else 'google/owlvit-base-patch32'
THRESH = 0.05
proc = AutoProcessor.from_pretrained(M)
model = AutoModelForZeroShotObjectDetection.from_pretrained(M).eval().cuda()

buckets = {}   # range bucket -> [hits, total]
for d in sorted(glob.glob('rover/data/raw_v4/*/'))[:45]:
    fj, ff, fg = (os.path.join(d, x) for x in ('scene_config.json','frames.jsonl','gt_pose.jsonl'))
    if not all(os.path.exists(x) for x in (fj, ff, fg)):
        continue
    cfg = json.load(open(fj))
    frames = [json.loads(l) for l in open(ff)]
    poses  = [json.loads(l) for l in open(fg)]
    pts = [p['t'] for p in poses]
    g = cfg['props'][cfg['goal_index']]
    tq = f"{g['color']} {g['shape']}"

    # sample frames across the episode -> a spread of true ranges
    for fr in frames[::max(1, len(frames)//6)]:
        fp = os.path.join(d, 'frames', f"{fr['i']:06d}.jpg")
        if not os.path.exists(fp):
            continue
        k = min(bisect.bisect_left(pts, fr['t']), len(poses)-1)
        p = poses[k]
        dx, dy = g['x']-p['x'], g['y']-p['y']
        c, s = math.cos(-p['yaw']), math.sin(-p['yaw'])
        bx, by = c*dx - s*dy, s*dx + c*dy
        rng, bear = math.hypot(bx, by), math.atan2(by, bx)
        if bx <= 0 or abs(bear) > math.radians(50):     # not in FOV, skip
            continue
        img = Image.open(fp).convert('RGB')
        inp = proc(text=[[tq]], images=img, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model(**inp)
        r = proc.post_process_grounded_object_detection(outputs=out,
            target_sizes=torch.tensor([img.size[::-1]]).cuda(), threshold=THRESH)[0]
        hit = len(r['scores']) > 0
        b = round(rng*2)/2                              # 0.5 m buckets
        e = buckets.setdefault(b, [0,0]); e[0]+=hit; e[1]+=1

print(f"model: {M}   threshold {THRESH}\n")
print(f"{'range (m)':>10s} {'recall':>14s}  n")
for b in sorted(buckets):
    h, n = buckets[b]
    print(f"{b:>9.1f}  {h}/{n:<4d} {100*h/n:5.0f}%  {'#'*int(20*h/n)}")
