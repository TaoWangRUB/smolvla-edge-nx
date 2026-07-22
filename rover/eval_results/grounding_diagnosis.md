# Grounding diagnosis (2026-07-20)

## Metrics recap
- **success N/10**: reached the commanded goal's 0.6 m ring without collision, over 10 unseen scenes (primary run only).
- **swap test N/8**: the real grounding test. Each scene is run twice — instruction on the goal, then on a same-shape/different-color hard negative in the *identical* layout. The pair passes only if the rover goes to the *commanded* object both times. It catches a policy that ignores language and drives to whatever is most salient.

## Results
| Model | success | swap |
|---|---|---|
| stage1_v2 (frozen backbone) | 5/10 | 2/8 |
| stage1b_v2 (full vision unfreeze) | 2/10 | 0/8 |

Full vision unfreeze **regressed** — a ~400M SigLIP fully unfrozen on ~1 epoch of 568 synthetic episodes drifted the pretrained features out from under the expert head. (Design D5 contingency 1 actually prescribes vision-encoder *LoRA*, not a full unfreeze.)

## Offline grounding probe (stage1_v2, 40 recorded episodes, no driving)
Feed the policy the same spawn frame under different instructions; measure whether the predicted waypoint chunk turns toward the commanded object.

- **directional accuracy 0.71** (chance 0.50) — it *does* steer to the correct side for off-axis targets. Language is not ignored wholesale.
- **offline swap-flip rate 0.10** (chance 0.25 — **below chance**) — for same-shape/different-color pairs on opposite sides, it almost never flips. Below chance ⇒ it steers to a **fixed salient object regardless of the color word**.
- **mean predicted-bearing change between goal vs hard-neg instruction: 12.7°** — near-invariant to the color word.

**Reading:** the policy grounds **shape** (0.71) but **not color** (0.10). It reads "crate vs pillar" and steers, but ignores "red vs blue" among same-shape objects.

## Root cause (data confound, not model capacity)
`scene_manager.sample()` placed the **goal in the forward camera cone (±0.8·HFOV, 2–7 m)** while all distractors (including the same-shape/different-color hard negative) were scattered **uniformly across the whole arena**. So the goal was systematically the most central/forward object. A policy learns the shortcut "drive to the salient object ahead" — which scores ~50% success and passes shape-distinguished cases, while color grounding is never required. The swap test (which commands the *peripheral* hard negative) exposes it.

Web context: SigLIP does encode color (the information is present in the tokens), but SmolVLA freezes the VLM and trains only the small action expert; instruction-following / fine attribute grounding is a documented weak point of that design, and single-SigLIP grounding is weaker than dual-encoder (SigLIP+DINOv2). So a frozen backbone + tiny expert failing to bind the color *word* to the right object — especially when the data lets it cheat on position — is expected.

## Fix (data-side, this rung of the ladder)
Place **all** candidate props in the visible cone with comparable centrality so the goal is not privileged, guaranteeing the same-shape/different-color twin is equally visible → **color becomes necessary** to disambiguate; the positional shortcut dies. Then regenerate the dataset, retrain (frozen backbone — the stable recipe), and re-run the swap test.

## Update 2026-07-21 — sampler fix did not restore color grounding

Regenerated `local/rover_vla_v3` (520 ep, confound-fixed sampler) and retrained the frozen
backbone (stage1_v3, 10k, loss ~0.16). Results:

| measure | stage1_v2 (v2 data) | stage1_v3 (v3 data) |
|---|---|---|
| closed-loop success (open_ground 9000-9009) | 5/10 | 3/10 * |
| closed-loop swap | 2/8 | 0/9 * |
| offline directional acc (on raw_v3 frames) | 0.62 | **0.71** |
| offline color swap-flip (on raw_v3 frames) | 0.27 | 0.18 |

\* Closed-loop v3 is on HARDER scenes (fixed sampler clusters props in the cone), so the
success/swap drop is largely scene difficulty, not model regression — the controlled probe
(both checkpoints on identical raw_v3 frames) is the fair comparison.

**Read:** the sampler fix removed a genuine shortcut and modestly improved shape/side grounding
(directional 0.62 -> 0.71), but **color-word binding stayed at chance** (swap-flip ~0.2, chance
0.25) for both models. Removing the data confound was necessary and correct, but it was **not**
the cause of the color failure. With the earlier full-unfreeze regression, the color-grounding
limit is now isolated to **model capacity**: frozen SmolVLA (single SigLIP + small action
expert) does not bind color attributes at ~500-episode scale.

**Next rung (D5 contingency 1, done correctly):** vision-encoder **LoRA** — low-rank adapters on
the SigLIP encoder, language model and base vision weights frozen (NOT the full unfreeze that
scrambled features). If LoRA is also flat: color-stressed data / richer color cues, then the
architecture fallback (Qwen2.5-VL + diffusion head, D5 contingency 4).

**Methodology fix:** freeze a single eval scene set across model versions. Using the
per-version sampler for eval made v3's closed-loop numbers incomparable to v2's (scene
difficulty changed with the sampler).

## Update 2026-07-21 (later) — constrained vision adaptation also failed

`stage1c_v3`: warm-start stage1_v3, unfreeze **top-2 of 16 vision layers** (LM + lower vision
frozen; 126M trainable, verified before launch), 10k steps, loss 0.117. This is the robust
stand-in for D5 contingency-1 LoRA (lerobot/peft integration was blocked: `diffusers` pins
peft>=0.17, and lerobot's factory treats `--policy.path=<ckpt> --use_peft` as "load an existing
adapter", not "add fresh LoRA").

Offline grounding probe, all three models on identical `raw_v3` frames:

| model | directional acc | color swap-flip (chance 0.25) |
|---|---|---|
| stage1_v2 (frozen, v2 data) | 0.62 | 0.27 |
| stage1_v3 (frozen, fixed data) | 0.71 | 0.18 |
| **stage1c_v3 (top-2 vision adapted)** | **0.75** | **0.05** |

**Vision adaptation made the model a better navigator and a worse colour-grounder.**
Directional accuracy is the best of the three (0.75) while swap-flip collapsed to 0.05 — far
*below* chance, and the predicted bearing still moves only ~15 deg when the colour word flips.
Below-chance swap-flip means the policy steers to a **fixed salient object regardless of the
instruction**, and adapting vision made it do that *more* confidently.

Closed-loop (open_ground seeds 9000-9009): **success 3/10, swap 0/9** — identical to
stage1_v3's 3/10, 0/9, confirming the probe: vision adaptation changed navigation quality but
not grounding. Log: `rover/eval_results/eval_stage1c_v3_open_ground.log`.

### Converging evidence: the vision encoder is not the bottleneck
Four interventions, none of which moved colour grounding:
1. data confound fix (v2 -> v3 sampler) — swap-flip 0.27 -> 0.18
2. frozen backbone — 0.18
3. full vision unfreeze — regressed (closed-loop 0/8)
4. constrained top-2 vision adaptation — 0.05

SigLIP demonstrably encodes colour, so the missing step is **binding the colour word to the
right object** — a cross-modal attention operation that happens inside the LM. And SmolVLA
**truncates the LM to the first 16 of 32 layers** (`num_vlm_layers=16`,
`text_model.layers = text_model.layers[:16]`). If fine attribute binding needs the discarded
upper layers, no amount of vision adaptation can recover it.

### Next experiment (priority): deeper LM
Raise `num_vlm_layers` (16 -> 32) and retrain from base — `MODE='C'` in
`rover/train_colab.ipynb` (needs an A100's VRAM; the expert resizes to match, so it is a
from-base run, not a warm start). If swap lifts, the truncation was the cause. If it does not,
the cheap levers are exhausted and the honest escalation is D5 contingency 4 (Qwen2.5-VL +
diffusion head).

## Update 2026-07-21 (later still) — REFRAMING: the colour conclusion was under-determined

### Mode C (deeper LM) result
`stage1d_deeplm` — `num_vlm_layers` 16→32 (model 450M→706M; the expert scales with the LM),
10k steps @ batch 32 on Colab A100, `rover_vla_v3`. Probe on identical `raw_v3` frames:

| model | LM layers | directional | colour swap-flip | bearing change |
|---|---|---|---|---|
| stage1_v3 | 16 | 0.71 | 0.18 | 15.3 deg |
| stage1c_v3 (vision adapted) | 16 | 0.75 | 0.05 | 14.8 deg |
| stage1d_deeplm | **32** | 0.71 | **0.27** | **22.3 deg** |

Best swap-flip so far and a ~50% jump in instruction sensitivity — but 0.27 vs chance 0.25 on
22 pairs is statistically indistinguishable. A faint signal that LM depth matters; **not a fix**.

### What the eval traces actually show
Per-episode analysis of the 10 eval traces (`rover/eval_traces/`):
- In **8/10 the rover ends 0.32–0.60 m from *some* prop** — a precise approach, so low-level
  path-following works.
- But `min_dist_to_commanded_ever ~= final_dist_to_commanded`: it **never heads toward the
  commanded target** — it commits elsewhere immediately.
- **Distance bias**: when the commanded target is far (distance rank 4–6 from spawn), the rover
  went to the *nearest* prop in **4 of 5** cases. When commanded is near (rank 2–3), it ends in
  the right neighbourhood.
- The **expert scores 10/10 on the identical seeds**, so every scene is solvable.

### Correction to the earlier conclusion
The previous entries concluded "colour-word binding is broken / the vision encoder is not the
bottleneck". That conclusion is **under-determined**: a policy dominated by proximity/saliency
produces the *same* swap score as one that cannot perceive colour. The swap test cannot separate
them while the commanded target and distractors sit at different ranges.

### Structural diagnosis (the likely root cause)
SmolVLA here is a **local, memoryless visuomotor policy** being asked to do **long-horizon goal
selection**:
1. `n_obs_steps=1` — one current frame, no temporal state. **If the goal leaves the ~100 deg FOV
   there is zero information about where it is**; the policy latches onto whatever is visible.
2. Chunk horizon = K*DT = 2.5 s ~= **1.25 m** at 0.5 m/s, against goals **2–7 m** away — up to
   5.6 independent replans must agree on a destination, with no memory to enforce that.
   This is the mechanism behind both observed symptoms: **wandering** and **serpentine driving**.
3. Design D3 specifies a **mission loop (0.1–1 Hz)** for global goal selection, deferred as
   M4-optional — so the policy was made to do the mission layer's job as well as its own.

### Experiment in flight: short horizon (`rover_vla_v4`)
Sampler placement range 2.0–7.0 m -> **2.0–3.5 m** (goal mean 2.80 m ~= 2.2 chunk-reaches),
1 filler; measured within-scene range spread 1.27 -> **0.88 m**. This keeps the goal in the FOV
for the whole approach *and* equalises range, so the swap test becomes a clean colour measure
for the first time. If success and swap both lift, the failure was **horizon/visibility**, not
colour perception.

### Long-term (independent of the above): the policy needs goal memory
Shortening the horizon *avoids* the memoryless limitation rather than fixing it. Options, in
cost order:
1. `n_obs_steps > 1` — short frame history; cheap, but won't cover a target out of view for seconds.
2. **Mission layer (D3)** holding the goal's location in the odom frame (the EKF already tracks
   it) and feeding the fast policy a persistent target — goal memory belongs in map/state, not
   in policy weights.
3. Explicit goal state in the observation (relative bearing/range once acquired) — i.e. the
   point-goal interface, now well-motivated rather than redundant.

## Update 2026-07-22 — reference model isolates the failure to the language→action pathway

Ran the released **OmniVLA-edge** (~108M ViNT-lineage navigation specialist; 6-frame history,
CLIP text, 2D goal-pose channel, MIT) zero-shot on our data, as the cheap decisive experiment
before more training. Three measurements, one story:

**1. Offline swap on 12 range-equalised same-shape/diff-colour pairs** (both props in front,
opposite sides, matched range — the clean colour condition v4 was built to create):

| modality | swap | mean bearing response |
|---|---|---|
| goal pose | **12/12** | 29.0° |
| language | 2/12 | **1.5°** |

(SmolVLA references on the same protocol: stage1_v3 15.3°, stage1d_deeplm 22.3°, swap ≈ chance.)
A navigation-pretrained model with a real language head is *less* instruction-sensitive than
our SmolVLA — language conditioning does nothing on this imagery.

**2. CLIP colour probe on projected prop crops**: 9/11 correct (chance 0.25); both misses are
crop artefacts (occlusion / frame edge), so 0.82 is a lower bound. Colour is present and
machine-legible — perception is not the broken stage.

**3. Closed-loop, seeds 9000–9009, same tracker/referee** (privileged goal via
`run_eval.py --send-goal`): pose **7/10** vs language **4/10** vs trained SmolVLA 3/10
(expert 10/10). Language misses end 9–15 m from the target — driving *away*, the same
no-information signature as our traces. Pose failures are obstacle-handling only (two 5–17 mm
transit grazes, one goal-blocked-by-prop hold).

**Executor forensics (first pose run read 6/10 with freezes — all three causes were ours, the
model was exonerated by replay: it emits ~1 m paths even for a goal at the origin):**
1. Stop-intent (near-zero) model paths → all-zero resampled chunk → tracker `at_end` latch =
   **permanent park** (9001/9009 frozen 40 s; 9007 froze at closest approach, 5 cm from a
   blocking prop — frozen rover ⇒ identical frames ⇒ identical prediction: a fixed point).
2. Server exceptions closed the socket without replying → silent tracker starvation.
3. Arrival stop 0.55 m + tracker GOAL_TOL 0.15 m = park at 0.70 m — 10 cm *outside* the 0.6 m
   ring; 8 episodes timed out at 0.69–0.70 m, ruler-consistent.
Fixed (recovery arc R ≥ 0.36 m, in-ring stop 0.40 m, always-reply errors; 32 unit checks) →
7/10, previously-frozen 9001 reaches cleanly.

**Reading.** Combining with the v4 horizon result (success 4/10 but swap unmoved at 1/9):
every stage of the pipeline is now individually measured — colour perception ✓ (CLIP 9/11),
goal-conditioned driving ✓ (12/12 offline, 7/10 closed-loop), language→steering ✗ in every
model tested (ours across 6 interventions; theirs zero-shot). The failure is not data, not
capacity (9× smaller wins with the right input), not visibility (v4), not perception — it is
the **binding of the instruction to a spatial target inside the policy**, and the measured
escape is to stop asking the policy to do it: acquisition selects (D9, 94%), a goal channel
steers (D10, tasks 2.11–2.12).
