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
