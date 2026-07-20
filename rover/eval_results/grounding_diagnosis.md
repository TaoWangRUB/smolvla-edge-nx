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
