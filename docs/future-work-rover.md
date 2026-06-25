# Future work — mobile rover embodiment

The on-hand rover is an Ackermann/mobile platform, **not** an arm. SmolVLA is pretrained on
SO-100/SO-101 manipulation data, so applying it to a mobile base is a change of embodiment —
different observation/action spaces, different action semantics (velocity/steering vs. joint
targets), no matching pretraining prior. That's a research project, not a two-week demo, so it
is explicitly scoped out of the gating artifact.

Sketch of what adapting it would take:

1. **Data.** Collect (or source) a rover dataset in LeRobot format with the rover's action
   space (e.g. linear/angular velocity or throttle/steer) and its camera streams.
2. **Action/observation spec.** Define a new policy action head / config matching the rover's
   DOF; the SO-101 action head does not transfer directly.
3. **Fine-tune / re-train.** Expect to fine-tune more aggressively (or train more layers) than
   the manipulation case, since the pretraining prior is weaker for this embodiment.
4. **Edge deployment.** The Xavier NX latency-engineering work from Phase 2 transfers directly
   — that's the reusable payoff and the reason to do manipulation first.

Framing for the writeup: "manipulation validates the VLA + edge pipeline; the rover is the
natural next embodiment, and the edge-deployment work carries over unchanged."
