# Deferred sampler patches

## `v5_wide_bearing.patch`

Band-stratified prop bearing: `BEARING_OUTER_FRAC` (default 0.4) of props land
in the OUTER band, 0.55-0.95 of half-FOV, i.e. far left/right rather than ahead.

**Why.** Sampling bearing uniformly and rejecting on the scene rectangle biases
hard toward centre — wide bearings only fit at short range, so they are rejected
more often. Measured on v4: **45% of goals within 10 deg, only 15% beyond 30 deg**
(max 42.3). A centre-clustered goal lets a policy succeed by driving roughly
straight, and never forces the turn that takes the goal out of the FOV — exactly
the case goal memory exists for. With the patch: 20% within 10 deg, 34% beyond
30 deg, max 47.4.

Applies to **every** prop, goal included. Stratifying only the goal would create
a "drive to the most peripheral object" shortcut — the mirror image of the
centrality shortcut removed on 2026-07-20. Verified: goal and distractor bearing
marginals stay matched (20.0% vs 22.3% in the inner bin).

Also halves a pre-existing corridor placement-failure rate (36/400 -> 18/400) by
falling back to unstratified sampling after 200 attempts: `corridor` is a 1.2 m
hallway where 30 deg at 2 m already exits the walls, so a peripheral band is
geometrically impossible there and must not become a hard failure.

**Held for v5, not applied to v4.** v4 is the short-horizon control: it tests
whether the M1 failure was horizon-driven, and changing bearing at the same time
would test two things at once. Peripheral goals are the point of v5, the dataset
for the memory/acquisition architecture.

```bash
git apply rover/datagen/patches/v5_wide_bearing.patch
```

Per-scene reachable spread (400 seeds, v4 ranges): open_ground max 47.4 deg /
34% beyond 30; parking_lot max 47.0 / 26%; corridor max 23.2 / 0% (walls).
