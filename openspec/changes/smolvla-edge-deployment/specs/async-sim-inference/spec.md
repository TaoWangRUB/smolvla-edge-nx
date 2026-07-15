## ADDED Requirements

### Requirement: Asynchronous inference control loop (SmolVLA §3.3, Algorithm 1)

The system SHALL implement the SmolVLA asynchronous inference stack in simulation: a
`RobotClient` consumes actions from a queue while a `PolicyServer` (in-process worker or
remote) computes the next chunk, triggered when the queue fraction drops below a threshold
`g ∈ [0, 1]`. Synchronous inference SHALL be expressible as the `g = 0` limit of the same loop.

#### Scenario: Async rollout in gym-aloha

- **WHEN** an operator runs `python -m smolvla_edge.eval --mode sim --inference async --g 0.7 --policy-path <checkpoint>`
- **THEN** the client pops one action per control tick, triggers a non-blocking chunk
  prediction when `|A_t|/n < g`, and keeps executing the existing queue while inference runs

#### Scenario: Inference still running at trigger

- **WHEN** the queue drops below the threshold but the previous chunk prediction has not completed
- **THEN** the client carries the current queue over unchanged and does not block

### Requirement: Chunk aggregation on overlapping timesteps

When a new chunk `Ã_t+1` arrives while actions remain in the current queue `A_t`, the system
SHALL aggregate the two on their overlapping timesteps via a pluggable function `f(A_t, Ã_t+1)`.

#### Scenario: Overlap merged

- **WHEN** a predicted chunk arrives and the live queue is non-empty
- **THEN** the updated queue equals `f(A_t, Ã_t+1)` on the overlap and the new chunk's tail beyond it

### Requirement: Joint-space observation similarity filter

The client SHALL drop candidate observations whose joint-space distance to the last processed
observation is below a threshold `ε`, EXCEPT when the action queue is empty, in which case the
most recent observation SHALL be processed regardless of similarity.

#### Scenario: Near-duplicate dropped

- **WHEN** the queue threshold triggers and the new observation is within `ε` of the last one sent
- **THEN** no prediction request is issued and the client keeps consuming the current queue

#### Scenario: Empty queue forces processing

- **WHEN** the action queue is empty
- **THEN** the latest observation is sent for prediction even if within `ε` of the last one

### Requirement: Paced control loop with idle-tick accounting

In simulation the client loop SHALL be paced at the environment's nominal control rate `Δt`,
and SHALL log per-tick queue size, observations sent vs. filtered, idle ticks (queue empty at
a tick boundary), and wall-clock episode time, so sync and async runs are comparable.

#### Scenario: Queue evolution logged

- **WHEN** an async eval run completes
- **THEN** the output JSON contains a per-tick queue-size trace sufficient to reproduce the
  paper's Figure 3 for `g ∈ {0, 0.7, 1.0}`

### Requirement: Sync vs async head-to-head report

The system SHALL produce a sync-vs-async comparison on the fine-tuned checkpoint (identical
seeds and episode count) reporting success rate, time-to-task-completion, observations
processed, and idle ticks.

#### Scenario: Comparison rows collated

- **WHEN** both sync and async eval runs have produced raw JSON
- **THEN** `benchmarks/collate.py` emits comparison rows into `benchmarks/results/summary.csv`
