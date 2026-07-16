## ADDED Requirements

### Requirement: C++ async control node equivalent to AsyncRunner

The system SHALL provide a C++ `rclcpp` node implementing the client side of SmolVLA §3.3
Algorithm 1 with the same semantics as `src/smolvla_edge/async_infer.py`: one control tick per
received `/observation` (the bridge owns the 50 Hz clock, so one observation == one tick), pop
one action per tick and publish it; after the pop, when `queue_size / n < g`, snapshot the
latest observation and issue a non-blocking `PredictChunk` gRPC call (at most one in flight);
when the queue is empty, the latest observation SHALL be processed regardless of the
similarity filter. Chunk visibility follows real gRPC arrival time (the Python runner's
virtual-time emulation is unnecessary under wall-clock pacing).
`g`, `epsilon`, `dt`, the aggregator and the server address SHALL be ROS parameters. Sync
inference SHALL be the `g = 0` configuration of the same node.

#### Scenario: Threshold trigger after pop

- **WHEN** a tick pops an action and the remaining queue fraction drops below `g` with no
  request in flight and the observation passes the similarity filter
- **THEN** the node issues exactly one non-blocking `PredictChunk` and keeps publishing queued
  actions on subsequent ticks

#### Scenario: Empty queue bypasses the filter

- **WHEN** the action queue is empty and no chunk has landed
- **THEN** the node processes the most recent observation regardless of `epsilon` and records
  idle ticks until a chunk arrives

### Requirement: Ported similarity filter and chunk aggregation

The node SHALL implement the `epsilon` joint-space similarity filter and the `new_wins`
overlap aggregation with formulas identical to `async_infer.py` (blend weights
`linspace(0.5, 1.0, m)` over the overlap), such that both stacks produce the same merged queue
for the same inputs.

#### Scenario: Aggregation parity on fixed vectors

- **WHEN** the C++ aggregation is fed a fixture of old-queue/new-chunk pairs exported from the
  Python implementation
- **THEN** the merged queues match the Python outputs within float tolerance (unit test)

### Requirement: Per-tick diagnostics for cross-stack comparison

The node SHALL publish per-tick events (tick index, queue depth, sent/filtered/merged/idle
flags, request round-trip time) on a diagnostics topic, recordable to compare runs
event-by-event against Python `AsyncRunner` traces.

#### Scenario: Trace comparison against the Python oracle

- **WHEN** the same episode seed is run on the Python stack and the ROS2 stack with equal
  `g`/`epsilon` and the Python policy server
- **THEN** recorded event sequences agree on trigger ticks, filter decisions and merge counts,
  and the closed-loop success rate over ≥ 50 episodes matches within binomial noise
