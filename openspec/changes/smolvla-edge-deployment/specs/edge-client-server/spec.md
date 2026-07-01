## ADDED Requirements

### Requirement: gRPC policy server on the workstation

The system SHALL serve the fine-tuned policy over gRPC from the Titan X workstation, using a
defined `policy.proto` contract, so a remote client can request actions for observations.

#### Scenario: Start the server

- **WHEN** an operator runs `python deploy/client_server/server.py --policy-path <checkpoint> --precision fp16` on the workstation
- **THEN** the server loads the policy and listens for gRPC action requests

### Requirement: Thin NX control client replaying held-out frames

The system SHALL provide an NX client that replays held-out dataset frames as observations, calls
the remote server, and requires no physical robot.

#### Scenario: Run the client against the server

- **WHEN** an operator runs `python deploy/client_server/client.py --server <workstation-ip>:50051 --out benchmarks/results/raw/client_server.json` on the NX
- **THEN** the client streams replayed observations to the server and receives actions

### Requirement: Report split round-trip latency

The client SHALL report round-trip latency split into server-compute time and network overhead,
producing the client/server benchmark tier.

#### Scenario: Latency breakdown emitted

- **WHEN** a client/server run completes
- **THEN** the output JSON records round-trip latency separated into server-compute vs. network
  overhead
