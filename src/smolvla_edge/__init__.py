"""smolvla_edge: fine-tune SmolVLA and deploy/benchmark it on the Jetson Xavier NX.

Modules:
    infer  - load a policy + dataset and run inference (stack smoke-test)
    eval   - run a held-out eval and report a success-rate number
    bench  - measure latency / throughput / memory for a deployment config
    common - shared helpers (device selection, timers, policy loading)
"""

__version__ = "0.1.0"
