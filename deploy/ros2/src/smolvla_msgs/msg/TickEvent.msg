# Per-tick diagnostics from the async client, comparable event-by-event with the Python
# AsyncRunner's trace (spec ros2-async-control).
int64 tick
int64 episode
int32 queue_depth       # before the pop
int32 queue_after       # after pop + any merge
bool sent               # PredictChunk triggered this tick
bool filtered           # similarity filter dropped the candidate observation
bool merged             # a chunk landed and was aggregated with the live queue
bool idle               # no action available: client held pose
float64 rtt_s           # round-trip of the chunk that landed this tick (0 if none)
float64 proc_ms         # wall time spent in this node's on_observation callback (client compute)
