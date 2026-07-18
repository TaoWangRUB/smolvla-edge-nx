# Chunk-prediction request over DDS — the all-ROS2 replacement for the gRPC PredictChunk
# hop. Correlated with PolicyChunk by request_id (one in flight per client, Algorithm 1).
int64 request_id
string task                   # instruction; server builds one predictor per task string
sensor_msgs/Image image_top   # "pixels.top", rgb8, raw sim/camera frame
float32[] agent_pos           # 14-D joint state
float64 client_send_ts        # client wall clock (diagnostics only)
