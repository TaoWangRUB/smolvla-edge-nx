# One control-tick observation from the sim bridge. Single message (image + joint state +
# tick) so the client never has to time-synchronize separate topics — mirrors the gRPC
# Observation message (design D2/D3).
int64 tick                    # sim tick index (0 == fresh episode: client must reset)
int64 episode                 # episode counter, seeds map 1:1 (reset(seed=episode))
sensor_msgs/Image image_top   # "pixels.top", rgb8, straight from SimEnv
float32[] agent_pos           # 14-D joint state ("agent_pos")
