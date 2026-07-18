# Action-chunk reply over DDS, mirrors the gRPC ActionChunk (+ request correlation).
int64 request_id
uint32 rows                   # chunk length (50)
uint32 cols                   # action dim (14)
float32[] data                # row-major (rows x cols)
float64 server_recv_ts        # server clock: request received
float64 server_send_ts        # server clock: reply sent (compute = send - recv)
