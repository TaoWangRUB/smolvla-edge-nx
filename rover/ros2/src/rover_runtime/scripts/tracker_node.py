#!/usr/bin/env python3
"""50 Hz Pure Pursuit tracker on body-frame waypoint chunks (task 2.5).

Chunk topic contract (/waypoint_chunk, std_msgs/Float32MultiArray):
  data = [capture_t, K, x0, y0, v0, x1, y1, v1, ...]
where (x_i, y_i, v_i) is the i-th waypoint in the BODY FRAME AT capture_t
(design D2) and capture_t is the sim-time stamp of the image the policy saw.

Latency compensation (design D3): on arrival, waypoints are transformed
into the world (odom) frame using the pose interpolated from history at
capture_t — executing in world frame then absorbs all motion since capture.
Each new chunk REPLACES the active path.

Hard limits enforced here, not upstream: v <= V_MAX, |w| <= v / R_MIN
(R_MIN = 0.341 m measured), and the staleness watchdog ramps speed to zero
as chunk age exceeds STALE_S (age > STALE_S + RAMP_S -> full stop).
"""

import bisect
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

R_MIN = 0.341
V_MAX = 0.8
STALE_S = 1.0
RAMP_S = 0.5
GOAL_TOL = 0.15


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Tracker(Node):
    def __init__(self):
        super().__init__('tracker_node', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.declare_parameter('odom_topic', '/ackermann/gt_odom')
        odom_topic = self.get_parameter('odom_topic').value

        self.pose_hist = deque()      # (t, x, y, yaw), ~3 s
        self.path = []                # [(x, y, v)] world frame
        self.wp_i = 0
        self.chunk_t = None           # capture time of active chunk
        self.pose = None

        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 50)
        self.create_subscription(Float32MultiArray, '/waypoint_chunk',
                                 self.on_chunk, 10)
        self.create_timer(0.02, self.tick)

    def on_odom(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p = m.pose.pose.position
        yaw = quat_yaw(m.pose.pose.orientation)
        self.pose = (p.x, p.y, yaw)
        self.pose_hist.append((t, p.x, p.y, yaw))
        while self.pose_hist and t - self.pose_hist[0][0] > 3.0:
            self.pose_hist.popleft()

    def pose_at(self, t):
        h = self.pose_hist
        if not h:
            return None
        ts = [r[0] for r in h]
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            return h[0][1:]
        if i >= len(h):
            return h[-1][1:]
        t0, x0, y0, w0 = h[i - 1]
        t1, x1, y1, w1 = h[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        dyaw = w1 - w0
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi
        return (x0 + a * (x1 - x0), y0 + a * (y1 - y0), w0 + a * dyaw)

    def on_chunk(self, m):
        d = list(m.data)
        if len(d) < 5:
            return
        capture_t, k = d[0], int(d[1])
        base = self.pose_at(capture_t)
        if base is None:
            self.get_logger().warn('chunk dropped: no pose history at capture time')
            return
        bx, by, byaw = base
        c, s = math.cos(byaw), math.sin(byaw)
        path = []
        for i in range(k):
            x, y, v = d[2 + 3 * i: 5 + 3 * i]
            path.append((bx + c * x - s * y, by + s * x + c * y, v))
        self.path = path            # replace, never queue (D3)
        self.wp_i = 0
        self.chunk_t = capture_t

    def tick(self):
        if self.pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        scale = 0.0
        if self.path and self.chunk_t is not None:
            age = now - self.chunk_t
            scale = 1.0 if age <= STALE_S else max(
                0.0, 1.0 - (age - STALE_S) / RAMP_S)

        if scale > 0.0:
            x, y, yaw = self.pose
            lookahead = 0.45
            while (self.wp_i < len(self.path) - 1 and
                   math.hypot(self.path[self.wp_i][0] - x,
                              self.path[self.wp_i][1] - y) < lookahead):
                self.wp_i += 1
            tx, ty, tv = self.path[self.wp_i]
            dist = math.hypot(tx - x, ty - y)
            at_end = (self.wp_i >= len(self.path) - 1 and dist < GOAL_TOL)
            if not at_end:
                alpha = math.atan2(ty - y, tx - x) - yaw
                while alpha > math.pi:
                    alpha -= 2 * math.pi
                while alpha < -math.pi:
                    alpha += 2 * math.pi
                v = min(max(tv, 0.12), V_MAX) * scale
                if abs(alpha) > 0.8:
                    v = min(v, 0.2)
                ld = max(0.2, dist)
                w = 2.0 * v * math.sin(alpha) / ld
                wmax = v / R_MIN
                msg.twist.linear.x = v
                msg.twist.angular.z = max(-wmax, min(wmax, w))
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(Tracker())


if __name__ == '__main__':
    main()
