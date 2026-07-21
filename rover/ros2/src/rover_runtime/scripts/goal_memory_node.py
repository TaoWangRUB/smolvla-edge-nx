#!/usr/bin/env python3
"""Goal memory / mission layer scaffold (task 5.1, design D3 mission loop).

Why this exists: the VLA policy is *memoryless* (`n_obs_steps=1` — and SmolVLA
never reads that field anyway, so frame history is not available without model
surgery). Its only input is the current frame, so **a goal that leaves the
~100 deg camera FOV is unrecoverable** — measured as the rover latching onto
whatever else is visible. Design D3 puts goal persistence in the *mission
loop*, not the policy: goal memory belongs in map/state.

This node holds the goal in the **odom frame** (world-fixed). As the rover
moves, odometry keeps the goal's relative bearing/range correct even when it is
out of view — i.e. "beyond-line-of-sight goals" per task 5.1.

Interfaces
  in   /ackermann/gt_odom            nav_msgs/Odometry   (sim; EKF odom on HW)
       /goal_memory/set_odom         Float32MultiArray [x, y]      goal in odom frame
       /goal_memory/set_relative     Float32MultiArray [x, y]      goal in CURRENT body
                                     frame (the natural output of a grounding
                                     module: "red crate is 3 m ahead, 20 deg left")
       /goal_memory/clear            Float32MultiArray (any)       forget the goal
  out  /goal_memory/relative         Float32MultiArray [range, bearing, bx, by]
       /waypoint_chunk               Float32MultiArray  same format the policy
                                     emits, so the EXISTING tracker drives to the
                                     remembered goal with no other changes.

Emitting the policy's own chunk format is the point: it demonstrates the full
architecture (ground once -> remember in odom -> chunk -> tracker) and lets the
mission layer stand in for the policy on the "go to a known goal" leg.
"""

import math

# ROS imports are guarded so the pure transform helpers below can be imported
# (and unit-tested) without a ROS environment — see
# rover/eval_results/test_goal_memory.py.
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float32MultiArray
    _HAS_ROS = True
except ImportError:      # pragma: no cover - exercised only outside ROS
    _HAS_ROS = False
    Node = object

K = 10           # waypoints per chunk (match the policy's action space)
DT = 0.25        # s between waypoints
CRUISE = 0.5     # m/s
GOAL_REACH = 0.6  # stop ring, matches the eval referee
R_MIN = 0.341     # measured min turn radius (feasibility clamp lives in tracker)


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def to_body(gx, gy, x, y, yaw):
    """Goal in odom frame -> body frame at pose (x, y, yaw)."""
    dx, dy = gx - x, gy - y
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


def to_odom(bx, by, x, y, yaw):
    """Goal in body frame at pose (x, y, yaw) -> odom frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * bx - s * by, y + s * bx + c * by


def straight_chunk(bx, by, cruise=CRUISE, k=K, dt=DT):
    """K body-frame (x, y, v) waypoints stepping toward (bx, by).

    Deliberately simple: the tracker owns curvature feasibility, and the point
    here is goal persistence, not path quality. Waypoints clamp at the goal so
    the chunk 'parks' once reached (v -> 0), same shape as a hindsight label.
    """
    dist = math.hypot(bx, by)
    if dist < 1e-6:
        return [(0.0, 0.0, 0.0)] * k
    ux, uy = bx / dist, by / dist
    out = []
    for i in range(1, k + 1):
        s = min(cruise * dt * i, max(0.0, dist - GOAL_REACH * 0.5))
        v = 0.0 if s >= dist - GOAL_REACH * 0.5 else cruise
        out.append((ux * s, uy * s, v))
    return out


class GoalMemory(Node):
    def __init__(self):
        super().__init__('goal_memory', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.declare_parameter('odom_topic', '/ackermann/gt_odom')
        self.declare_parameter('emit_chunk', True)
        odom_topic = self.get_parameter('odom_topic').value
        self.emit_chunk = bool(self.get_parameter('emit_chunk').value)

        self.pose = None          # (x, y, yaw) in odom
        self.goal_odom = None     # (gx, gy) in odom — the memory

        self.pub_rel = self.create_publisher(Float32MultiArray,
                                             '/goal_memory/relative', 10)
        self.pub_chunk = self.create_publisher(Float32MultiArray,
                                               '/waypoint_chunk', 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 50)
        self.create_subscription(Float32MultiArray, '/goal_memory/set_odom',
                                 self.on_set_odom, 10)
        self.create_subscription(Float32MultiArray, '/goal_memory/set_relative',
                                 self.on_set_relative, 10)
        self.create_subscription(Float32MultiArray, '/goal_memory/clear',
                                 self.on_clear, 10)
        self.create_timer(0.1, self.tick)   # 10 Hz; mission loop is 0.1-1 Hz
        self.last_t = None

    def on_odom(self, m):
        p = m.pose.pose.position
        self.pose = (p.x, p.y, quat_yaw(m.pose.pose.orientation))
        self.last_t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def on_set_odom(self, m):
        if len(m.data) >= 2:
            self.goal_odom = (float(m.data[0]), float(m.data[1]))
            self.get_logger().info(f'goal set (odom): {self.goal_odom}')

    def on_set_relative(self, m):
        """Grounding modules speak body frame; convert once and remember."""
        if len(m.data) >= 2 and self.pose is not None:
            self.goal_odom = to_odom(float(m.data[0]), float(m.data[1]), *self.pose)
            self.get_logger().info(f'goal set (from body frame) -> odom {self.goal_odom}')

    def on_clear(self, _m):
        self.goal_odom = None
        self.get_logger().info('goal cleared')

    def tick(self):
        if self.pose is None or self.goal_odom is None:
            return
        bx, by = to_body(*self.goal_odom, *self.pose)
        rng, bear = math.hypot(bx, by), math.atan2(by, bx)

        rel = Float32MultiArray()
        rel.data = [float(rng), float(bear), float(bx), float(by)]
        self.pub_rel.publish(rel)

        if self.emit_chunk and self.last_t is not None:
            ch = straight_chunk(bx, by)
            out = Float32MultiArray()
            out.data = [float(self.last_t), float(len(ch))] + [
                float(v) for wp in ch for v in wp]
            self.pub_chunk.publish(out)


def main():
    rclpy.init()
    rclpy.spin(GoalMemory())


if __name__ == '__main__':
    main()
