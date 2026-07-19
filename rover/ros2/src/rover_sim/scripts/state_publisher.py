#!/usr/bin/env python3
"""Publish the policy-facing state vector (task 1.5): /observation/state.

Layout (Float32MultiArray, 50 Hz, sim time):
  [0] speed      m/s    — |v_xy| from GT odom linear twist (verified reliable)
  [1] yaw_rate   rad/s  — finite-differenced from GT pose *history* (the
                          OdometryPublisher twist.angular field is unreliable
                          at high curvature — measured in M0)
  [2] steering   rad    — equivalent bicycle angle from the two knuckle
                          joints: cot(d) = (cot(d_l) + cot(d_r)) / 2

/observation is by convention {/vla_camera/image, /vla_camera/camera_info,
/observation/state}; commands enter on /cmd_vel (TwistStamped). See
rover/README.md for the full contract and clock discipline.
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

YAW_WINDOW_S = 0.12


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class StatePublisher(Node):
    def __init__(self):
        super().__init__(
            'state_publisher',
            parameter_overrides=[Parameter('use_sim_time',
                                           Parameter.Type.BOOL, True)])
        self.pub = self.create_publisher(Float32MultiArray,
                                         '/observation/state', 10)
        self.create_subscription(Odometry, '/ackermann/gt_odom', self.on_odom, 50)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 50)
        self.yaw_hist = deque()
        self.steering = 0.0

    def on_joints(self, m):
        d = dict(zip(m.name, m.position))
        dl = d.get('ackermann/front_left_wheel_steering_joint')
        dr = d.get('ackermann/front_right_wheel_steering_joint')
        if dl is None or dr is None:
            return
        if abs(dl) < 1e-4 and abs(dr) < 1e-4:
            self.steering = 0.0
            return
        cots = []
        for a in (dl, dr):
            t = math.tan(a)
            if abs(t) > 1e-6:
                cots.append(1.0 / t)
        if cots:
            self.steering = math.atan(1.0 / (sum(cots) / len(cots)))

    def on_odom(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        yaw = quat_yaw(m.pose.pose.orientation)
        self.yaw_hist.append((t, yaw))
        while self.yaw_hist and t - self.yaw_hist[0][0] > YAW_WINDOW_S:
            self.yaw_hist.popleft()

        yaw_rate = 0.0
        if len(self.yaw_hist) >= 2:
            t0, y0 = self.yaw_hist[0]
            dt = t - t0
            if dt > 1e-6:
                dy = yaw - y0
                while dy > math.pi:
                    dy -= 2 * math.pi
                while dy < -math.pi:
                    dy += 2 * math.pi
                yaw_rate = dy / dt

        v = m.twist.twist.linear
        out = Float32MultiArray()
        out.data = [float(math.hypot(v.x, v.y)), float(yaw_rate),
                    float(self.steering)]
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(StatePublisher())


if __name__ == '__main__':
    main()
