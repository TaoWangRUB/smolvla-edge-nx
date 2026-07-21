#!/usr/bin/env python3
"""Goal acquisition: instruction + image -> body-frame goal (design D9 step c).

The stage that replaces what SmolVLA never learned. M1 measured the policy
selecting the commanded object at roughly chance; this path measured **94%**
across scene families offline (open_ground 98%, corridor 89%) with 0.13-0.22 m
median position error -- see design.md "D9 validation".

    instruction --> target phrase --> open-vocab detector --> bbox
                                                                |
                                        ground-plane projection v
                                              /goal_memory/set_relative

It publishes the goal **once**, in the body frame, and then stops: goal_memory
converts it to the odom frame and holds it there, so the goal survives leaving
the camera FOV. That is the entire point of the split -- acquisition is a
mission-loop event (0.1-1 Hz per design D3), not a control-loop one.

Two findings from offline validation are load-bearing here; changing either
will silently degrade acquisition:

  * ONE query per forward pass. Batching several phrases lets OWL-ViT's queries
    suppress one another -- 13/39 targets scored EXACTLY 0.0 batched and
    0.10-0.29 alone. A confidence-threshold sweep cannot see this.
  * OWLv2, not OWL-ViT. `owlvit-base-patch32` (the NanoOWL backbone) gets 22%
    recall here, and recall is FLAT with range (29% at 0.5 m), so neither
    tiling nor driving closer rescues it. Cost is ~12x; affordable only because
    acquisition is one-shot. On-NX feasibility is UNPROVEN -- see task 5.5.

Interfaces
  in   /vla_camera/image            sensor_msgs/Image
       /instruction                 std_msgs/String   e.g. "drive to the red crate"
  out  /goal_memory/set_relative    Float32MultiArray [bx, by]
       /goal_acquisition/debug      Float32MultiArray [score, x1, y1, x2, y2, bx, by]
"""

import math
import threading

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray, String
    _HAS_ROS = True
except ImportError:      # pragma: no cover - exercised only outside ROS
    _HAS_ROS = False
    Node = object

from goal_projection import OBJECT_RADIUS, bbox_to_body_groundplane

COLORS = ('red', 'blue', 'green', 'yellow')
SHAPES = ('barrel', 'crate', 'ball', 'pillar')
DEFAULT_MODEL = 'google/owlv2-base-patch16-ensemble'
MIN_SCORE = 0.05


def parse_instruction(text):
    """'drive to the red crate' -> ('red', 'crate'), or None.

    Template-independent by construction: it scans for the closed colour/shape
    vocabulary rather than matching the 18 phrasings, so held-out templates
    ("your destination is the ...") parse identically to trained ones.
    """
    t = (text or '').lower()
    color = next((c for c in COLORS if c in t), None)
    shape = next((s for s in SHAPES if s in t), None)
    if color is None or shape is None:
        return None
    return color, shape


def detect_target(model, proc, img, phrase, device, threshold=MIN_SCORE):
    """Best box for `phrase`, queried ALONE. Returns (bbox, score) or None."""
    import torch
    inputs = proc(text=[[phrase]], images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        outputs=out,
        target_sizes=torch.tensor([img.size[::-1]]).to(device),
        threshold=threshold)[0]
    best, best_score = None, -1.0
    for score, box in zip(res['scores'], res['boxes']):
        if float(score) > best_score:
            best, best_score = [float(v) for v in box], float(score)
    return (best, best_score) if best is not None else None


class GoalAcquisition(Node):
    def __init__(self):
        super().__init__('goal_acquisition', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.declare_parameter('model', DEFAULT_MODEL)
        self.declare_parameter('min_score', MIN_SCORE)
        self.declare_parameter('period_s', 2.0)
        self.declare_parameter('lock_once', True)

        self.min_score = float(self.get_parameter('min_score').value)
        self.lock_once = bool(self.get_parameter('lock_once').value)
        self.target = None       # (color, shape) from the instruction
        self.frame = None        # latest PIL image
        self.locked = False
        self._busy = False       # inference is seconds long; never re-enter

        self.pub_goal = self.create_publisher(Float32MultiArray,
                                              '/goal_memory/set_relative', 10)
        self.pub_dbg = self.create_publisher(Float32MultiArray,
                                             '/goal_acquisition/debug', 10)
        self.create_subscription(String, '/instruction', self.on_instruction, 10)
        self.create_subscription(Image, '/vla_camera/image', self.on_image, 1)

        self.get_logger().info(f'loading detector {self.get_parameter("model").value} ...')
        self._load()
        self.create_timer(float(self.get_parameter('period_s').value), self.tick)

    def _load(self):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor)
        name = self.get_parameter('model').value
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            name).eval().to(self.device)
        self.get_logger().info(f'detector ready on {self.device}')

    def on_instruction(self, m):
        tgt = parse_instruction(m.data)
        if tgt is None:
            self.get_logger().warn(f'cannot parse instruction: {m.data!r}')
            return
        if tgt != self.target:
            self.target, self.locked = tgt, False   # new order -> re-acquire
            self.get_logger().info(f'target: {tgt[0]} {tgt[1]}')

    def on_image(self, m):
        from PIL import Image as PILImage
        if m.encoding not in ('rgb8', 'bgr8'):
            return
        buf = bytes(m.data)
        img = PILImage.frombytes('RGB', (m.width, m.height), buf)
        self.frame = img if m.encoding == 'rgb8' else img.convert('RGB')

    def tick(self):
        if (self.target is None or self.frame is None or self._busy
                or (self.locked and self.lock_once)):
            return
        self._busy = True
        # Detector inference is seconds long -- off the executor thread, or
        # /clock and the 50 Hz tracker stall behind it.
        threading.Thread(target=self._acquire, args=(self.frame,),
                         daemon=True).start()

    def _acquire(self, img):
        try:
            color, shape = self.target
            hit = detect_target(self.model, self.proc, img, f'{color} {shape}',
                                self.device, self.min_score)
            if hit is None:
                self.get_logger().info(f'no detection for "{color} {shape}"')
                return
            bbox, score = hit
            proj = bbox_to_body_groundplane(
                bbox, object_radius=OBJECT_RADIUS.get(shape, 0.0))
            if proj is None:
                self.get_logger().warn('bbox base at/above horizon; no ground hit')
                return
            bx, by = proj
            msg = Float32MultiArray()
            msg.data = [float(bx), float(by)]
            self.pub_goal.publish(msg)

            dbg = Float32MultiArray()
            dbg.data = [float(score)] + [float(v) for v in bbox] + [float(bx), float(by)]
            self.pub_dbg.publish(dbg)

            self.locked = True
            self.get_logger().info(
                f'acquired "{color} {shape}" score={score:.3f} -> body '
                f'({bx:.2f}, {by:.2f}) range={math.hypot(bx, by):.2f} m '
                f'bearing={math.degrees(math.atan2(by, bx)):.1f} deg')
        finally:
            self._busy = False


def main():
    rclpy.init()
    rclpy.spin(GoalAcquisition())


if __name__ == '__main__':
    main()
