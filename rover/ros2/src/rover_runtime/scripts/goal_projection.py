#!/usr/bin/env python3
"""Bounding box -> body-frame goal (design D9, acquisition geometry).

Two ways to turn an open-vocabulary detector's box into a metric goal:

  bbox_to_body_groundplane()  monocular, no depth sensor. Props rest on the
      ground, so the box's BOTTOM edge is where the object meets the floor.
      Ray-cast that pixel to the ground plane using the known camera height and
      pitch. Used in sim (the VLA camera is RGB-only) and as a hardware
      fallback when depth is missing/invalid.

  bbox_to_body_depth()        uses a depth reading at the box centroid (the
      D435i on the real rover). More robust for objects that are occluded at
      the base or not ground-resting.

Camera model: pinhole, principal point at image centre, fx derived from HFOV.
Frames: body/REP-103 (x forward, y left). Camera optical axis is +x_body when
pitch = 0; positive pitch tilts the camera DOWN.

Defaults match rover_sim's VLA camera (OV9782-locked): 1280x800, HFOV 100 deg,
height 0.15 m, pitch 0.
"""

import math

CAM_W = 1280
CAM_H = 800
CAM_HFOV = math.radians(100.0)
CAM_HEIGHT = 0.15      # metres above ground
CAM_PITCH = 0.0        # radians, positive = tilted down


def focal_px(width=CAM_W, hfov=CAM_HFOV):
    """fx in pixels from horizontal FOV."""
    return (width / 2.0) / math.tan(hfov / 2.0)


def pixel_rays(u, v, width=CAM_W, height=CAM_H, hfov=CAM_HFOV):
    """Pixel -> unit-ish ray in body frame (x fwd, y left, z up), pitch=0.

    Returns the *direction* (not normalised); scale is resolved by the caller.
    """
    f = focal_px(width, hfov)
    cx, cy = width / 2.0, height / 2.0
    # camera: +x right, +y down, +z forward  ->  body: x=z, y=-x_cam, z=-y_cam
    x_cam = (u - cx) / f
    y_cam = (v - cy) / f
    return 1.0, -x_cam, -y_cam


def bbox_to_body_groundplane(bbox, cam_height=CAM_HEIGHT, pitch=CAM_PITCH,
                             width=CAM_W, height=CAM_H, hfov=CAM_HFOV):
    """(x1, y1, x2, y2) pixels -> (x, y) metres in body frame, or None.

    Uses the bottom-centre pixel (object's ground contact). Returns None when
    the ray does not descend (points at or above the horizon), which is the
    correct failure for a box whose base is not visible.
    """
    x1, y1, x2, y2 = bbox
    u = 0.5 * (x1 + x2)
    v = max(y1, y2)                      # bottom edge in image coords
    dx, dy, dz = pixel_rays(u, v, width, height, hfov)

    if pitch:                            # rotate ray about body y by +pitch (down)
        c, s = math.cos(pitch), math.sin(pitch)
        dx, dz = c * dx + s * dz, -s * dx + c * dz

    if dz >= -1e-9:                      # not pointing down -> no ground hit
        return None
    t = cam_height / (-dz)               # scale until the ray reaches z = -cam_height
    return dx * t, dy * t


def bbox_to_body_depth(bbox, depth_m, pitch=CAM_PITCH,
                       width=CAM_W, height=CAM_H, hfov=CAM_HFOV):
    """(bbox, depth at centroid) -> (x, y) metres in body frame.

    `depth_m` is range along the camera's optical axis (RealSense convention).
    """
    x1, y1, x2, y2 = bbox
    u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    dx, dy, dz = pixel_rays(u, v, width, height, hfov)
    if pitch:
        c, s = math.cos(pitch), math.sin(pitch)
        dx, dz = c * dx + s * dz, -s * dx + c * dz
    # dx is 1.0 before pitch, so scaling by depth puts the point at that range
    return dx * depth_m, dy * depth_m


def body_to_bearing_range(bx, by):
    return math.atan2(by, bx), math.hypot(bx, by)
