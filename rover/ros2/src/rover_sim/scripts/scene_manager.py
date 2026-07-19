#!/usr/bin/env python3
"""Per-episode scene randomization for rover_sim (task 1.4, minimal hooks).

Samples an EpisodeConfig (seed-deterministic) and applies it to a *running*
gz world through gz services — no simulator restart between episodes:

  - props:        /world/<scene>/create   (EntityFactory, SDF string)
  - cleanup:      /world/<scene>/remove   (deterministic ep_* names)
  - ground color: spawned ep_ground slab over the base plane
  - sun:          /world/<scene>/light_config
  - spawn pose:   /world/<scene>/set_pose on the "ackermann" model

Scene construction guarantees language is *necessary* (spec: rover-sim-environment):
the goal prop always has a same-color/different-shape distractor and a
same-shape/different-color distractor. The full config (JSON) is the episode
metadata the recorder logs (task 1.7) and the failure-slicing key (D6).

Not yet randomized (M1+): weather, camera exposure/extrinsic jitter, sensor
noise — hooks belong to the recorder/augmentation stage, noted in tasks.md.

CLI:
  scene_manager.py apply --scene open_ground --seed 42 [--out ep.json]
  scene_manager.py clear --scene open_ground
"""

import argparse
import dataclasses
import json
import math
import random
import subprocess
import sys

MAX_PROPS = 10
SHAPES = {
    'barrel': ('cylinder', {'radius': 0.15, 'length': 0.5}),
    'pillar': ('cylinder', {'radius': 0.08, 'length': 0.8}),
    'crate': ('box', {'size': (0.4, 0.4, 0.4)}),
    'ball': ('sphere', {'radius': 0.15}),
}
COLORS = {
    'red': (0.8, 0.05, 0.05),
    'blue': (0.05, 0.05, 0.8),
    'green': (0.05, 0.55, 0.05),
    'yellow': (0.85, 0.75, 0.05),
}
# Prop placement region per scene family: (x_min, x_max, y_min, y_max)
SCENE_BOUNDS = {
    'open_ground': (2.0, 8.0, -3.0, 3.0),
    'corridor': (2.0, 7.5, -0.6, 0.6),
    'parking_lot': (2.0, 7.0, -2.0, 2.0),
    'props_ground': (2.0, 8.0, -3.0, 3.0),
}
CAM_HALF_FOV = math.radians(50.0)


@dataclasses.dataclass
class Prop:
    name: str
    shape: str
    color: str
    x: float
    y: float
    yaw: float


@dataclasses.dataclass
class EpisodeConfig:
    seed: int
    scene: str
    sun_elevation_deg: float
    sun_azimuth_deg: float
    sun_intensity: float
    ground_rgb: tuple
    spawn: tuple            # (x, y, yaw)
    props: list             # [Prop]
    goal_index: int
    instruction: str

    def to_json(self):
        d = dataclasses.asdict(self)
        return json.dumps(d, indent=2)


def sample(scene: str, seed: int) -> EpisodeConfig:
    rng = random.Random(seed)
    x0, x1, y0, y1 = SCENE_BOUNDS[scene]
    spawn = (0.0, rng.uniform(-0.3, 0.3) if scene != 'corridor' else 0.0,
             rng.uniform(-0.15, 0.15))

    def place(existing, min_sep=0.7):
        for _ in range(200):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if math.hypot(x - spawn[0], y - spawn[1]) < 1.2:
                continue
            if all(math.hypot(x - p.x, y - p.y) >= min_sep for p in existing):
                return x, y
        raise RuntimeError('placement failed; loosen bounds or min_sep')

    props = []
    goal_shape = rng.choice(list(SHAPES))
    goal_color = rng.choice(list(COLORS))

    # Goal must be visible from spawn: inside the camera cone, 2-7 m out.
    for _ in range(200):
        dist = rng.uniform(2.0, min(7.0, x1))
        bearing = spawn[2] + rng.uniform(-CAM_HALF_FOV * 0.8, CAM_HALF_FOV * 0.8)
        gx = spawn[0] + dist * math.cos(bearing)
        gy = spawn[1] + dist * math.sin(bearing)
        if x0 <= gx <= x1 and y0 <= gy <= y1:
            break
    props.append(Prop('ep_prop_0', goal_shape, goal_color, gx, gy,
                      rng.uniform(0, math.pi)))

    # Hard negatives (spec requirement): share color, and share shape.
    other_shapes = [s for s in SHAPES if s != goal_shape]
    other_colors = [c for c in COLORS if c != goal_color]
    hard = [(rng.choice(other_shapes), goal_color),
            (goal_shape, rng.choice(other_colors))]
    for i, (shape, color) in enumerate(hard, start=1):
        x, y = place(props)
        props.append(Prop(f'ep_prop_{i}', shape, color, x, y,
                          rng.uniform(0, math.pi)))

    # Filler clutter — never an exact (shape, color) duplicate of the goal,
    # or the instruction stops identifying a unique target.
    for i in range(3, 3 + rng.randint(1, 3)):
        x, y = place(props)
        while True:
            shape, color = rng.choice(list(SHAPES)), rng.choice(list(COLORS))
            if (shape, color) != (goal_shape, goal_color):
                break
        props.append(Prop(f'ep_prop_{i}', shape, color, x, y,
                          rng.uniform(0, math.pi)))

    return EpisodeConfig(
        seed=seed,
        scene=scene,
        sun_elevation_deg=rng.uniform(20.0, 80.0),
        sun_azimuth_deg=rng.uniform(0.0, 360.0),
        sun_intensity=rng.uniform(0.5, 1.15),
        ground_rgb=(rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8),
                    rng.uniform(0.2, 0.8)),
        spawn=spawn,
        props=props,
        goal_index=0,
        instruction=f'drive to the {goal_color} {goal_shape}',
    )


# ---------------------------------------------------------------- gz plumbing

def gz_service(world: str, service: str, reqtype: str, req: str,
               reptype: str = 'gz.msgs.Boolean', timeout_ms: int = 4000,
               attempts: int = 3):
    """One gz service call with retries: each CLI invocation does fresh
    transport discovery, which intermittently exceeds the timeout under
    batch load (measured ~1/3 of calls failing at 3 s during datagen)."""
    cmd = ['gz', 'service', '-s', f'/world/{world}/{service}',
           '--reqtype', reqtype, '--reptype', reptype,
           '--timeout', str(timeout_ms), '--req', req]
    msg = ''
    for _ in range(attempts):
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode == 0 and 'data: true' in out.stdout:
            return True, out.stdout.strip()
        msg = out.stdout.strip() or out.stderr.strip()
    return False, msg


def prop_sdf(p: Prop) -> str:
    kind, dims = SHAPES[p.shape]
    r, g, b = COLORS[p.color]
    if kind == 'cylinder':
        geom = (f"<cylinder><radius>{dims['radius']}</radius>"
                f"<length>{dims['length']}</length></cylinder>")
        z = dims['length'] / 2
    elif kind == 'sphere':
        geom = f"<sphere><radius>{dims['radius']}</radius></sphere>"
        z = dims['radius']
    else:
        sx, sy, sz = dims['size']
        geom = f'<box><size>{sx} {sy} {sz}</size></box>'
        z = dims['size'][2] / 2
    mat = (f'<material><ambient>{r} {g} {b} 1</ambient>'
           f'<diffuse>{r} {g} {b} 1</diffuse></material>')
    return (f'<?xml version="1.0"?><sdf version="1.9">'
            f'<model name="{p.name}"><static>true</static>'
            f'<pose>{p.x} {p.y} {z} 0 0 {p.yaw}</pose>'
            f'<link name="link">'
            f'<collision name="c"><geometry>{geom}</geometry></collision>'
            f'<visual name="v"><geometry>{geom}</geometry>{mat}</visual>'
            f'</link></model></sdf>')


def ground_sdf(rgb) -> str:
    r, g, b = rgb
    # 10 mm thick, top at z=+10 mm: enough separation to avoid z-fighting
    # with the base ground plane at grazing view angles (measured artifact).
    return ('<?xml version="1.0"?><sdf version="1.9">'
            '<model name="ep_ground"><static>true</static>'
            '<pose>4 0 0.005 0 0 0</pose><link name="link">'
            '<visual name="v"><geometry><box><size>30 30 0.01</size></box></geometry>'
            f'<material><ambient>{r} {g} {b} 1</ambient>'
            f'<diffuse>{r} {g} {b} 1</diffuse>'
            '<specular>0.05 0.05 0.05 1</specular></material>'
            '</visual></link></model></sdf>')


def spawn_model(world: str, sdf: str):
    escaped = sdf.replace('"', '\\"')
    return gz_service(world, 'create', 'gz.msgs.EntityFactory',
                      f'sdf: "{escaped}"')


def clear_episode(world: str):
    # attempts=1: removing an absent entity returns a fast negative — do not
    # retry those, or the stateless clear costs ~12 s per episode.
    removed = 0
    for name in ['ep_ground'] + [f'ep_prop_{i}' for i in range(MAX_PROPS)]:
        ok, _ = gz_service(world, 'remove', 'gz.msgs.Entity',
                           f'name: "{name}" type: MODEL', attempts=1)
        removed += bool(ok)
    ok, _ = gz_service(world, 'remove', 'gz.msgs.Entity',
                       'name: "ep_light" type: LIGHT', attempts=1)
    return removed + bool(ok)


def set_sun(world: str, cfg: EpisodeConfig):
    # /world/*/light_config acks but never reaches the sensors render scene
    # (measured); spawning a fresh light entity per episode does work.
    el = math.radians(cfg.sun_elevation_deg)
    az = math.radians(cfg.sun_azimuth_deg)
    d = (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az),
         -math.sin(el))
    i = cfg.sun_intensity
    sdf = ('<?xml version="1.0"?><sdf version="1.9">'
           '<light name="ep_light" type="directional">'
           '<cast_shadows>true</cast_shadows>'
           f'<diffuse>{i:.3f} {i:.3f} {i * 0.97:.3f} 1</diffuse>'
           '<specular>0.2 0.2 0.2 1</specular>'
           f'<direction>{d[0]:.4f} {d[1]:.4f} {d[2]:.4f}</direction>'
           '</light></sdf>')
    return spawn_model(world, sdf)


def teleport_rover(world: str, spawn):
    x, y, yaw = spawn
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
    req = (f'name: "ackermann" position {{x: {x} y: {y} z: 0.06}} '
           f'orientation {{z: {qz:.6f} w: {qw:.6f}}}')
    return gz_service(world, 'set_pose', 'gz.msgs.Pose', req)


# Scenes whose floor detail (e.g. parking bay lines) must stay visible skip
# the ground-color slab; their appearance varies via lighting until M1's
# texture randomization.
NO_SLAB_SCENES = {'parking_lot'}


def apply(cfg: EpisodeConfig):
    world = cfg.scene
    failures = []
    clear_episode(world)
    if cfg.scene not in NO_SLAB_SCENES:
        ok, msg = spawn_model(world, ground_sdf(cfg.ground_rgb))
        if not ok:
            failures.append(f'ground: {msg}')
    for p in cfg.props:
        ok, msg = spawn_model(world, prop_sdf(p))
        if not ok:
            failures.append(f'{p.name}: {msg}')
    ok, msg = set_sun(world, cfg)
    if not ok:
        failures.append(f'sun: {msg}')
    ok, msg = teleport_rover(world, cfg.spawn)
    if not ok:
        failures.append(f'teleport: {msg}')
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('apply')
    a.add_argument('--scene', required=True, choices=list(SCENE_BOUNDS))
    a.add_argument('--seed', type=int, required=True)
    a.add_argument('--out', help='write episode config JSON here')
    c = sub.add_parser('clear')
    c.add_argument('--scene', required=True, choices=list(SCENE_BOUNDS))
    args = ap.parse_args()

    if args.cmd == 'clear':
        n = clear_episode(args.scene)
        print(f'removed {n} episode entities')
        return

    cfg = sample(args.scene, args.seed)
    failures = apply(cfg)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(cfg.to_json())
    print(cfg.to_json())
    if failures:
        print(f'FAILED: {failures}', file=sys.stderr)
        sys.exit(1)
    print(f'# episode applied: scene={cfg.scene} seed={cfg.seed} '
          f'instruction={cfg.instruction!r}', file=sys.stderr)


if __name__ == '__main__':
    main()
