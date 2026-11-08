"""The MuJoCo scene of Holosoma's G1 velocity task (upstream 79ba714, `simulator/mujoco/scene_manager.py`): the robot on the
terrain, as the MjModel that the simulator (sim.py) runs. `build` compiles the scene and `spec` returns it before compilation.

Holosoma's MuJoCo backends compile: a world with timestep 0.005 and MuJoCo's default solver options (Euler, Newton, 100
iterations, 50 line-search iterations, pyramidal cone, tolerance 1e-8); a terrain geom named `floor` (friction
[0.7, 0.005, 0.001], solimp [0.99, 0.99, 0.01, 0.5, 2], solref [0.001, 1], contype 2 / conaffinity 1); and the G1 MJCF
(`data/robots/g1/g1_29dof.xml`) attached with prefix `robot_`, its collision geoms set to conaffinity 2 so that the robot
collides with the terrain only. The MJCF's 14 explicit sole pairs (`<pair geom2="floor" solref="0.01 1" friction="0.8 0.8"/>`)
bind to that floor and carry their own contact parameters.

Here the terrain is one heightfield per terrain tile, each centered on its tile, and the flat border is four boxes. The reason
is that MuJoCo Warp's heightfield collision loses float32 precision with the distance from the heightfield's center. On
Holosoma's single 160 x 240 m field, robots on the corner tiles blow up: 511 of 512 robots within 20 s under violent random
actions and 10 of 512 under normal ones, against 14 and 0 on the central tiles. With one field per tile it is 12 and 0 on the
corner tiles (tools/stress_sim.py). Holosoma's README reports the same failure for PPO on MuJoCo Warp at full terrain scale.

An explicit pair names one terrain geom, and there are now many (one per tile, and the border). So the sole pairs become dynamic
contacts between the sole geoms and every terrain geom: the sole geom gets priority, so its contacts use the pairs' parameters
(friction 0.8, solref 0.01 1, MuJoCo's default solimp). These are the same contacts with the same parameters
(tests/test_scene.py), and friction randomization writes the sole geoms' friction.

Options of `build` and `spec`:
  foot      "capsules" (Holosoma's 7 thin capsules per sole) or "box" (RL-X's G1 box foot: half-sizes 0.095 x 0.027 x 0.008
            at (0.04, 0, -0.0274) in the ankle-roll frame, same sole height), with the same contact parameters
  terrain   None for a plane, or a `terrain.Terrain` (heightfield samples exactly one pixel apart)
  tiles     False: one heightfield for the whole terrain (Holosoma's layout, for comparisons)
  pairs     True: keep the explicit pairs (needs a single terrain geom; Holosoma's structure, for the equivalence test)
  visual    False drops the visual meshes (never collide; MuJoCo would still place them every step)
  sensors   False drops the MJCF's IMU and lidar-frame sensors: the task reads qpos / qvel, and the accelerometer would cost
            a post-constraint pass (cacc, cfrc_int, cfrc_ext) every substep
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from . import terrain as terrain_mod

# Holosoma's G1 MJCF, unmodified; the environment variable XSOMA_G1_XML overrides the path. Its visual meshes (assets/meshes,
# tools/download_meshes.sh) are only loaded with visual=True.
G1_XML = os.environ.get("XSOMA_G1_XML") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "g1_29dof.xml")

FLOOR = {"friction": [0.7, 0.005, 0.001], "solimp": [0.99, 0.99, 0.01, 0.5, 2], "solref": [0.001, 1]}  # Holosoma's terrain geom
# Holosoma's explicit sole-floor pairs; friction is (sliding twice, torsional, rolling twice), as in MuJoCo's <pair>
PAIR = {"friction": [0.8, 0.8, 0.005, 0.0001, 0.0001], "solref": [0.01, 1], "solimp": [0.9, 0.95, 0.001, 0.5, 2], "condim": 3}
BOX_FOOT = {"size": [0.095, 0.027, 0.008], "pos": [0.04, 0.0, -0.0274]}  # half-sizes, and position in the ankle-roll frame
BASE = 0.1  # heightfield depth below its lowest sample


def _hfield(world, name, heights, x0, y0, hs):
    """A heightfield geom whose samples (nx, ny), heights in meters, sit hs apart starting at world (x0, y0)."""
    nx, ny = heights.shape
    lo, hi = float(heights.min()), float(heights.max())
    span = max(hi - lo, 1e-6)
    rx, ry = (nx - 1) * hs / 2, (ny - 1) * hs / 2
    # MuJoCo's size is (x radius, y radius, elevation range, base depth); the elevations are stored in [0, 1], rows along y
    world.add_hfield(name=name, nrow=ny, ncol=nx, size=[rx, ry, span, BASE], userdata=((heights - lo) / span).T.astype(np.float32).ravel().tolist())
    return world.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name, pos=[x0 + rx, y0 + ry, lo])


def _terrain_geoms(world, terrain: terrain_mod.Terrain, tiles: bool):
    """The terrain geoms: with `tiles`, one heightfield per terrain tile and four flat border boxes; otherwise one heightfield
    for the whole terrain."""
    cfg, h = terrain.cfg, terrain.heights_m
    hs, x0 = cfg.horizontal_scale, terrain.x0
    if not tiles:
        return [_hfield(world, "floor", h, x0, x0, hs)]
    tp, b = int(cfg.tile / hs), int(cfg.border / hs)
    R, C = cfg.num_rows, cfg.num_cols
    geoms = []
    for i in range(R):
        for j in range(C):  # samples of the tile plus the first of the next; the outer tiles also take the last border sample
            xa, ya = b + i * tp - (i == 0), b + j * tp - (j == 0)
            geoms.append(_hfield(world, f"floor_{i}_{j}", h[xa : b + (i + 1) * tp + 1, ya : b + (j + 1) * tp + 1], x0 + xa * hs, x0 + ya * hs, hs))
    # the flat border (height 0) outside the tiles' samples [(b - 1) hs, (b + n tp) hs]: four boxes with their top at z = 0
    lo_x, hi_x, lo_y, hi_y = x0 + (b - 1) * hs, x0 + (b + R * tp) * hs, x0 + (b - 1) * hs, x0 + (b + C * tp) * hs
    end_x, end_y = x0 + (h.shape[0] - 1) * hs, x0 + (h.shape[1] - 1) * hs
    for name, (xa, xb, ya, yb) in {"left": (x0, lo_x, x0, end_y), "right": (hi_x, end_x, x0, end_y), "front": (lo_x, hi_x, x0, lo_y),
                                   "back": (lo_x, hi_x, hi_y, end_y)}.items():  # fmt: skip
        geoms.append(world.worldbody.add_geom(name=f"floor_border_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(xb - xa) / 2, (yb - ya) / 2, BASE],
                                              pos=[(xa + xb) / 2, (ya + yb) / 2, -BASE]))  # fmt: skip
    return geoms


def build(foot: str = "capsules", terrain: terrain_mod.Terrain | None = None, tiles: bool = True, pairs: bool = False, visual: bool = False,
          sensors: bool = False, iterations: int = 100, ls_iterations: int = 50, ccd_tolerance: float = 1e-6, xml: str = G1_XML) -> mujoco.MjModel:  # fmt: skip
    """Compile the scene of `spec`, which takes the same options."""
    return spec(foot, terrain, tiles, pairs, visual, sensors, iterations, ls_iterations, ccd_tolerance, xml).compile()


def spec(foot: str = "capsules", terrain: terrain_mod.Terrain | None = None, tiles: bool = True, pairs: bool = False, visual: bool = False,
         sensors: bool = False, iterations: int = 100, ls_iterations: int = 50, ccd_tolerance: float = 1e-6, xml: str = G1_XML) -> mujoco.MjSpec:  # fmt: skip
    """The scene before compilation (with build's options), for example to restyle it for rendering."""
    world = mujoco.MjSpec()
    world.copy_during_attach = True
    world.option.timestep = 0.005
    world.option.iterations = iterations
    world.option.ls_iterations = ls_iterations
    world.option.ccd_tolerance = ccd_tolerance
    if terrain is None:
        floors = [world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])]
    else:
        floors = _terrain_geoms(world, terrain, tiles)
    for f in floors:
        f.friction, f.solimp, f.solref = FLOOR["friction"], FLOOR["solimp"], FLOOR["solref"]
        f.contype, f.conaffinity = 2, 1
    assert not pairs or len(floors) == 1, "explicit pairs bind to a single terrain geom"

    robot = mujoco.MjSpec.from_file(xml)
    if not visual:  # drop the visual-only geoms (they never collide) and the meshes
        for g in [g for g in robot.geoms if g.contype == 0 and g.conaffinity == 0]:
            robot.delete(g)
        for m in list(robot.meshes):
            robot.delete(m)
    if not sensors:
        for s in list(robot.sensors):
            robot.delete(s)
    if foot == "box":  # replace the capsule soles and their pairs by one box per foot
        for p in list(robot.pairs):
            robot.delete(p)
        for g in [g for g in robot.geoms if "foot" in g.name and g.name.endswith("_collision")]:
            robot.delete(g)
        for side in ("left", "right"):
            body = robot.body(f"{side}_ankle_roll_link")
            body.add_geom(name=f"{side}_foot_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=BOX_FOOT["size"], pos=BOX_FOOT["pos"])
            if pairs:
                robot.add_pair(name=f"{side}_foot_box_floor", geomname1=f"{side}_foot_box", geomname2="floor", **PAIR)
    elif foot != "capsules":
        raise ValueError(f"foot must be 'capsules' or 'box', not {foot!r}")
    if not pairs:  # sole contacts with every terrain geom, with the explicit pairs' parameters
        soles = {p.geomname1 for p in robot.pairs} | {f"{s}_foot_box" for s in ("left", "right") if foot == "box"}
        for p in list(robot.pairs):
            robot.delete(p)
        for g in robot.geoms:
            if g.name in soles:
                g.priority, g.condim = 1, PAIR["condim"]  # priority makes MuJoCo use this geom's parameters in its contacts
                g.friction = [PAIR["friction"][0], PAIR["friction"][2], PAIR["friction"][3]]  # (sliding, torsional, rolling)
                g.solref, g.solimp = PAIR["solref"], PAIR["solimp"]
    # Holosoma's self-collision switch: robot geoms (contype 1) get conaffinity 2, so they hit the terrain (contype 2,
    # conaffinity 1) but not each other
    for g in robot.geoms:
        if g.contype == 1 and g.conaffinity == 1:
            g.conaffinity = 2
    world.attach(robot, frame=world.worldbody.add_frame(), prefix="robot_")
    return world


def sole_geoms(m: mujoco.MjModel, feet=("robot_left_ankle_roll_link", "robot_right_ankle_roll_link")):
    """The colliding geoms of the feet (the robot side of the sole contacts)."""
    ids = [m.body(f).id for f in feet]
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and m.geom_contype[g] > 0]
