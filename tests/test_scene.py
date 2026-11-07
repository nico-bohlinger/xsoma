"""Scene equivalences on MuJoCo's CPU engine (run: python tests/test_scene.py, or pytest).

  pairs   Holosoma's explicit sole-floor pairs vs the priority contacts XSoma uses: the same contacts (geoms, distance,
          position, frame, friction, solref, solimp) and the same constraint solution (qacc), on a plane and on the heightfield
  tiles   one heightfield per tile (and border boxes) vs one heightfield for the whole terrain: the same surface under mj_ray;
          inside a tile the same contacts and constraint solution; on a tile boundary the contacts of two fields (or of a border
          box) for those of one, the same surface discretized differently
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from xsoma import scene, task  # noqa: E402
from xsoma import terrain as terrain_mod  # noqa: E402

TERRAIN = terrain_mod.generate()


def states(m, n, rng, xy=None, tile_edges=False):
    """Robots in the default pose (q0 x U[0.9, 1.1]) with the soles 0-1 cm into the ground, random yaw, over random points of
    the tiles (or on tile boundaries and the border's edge)."""
    out = []
    d = mujoco.MjData(m)
    for k in range(n):
        if xy is not None:
            x, y = xy[k]
        elif tile_edges:
            t = TERRAIN.cfg.tile
            x = t * rng.randint(0, TERRAIN.cfg.num_rows + 1) + rng.uniform(-0.15, 0.15)
            y = t * rng.uniform(0, TERRAIN.cfg.num_cols)
        else:
            x, y = rng.uniform(0, TERRAIN.cfg.tile * TERRAIN.cfg.num_rows), rng.uniform(0, TERRAIN.cfg.tile * TERRAIN.cfg.num_cols)
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[:] = 0
        d.qpos[:3] = [x, y, 1.0]
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[7:] = task.DEFAULT_Q * rng.uniform(0.9, 1.1, task.J)
        mujoco.mj_kinematics(m, d)
        feet = [m.body(f"robot_{b}").id for b in task.FOOT_POINTS]
        clear = min(z - (_height(fx, fy) if m.nhfield else 0.0) for fx, fy, z in d.xpos[feet])
        d.qpos[2] -= clear + 0.0035 + rng.uniform(0, 0.01)  # the lower foot point 3.5-13.5 mm into the ground (soles 2 mm above it)
        out.append(d.qpos.copy())
    return np.asarray(out)


def _height(x, y):
    import jax.numpy as jnp

    return float(terrain_mod.height(jnp.asarray(TERRAIN.heights_m, jnp.float32), TERRAIN.x0, TERRAIN.cfg.horizontal_scale, x, y))


def contacts(m, qpos):
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    rows = []
    for c in d.contact[: d.ncon]:
        g = sorted((int(c.geom1), int(c.geom2)), key=lambda x: m.geom_bodyid[x] > 0)  # (terrain, robot)
        rows.append((m.geom(g[1]).name, round(float(c.pos[0]), 6), round(float(c.pos[1]), 6), c.dist, c.dim, *c.friction, *c.solref, *c.solimp, *c.frame))
    return sorted(rows), d.qacc.copy()


def test_pairs(n=40):
    rng = np.random.RandomState(0)
    for ter in (None, TERRAIN):
        a = scene.build(terrain=ter, tiles=False, pairs=True)
        b = scene.build(terrain=ter, tiles=False, pairs=False)
        assert a.npair == 14 and b.npair == 0
        worst_qacc, n_con = 0.0, 0
        for q in states(a, n, rng):
            ca, qa = contacts(a, q)
            cb, qb = contacts(b, q)
            assert len(ca) == len(cb) > 0, (len(ca), len(cb))
            assert all(x[0] == y[0] for x, y in zip(ca, cb))
            np.testing.assert_allclose(np.asarray([x[1:] for x in ca], float), np.asarray([y[1:] for y in cb], float), atol=1e-9)
            worst_qacc = max(worst_qacc, np.abs(qa - qb).max() / max(np.abs(qa).max(), 1.0))
            n_con += len(ca)
        assert worst_qacc < 1e-6, worst_qacc
        print(f"pairs == priority contacts ({'heightfield' if ter else 'plane'}): {n_con} contacts identical, qacc rel. error {worst_qacc:.1e}")


def test_tiles(n=40):
    rng = np.random.RandomState(1)
    one = scene.build(terrain=TERRAIN, tiles=False)
    many = scene.build(terrain=TERRAIN, tiles=True)
    print(f"terrain geoms: {sum(one.geom_bodyid == 0)} vs {sum(many.geom_bodyid == 0)}, hfield samples {one.nhfielddata} vs {many.nhfielddata}")
    d1, d2, gid = mujoco.MjData(one), mujoco.MjData(many), np.zeros(1, np.int32)
    pts = np.concatenate((rng.uniform([-1, -1], [81, 161], (3000, 2)), rng.uniform([-0.3, -0.3], [0.3, 0.3], (1000, 2)) + rng.randint(0, 11, (1000, 1)) * [8, 0]
                          + rng.uniform(0, 160, (1000, 1)) * [0, 1], rng.uniform([-40, -40], [120, 200], (1000, 2))))  # fmt: skip
    err = 0.0
    for x, y in pts:
        r1 = mujoco.mj_ray(one, d1, np.asarray([x, y, 1.0]), np.asarray([0.0, 0, -1]), None, 1, -1, gid)
        r2 = mujoco.mj_ray(many, d2, np.asarray([x, y, 1.0]), np.asarray([0.0, 0, -1]), None, 1, -1, gid)
        err = max(err, abs(r1 - r2))
    assert err < 1e-5, err
    print(f"tiles == one heightfield: surface under mj_ray within {err:.1e} m at {len(pts)} points (tiles, their edges, the border)")
    worst = {}
    for edges in (False, True):
        errs = []
        for q in states(one, n, rng, tile_edges=edges):
            ca, qa = contacts(one, q)
            cb, qb = contacts(many, q)
            errs.append(np.abs(qa - qb).max() / max(np.abs(qa).max(), 1.0))
            if not edges:  # inside a tile: the same contacts (up to float rounding of the per-tile elevation scale)
                assert len(ca) == len(cb) and all(x[0] == y[0] for x, y in zip(ca, cb))
        worst[edges] = (np.median(errs), max(errs))
    print(f"tiles == one heightfield: qacc rel. error inside tiles median {worst[False][0]:.1e} max {worst[False][1]:.1e} (same contacts); "
          f"on tile boundaries median {worst[True][0]:.1e} max {worst[True][1]:.1e} (two fields' contacts, or a border box's, for one)")
    assert worst[False][1] < 1e-2, worst


if __name__ == "__main__":
    test_pairs()
    test_tiles()
