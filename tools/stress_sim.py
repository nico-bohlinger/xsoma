"""Physics failures of the simulator under hard actions: diverged states (non-finite or |qvel| > sim.DIVERGED) per configuration.

For each (terrain, action) setting: `steps` steps of n envs with actions a ~ sigma N(0, 1) (gauss) or sigma sign(N(0, 1))
redrawn every `hold` steps (bang); reports the share of env steps whose state is non-finite, the largest finite |qvel| (root
linear, root angular, joints) and its 99.99 % quantile, and where the first non-finite state appeared (terrain tile type).

python tools/stress_sim.py [n=512] [steps=1000] [grid=field:one,tiles;tiles:corner,center;sigma:0.5,5]
  grid keys: terrain (mix, plane), scale (1.0, 0.5), field (one: Holosoma's single heightfield, tiles: XSoma's one per tile),
  tiles (all, corner, center: where the robots stand), act (gauss, bang), sigma, ccd (collision tolerance)
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xsoma import task  # noqa: E402
from xsoma.sim import Sim, SimCfg  # noqa: E402


def run(n, steps, terrain, scale, act, sigma, hold=4, tiles="all", ccd=1e-6, field="tiles"):
    sim = Sim(SimCfg(num_envs=n, terrain=terrain, terrain_scale=scale, ccd_tolerance=ccd, tiles=field == "tiles"))
    s = jax.jit(sim.init)(jax.random.PRNGKey(0))
    if tiles != "all" and sim.terrain is not None:  # every env on the heightfield's central tiles, or on its corner tiles
        R, C = sim.terrain.cfg.num_rows, sim.terrain.cfg.num_cols
        pick = [(R // 2 - 1, C // 2 - 1), (R // 2, C // 2)] if tiles == "center" else [(0, 0), (R - 1, C - 1), (0, C - 1), (R - 1, 0)]
        org = np.stack([sim.terrain.origins[i, j] for i, j in pick])[np.arange(n) % len(pick)].astype(np.float32)
        s = s._replace(world=s.world._replace(origin=jnp.asarray(org)))
        s = s._replace(t=jnp.full((n,), task.MAX_EPISODE, jnp.int32))  # every env times out at the next step and respawns there
        s, _ = jax.jit(sim.step)(s, jnp.zeros((n, task.J)))

    def body(carry, k):
        s, a = carry
        k1, k2 = jax.random.split(k)
        z = jax.random.normal(k1, (n, task.J))
        new = sigma * (jnp.sign(z) if act == "bang" else z)
        a = jnp.where(jax.random.uniform(k2, (n, 1)) < 1.0 / hold, new, a)
        s, tr = sim.step(s, a)
        bad = tr.diverged  # the simulator resets these envs
        av = jnp.where(bad[:, None], 0.0, jnp.abs(tr.qvel1))
        mx = jnp.stack((av[:, :3].max(-1), av[:, 3:6].max(-1), av[:, 6:].max(-1)), -1)
        return (s, a), (bad, mx, tr.qpos[:, :2])

    prog = jax.jit(lambda s, k: jax.lax.scan(body, (s, jnp.zeros((n, task.J))), jax.random.split(k, steps))[1])
    bad, mx, xy = (np.asarray(x) for x in prog(s, jax.random.PRNGKey(1)))
    first = np.argwhere(bad)
    where = ""
    if len(first) and sim.terrain is not None:
        t, e = first[0]
        x, y = xy[t, e]
        i = int((x - sim.terrain.x0 - sim.terrain.cfg.border) // sim.terrain.cfg.tile)
        j = int((y - sim.terrain.x0 - sim.terrain.cfg.border) // sim.terrain.cfg.tile)
        ok = 0 <= i < sim.terrain.cfg.num_rows and 0 <= j < sim.terrain.cfg.num_cols
        where = f", first at step {t} env {e} xy ({x:.1f}, {y:.1f}) tile {(i, j)} {sim.terrain.types[i, j] if ok else 'border'}"
    q = np.quantile(mx.reshape(-1, 3), 0.9999, axis=0)
    print(f"{terrain:5s} x{scale:<4} field {field:5s} {act:5s} sigma {sigma:<4} robots {tiles:6s} ccd {ccd:.0e}: diverged {bad.mean():.5f} of env steps ({bad.any(0).sum()} envs), "
          f"max |qvel| root lin {mx[..., 0].max():.1f} ang {mx[..., 1].max():.1f} joints {mx[..., 2].max():.1f}, "
          f"99.99 % {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}{where}", flush=True)  # fmt: skip


if __name__ == "__main__":
    n, steps = int(kw.get("n", 512)), int(kw.get("steps", 1000))
    grid = [(k, v.split(",")) for k, v in (g.split(":", 1) for g in kw.get("grid", "field:one,tiles;tiles:corner,center;sigma:0.5,5").split(";"))]
    for combo in itertools.product(*[vs for _, vs in grid]):
        c = dict(zip([k for k, _ in grid], combo))
        if c.get("terrain") == "plane" and c.get("scale") not in (None, "1.0"):
            continue
        run(n, steps, c.get("terrain", "mix"), float(c.get("scale", 1.0)), c.get("act", "gauss"), float(c.get("sigma", 0.5)), tiles=c.get("tiles", "all"),
            ccd=float(c.get("ccd", 1e-6)), field=c.get("field", "tiles"))
