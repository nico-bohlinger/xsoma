"""Checks of sim.py against MuJoCo itself, then a smoke run.

  fk        the task's frames (foot contact points, ankle-roll links, termination colliders) vs mj_kinematics on random poses
  height    terrain.height vs mj_ray straight down onto the heightfield at random points of the robots' tiles
  contact   collider forces decoded from the contact pool vs MuJoCo Warp's cfrc_ext (sensors kept, so the post-constraint
            pass runs), on robots dropped in random poses
  run       init and `steps` steps with random actions: finite state, termination / time-out counts, env steps per second

python tools/check_sim.py [n=256] [terrain=mix] [foot=capsules] [steps=200] [sigma=0.5]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402

from xsoma import scene, task  # noqa: E402
from xsoma.rot import quat_to_mat  # noqa: E402
from xsoma.sim import Sim, SimCfg  # noqa: E402

N, steps, sigma = int(kw.get("n", 256)), int(kw.get("steps", 200)), float(kw.get("sigma", 0.5))
sim = Sim(SimCfg(num_envs=N, terrain=kw.get("terrain", "mix"), foot=kw.get("foot", "capsules")))
m = sim.m
rng = np.random.RandomState(0)


def random_qpos(n, z=(0.3, 1.0)):
    qpos = np.zeros((n, m.nq), np.float32)
    org = sim.terrain.origins.reshape(-1, 3)[rng.randint(0, sim.terrain.origins[..., 0].size, n)] if sim.terrain else np.zeros((n, 3))
    qpos[:, :2] = org[:, :2] + rng.uniform(-2, 2, (n, 2))
    qpos[:, 2] = org[:, 2] + rng.uniform(*z, n)
    quat = rng.normal(size=(n, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    qpos[:, 7:] = task.DEFAULT_Q + rng.uniform(-1, 1, (n, task.J))
    return qpos


# ------------------------------------------------------------------ fk
qpos = random_qpos(64)
rb = jax.jit(lambda q: sim.view(q, jnp.zeros((q.shape[0], m.nv)), jnp.asarray(sim.heights)).robot)(qpos)
d = mujoco.MjData(m)
err = {"foot_p": 0.0, "feet_p": 0.0, "feet_R": 0.0, "colliders": 0.0}
fp = [m.body(f"robot_{b}").id for b in task.FOOT_POINTS]
fe = [m.body(f"robot_{b}").id for b in task.FEET]
P, Rs = (np.asarray(x) for x in jax.jit(lambda q: sim.kin.forward(q[:, :3], quat_to_mat(q[:, 3:7]), q[:, 7:]))(qpos))
pts = P[:, sim.coll_frame] + np.einsum("nkij,kj->nki", Rs[:, sim.coll_frame], sim.coll_local)  # collider sphere centres
for i in range(len(qpos)):
    d.qpos[:] = qpos[i]
    mujoco.mj_kinematics(m, d)
    err["foot_p"] = max(err["foot_p"], np.abs(np.asarray(rb.foot_p[i]) - d.xpos[fp]).max())
    err["feet_p"] = max(err["feet_p"], np.abs(np.asarray(rb.feet_p[i]) - d.xpos[fe]).max())
    err["feet_R"] = max(err["feet_R"], np.abs(np.asarray(rb.feet_R[i]) - d.xmat[fe].reshape(-1, 3, 3)).max())
    ref = []  # the same centres from MuJoCo's geom frames: a sphere's centre, a capsule's two end centres (+axis first)
    for g in sim.collider_geoms:
        ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
        ref += [d.geom_xpos[g]] if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_SPHERE else [d.geom_xpos[g] + s * m.geom_size[g, 1] * ax for s in (1, -1)]
    err["colliders"] = max(err["colliders"], np.abs(pts[i] - np.stack(ref)).max())
print("fk max abs error:", {k: f"{v:.2e}" for k, v in err.items()}, flush=True)

# ------------------------------------------------------------------ height
if sim.terrain is not None:
    org = sim.terrain.origins.reshape(-1, 3)
    pts = org[rng.randint(0, len(org), 4000), :2] + rng.uniform(-4, 4, (4000, 2))
    ours = np.asarray(jax.jit(lambda x, y: sim.height(jnp.asarray(sim.heights), x, y))(pts[:, 0], pts[:, 1]))
    ray, gid = [], np.zeros(1, np.int32)
    for x, y in pts:
        dist = mujoco.mj_ray(m, d, np.asarray([x, y, 2.0]), np.asarray([0.0, 0.0, -1.0]), None, 1, -1, gid)
        ray.append(2.0 - dist if gid[0] >= 0 and m.geom_bodyid[gid[0]] == 0 else np.nan)  # a terrain geom
    ray = np.asarray(ray)
    e = np.abs(ours - ray)
    print(f"height vs mj_ray: max abs error {np.nanmax(e):.2e} m, mean {np.nanmean(e):.2e}, misses {np.isnan(ray).sum()}, "
          f"range {ray.min():.3f}..{ray.max():.3f} m", flush=True)  # fmt: skip

# ------------------------------------------------------------------ contact
ms = scene.build(sim.cfg.foot, sim.terrain, sensors=True)
mxs = mjx.put_model(ms, impl="warp")
dxs = mjx.make_data(ms, impl="warp", naconmax=N * 96, njmax=576)
q0 = random_qpos(N, z=(0.05, 0.5))
data = jax.vmap(lambda q: dxs.replace(qpos=q))(jnp.asarray(q0))
step = jax.jit(jax.vmap(lambda d_: mjx.step(mxs, d_)))
for _ in range(3):
    data = step(data)
F = np.asarray(jax.jit(sim.collider_forces)(data))
bodies = [int(ms.geom_bodyid[g]) for g in sim.collider_geoms]
ref = np.asarray(data._impl.cfrc_ext)[:, bodies, 3:6]
e = np.linalg.norm(F - ref, axis=-1)
on = np.linalg.norm(ref, axis=-1) > 1.0
print(f"collider forces vs cfrc_ext: max abs error {e.max():.3e} N (largest force {np.linalg.norm(ref, axis=-1).max():.0f} N), "
      f"termination agreement {(np.linalg.norm(F, axis=-1) > 1.0).any(-1).__eq__(on.any(-1)).mean():.4f}, "
      f"terminated {on.any(-1).mean():.2f}", flush=True)  # fmt: skip

# ------------------------------------------------------------------ run
t0 = time.time()
init = jax.jit(sim.init)
s = init(jax.random.PRNGKey(0))
jax.block_until_ready(s)
t_init = time.time() - t0


def rollout(s, key):
    def body(s, k):
        a = sigma * jax.random.normal(k, (N, task.J))
        s, tr = sim.step(s, a)
        return s, (tr.stats, jnp.isfinite(s.data.qpos).all(), jnp.isfinite(s.obs).all())

    return jax.lax.scan(body, s, jax.random.split(key, steps))


t0 = time.time()
prog = jax.jit(rollout, donate_argnums=(0,)).lower(s, jax.random.PRNGKey(1)).compile()
t_compile = time.time() - t0
s, (st, fin_q, fin_o) = prog(s, jax.random.PRNGKey(1))
jax.block_until_ready(s)
t0 = time.time()
s, (st, fin_q, fin_o) = prog(s, jax.random.PRNGKey(2))
jax.block_until_ready(s)
dt = time.time() - t0
st = jax.device_get(st)
done = st["done"].sum()
print(f"run: init {t_init:.0f} s, compile {t_compile:.0f} s, {N * steps / dt / 1e3:.1f}k env steps/s ({1e3 * dt / steps:.2f} ms per step), "
      f"finite qpos {bool(np.asarray(fin_q).all())} obs {bool(np.asarray(fin_o).all())}", flush=True)  # fmt: skip
print(f"     per step: reward {st['reward'].mean():.4f}, finished {done / steps / N:.4f} (contact {st['terminated'].sum() / max(done, 1):.2f}), "
      f"mean length {st['len_sum'].sum() / max(done, 1):.1f}, pushes {st['pushes'].sum() / steps / N:.4f}, "
      f"geometric termination agrees {st['term_geom'].mean():.4f}, contact pool {st['nacon'].max():.3f}, "
      f"scale {float(st['scale'][-1]):.4f}, avg len {float(st['avg_len'][-1]):.1f}", flush=True)  # fmt: skip
rew = st["rew_sums"].sum(0) / max(done, 1)
print("     Episode/rew_*:", {k: round(float(v), 4) for k, v in zip(task.TERMS, rew)}, flush=True)
