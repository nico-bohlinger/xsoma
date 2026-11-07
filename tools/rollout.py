"""Record a deterministic rollout of a policy in the simulator for rendering (tools/render_gif.py) or inspection.

python tools/rollout.py ckpt=<path.npz> [out=rollout.npz] [seconds=20 (at most one 20-s episode)] [n=16] [scale=0.5] [tile=2,4]
                       [plan=default] [seed=0]

All n envs start on one tile and follow a scripted command plan (instead of random commands), with training's randomization and
pushes. The recording holds, at 50 Hz, every env's qpos and command, the terrain heights and the tile layout, and per env whether
and when it fell. Commands of the plan "default" (vx, vy, wz), each for 4 s: walk forward, turn left while walking, walk
sideways, walk backward, stand.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xsoma import evaluate as ev  # noqa: E402
from xsoma import task  # noqa: E402

PLANS = {"default": [(0.8, 0.0, 0.0), (0.6, 0.0, 0.8), (0.0, 0.6, 0.0), (-0.6, 0.0, 0.0), (0.0, 0.0, 0.0)]}


def main():
    n, seconds = int(kw.get("n", 16)), float(kw.get("seconds", 20))
    sim = ev.make_sim(n, float(kw.get("scale", 0.5)), True, "fast_sac")
    kind, normalize, params = ev.load(kw["ckpt"])
    act = jax.tree_util.Partial(ev.act_fn(kind, normalize), params)
    plan = np.asarray(PLANS[kw.get("plan", "default")], np.float32)
    steps = min(int(seconds / task.DT), task.MAX_EPISODE)  # one episode: no time-out inside the recording
    per = max(1, steps // len(plan))
    cmds = jnp.asarray(plan[np.minimum(np.arange(steps) // per, len(plan) - 1)])
    s = sim.init(jax.random.PRNGKey(int(kw.get("seed", 0))))
    if sim.terrain is not None:  # every env on one tile
        i, j = (int(x) for x in kw.get("tile", "2,4").split(","))
        org = jnp.broadcast_to(jnp.asarray(sim.terrain.origins[i, j], jnp.float32), (n, 3))
        s = s._replace(world=s.world._replace(origin=org), t=jnp.full((n,), task.MAX_EPISODE, jnp.int32))
        s, _ = sim.step(s, jnp.zeros((n, task.J)))  # every env times out and respawns there

    def body(s, cmd):
        c = jnp.broadcast_to(cmd, (n, 3))
        phase = task.gait_phase(s.t, s.freq, s.offset, c)
        s = s._replace(cmd=c, phase=phase)
        view = sim.view(s.data.qpos, s.data.qvel, s.world.heights)
        obs, _ = task.observations(view.robot, s.a_last, c, phase, jax.random.fold_in(s.key, 7))
        s1, tr = sim.step(s, act(obs))
        return s1, (tr.qpos1, tr.terminated)

    s, (qpos, fell) = jax.jit(lambda s: jax.lax.scan(body, s, cmds))(s)
    qpos, fell = np.asarray(qpos), np.asarray(fell)
    first = np.where(fell.any(0), fell.argmax(0), -1)
    out = kw.get("out", "rollout.npz")
    np.savez(out, qpos=qpos, cmd=np.asarray(cmds), fell_at=first, heights=sim.heights, x0=sim.terrain.x0 if sim.terrain else 0.0,
             hs=sim.terrain.cfg.horizontal_scale if sim.terrain else 0.1, scale=sim.cfg.terrain_scale, seed=sim.cfg.seed, dt=task.DT)  # fmt: skip
    print(f"{steps} steps x {n} envs -> {out}; fell: {(first >= 0).sum()} of {n} (at steps {sorted(first[first >= 0].tolist())[:8]})")


if __name__ == "__main__":
    main()
