"""Record a deterministic rollout of a policy in the simulator for rendering (tools/render_gif.py) or inspection.

python tools/rollout.py ckpt=<path.npz> [out=rollout.npz] [seconds=20 (at most one 20-s episode)] [n=16] [scale=0.5] [tile=2,4]
                       [plan=default] [ramp=0] [pushes=1] [seed=0]

All n envs start on one tile and follow a scripted command plan (instead of random commands), with training's randomization and,
unless pushes=0, its pushes. ramp=<s> blends from one command of the plan to the next over that many seconds. The recording holds,
at 50 Hz, every env's qpos and command, the terrain heights and the tile layout, and per env whether and when it fell.
Plans are lists of (vx, vy, wz, seconds). "default": walk forward, turn left while walking, walk sideways, walk backward and
stand, 4 s each. "tour" (the release video and GIF, with ramp=0.8 pushes=0): start from standing, walk, curve left, walk,
step sideways, curve right, slow down and stand.
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

PLANS = {
    "default": [(0.8, 0.0, 0.0, 4), (0.6, 0.0, 0.8, 4), (0.0, 0.6, 0.0, 4), (-0.6, 0.0, 0.0, 4), (0.0, 0.0, 0.0, 4)],
    "tour": [(0.0, 0.0, 0.0, 1), (0.7, 0.0, 0.0, 4), (0.6, 0.0, 0.6, 3), (0.7, 0.0, 0.0, 2.5), (0.0, -0.5, 0.0, 2.5),
             (0.6, 0.0, -0.6, 3), (0.3, 0.0, 0.0, 1.5), (0.0, 0.0, 0.0, 2.5)],
}


def command_plan(plan, steps, ramp):
    """(steps, 3) commands: each entry of the plan for its seconds, blended linearly from the previous one over `ramp` s."""
    cmd = np.asarray([c[:3] for c in plan], np.float32)
    ends = np.cumsum([c[3] for c in plan]) / task.DT  # the step at which each entry ends
    t = np.arange(steps)
    i = np.minimum(np.searchsorted(ends, t, side="right"), len(plan) - 1)
    start = np.where(i > 0, ends[np.maximum(i - 1, 0)], 0.0)
    w = np.clip((t - start) / max(ramp / task.DT, 1e-9), 0.0, 1.0)[:, None] if ramp > 0 else 1.0
    return jnp.asarray(cmd[np.maximum(i - 1, 0)] + w * (cmd[i] - cmd[np.maximum(i - 1, 0)]))


def main():
    n, seconds = int(kw.get("n", 16)), float(kw.get("seconds", 20))
    sim = ev.make_sim(n, float(kw.get("scale", 0.5)), True, "fast_sac")
    kind, normalize, params = ev.load(kw["ckpt"])
    act = jax.tree_util.Partial(ev.act_fn(kind, normalize), params)
    steps = min(int(seconds / task.DT), task.MAX_EPISODE)  # one episode: no time-out inside the recording
    cmds = command_plan(PLANS[kw.get("plan", "default")], steps, float(kw.get("ramp", 0)))
    pushes = bool(int(kw.get("pushes", 1)))
    s = sim.init(jax.random.PRNGKey(int(kw.get("seed", 0))))
    if sim.terrain is not None:  # every env on one tile
        i, j = (int(x) for x in kw.get("tile", "2,4").split(","))
        org = jnp.broadcast_to(jnp.asarray(sim.terrain.origins[i, j], jnp.float32), (n, 3))
        s = s._replace(world=s.world._replace(origin=org), t=jnp.full((n,), task.MAX_EPISODE, jnp.int32))
        s, _ = sim.step(s, jnp.zeros((n, task.J)))  # every env times out and respawns there

    def body(s, cmd):
        if not pushes:  # a push comes when the counter reaches the env's interval, which a zeroed counter never does
            s = s._replace(push_t=jnp.zeros_like(s.push_t))
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
