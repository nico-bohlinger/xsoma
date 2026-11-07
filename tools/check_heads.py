"""Check that imagination evaluates the task exactly as the simulator does, on real transitions.

Runs n envs for `steps` steps (random actions, or a checkpoint's policy plus noise) and recomputes, from each recorded pre-step
state and the true post-physics state, what train.XSoma.imagine computes from world-model states:
  reward       task.reward of the post-physics state (forward kinematics, exact terrain lookup, the phase and command the policy
               saw, the current penalty scale) vs the simulator's reward (expected exact up to float rounding)
  phase        task.gait_phase of the next step (episode counter + 1, the same frequency, offsets and command) vs the simulator's
               next phase, on steps without a reset or a command resample
  observation  the next critic observation from the next state (after pushes) vs the simulator's, on steps without a reset
  termination  the geometric termination imagination uses (a termination collider below the terrain) vs the simulator's
               contact-force termination
python tools/check_heads.py [n=512] [steps=300] [ckpt=<path.npz>] [noise=0.3] [scale=1.0]
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
from xsoma.sim import Sim, SimCfg, unflatten  # noqa: E402


def main():
    n, steps, noise = int(kw.get("n", 512)), int(kw.get("steps", 300)), float(kw.get("noise", 0.3))
    sim = Sim(SimCfg(num_envs=n, terrain_scale=float(kw.get("scale", 1.0)), preset="fast_sac", random_start=True))
    if "ckpt" in kw:
        kind, normalize, params = ev.load(kw["ckpt"])
        policy = jax.tree_util.Partial(ev.act_fn(kind, normalize), params)
    else:
        policy = jax.tree_util.Partial(lambda o: jnp.zeros((o.shape[0], task.J)))

    def body(s, k):
        a = policy(s.obs) + noise * jax.random.normal(k, (n, task.J))
        s1, tr = sim.step(s, a)
        heights = s.world.heights
        v1 = sim.view_state(unflatten(tr.qpos1, tr.qvel1), heights)
        r = task.reward(task.reward_terms(v1.robot, v1.foot_h, a, s.a_last, s.cmd, s.phase), sim.reward_cfg, s.scale)
        phase1 = task.gait_phase(s.t + 1, s.freq, s.offset, s.cmd)
        vn = sim.view(s1.data.qpos, s1.data.qvel, heights)  # the next state, after resets and pushes
        _, crit = task.observations(vn.robot, a, s1.cmd, s1.phase)
        done = tr.terminated | tr.truncated
        same = ~done & ~tr.diverged
        resample = (s.t + 1) % task.RESAMPLE == 0
        out = {"r_err": jnp.where(same, jnp.abs(r - tr.reward), 0.0).max(), "r_scale": jnp.abs(tr.reward).max(),
               "phase_err": jnp.where((same & ~resample)[:, None], jnp.abs(jnp.angle(jnp.exp(1j * (phase1 - s1.phase)))), 0.0).max(),
               "obs_err": jnp.where(same[:, None], jnp.abs(crit - s1.critic), 0.0).max(),
               "term_agree": ((tr.clearance.min(-1) < 0) == tr.terminated).mean(), "term": tr.terminated.mean(),
               "geo_term": (tr.clearance.min(-1) < 0).mean(), "n_same": same.sum()}  # fmt: skip
        return s1, out

    s = jax.jit(sim.init)(jax.random.PRNGKey(0))
    _, out = jax.jit(lambda s: jax.lax.scan(body, s, jax.random.split(jax.random.PRNGKey(1), steps)))(s)
    out = jax.device_get(out)
    print(f"{n} envs x {steps} steps ({'policy ' + kw['ckpt'] if 'ckpt' in kw else 'zero policy'} + noise {noise}), terrain x{sim.cfg.terrain_scale}")
    print(f"reward:      max abs error {out['r_err'].max():.2e} (rewards up to {out['r_scale'].max():.3f})")
    print(f"phase:       max abs error {out['phase_err'].max():.2e} rad")
    print(f"observation: max abs error {out['obs_err'].max():.2e} (critic, no noise)")
    print(f"termination: geometric vs contact agree on {out['term_agree'].mean():.4f} of env steps "
          f"(contact {out['term'].mean():.4f}, geometric {out['geo_term'].mean():.4f} per step)")  # fmt: skip


if __name__ == "__main__":
    main()
