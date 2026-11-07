"""Throughput of the simulator: env steps per second of sim.Sim.step, Holosoma's whole task step (four 5-ms physics substeps
with the PD law, contact-force terminations, rewards, resets, curriculum, commands, pushes, observations), with random actions.

    python tools/bench_sim.py [n=4096] [steps=100] [reps=5] [scale=0.5] [sigma=0.5]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from xsoma import task  # noqa: E402
from xsoma.sim import Sim, SimCfg  # noqa: E402


def main():
    n, steps, reps, sigma = int(kw.get("n", 4096)), int(kw.get("steps", 100)), int(kw.get("reps", 5)), float(kw.get("sigma", 0.5))
    sim = Sim(SimCfg(num_envs=n, terrain_scale=float(kw.get("scale", 0.5)), preset="fast_sac"))

    def run(s, key):
        def body(s, k):
            s, tr = sim.step(s, sigma * jax.random.normal(k, (n, task.J)))
            return s, (tr.terminated | tr.truncated).mean()

        return jax.lax.scan(body, s, jax.random.split(key, steps))

    s = sim.init(jax.random.PRNGKey(0))
    t = time.time()
    prog = jax.jit(run, donate_argnums=(0,)).lower(s, jax.random.PRNGKey(1)).compile()
    compile_s = time.time() - t
    s, _ = prog(s, jax.random.PRNGKey(2))
    jax.block_until_ready(s)
    t = time.time()
    for r in range(reps):
        s, done = prog(s, jax.random.PRNGKey(3 + r))
    jax.block_until_ready(s)
    dt = (time.time() - t) / reps
    print(f"{n} envs: {n * steps / dt / 1e3:.0f}k env steps/s ({1e3 * dt / steps:.2f} ms per step of all envs), episodes ending per step "
          f"{float(jnp.mean(done)):.3f}, compile {compile_s:.0f} s, {jax.devices()[0].device_kind}")


if __name__ == "__main__":
    main()
