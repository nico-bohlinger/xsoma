"""Imagination against the simulator on one GPU: how fast each produces steps of experience for n robots under the policy.

  simulator    the policy acts on the observations (with the exploration noise of collection), then sim.Sim.step runs
               Holosoma's whole task step: four 5-ms physics substeps with the PD law, contact-force terminations, rewards,
               resets, commands, pushes and observations
  imagination  the step of a policy update's rollout (XSoma.imagine) without the gradient: the policy acts on the
               observations, the world model predicts the next state, and Holosoma's reward and termination are computed on
               it with forward kinematics and terrain lookups
  update       a whole policy update for comparison: the imagined rollout with its backward pass (steps of the rollout per
               second)

Each runs `steps` steps of all n robots inside one compiled program, from the same states of the simulator, with the
training defaults (bf16 matrix multiplications in imagination, cuBLAS GEMMs). The policy and the world model are freshly
initialized, as their speed does not depend on what they have learned.

    python tools/bench_imagination.py [n=2048,4096,8192,16384] [steps=48] [reps=10] [scale=0.5]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false").strip()  # as train.py

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from xsoma import nets, task  # noqa: E402
from xsoma.train import Config, XSoma  # noqa: E402

START_KEYS = ("s_qpos", "s_qvel", "xi", "delay", "offset", "a_last", "cmd", "freq", "phase", "t")


def timed(fn, reps):
    """Seconds per call of fn() (which returns arrays), after one warm-up call."""
    jax.block_until_ready(fn())
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / reps


def bench(n, steps, reps, scale, cache):
    cfg = Config(num_envs=n, pi_batch=n, horizon=steps, terrain_scale=scale, buffer_steps=8, jax_cache=cache)
    xs = XSoma(cfg)
    carry = xs.init(jax.random.PRNGKey(0))
    env, pi, wm, wm_b = carry["env"], carry["pi"], carry["wm"], carry["wm_b"]
    # imagination starts from one simulator step of the initial states, recorded as collection records it
    s1, tr = xs.sim.step(env, jnp.zeros((n, task.J)))
    rec, _ = xs.record(env, s1, tr)
    start = {k: rec[k] for k in START_KEYS}

    def sim_run(s, key):
        def body(s, k):
            a = nets.pi_apply(pi, s.obs) + cfg.action_noise * jax.random.normal(k, (n, task.J))
            s, tr = xs.sim.step(s, a)
            return s, tr.reward.mean()

        return jax.lax.scan(body, s, jax.random.split(key, steps))

    def imag_run(key):
        return xs.imagine(pi, wm, wm_b, start, jax.random.split(key, steps), env.scale, env.world.heights)[0]

    def update_run(key):
        return jax.grad(lambda p: xs.imagine(p, wm, wm_b, start, jax.random.split(key, steps), env.scale, env.world.heights)[0])(pi)

    sim_prog = jax.jit(sim_run)
    imag_prog, update_prog = jax.jit(imag_run), jax.jit(update_run)
    key = jax.random.PRNGKey(1)
    rate = lambda t: n * steps / t  # noqa: E731
    t_sim = timed(lambda: sim_prog(s1, key), reps)
    t_imag = timed(lambda: imag_prog(key), reps)
    print(f"{n:6d} robots | simulator {rate(t_sim) / 1e3:7.0f}k steps/s ({1e3 * t_sim / steps:6.3f} ms per step) | imagination "
          f"{rate(t_imag) / 1e3:7.0f}k steps/s ({1e3 * t_imag / steps:6.3f} ms per step): {t_sim / t_imag:5.1f}x the simulator", flush=True)
    try:  # the backward pass stores every imagined step's activations: n = 2048 needs more than a 12-GB GPU
        t_upd = timed(lambda: update_prog(key), reps)
        print(f"{n:6d} robots | update with backward pass {rate(t_upd) / 1e3:7.0f}k steps/s ({1e3 * t_upd:6.1f} ms per update): "
              f"{t_sim / t_upd:4.1f}x the simulator", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{n:6d} robots | update with backward pass: {type(e).__name__}: {str(e).splitlines()[0][:120]}", flush=True)


def main():
    steps, reps, scale = int(kw.get("steps", 48)), int(kw.get("reps", 10)), float(kw.get("scale", 0.5))
    print(f"{jax.devices()[0].device_kind}, {steps} steps per program, terrain scale {scale}", flush=True)
    for n in [int(x) for x in kw.get("n", "2048,4096,8192,16384").split(",")]:
        bench(n, steps, reps, scale, kw.get("jax_cache", ""))


if __name__ == "__main__":
    main()
