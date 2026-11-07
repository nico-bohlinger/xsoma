"""Deterministic evaluation of policies in the simulator: every env runs one full episode (up to 1001 steps, 20 s) with the
policy's mean action, training's randomization (masses, friction, PD gains, delay, pushes) and terrain, all envs starting at step 0.

Per policy: Holosoma's logged metrics (Episode/rew_<term> = the episode's sum of w dt r / 20 s, averaged over episodes; penalties
at full weight, penalty scale 1), the mean episode length, the share of episodes that reach the time-out without a fall, and the
mean velocity-tracking errors while alive. Policies: XSoma checkpoints (train.save_ckpt) and Holosoma's PPO / FastSAC actors
(converted by tools/holosoma_ckpt.py), evaluated alike.

    python -m xsoma.evaluate <run dir or checkpoint> [...] [--n 2048] [--scale 0.5] [--sole_friction 0] [--preset fast_sac]

A run directory is evaluated checkpoint by checkpoint (XSoma: ckpt/*.npz; Holosoma: hs/**/model_*.npz), each with its wall-clock
time since the run's launch, and the records are appended to <run>/eval.jsonl. The defaults are the setting Holosoma's MJWarp
baselines train in: terrain scale 0.5, sole friction fixed at 0.8.
"""

import argparse
import glob
import json
import os

if __name__ == "__main__":  # standalone: JAX allocates on demand, so MuJoCo Warp's own buffers (more contacts when robots fall) fit
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from . import nets, task  # noqa: E402
from .sim import Sim, SimCfg  # noqa: E402


def evaluate(sim: Sim, act, key, steps: int = task.MAX_EPISODE + 2):
    """act(obs) -> actions; per-env first-episode term sums, lengths, falls, ended flags and mean tracking errors."""
    N = sim.N
    s = sim.init(key)
    s = s._replace(scale=jnp.ones_like(s.scale))  # penalties at full weight

    def body(carry, _):
        s, sums, ended, length, fell, err_v, err_w, alive_steps = carry
        s1, tr = sim.step(s, act(s.obs))
        live = ~ended
        sums = sums + jnp.where(live[:, None], tr.parts, 0.0)
        R = sim.view(tr.qpos1, tr.qvel1, s.world.heights).robot
        v_b = jnp.einsum("nji,nj->ni", R.R, R.v)
        err_v = err_v + jnp.where(live, jnp.linalg.norm(s.cmd[:, :2] - v_b[:, :2], axis=-1), 0.0)
        err_w = err_w + jnp.where(live, jnp.abs(s.cmd[:, 2] - R.w[:, 2]), 0.0)
        alive_steps = alive_steps + live
        end = (tr.terminated | tr.truncated) & live
        length = jnp.where(end, s.t + 1, length)
        fell = fell | (tr.terminated & live)
        return (s1._replace(scale=jnp.ones_like(s1.scale)), sums, ended | end, length, fell, err_v, err_w, alive_steps), None

    z = jnp.zeros((N,))
    init = (s, jnp.zeros((N, len(task.TERMS))), jnp.zeros((N,), bool), z, jnp.zeros((N,), bool), z, z, z)
    out, _ = jax.lax.scan(body, init, None, length=steps)
    _, sums, ended, length, fell, err_v, err_w, alive_steps = out
    return sums, length, fell, ended, err_v / jnp.maximum(alive_steps, 1), err_w / jnp.maximum(alive_steps, 1)


def load(path, which="avg"):
    """(kind, normalize, parameters) of an XSoma checkpoint or a converted Holosoma PPO / FastSAC actor."""
    z = np.load(path)
    if "kind" not in z.files:
        return "xsoma", False, load_policy(path, which)
    params = {k: jnp.asarray(z[k]) for k in z.files if k not in ("kind", "iteration", "normalize", "wall_s")}
    return str(z["kind"]), bool(z["normalize"]) if "normalize" in z.files else False, params


def act_fn(kind, normalize=False):
    """(parameters, obs) -> deterministic action; the parameters are an argument, so one compilation serves a policy type."""
    if kind == "xsoma":
        return nets.pi_apply

    def dot(p, h, i):
        return jnp.dot(h, p[f"w{i}"], precision=jax.lax.Precision.HIGHEST) + p[f"b{i}"]

    if kind == "ppo":  # Holosoma PPO's actor mean

        def act(p, o):
            h = o
            for i in range(3):
                h = jax.nn.elu(dot(p, h, i))
            return dot(p, h, 3)

        return act

    def act(p, o):  # Holosoma FastSAC: observation normalizer, LayerNorm-SiLU MLP, tanh scaled per joint
        h = (o - p["mean"]) / (p["std"] + 0.01) if normalize else o
        for i in range(3):
            h = dot(p, h, i)
            mu, var = h.mean(-1, keepdims=True), h.var(-1, keepdims=True)
            h = jax.nn.silu((h - mu) * jax.lax.rsqrt(var + 1e-5) * p[f"g{i}"] + p[f"h{i}"])
        return jnp.tanh(dot(p, h, 3)) * p["scale"] + p["bias"]

    return act


def checkpoints(run):
    """(path, wall-clock seconds since launch, env steps or iteration) of a run directory's checkpoints."""
    out = []
    for p in sorted(glob.glob(os.path.join(run, "ckpt", "*.npz"))):  # XSoma
        meta = json.load(open(p[: -len(".npz")] + ".json"))
        out.append((p, meta.get("wall_s", meta["train_s"]), meta["env_steps"]))
    if not out:  # Holosoma: model_*.pt converted to .npz (tools/holosoma_ckpt.py stores the time since launch)
        for p in sorted(glob.glob(os.path.join(run, "hs", "**", "model_*.npz"), recursive=True)):
            z = np.load(p)
            if "wall_s" in z.files:
                wall = float(z["wall_s"])
            else:
                wall = os.path.getmtime(p[: -len(".npz")] + ".pt") - float(open(os.path.join(run, "t_start")).read().split()[0])
            out.append((p, wall, int(z["iteration"])))
    return out


def load_policy(path, which="avg"):
    z = np.load(path)
    n = 1 + max(int(k.split("/")[1]) for k in z.files if k.startswith(which + "/"))
    return [{k.split("/")[2]: jnp.asarray(z[k]) for k in z.files if k.startswith(f"{which}/{i}/")} for i in range(n)]


def make_sim(n=2048, scale=0.5, sole_friction=False, preset="fast_sac", terrain="mix"):
    """The evaluation simulator; the defaults are the setting Holosoma's MJWarp baselines train in (terrain scale 0.5, sole
    friction fixed at 0.8)."""
    return Sim(SimCfg(num_envs=n, terrain=terrain, terrain_scale=scale, preset=preset, sole_friction=sole_friction, random_start=False))


def run_all(sim: Sim, jobs, seed=1, which="avg", out=None):
    """Evaluate (path, wall_s, steps, run) jobs; one compilation per policy type. Yields the result records; a job with a run
    directory appends its record to <run>/<out>.jsonl."""
    progs, key = {}, jax.random.PRNGKey(seed)
    for path, wall, steps, run in jobs:
        kind, normalize, params = load(path, which)
        if (kind, normalize) not in progs:
            act = act_fn(kind, normalize)
            progs[(kind, normalize)] = jax.jit(lambda p, k, act=act: evaluate(sim, jax.tree_util.Partial(act, p), k))
        sums, length, fell, ended, ev, ew = jax.device_get(progs[(kind, normalize)](params, key))
        rew = sums.mean(0) / (task.MAX_EPISODE * task.DT)
        rec = {"ckpt": path, "kind": kind, "wall_s": wall, "steps": steps, "n": sim.N, "scale": sim.cfg.terrain_scale, "sole_friction": sim.cfg.sole_friction,
               "len": float(length.mean()), "success": float(1 - fell.mean()), "ended": float(ended.mean()), "err_v": float(ev.mean()),
               "err_w": float(ew.mean()), **{f"rew_{k}": float(v) for k, v in zip(task.TERMS, rew)}}  # fmt: skip
        if run and out:
            with open(os.path.join(run, out + ".jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        yield rec


def main(argv=None):
    p = argparse.ArgumentParser(description="Deterministic evaluation of XSoma runs, Holosoma runs or single checkpoints.")
    p.add_argument("paths", nargs="+", help="run directories or .npz checkpoints")
    p.add_argument("--n", type=int, default=2048, help="envs, one episode each")
    p.add_argument("--scale", type=float, default=0.5, help="terrain scale")
    p.add_argument("--sole_friction", type=int, default=0, help="1: the soles' friction is randomized as well")
    p.add_argument("--preset", default="fast_sac", help="reward preset (the logged metrics; both presets weigh tracking alike)")
    p.add_argument("--terrain", default="mix")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--which", default="avg", help="XSoma checkpoints: avg (the averaged policy) or live")
    p.add_argument("--out", default="eval", help="records go to <run>/<out>.jsonl")
    a = p.parse_args(argv)
    sim = make_sim(a.n, a.scale, bool(a.sole_friction), a.preset, a.terrain)
    jobs = []
    for path in a.paths:
        if os.path.isdir(path):
            jobs += [(c, t, steps, path) for c, t, steps in checkpoints(path)]
        else:
            jobs.append((path, None, None, None))
    for rec in run_all(sim, jobs, a.seed, a.which, a.out):
        t = f"{rec['wall_s'] / 60:.1f} min" if rec["wall_s"] is not None else ""
        print(f"[eval] {rec['ckpt']} {t}: length {rec['len']:.0f} no fall {rec['success']:.3f} velocity error {rec['err_v']:.3f} m/s "
              f"yaw-rate error {rec['err_w']:.3f} rad/s | Holosoma's tracking terms lin {rec['rew_tracking_lin_vel']:.3f} "
              f"ang {rec['rew_tracking_ang_vel']:.3f}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
