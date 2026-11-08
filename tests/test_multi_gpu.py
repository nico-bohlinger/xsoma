"""The multi-GPU layout (xsoma/multi_gpu.py) against the one-GPU policy update, on the CPU with 4 emulated devices and a toy in place
of the simulator, the world model and the imagined return (run: python tests/test_multi_gpu.py, or pytest, which starts it as its
own process so that JAX sees 4 devices). The toy's matrix products are float32, so batched and unbatched passes agree to rounding.

  message      the starts the actor packs for the learners unpack to exactly the starts it drew
  one learner  one learner (2 GPUs) equals train.XSoma.policy_update applied pi_updates times to the same starts
  learners     2 and 3 learners equal a plain reference: in each round one gradient per learner from the same policy, applied in
               the learners' order as steps of the same clip, gradient gate, Adam step and average; the learners' copies are
               bitwise identical
  gate         a round whose world model failed the gate leaves the learners' state as it was
  loop         the whole layout (actor thread, learners, log.jsonl, checkpoints) for a few seconds of training
"""

import dataclasses
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import NamedTuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from xsoma import buffer as rb  # noqa: E402
from xsoma import multi_gpu as mg  # noqa: E402
from xsoma import nets, optim, task, train  # noqa: E402

N, T = 64, 40  # envs and replay-buffer length of the toy


class World(NamedTuple):
    heights: jax.Array


class Env(NamedTuple):
    scale: jax.Array
    world: World
    x: jax.Array


def mlp32(layers, h, act):
    """nets.mlp_apply in float32 at the highest precision."""
    for i, layer in enumerate(layers):
        h = jnp.dot(h, layer["w"], precision=jax.lax.Precision.HIGHEST) + layer["b"]
        if i == len(layers) - 1:
            return h
        if "ln_g" in layer:
            mu = h.mean(-1, keepdims=True)
            h = (h - mu) * jax.lax.rsqrt(((h - mu) ** 2).mean(-1, keepdims=True) + 1e-5) * layer["ln_g"] + layer["ln_b"]
        h = act(h)


class Toy(train.XSoma):
    """train.XSoma with a toy simulator and world model: the real policy, optimizer, replay buffer, start sampling and policy
    update, and an imagined return that reads every start field, the world model, the penalty scale and the terrain."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.wm_in = 7

    def init(self, key):
        ks = jax.random.split(key, 3)
        pi = nets.pi_init(ks[0], task.ACTOR_DIM, task.J)
        wm = {"net": nets.mlp_init(ks[1], [7, 16, 5], layer_norm=True)}
        self.pi_mask = jax.tree.map(lambda _: False, pi)
        rec = {"s_qpos": jnp.zeros((N, 36)), "s_qvel": jnp.zeros((N, 35)), "xi": jnp.zeros((N, 72)), "a_last": jnp.zeros((N, task.J)),
               "cmd": jnp.zeros((N, 3)), "phase": jnp.zeros((N, 2)), "t": jnp.zeros((N,), jnp.int32), "freq": jnp.zeros((N,)),
               "offset": jnp.zeros((N, 2)), "delay": jnp.zeros((N,), jnp.int32), "done": jnp.zeros((N,)), "ok": jnp.ones((N,))}  # fmt: skip
        return {"env": Env(jnp.float32(0.5), World(jnp.ones((3, 3), jnp.float32)), jnp.zeros((N, 4))), "buf": rb.init(T, rec), "pi": pi,
                "avg": jax.tree.map(jnp.copy, pi), "wm": wm, "wm_b": {k: nets.rn_init(n) for k, n in (("in_norm", 7), ("out_norm", 5), ("inc_norm", 5))},
                "opt_pi": optim.init(pi, self.pi_mask), "opt_wm": optim.init(wm, {"net": jax.tree.map(lambda _: False, wm["net"])}),
                "key": ks[2], "it": jnp.zeros((), jnp.int32), "env_steps": jnp.zeros((), jnp.float32), "nonfinite": jnp.zeros((), jnp.int32),
                "wm_ema": jnp.zeros((), jnp.float32), "gn_ema": jnp.zeros((), jnp.float32), "wm_ok": jnp.ones((), bool)}  # fmt: skip

    def collect_iter(self, carry):
        key, k = jax.random.split(carry["key"])
        a = nets.pi_apply(carry["pi"], jax.random.normal(k, (N, task.ACTOR_DIM)))
        rec = {"s_qpos": jnp.concatenate([a, a[:, :7]], -1), "s_qvel": jnp.concatenate([a, a[:, :6]], -1), "xi": jnp.tile(a[:, :24], (1, 3)),
               "a_last": a, "cmd": a[:, :3], "phase": a[:, 3:5], "t": (jnp.abs(a[:, 0]) * 1000).astype(jnp.int32) % 1000, "freq": a[:, 5],
               "offset": a[:, 6:8], "delay": (a[:, 8] > 0).astype(jnp.int32), "done": jnp.zeros((N,)), "ok": jnp.ones((N,))}  # fmt: skip
        stats = {"done": jnp.float32(3), "len_sum": jnp.float32(30), "terminated": jnp.float32(1), "rew_sums": jnp.ones(len(task.TERMS)),
                 "reward": a.mean(), "scale": carry["env"].scale, "avg_len": jnp.float32(10), "term_geom": jnp.float32(0), "nacon": jnp.float32(5),
                 "nonfinite_env": jnp.float32(0), "diverged": jnp.float32(0)}  # fmt: skip
        return {**carry, "key": key, "buf": rb.add(carry["buf"], rec), "env_steps": carry["env_steps"] + N}, stats

    def wm_iter(self, carry):
        key, k = jax.random.split(carry["key"])
        wm = jax.tree.map(lambda w: w + 1e-3 * jax.random.normal(k, w.shape), carry["wm"])
        return {**carry, "key": key, "wm": wm, "wm_ok": jnp.ones((), bool)}, {"wm": jnp.float32(0.1), "wm8": jnp.float32(0.2)}

    def wm_eval(self, carry, key):
        return {k: jnp.float32(0) for k in ("h_err8", "h_errK", "term_env", "term_wm")}

    def imagine(self, pi, wm, wm_b, start, keys, scale, heights):
        assert start["s_qpos"].shape == (self.cfg.pi_batch, 36)
        f = lambda x: x.reshape(x.shape[0], -1).astype(jnp.float32)  # noqa: E731
        x = jnp.concatenate([f(start[k]) for k in mg.START_KEYS], -1)[:, : task.ACTOR_DIM] / 10
        a = mlp32(pi, x, jax.nn.elu)
        y = mlp32(wm["net"], nets.rn_apply(wm_b["in_norm"], a[:, :7]), jax.nn.silu)
        G = -jnp.mean((a - 0.1) ** 2, -1) + scale * jnp.tanh(y).mean(-1) + heights.mean()
        return -G.mean(), {"G": G.mean(), "r": G.mean() / 2, "surv": jnp.float32(1)}


def setup(L, rounds, pi_batch=12, first=True, active=True):
    """A toy learner after T collected steps, the actor's message for `rounds` rounds of L learners, and the learners' program."""
    cfg = dataclasses.replace(train.Config(), num_envs=N, pi_batch=pi_batch, pi_updates=rounds * L, wm_only_iters=0 if first else -1, horizon=4)
    xs = Toy(cfg)
    carry = xs.init(jax.random.PRNGKey(0))
    for _ in range(T):
        carry, _ = xs.collect_iter(carry)
    key = carry["key"]
    carry, starts, rep, _ = jax.jit(mg.actor_message(xs, L, rounds))(carry)
    drawn = jax.vmap(xs.sample_starts, in_axes=(None, 0))(carry["buf"], jax.random.split(jax.random.split(key)[1], rounds * L))
    st = jax.eval_shape(xs.sample_starts, carry["buf"], carry["key"])
    unpack_starts = mg.make_unpack({k: jax.ShapeDtypeStruct(st[k].shape[1:], st[k].dtype) for k in mg.START_KEYS})
    unpack_rep = mg.make_unpack(jax.eval_shape(lambda c: {"wm": c["wm"], "wm_b": c["wm_b"], "scale": c["env"].scale, "active": c["wm_ok"],
                                                         "first": c["wm_ok"]}, carry))  # fmt: skip
    if not active:
        rep = mg.pack({**unpack_rep(rep), "active": jnp.zeros((), bool)})
    devs = jax.devices()[1 : L + 1]
    mesh = Mesh(np.array(devs), ("l",))
    rep_s = NamedSharding(mesh, P())
    sL = jax.make_array_from_single_device_arrays((rounds, L * pi_batch, starts[0].shape[-1]), NamedSharding(mesh, P(None, "l")),
                                                  [jax.device_put(s, d) for s, d in zip(starts, devs)])  # fmt: skip
    rL = jax.make_array_from_single_device_arrays(rep.shape, rep_s, [jax.device_put(rep, d) for d in devs])
    cL = jax.device_put({**{k: carry[k] for k in mg.LEARNER_KEYS}, "key": jax.random.PRNGKey(5)}, rep_s)
    learn = jax.jit(mg.learner_program(xs, mesh, rounds, unpack_starts, unpack_rep))
    return xs, carry, starts, drawn, unpack_starts, unpack_rep(rep), learn(cL, sL, rL, jax.device_put(carry["env"].world.heights, rep_s))


def largest(a, b):
    return max(float(jnp.abs(jax.device_put(x, jax.devices()[0]) - jax.device_put(y, jax.devices()[0])).max()) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def copies(tree):
    """The largest difference between the learners' copies of every leaf."""
    return max(float(np.abs(np.asarray(s.data) - np.asarray(x.addressable_shards[0].data)).max()) for x in jax.tree.leaves(tree) for s in x.addressable_shards)


def check_message(L=3, rounds=2):
    xs, carry, starts, drawn, unpack_starts, m, out = setup(L, rounds)
    back = unpack_starts(jnp.stack(starts, 1).reshape(rounds * L, xs.cfg.pi_batch, -1))  # (round, learner) order, as drawn
    err = max(float(jnp.abs(back[k].astype(jnp.float32) - drawn[k].astype(jnp.float32)).max()) for k in mg.START_KEYS)
    assert err == 0 and all(back[k].dtype == drawn[k].dtype for k in mg.START_KEYS), err
    print(f"message: {rounds} rounds x {L} learners x {xs.cfg.pi_batch} starts unpack exactly to the drawn starts")


def check_one_learner(rounds=2):
    for first in (True, False):
        xs, carry, starts, drawn, unpack_starts, m, (cL, pi_flat, stats) = setup(1, rounds, first=first)
        ref = {**carry, "it": jnp.zeros((), jnp.int32) if first else jnp.ones((), jnp.int32)}  # it == wm_only_iters: the average copies
        for r in range(rounds):
            sub, _ = xs.policy_update(ref, unpack_starts(starts[0][r]), jax.random.PRNGKey(r))
            ref = {**ref, **sub}
        moved = largest(ref["pi"], carry["pi"])
        err = max(largest(cL[k], ref[k]) for k in ("pi", "avg", "opt_pi", "gn_ema"))
        assert moved > 1e-4 and err < 1e-6, (moved, err)
        assert largest(mg.make_unpack(carry["pi"])(jax.device_put(pi_flat, jax.devices()[0])), ref["pi"]) < 1e-6
        print(f"one learner == train.XSoma.policy_update x {rounds} ({'first update' if first else 'later update'}): largest difference "
              f"{err:.1e} (the policy moved {moved:.1e})")  # fmt: skip


def check_learners(rounds=2):
    for L in (2, 3):
        xs, carry, starts, drawn, unpack_starts, m, (cL, _, _) = setup(L, rounds)
        c = xs.cfg
        pi, opt, avg, gn_ema = carry["pi"], carry["opt_pi"], carry["avg"], carry["gn_ema"]
        loss = lambda p, st: xs.imagine(p, m["wm"], m["wm_b"], unpack_starts(st), None, m["scale"], carry["env"].world.heights)[0]  # noqa: E731
        for r in range(rounds):
            grads = [jax.grad(loss)(pi, starts[l][r]) for l in range(L)]  # all from the round's policy
            for g in grads:
                g, gn = optim.clip_by_global_norm(g, c.pi_clip)
                f0 = gn_ema <= 0
                ok = jnp.isfinite(gn) & (f0 | (gn < c.pi_gn_gate * gn_ema))
                gn_ema = jnp.where(ok, jnp.where(f0, gn, 0.98 * gn_ema + 0.02 * gn), 1.01 * gn_ema)
                pi, opt = optim.step(pi, jax.tree.map(lambda x: jnp.where(ok, x, 0.0), g), opt, xs.pi_mask, c.pi_lr, betas=(0.9, 0.95), ok=ok)
                avg = optim.ema(avg, pi, c.pi_ema, m["first"])
        moved = largest(pi, carry["pi"])
        err = max(largest(cL["pi"], pi), largest(cL["avg"], avg))
        same = copies({k: cL[k] for k in ("pi", "avg", "opt_pi")})
        assert moved > 1e-4 and err < 1e-6 and same == 0, (moved, err, same)
        print(f"{L} learners == reference ({rounds} rounds, {rounds * L} steps): largest difference {err:.1e} (the policy moved {moved:.1e}); "
              f"copies identical")  # fmt: skip


def check_gate():
    xs, carry, starts, drawn, unpack_starts, m, (cL, _, stats) = setup(3, 1, active=False)
    err = max(largest(cL[k], carry[k]) for k in ("pi", "avg", "opt_pi", "gn_ema"))
    assert err == 0 and all(float(v) == 0 for v in stats.values()), err
    print("gate: a round with a failed world model leaves the learners' state unchanged")


def check_loop():
    log_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(log_dir, "ckpt"))
    cfg = dataclasses.replace(train.Config(), layout="AL", num_envs=N, buffer_steps=T, pi_batch=12, horizon=4, wm_only_iters=5, warmup_steps=8,
                              log_every=5, minutes=0.05, ckpt_minutes=0.02, eval_n=0, log_dir=log_dir)  # fmt: skip
    real, mg.XSoma = mg.XSoma, Toy
    try:
        mg.run(cfg, time.time())
    finally:
        mg.XSoma = real
    logs = [json.loads(line) for line in open(os.path.join(log_dir, "log.jsonl"))]
    ckpts = sorted(glob.glob(os.path.join(log_dir, "ckpt", "*.npz")))
    assert logs and ckpts and logs[-1]["it"] > cfg.wm_only_iters and logs[-1]["G"] != 0, (len(logs), len(ckpts))
    print(f"loop: {logs[-1]['it']} iterations on 1 actor + {len(jax.devices()) - 1} learners, {len(logs)} log lines, {len(ckpts)} checkpoints")


def test_multi_gpu():
    """pytest: run this file in a fresh process (JAX must start with 4 CPU devices)."""
    subprocess.run([sys.executable, os.path.abspath(__file__)], check=True)


if __name__ == "__main__":
    assert len(jax.devices()) >= 4, f"needs 4 devices, found {jax.devices()}"
    check_message()
    check_one_learner()
    check_learners()
    check_gate()
    check_loop()
    print("all multi-GPU checks pass")
