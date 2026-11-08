"""XSoma on several GPUs, layout AL (`--layout AL`): GPU 0 is the actor, every other visible GPU (or `learners` of them) a policy
learner.

An iteration:
  actor (GPU 0)          collects `env_steps` steps of all envs with the newest policy it has and fits the world model `wm_batches`
                         times (train.XSoma's collection and world-model programs, unchanged), then draws the imagination starts
                         of the learners' next round from the replay buffer, as train.XSoma.sample_starts draws them, and sends
                         them to the learners together with the world model;
  learners (GPU 1 to L)  update the policy on the starts and the world model of the actor's previous iteration, in rounds: every
                         learner takes one gradient of `pi_batch` imagined rollouts from the same policy, and all learners apply
                         all L gradients one after another, as L steps of the one-GPU update (clip, gradient gate, Adam step,
                         average), so their copies of the policy stay identical. An iteration has ceil(pi_updates / L) rounds.
With one learner (2 GPUs) the updates are those of one GPU, two per iteration. With three learners (4 GPUs) an iteration has one
round of three steps: the second and the third step use a gradient taken one and two steps earlier. Splitting one update's
rollouts over the learners instead would gain little, because an update is bound by kernel latency rather than by its batch
(on an H100 ~40 ms with 2048 rollouts and ~29 ms with 683). On any number of GPUs, the updates see the world model and the starts
of the previous iteration (one GPU: of the current one), and the actor collects with the policy of the learners' previous round.

The host: one thread starts the actor's programs, the main thread the learners'. With one thread for both the GPUs would take
turns, because a dispatch returns only once its program is launched: the learners' program launches thousands of short kernels,
and the collection runs MuJoCo Warp through Python callbacks. The actor's thread is the only one that touches GPU 0, where MuJoCo
Warp captures CUDA graphs: a copy into that GPU from another thread during a capture breaks it. The learners' rounds are one
program over all learner GPUs (shard_map). The main thread also writes train.main's log lines, log.jsonl, checkpoints and the
final evaluation, in the same formats, so evaluate.py and tools/make_figures.py read multi-GPU runs as they read one-GPU runs.
"""

from __future__ import annotations

import gc
import json
import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P

from . import optim, task
from .train import XSoma, save_ckpt

START_KEYS = ("s_qpos", "s_qvel", "xi", "a_last", "cmd", "phase", "t", "freq", "offset", "delay")  # what imagination reads of a start
LEARNER_KEYS = ("pi", "opt_pi", "avg", "gn_ema", "nonfinite")  # the learners' part of the training state
PI_STATS = ("G", "r", "surv", "gn", "gn_gated")  # the statistics of the policy updates
STOP = object()  # tells the actor's thread to end


# ---------------------------------------------------------------- packing
def pack(tree):
    """All leaves of a pytree as one float32 vector: one copy between GPUs instead of one per leaf, and always a fresh buffer,
    never one that a donated training state owns."""
    return jnp.concatenate([x.astype(jnp.float32).ravel() for x in jax.tree.leaves(tree)])


def make_unpack(template):
    """The inverse of pack for pytrees shaped like `template`. The leaves lie along the last axis; leading axes are kept."""
    leaves, tdef = jax.tree.flatten(template)
    spec = [(x.shape, x.dtype, int(np.prod(x.shape))) for x in leaves]

    def unpack(v):
        out, i = [], 0
        for shape, dtype, n in spec:
            out.append(v[..., i : i + n].reshape(v.shape[:-1] + shape).astype(dtype))
            i += n
        return jax.tree.unflatten(tdef, out)

    return unpack


# ---------------------------------------------------------------- programs
def actor_message(xs: XSoma, L: int, rounds: int):
    """The program that ends an actor iteration. It draws the imagination starts of the learners' next `rounds` rounds, `pi_batch`
    per learner and round, and packs each learner's share into one array (rounds, pi_batch, d); it packs the world model, its
    normalizers, the penalty scale and the two update flags into one vector; and it advances the iteration counter (pi_iter's job on
    one GPU)."""
    c = xs.cfg

    def message(carry):
        key, k_s = jax.random.split(carry["key"])
        st = jax.vmap(xs.sample_starts, in_axes=(None, 0))(carry["buf"], jax.random.split(k_s, rounds * L))
        leaves = jax.tree.leaves({k: st[k] for k in START_KEYS})  # in the order make_unpack reads them
        rows = jnp.concatenate([x.reshape(x.shape[:2] + (-1,)).astype(jnp.float32) for x in leaves], -1).reshape(rounds, L, c.pi_batch, -1)
        it = carry["it"]
        active = it >= c.wm_only_iters
        rep = pack({"wm": carry["wm"], "wm_b": carry["wm_b"], "scale": carry["env"].scale, "active": active & carry["wm_ok"],
                    "first": it == c.wm_only_iters})  # fmt: skip
        stats = {"wm_gated": (active & ~carry["wm_ok"]).astype(jnp.float32)}
        return {**carry, "key": key, "it": it + 1}, tuple(rows[:, l] for l in range(L)), rep, stats

    return message


def learner_program(xs: XSoma, mesh: Mesh, rounds: int, unpack_starts, unpack_rep):
    """The learners' program: `rounds` rounds of policy updates, as one shard_map program over the learner GPUs. It returns the
    learners' new state, the policy packed for the actor, and the statistics averaged over the round's gradients. A round whose
    world model failed the gate (the message's `active` flag) leaves the learners' state as it was; before `wm_only_iters` the host
    does not start the program at all."""
    c = xs.cfg

    def apply(sub, g, first):
        """train.XSoma.policy_update after its gradient: the clip, the gradient gate, the Adam step and the policy average."""
        grads, gn = optim.clip_by_global_norm(g, c.pi_clip)
        finite = jnp.isfinite(gn)
        f0 = sub["gn_ema"] <= 0
        ok = finite & (f0 | (c.pi_gn_gate <= 0) | (gn < c.pi_gn_gate * sub["gn_ema"]))
        gn_ema = jnp.where(ok, jnp.where(f0, gn, 0.98 * sub["gn_ema"] + 0.02 * gn), 1.01 * sub["gn_ema"])
        grads = jax.tree.map(lambda x: jnp.where(ok, x, 0.0), grads)
        pi, opt_pi = optim.step(sub["pi"], grads, sub["opt_pi"], xs.pi_mask, c.pi_lr, betas=(0.9, 0.95), ok=ok)
        avg = optim.ema(sub["avg"], pi, c.pi_ema, first)
        out = {"pi": pi, "opt_pi": opt_pi, "avg": avg, "gn_ema": gn_ema, "nonfinite": sub["nonfinite"] + (~finite).astype(jnp.int32)}
        return out, {"gn": gn, "gn_gated": (finite & ~ok).astype(jnp.float32)}

    def local(cL, starts, rep, heights):
        """One learner's view: the learners' state and the message are copies, `starts` is this learner's share."""
        m = unpack_rep(rep)
        key, k_p = jax.random.split(cL["key"])
        k_p = jax.random.fold_in(k_p, jax.lax.axis_index("l"))  # each learner its own observation noise and commands
        sub0 = {k: cL[k] for k in LEARNER_KEYS}

        def round_(sub, x):
            st, k = x
            (_, aux), g = jax.value_and_grad(xs.imagine, has_aux=True)(sub["pi"], m["wm"], m["wm_b"], unpack_starts(st), jax.random.split(k, c.horizon),
                                                                         m["scale"], heights)  # fmt: skip
            # every learner applies every learner's gradient, in the learners' order
            sub, s2 = jax.lax.scan(lambda s, gl: apply(s, gl, m["first"]), sub, jax.lax.all_gather(g, "l"))
            return sub, {**jax.lax.pmean(aux, "l"), **jax.tree.map(jnp.mean, s2)}

        sub, st = jax.lax.scan(round_, sub0, (starts, jax.random.split(k_p, rounds)))
        ok = m["active"]
        sub = jax.tree.map(lambda new, old: jnp.where(ok, new, old), sub, sub0)
        return {**sub, "key": key}, pack(sub["pi"]), {k: jnp.where(ok, v.mean(), 0.0) for k, v in st.items()}

    rep = P()
    return jax.shard_map(local, mesh=mesh, in_specs=(rep, P(None, "l"), rep, rep), out_specs=(rep, rep, rep), check_vma=False)


# ---------------------------------------------------------------- training
def run(cfg, t_launch):
    """Train with layout AL for `minutes`: train.main's loop, log, checkpoints and final evaluation, on L + 1 GPUs."""
    devs = jax.devices()
    L = cfg.learners or len(devs) - 1
    assert cfg.layout == "AL", f"unknown layout {cfg.layout!r} (1GPU | AL)"
    assert 1 <= L <= len(devs) - 1, f"layout AL with {L} learners needs {L + 1} GPUs, found {len(devs)}"
    rounds = math.ceil(cfg.pi_updates / L)
    devA, devL = devs[0], devs[1 : L + 1]
    mesh = Mesh(np.array(devL), ("l",))
    rep_s, shard_s = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "l"))

    t_build = time.time()
    xs = XSoma(cfg)  # the simulator lives on the default device, GPU 0
    cA = xs.init(jax.random.PRNGKey(cfg.seed))  # the actor's state: train.XSoma's carry; its policy is the one the actor acts with
    cL = jax.device_put({**{k: cA[k] for k in LEARNER_KEYS}, "key": jax.random.fold_in(cA["key"], 7)}, rep_s)
    heights = jax.device_put(cA["env"].world.heights, rep_s)
    jax.block_until_ready((cL, heights))  # copied before the actor's first program takes over (donates) the state they come from
    print(f"[xsoma] {cfg.terrain} x{cfg.terrain_scale} {cfg.foot} {cfg.preset}, {cfg.num_envs} envs, world-model input {xs.wm_in}, "
          f"buffer {sum(v.nbytes for v in jax.tree.leaves(cA['buf'])) / 2**30:.2f} GiB | layout AL: actor {devA}, learners "
          f"{', '.join(str(d) for d in devL)}; {rounds} round(s) of {L} policy update(s) per iteration", flush=True)  # fmt: skip

    message = actor_message(xs, L, rounds)
    _, starts_aval, rep_aval, _ = jax.eval_shape(message, cA)
    st_aval = jax.eval_shape(xs.sample_starts, cA["buf"], cA["key"])
    unpack_starts = make_unpack({k: jax.ShapeDtypeStruct(st_aval[k].shape[1:], st_aval[k].dtype) for k in START_KEYS})
    unpack_rep = make_unpack(jax.eval_shape(lambda cr: {"wm": cr["wm"], "wm_b": cr["wm_b"], "scale": cr["env"].scale, "active": cr["wm_ok"],
                                                       "first": cr["wm_ok"]}, cA))  # fmt: skip
    unpack_pi = make_unpack(jax.eval_shape(lambda p: p, cA["pi"]))
    d = starts_aval[0].shape[-1]
    starts_L = jax.ShapeDtypeStruct((rounds, L * cfg.pi_batch, d), jnp.float32, sharding=shard_s)
    rep_L = jax.ShapeDtypeStruct(rep_aval.shape, rep_aval.dtype, sharding=rep_s)
    pi_flat = jax.ShapeDtypeStruct((sum(x.size for x in jax.tree.leaves(cA["pi"])),), jnp.float32, sharding=SingleDeviceSharding(devA))
    t_c = time.time()  # lowered one after the other, compiled in parallel threads (as train.XSoma.compile_programs)
    lowered = {"collect": jax.jit(xs.collect_iter, donate_argnums=(0,)).lower(cA), "wm": jax.jit(xs.wm_iter, donate_argnums=(0,)).lower(cA),
               "message": jax.jit(message, donate_argnums=(0,)).lower(cA), "unpack": jax.jit(unpack_pi).lower(pi_flat),
               "learn": jax.jit(learner_program(xs, mesh, rounds, unpack_starts, unpack_rep), donate_argnums=(0, 1, 2)).lower(cL, starts_L, rep_L, heights)}  # fmt: skip
    if cfg.wm_eval:
        lowered["wm_eval"] = jax.jit(lambda cr: xs.wm_eval(cr, jax.random.fold_in(cr["key"], 3))).lower(cA)
    with ThreadPoolExecutor(len(lowered)) as ex:
        progs = dict(zip(lowered, ex.map(lambda low: low.compile(), lowered.values())))
    print(f"[xsoma] build {t_c - t_build:.0f} s, compile {time.time() - t_c:.0f} s, start-up {time.time() - t_launch:.0f} s since launch", flush=True)

    q_pi, q_msg = queue.Queue(), queue.Queue()  # main thread -> actor: policies; actor -> main thread: messages and statistics

    def actor(cA):
        """The actor's thread: iteration i collects with the policy of the learners' round i - 1 (iterations 0 and 1: the initial
        one), then sends its message; every log_every iterations it adds its statistics, fetched to the host."""
        try:
            for _ in range(-(-cfg.warmup_steps // cfg.env_steps)):  # warm-up collection, as in train.main
                cA, _ = progs["collect"](cA)
            stats, i = [], 0
            while True:
                if i >= 2:
                    pi = q_pi.get()
                    if pi is STOP:
                        return
                    if pi is not None:  # None: no round ran (world-model-only iterations), the policy is unchanged
                        cA = {**cA, "pi": progs["unpack"](jax.device_put(pi.addressable_shards[0].data, devA))}
                cA, cs = progs["collect"](cA)
                cA, ws = progs["wm"](cA)
                cA, starts, rep, ms = progs["message"](cA)
                sL = jax.make_array_from_single_device_arrays(starts_L.shape, shard_s, [jax.device_put(s, dv) for s, dv in zip(starts, devL)])
                rL = jax.make_array_from_single_device_arrays(rep.shape, rep_s, [jax.device_put(rep, dv) for dv in devL])
                stats.append((cs, {**ws, **ms}))
                log = None
                if (i + 1) % cfg.log_every == 0:
                    log = jax.device_get((stats, progs["wm_eval"](cA) if cfg.wm_eval else {}, cA["it"], cA["env_steps"]))
                    stats = []
                q_msg.put((sL, rL, log))
                i += 1
        except Exception as e:  # noqa: BLE001  (raised again in the main thread)
            q_msg.put(e)

    def receive():
        msg = q_msg.get()
        if isinstance(msg, Exception):
            raise RuntimeError("the actor's thread failed") from msg
        return msg

    gc.collect()
    gc.freeze()  # Python's generation-2 collections would otherwise stall the host
    logf = open(os.path.join(cfg.log_dir, "log.jsonl"), "a", buffering=1)
    t0 = time.time()  # the training clock: compilation is over, the warm-up counts
    thread = threading.Thread(target=actor, args=(cA,), daemon=True)
    thread.start()
    del cA  # from here on only the actor's thread holds (and touches) GPU 0's state
    sL, rL, _ = receive()  # actor iteration 0
    zeros = {k: 0.0 for k in PI_STATS}
    lstats, i = [zeros], 1  # the learners' statistics per iteration (iteration 0 has no round)
    last_ckpt, n_ckpt, prev_s, prev_env = 0.0, 0, 0.0, 0.0
    while True:
        if i - 1 >= cfg.wm_only_iters:  # round i, on the message of actor iteration i - 1 (whose counter is i - 1)
            cL, pi, st = progs["learn"](cL, sL, rL, heights)
            lstats.append(st)
            q_pi.put(pi)  # for actor iteration i + 1
        else:
            lstats.append(zeros)
            q_pi.put(None)
        sL, rL, log = receive()  # actor iteration i
        i += 1
        if log is None:
            continue
        actor_stats, ev, it, env_steps = log
        cs = [s[0] for s in actor_stats]
        ls = [{**s[1], **p} for s, p in zip(actor_stats, jax.device_get(lstats))]
        lstats = []
        # from here as in train.main
        n_done = max(sum(float(x["done"]) for x in cs), 1.0)
        rew = np.sum([x["rew_sums"] for x in cs], 0) / n_done
        stats = {"ep_len": sum(float(x["len_sum"]) for x in cs) / n_done, "done": n_done,
                 "contact": sum(float(x["terminated"]) for x in cs) / n_done, **{f"rew_{k}": float(v) for k, v in zip(task.TERMS, rew)},
                 **{k: float(np.mean([x[k] for x in cs])) for k in ("reward", "scale", "avg_len", "term_geom", "nacon", "nonfinite_env")},
                 "diverged": sum(float(x["diverged"]) for x in cs),
                 **{k: float(np.mean([x[k] for x in ls])) for k in ls[0]}, **{k: float(v) for k, v in ev.items()}}  # fmt: skip
        train_s = time.time() - t0
        it, env_steps = int(it), float(env_steps)
        rec = {"it": it, "train_s": train_s, "wall_s": time.time() - t_launch, "env_steps": env_steps, "nonfinite": int(cL["nonfinite"]), **stats}
        logf.write(json.dumps(rec) + "\n")
        print(f"it {it} | {train_s / 60:.2f} min | {env_steps / 1e6:.1f}M steps | len {stats['ep_len']:.0f} lin {stats['rew_tracking_lin_vel']:.3f} "
              f"ang {stats['rew_tracking_ang_vel']:.3f} r {stats['reward']:.4f} s {stats['scale']:.3f} | wm {stats['wm']:.4f} e8 {stats['wm8']:.4f} | "
              f"G {stats['G']:.3f} r_im {stats['r']:.4f} surv {stats['surv']:.3f} gn {stats['gn']:.2f} | {cfg.log_every / max(train_s - prev_s, 1e-9):.1f} it/s "
              f"{(env_steps - prev_env) / max(train_s - prev_s, 1e-9) / 1e3:.0f}k steps/s"
              + (f" | h_err {stats['h_err8'] * 100:.1f}/{stats['h_errK'] * 100:.1f} cm term env {stats['term_env']:.3f} wm {stats['term_wm']:.3f}" if cfg.wm_eval else "")
              + (f" | gated wm {stats['wm_gated']:.2f} gn {stats['gn_gated']:.2f}" if stats["wm_gated"] or stats["gn_gated"] else "")
              + (f" | diverged {stats['diverged']:.0f}" if stats["diverged"] else ""),
              flush=True)  # fmt: skip
        prev_s, prev_env = train_s, env_steps
        if train_s - last_ckpt >= 60 * cfg.ckpt_minutes or train_s >= 60 * cfg.minutes:
            save_ckpt(os.path.join(cfg.log_dir, "ckpt", f"{n_ckpt:04d}"), cL["avg"], cL["pi"],
                      {"it": it, "train_s": train_s, "wall_s": time.time() - t_launch, "env_steps": env_steps})  # fmt: skip
            last_ckpt, n_ckpt = train_s, n_ckpt + 1
        if train_s >= 60 * cfg.minutes:
            break
    q_pi.put(STOP)
    thread.join()  # the actor finishes its iteration before the evaluation uses GPU 0
    print(f"done: {it} iterations, {env_steps / 1e6:.1f}M env steps, {train_s / 60:.2f} min", flush=True)
    diff = max(float(np.abs(np.asarray(s.data) - np.asarray(x.addressable_shards[0].data)).max())
               for x in jax.tree.leaves({k: cL[k] for k in ("pi", "avg", "opt_pi")}) for s in x.addressable_shards)  # fmt: skip
    if diff:
        print(f"[xsoma] warning: the learners' copies of the training state differ by up to {diff:.3g}", flush=True)
    if cfg.eval_n > 0:
        from . import evaluate

        del cL, progs, xs  # free the training state before the evaluation builds its own simulator
        gc.collect()
        evaluate.main([cfg.log_dir, "--n", str(cfg.eval_n), "--scale", str(cfg.eval_scale), "--sole_friction", str(cfg.eval_sole_friction),
                       "--preset", cfg.preset])  # fmt: skip
