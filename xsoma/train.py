"""XSoma training: a world model of the robot and a policy trained by backpropagation through imagined rollouts in it, on
Holosoma's G1 velocity-tracking task.

One iteration:
  1. collect `env_steps` steps of all envs with the policy plus Gaussian action noise (sim.Sim) into a replay buffer;
  2. fit the world model `wm_batches` times on 8-step windows of real transitions (open loop, the recorded PD targets);
  3. update the policy `pi_updates` times: from replayed states, roll the policy out `horizon` steps in the world model and
     ascend the discounted sum of Holosoma's rewards along it. Observations, rewards and terminations of the imagined states
     are the task's own functions (task.py) on forward kinematics and the exact terrain surface (sim.Sim.view_state), so the
     gradient flows through the world model, the policy and the reward alike. There is no critic.
The three parts run as donated, ahead-of-time compiled programs, compiled in parallel; the host only dispatches them and fetches
statistics every `log_every` iterations. Checkpoints record the wall-clock time since launch, start-up and compilation included.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

from . import buffer as rb
from . import nets, optim, task
from .sim import Sim, SimCfg, unflatten


@dataclasses.dataclass
class Config:
    # task and simulator (sim.SimCfg)
    num_envs: int = 4096
    terrain: str = "mix"  # mix: Holosoma's terrain_locomotion_mix | plane
    terrain_scale: float = 0.5  # Holosoma's terrain scale factor; 0.5 as its MJWarp runs (1.0 trains too, see scene.py)
    foot: str = "capsules"  # capsules: Holosoma's MJCF soles | box: one box per sole
    preset: str = "fast_sac"  # Holosoma's reward and penalty-curriculum preset: fast_sac (g1_29dof_fast_sac) | ppo (g1_29dof);
    #                           they differ in the alive weight (10 vs 1), which a 1-s horizon needs to value not falling
    sole_friction: int = 1  # friction randomization reaches the soles (Holosoma's MuJoCo backends keep them at 0.8)
    random_start: int = 1  # episode counters start uniform in [0, 1000), as Holosoma's PPO
    # schedule
    minutes: float = 20.0  # training time, start-up excluded
    env_steps: int = 1  # env steps of all envs per iteration
    buffer_steps: int = 200  # replay buffer length in env steps (x num_envs transitions)
    warmup_steps: int = 32  # env steps collected before the first iteration
    wm_only_iters: int = 200  # iterations that fit only the world model
    # world model
    wm_hidden: str = "512,512,512"
    wm_window: int = 8
    wm_batches: int = 3
    wm_batch: int = 2048
    wm_lr: float = 4e-4
    wm_muon: int = 1  # Muon on the hidden weight matrices, Adam elsewhere (BlinkMimic's MuonAdam)
    wm_gate: float = 2.0  # skip an iteration's policy updates while the world-model loss exceeds wm_gate x its running average
    wm_eval: int = 1  # once per log interval, check the world model open loop on 32-step windows of real data
    # policy
    horizon: int = 48  # imagined steps per update (0.96 s)
    pi_batch: int = 2048  # imagined rollouts per update
    pi_updates: int = 2  # policy updates per iteration
    pi_lr: float = 3e-4
    pi_clip: float = 1.0  # gradient-norm clip
    pi_gn_gate: float = 10.0  # skip an update whose gradient norm exceeds pi_gn_gate x the running average
    pi_ema: float = 0.995  # the evaluated policy is an exponential moving average of the trained one
    gamma: float = 0.99
    action_noise: float = 0.3  # std of the collected actions' Gaussian exploration noise
    det_share: float = 0.5  # share of envs (spread over the terrain) that collect with the policy's action, without noise: with
    #                        noisy collection only, policies trained past ~45 min come to rely on the noise and fall without it
    cmd_fresh: float = 0.5  # share of imagined rollouts that start with a freshly drawn command
    term_tau: float = 0.02  # imagined termination: exact forward, gradient of sigmoid(-clearance / term_tau)
    # runtime
    imag_unroll: int = 4  # unrolling of the imagination loop (speed saturates at 4; compile time grows with it)
    imag_remat: str = "none"  # rematerialization of the imagination steps: none | dots | full (less memory, slower)
    mm_bf16: str = "imag"  # where matmuls output bf16, so their backward passes run on bf16 tensor cores: "" nowhere, "imag" in
    #                        imagination, "all" also in the world-model fits (collection and evaluation always output fp32)
    log_every: int = 50
    ckpt_minutes: float = 1.0
    eval_n: int = 2048  # > 0: after training, evaluate every checkpoint with this many envs (evaluate.py) into <log_dir>/eval.jsonl
    eval_scale: float = 0.5  # evaluation terrain scale
    eval_sole_friction: int = 0  # 0: soles at friction 0.8 in evaluation, as in Holosoma's MuJoCo training
    seed: int = 0
    log_dir: str = "runs/xsoma"
    jax_cache: str = "~/.cache/xsoma/jax"  # persistent compilation cache ("" for none): later runs load their programs
    xla_flags: str = "--xla_gpu_enable_triton_gemm=false"  # extra XLA flags, comma-separated, set before JAX's backend starts
    #                                                        (cuBLAS for every GEMM: faster on Hopper, neutral elsewhere)


# floors of the world model's per-dimension error scale: position, rotation, linear and angular velocity (3 each), q, qd
ERR_FLOOR = ((3, 3e-3), (3, 5e-3), (3, 2e-2), (3, 5e-2))


def widths(s: str):
    return tuple(int(x) for x in s.split(","))


def set_jax(cache_dir: str = ""):
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", os.path.expanduser(cache_dir))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


def target_q(a):
    return task.DEFAULT_Q + task.ACTION_SCALE * a


class XSoma:
    def __init__(self, cfg: Config):
        self.cfg = c = cfg
        set_jax(c.jax_cache)
        self.sim = Sim(SimCfg(num_envs=c.num_envs, terrain=c.terrain, terrain_scale=c.terrain_scale, foot=c.foot, preset=c.preset,
                              sole_friction=bool(c.sole_friction), random_start=bool(c.random_start), seed=42 + c.seed))  # fmt: skip
        self.J = task.J
        self.n_soles = len(self.sim.sole_radius)
        J = self.J
        self.err_floor = np.concatenate([np.full((n,), v, np.float32) for n, v in ERR_FLOOR] + [np.full((J,), 3e-3, np.float32), np.full((J,), 5e-2, np.float32)])

    # ------------------------------------------------------------------ state
    def init(self, key):
        c, J = self.cfg, self.J
        ks = jax.random.split(key, 4)
        env = self.sim.init(ks[0])
        rec_t = jax.eval_shape(lambda s: self.record(s, *self.sim.step(s, jnp.zeros((c.num_envs, J))))[0], env)
        rec_t = rb.flatten({**rec_t, "ok": jax.ShapeDtypeStruct((c.num_envs,), jnp.float32)})
        self.xi_dim = rec_t["xi"].shape[-1]
        self.wm_in = (13 + 2 * J) + self.n_soles + J + self.xi_dim  # invariant state, sole clearances, PD error, dynamics context
        pi = nets.pi_init(ks[1], task.ACTOR_DIM, J)
        wm, wm_b = nets.wm_init(ks[2], self.wm_in, 12 + 2 * J, widths(c.wm_hidden))
        self.pi_mask = jax.tree.map(lambda _: False, pi)
        self.wm_mask = {"net": nets.muon_mask(wm["net"]) if c.wm_muon else jax.tree.map(lambda _: False, wm["net"])}
        carry = {"env": env, "buf": rb.init(c.buffer_steps, {k: jnp.zeros(v.shape, v.dtype) for k, v in rec_t.items()}),
                 "pi": pi, "avg": jax.tree.map(jnp.copy, pi), "wm": wm, "wm_b": wm_b,
                 "opt_pi": optim.init(pi, self.pi_mask), "opt_wm": optim.init(wm, self.wm_mask),
                 "key": ks[3], "it": jnp.zeros((), jnp.int32), "env_steps": jnp.zeros((), jnp.float32), "nonfinite": jnp.zeros((), jnp.int32),
                 "wm_ema": jnp.zeros((), jnp.float32), "gn_ema": jnp.zeros((), jnp.float32), "wm_ok": jnp.ones((), bool)}  # fmt: skip
        strong = lambda x: jnp.asarray(x, dtype=x.dtype) if isinstance(x, jax.Array) and x.weak_type else x  # noqa: E731
        return jax.tree.map(strong, carry)

    # ------------------------------------------------------------------ collection
    def record(self, s, s1, tr):
        """One step's record: the state the action was taken in with its task context (an imagination start), the PD input
        and the state after the physics (a world-model transition), the dynamics context, flags."""
        rec = {"s_qpos": tr.qpos, "s_qvel": tr.qvel, "s1_qpos": tr.qpos1, "s1_qvel": tr.qvel1, "target": tr.target, "xi": self.sim.context(s),
               "a_last": s.a_last, "cmd": s.cmd, "phase": s.phase, "t": s.t, "freq": s.freq, "offset": s.offset, "delay": s.delay,
               "done": (tr.terminated | tr.truncated).astype(jnp.float32), "term": tr.terminated.astype(jnp.float32), "r": tr.reward}  # fmt: skip
        return rec, tr.stats

    def ingest(self, carry, flat):
        """Add one step of all envs' transitions to the replay buffer and to the world model's normalizers."""
        rec, heights = rb.nest(flat), carry["env"].world.heights
        S0, S1 = unflatten(rec["s_qpos"], rec["s_qvel"]), unflatten(rec["s1_qpos"], rec["s1_qvel"])
        v0 = self.sim.view_state(S0, heights)
        wb, ok = carry["wm_b"], rec["ok"] > 0
        upd = lambda st, x: nets.rn_update(st, x, weights=ok)  # noqa: E731
        wm_b = {"in_norm": upd(wb["in_norm"], nets.wm_inputs(S0, v0.height, v0.soles, target_q(rec["target"]), rec["xi"])),
                "out_norm": upd(wb["out_norm"], nets.wm_targets(S0, S1, task.DT)), "inc_norm": upd(wb["inc_norm"], nets.increments(S0, S1))}  # fmt: skip
        return {**carry, "buf": rb.add(carry["buf"], flat), "wm_b": wm_b, "env_steps": carry["env_steps"] + self.cfg.num_envs}

    def collect_step(self, carry, key):
        c = self.cfg
        s = carry["env"]
        noisy = (jnp.arange(c.num_envs) * c.det_share) % 1.0 >= c.det_share  # every 1 / det_share-th env acts without noise
        a = nets.pi_apply(carry["pi"], s.obs) + jnp.where(noisy, c.action_noise, 0.0)[:, None] * jax.random.normal(key, (c.num_envs, self.J))
        s1, tr = self.sim.step(s, a)
        rec, stats = self.record(s, s1, tr)
        flat = rb.sanitize(rb.flatten(rec))
        flat["ok"] = flat["ok"] * (1 - tr.diverged.astype(jnp.float32))
        stats = {**stats, "nonfinite_env": 1 - flat["ok"].mean()}
        return self.ingest({**carry, "env": s1}, flat), stats

    # ------------------------------------------------------------------ world model
    def wm_rollout(self, wm, wm_b, S, targets, xis, heights, unroll=True):
        """Open-loop world-model states over a window's PD inputs, (B, k, ...) per state part."""

        def step(S, x):
            tgt, xi = x
            v = self.sim.view_state(S, heights)
            S1 = nets.wm_step(wm, wm_b, S, v.height, v.soles, target_q(tgt), xi, task.DT, out_bf16=self.cfg.mm_bf16 == "all")
            return S1, S1

        _, Ss = jax.lax.scan(step, S, (jnp.swapaxes(targets, 0, 1), jnp.swapaxes(xis, 0, 1)), unroll=unroll)
        return tuple(jnp.swapaxes(P, 0, 1) for P in Ss)

    def wm_loss(self, wm, wm_b, w, heights):
        S_hat = self.wm_rollout(wm, wm_b, unflatten(w["s_qpos"][:, 0], w["s_qvel"][:, 0]), w["target"], w["xi"], heights)
        S_true = unflatten(w["s1_qpos"], w["s1_qvel"])
        std = jnp.maximum(nets.rn_std(wm_b["inc_norm"], 1e-4), self.err_floor)
        e = ((nets.state_error(S_hat, S_true) / std) ** 2).mean(-1)
        return (e * w["mask"]).sum() / w["mask"].shape[0], e[:, -1]

    def wm_update(self, carry, key):
        c = self.cfg
        w = rb.sample_windows(carry["buf"], key, c.wm_batch, c.wm_window)
        (loss, e_last), grads = jax.value_and_grad(self.wm_loss, has_aux=True)(carry["wm"], carry["wm_b"], w, carry["env"].world.heights)
        grads, _ = optim.clip_by_global_norm(grads, 1.0)
        wm, opt = optim.step(carry["wm"], grads, carry["opt_wm"], self.wm_mask, c.wm_lr, betas=(0.9, 0.999))
        m = w["mask"][:, -1]
        return {**carry, "wm": wm, "opt_wm": opt}, {"wm": loss / c.wm_window, "wm8": (e_last * m).sum() / jnp.maximum(m.sum(), 1)}

    def wm_eval(self, carry, key, B: int = 512, K: int = 32):
        """Open-loop rollouts from real states with the recorded PD inputs over K steps (valid parts of the windows only):
        pelvis height error at steps 8 and K, and the termination rate the env saw vs the one the model's states imply."""
        heights = carry["env"].world.heights
        w = rb.sample_windows(carry["buf"], key, B, K)
        S_hat = self.wm_rollout(carry["wm"], carry["wm_b"], unflatten(w["s_qpos"][:, 0], w["s_qvel"][:, 0]), w["target"], w["xi"], heights,
                                unroll=2)  # fmt: skip
        v_hat = self.sim.view_state(S_hat, heights)
        v_true = self.sim.view(w["s1_qpos"], w["s1_qvel"], heights)
        herr = jnp.abs(v_hat.height - v_true.height)
        m = w["mask"]
        mean = lambda x, j: (x[:, j] * m[:, j]).sum() / jnp.maximum(m[:, j].sum(), 1)  # noqa: E731
        tp = (v_hat.clearance.min(-1) < 0).astype(jnp.float32)
        return {"h_err8": mean(herr, 7), "h_errK": mean(herr, K - 1), "term_env": (w["term"] * m).sum() / jnp.maximum(m.sum(), 1),
                "term_wm": (tp * m).sum() / jnp.maximum(m.sum(), 1)}  # fmt: skip

    # ------------------------------------------------------------------ imagination
    def sample_starts(self, buf, key):
        """Imagination starts: replayed states with their task context; a share gets a fresh command (and gait frequency) and
        its gait phase recomputed, since a command can change at any state (every 500 steps, and on the first step)."""
        c, B = self.cfg, self.cfg.pi_batch
        k_s, k_u, k_c = jax.random.split(key, 3)
        start = rb.nest(rb.sample_starts(buf, k_s, B))
        cmd, freq = task.sample_commands(k_c, B)
        fresh = jax.random.uniform(k_u, (B,)) < c.cmd_fresh
        start["cmd"] = jnp.where(fresh[:, None], cmd, start["cmd"])
        start["freq"] = jnp.where(fresh, freq, start["freq"])
        start["phase"] = jnp.where(fresh[:, None], task.gait_phase(start["t"], start["freq"], start["offset"], start["cmd"]), start["phase"])
        return start

    def termination(self, clearance):
        """Imagined termination: forward exactly the geometric event (a termination collider below the terrain surface; agrees
        with the env's contact-force termination on ~99 % of steps), gradient of the sigmoid of the smallest clearance."""
        cmin = clearance.min(-1)
        soft = jax.nn.sigmoid(-cmin / self.cfg.term_tau)
        hard = (cmin < 0).astype(jnp.float32)
        return hard + soft - jax.lax.stop_gradient(soft)

    def imagine(self, pi, wm, wm_b, start, keys, scale, heights):
        """Negative mean discounted return of `horizon` imagined steps from `start`, the task's step order: the policy acts on
        the observation, the world model moves the robot, the reward and termination are read on the new state, then the gait
        phase advances and the command resamples every 500 steps."""
        c, B = self.cfg, self.cfg.pi_batch
        bf16 = c.mm_bf16 in ("imag", "all")
        rcfg = self.sim.reward_cfg
        S0 = unflatten(start["s_qpos"], start["s_qvel"])
        xi, delay, offset = start["xi"], start["delay"], start["offset"]

        def step(carry, k):
            S, v, a_last, cmd, freq, phase, t = carry
            k_o, k_c = jax.random.split(k)
            obs, _ = task.observations(v.robot, a_last, cmd, phase, k_o)
            a = nets.pi_apply(pi, obs, out_bf16=bf16)
            tgt = jnp.clip(jnp.where(delay[:, None] == 0, a, a_last), -task.ACTION_CLIP, task.ACTION_CLIP)
            S1 = nets.wm_step(wm, wm_b, S, v.height, v.soles, target_q(tgt), xi, task.DT, out_bf16=bf16)
            v1 = self.sim.view_state(S1, heights)
            r = task.reward(task.reward_terms(v1.robot, v1.foot_h, a, a_last, cmd, phase), rcfg, scale)
            d = self.termination(v1.clearance)
            t1 = t + 1
            phase1 = task.gait_phase(t1, freq, offset, cmd)
            cmd2, freq2 = task.sample_commands(k_c, B)
            new = t1 % task.RESAMPLE == 0
            cmd1, freq1 = jnp.where(new[:, None], cmd2, cmd), jnp.where(new, freq2, freq)
            return (S1, v1, a, cmd1, freq1, phase1, t1), (r, d)

        if c.imag_remat == "dots":
            step = jax.checkpoint(step, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        elif c.imag_remat == "full":
            step = jax.checkpoint(step)
        init = (S0, self.sim.view_state(S0, heights), start["a_last"], start["cmd"], start["freq"], start["phase"], start["t"])
        _, (r, d) = jax.lax.scan(step, init, keys, unroll=c.imag_unroll)
        alive = jnp.cumprod(1 - d, 0)
        alive_before = jnp.concatenate((jnp.ones_like(alive[:1]), alive[:-1]), 0)  # a step's reward counts if alive before it
        disc = c.gamma ** jnp.arange(c.horizon, dtype=jnp.float32)[:, None]
        G = (disc * alive_before * r).sum(0)
        return -G.mean(), {"G": G.mean(), "r": (alive_before * r).sum(0).mean() / c.horizon, "surv": alive[-1].mean()}

    def policy_update(self, carry, start, key):
        c = self.cfg
        env = carry["env"]
        (_, aux), grads = jax.value_and_grad(self.imagine, has_aux=True)(carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(key, c.horizon),
                                                                         env.scale, env.world.heights)  # fmt: skip
        grads, gn = optim.clip_by_global_norm(grads, c.pi_clip)
        finite = jnp.isfinite(gn)
        first = carry["gn_ema"] <= 0
        ok = finite & (first | (c.pi_gn_gate <= 0) | (gn < c.pi_gn_gate * carry["gn_ema"]))
        gn_ema = jnp.where(ok, jnp.where(first, gn, 0.98 * carry["gn_ema"] + 0.02 * gn), 1.01 * carry["gn_ema"])
        grads = jax.tree.map(lambda g: jnp.where(ok, g, 0.0), grads)
        pi, opt_pi = optim.step(carry["pi"], grads, carry["opt_pi"], self.pi_mask, c.pi_lr, betas=(0.9, 0.95), ok=ok)
        avg = optim.ema(carry["avg"], pi, c.pi_ema, carry["it"] == c.wm_only_iters)
        stats = {**aux, "gn": gn, "gn_gated": (finite & ~ok).astype(jnp.float32)}
        return {"pi": pi, "opt_pi": opt_pi, "avg": avg, "gn_ema": gn_ema, "nonfinite": carry["nonfinite"] + (~finite).astype(jnp.int32)}, stats

    # ------------------------------------------------------------------ iteration programs
    def collect_iter(self, carry):
        """env_steps steps of all envs; per-step statistics, summed (counts) or averaged (rates) over the steps."""
        key, k_c = jax.random.split(carry["key"])
        carry, st = jax.lax.scan(self.collect_step, {**carry, "key": key}, jax.random.split(k_c, self.cfg.env_steps))
        summed = ("done", "terminated", "len_sum", "rew_sums", "pushes", "diverged")
        return carry, {k: (v.sum(0) if k in summed else v.mean(0)) for k, v in st.items()}

    def wm_iter(self, carry):
        """wm_batches world-model fits, and the gate that keeps the policy from training on a model whose loss jumped."""
        c = self.cfg
        key, k_w = jax.random.split(carry["key"])
        carry, wstats = jax.lax.scan(self.wm_update, {**carry, "key": key}, jax.random.split(k_w, c.wm_batches))
        wm_now = wstats["wm"].mean()
        first = carry["wm_ema"] <= 0
        healthy = first | (c.wm_gate <= 0) | (wm_now < c.wm_gate * carry["wm_ema"])
        wm_ema = jnp.where(healthy, jnp.where(first, wm_now, 0.98 * carry["wm_ema"] + 0.02 * wm_now), 1.005 * carry["wm_ema"])
        return {**carry, "wm_ema": wm_ema, "wm_ok": healthy}, {"wm": wm_now, "wm8": wstats["wm8"][-1]}

    def pi_iter(self, carry):
        """pi_updates policy updates (after wm_only_iters iterations, while the world model is healthy)."""
        c = self.cfg
        key, k_p = jax.random.split(carry["key"])
        names = ("pi", "opt_pi", "avg", "gn_ema", "nonfinite")

        def one(sub, k):
            k_s, k_u = jax.random.split(k)
            return self.policy_update({**carry, **sub}, self.sample_starts(carry["buf"], k_s), k_u)

        def do(sub):
            sub, st = jax.lax.scan(one, sub, jax.random.split(k_p, c.pi_updates))
            return sub, jax.tree.map(jnp.mean, st)

        def skip(sub):
            return sub, {k: jnp.zeros((), jnp.float32) for k in ("G", "r", "surv", "gn", "gn_gated")}

        active = carry["it"] >= c.wm_only_iters
        sub, pstats = jax.lax.cond(active & carry["wm_ok"], do, skip, {k: carry[k] for k in names})
        return {**carry, **sub, "key": key, "it": carry["it"] + 1}, {**pstats, "wm_gated": (active & ~carry["wm_ok"]).astype(jnp.float32)}

    def compile_programs(self, carry):
        """AOT-compile an iteration as three donated programs (collection, world-model fits, policy updates) plus the world-model
        check: lowered one after the other (tracing holds the GIL), compiled in parallel threads (XLA's backend compile releases
        it), so the start-up waits for the largest program, not for their sum."""
        lowered = {"collect": jax.jit(self.collect_iter, donate_argnums=(0,)).lower(carry),
                   "wm": jax.jit(self.wm_iter, donate_argnums=(0,)).lower(carry),
                   "pi": jax.jit(self.pi_iter, donate_argnums=(0,)).lower(carry)}  # fmt: skip
        if self.cfg.wm_eval:
            lowered["wm_eval"] = jax.jit(lambda cr: self.wm_eval(cr, jax.random.fold_in(cr["key"], 3))).lower(carry)
        with ThreadPoolExecutor(len(lowered)) as ex:
            return dict(zip(lowered, ex.map(lambda low: low.compile(), lowered.values())))


def save_ckpt(path, pi_avg, pi_live, meta):
    flat = {}
    for name, layers in (("avg", pi_avg), ("live", pi_live)):
        for i, layer in enumerate(layers):
            for k, v in layer.items():
                flat[f"{name}/{i}/{k}"] = np.asarray(v)
    np.savez(path + ".tmp.npz", **flat)
    os.replace(path + ".tmp.npz", path + ".npz")
    with open(path + ".json", "w") as f:
        json.dump(meta, f)


def parse(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Train XSoma on Holosoma's G1 velocity-tracking task.")
    for f in dataclasses.fields(Config):
        p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return Config(**vars(p.parse_args(argv)))


def main(argv=None):
    t_launch = time.time()  # the wall clock of a comparison counts from here (start-up and compilation included)
    cfg = parse(argv)
    flags = [f for f in cfg.xla_flags.split(",") if f.startswith("--")]
    if flags:
        os.environ["XLA_FLAGS"] = " ".join([os.environ.get("XLA_FLAGS", "")] + flags).strip()
    os.makedirs(os.path.join(cfg.log_dir, "ckpt"), exist_ok=True)
    with open(os.path.join(cfg.log_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
    t_build = time.time()
    xs = XSoma(cfg)
    carry = xs.init(jax.random.PRNGKey(cfg.seed))
    print(f"[xsoma] {cfg.terrain} x{cfg.terrain_scale} {cfg.foot} {cfg.preset}, {cfg.num_envs} envs, world-model input {xs.wm_in}, "
          f"buffer {sum(v.nbytes for v in jax.tree.leaves(carry['buf'])) / 2**30:.2f} GiB, device {jax.devices()[0]}", flush=True)  # fmt: skip
    t_c = time.time()
    progs = xs.compile_programs(carry)
    print(f"[xsoma] build {t_c - t_build:.0f} s, compile {time.time() - t_c:.0f} s, start-up {time.time() - t_launch:.0f} s since launch", flush=True)
    gc.collect()
    gc.freeze()  # Python's generation-2 collections would otherwise stall the host loop
    logf = open(os.path.join(cfg.log_dir, "log.jsonl"), "a", buffering=1)
    t0 = time.time()
    for _ in range(-(-cfg.warmup_steps // cfg.env_steps)):
        carry, _ = progs["collect"](carry)
    last_ckpt, n_ckpt, prev_s, prev_env = 0.0, 0, 0.0, 0.0
    while True:
        cs, ls = [], []
        for _ in range(cfg.log_every):  # dispatched asynchronously; the host waits only for the statistics below
            carry, cstats = progs["collect"](carry)
            carry, wstats = progs["wm"](carry)
            carry, pstats = progs["pi"](carry)
            cs.append(cstats)
            ls.append({**wstats, **pstats})
        cs, ls, ev = jax.device_get((cs, ls, progs["wm_eval"](carry) if cfg.wm_eval else {}))
        n_done = max(sum(float(x["done"]) for x in cs), 1.0)
        rew = np.sum([x["rew_sums"] for x in cs], 0) / n_done
        stats = {"ep_len": sum(float(x["len_sum"]) for x in cs) / n_done, "done": n_done,
                 "contact": sum(float(x["terminated"]) for x in cs) / n_done, **{f"rew_{k}": float(v) for k, v in zip(task.TERMS, rew)},
                 **{k: float(np.mean([x[k] for x in cs])) for k in ("reward", "scale", "avg_len", "term_geom", "nacon", "nonfinite_env")},
                 "diverged": sum(float(x["diverged"]) for x in cs),
                 **{k: float(np.mean([x[k] for x in ls])) for k in ls[0]}, **{k: float(v) for k, v in ev.items()}}  # fmt: skip
        train_s = time.time() - t0
        it, env_steps = int(carry["it"]), float(carry["env_steps"])
        rec = {"it": it, "train_s": train_s, "wall_s": time.time() - t_launch, "env_steps": env_steps, "nonfinite": int(carry["nonfinite"]), **stats}
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
            save_ckpt(os.path.join(cfg.log_dir, "ckpt", f"{n_ckpt:04d}"), carry["avg"], carry["pi"],
                      {"it": it, "train_s": train_s, "wall_s": time.time() - t_launch, "env_steps": env_steps})  # fmt: skip
            last_ckpt, n_ckpt = train_s, n_ckpt + 1
        if train_s >= 60 * cfg.minutes:
            break
    print(f"done: {it} iterations, {env_steps / 1e6:.1f}M env steps, {train_s / 60:.2f} min", flush=True)
    if cfg.eval_n > 0:
        from . import evaluate

        del carry, progs, xs
        gc.collect()
        evaluate.main([cfg.log_dir, "--n", str(cfg.eval_n), "--scale", str(cfg.eval_scale), "--sole_friction", str(cfg.eval_sole_friction),
                       "--preset", cfg.preset])  # fmt: skip


if __name__ == "__main__":
    main()
