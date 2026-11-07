"""Device replay buffer of env transitions laid out (time, env), as XMimic's: fill level and write head on the device.

Leaves are the flattened transition record (see train.XSoma.record). The dynamics context xi is stored in float16; states,
actions and the task context in float32 / int32. A transition j ends a world-model window when its env was done, or when the
next transition does not start where it ended (a push overwrote the root velocity in between).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

HALF = ("xi",)


def flatten(rec):
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            out.update({f"{k}/{k2}": v2 for k2, v2 in v.items()})
        else:
            out[k] = v
    return out


def nest(flat):
    out = {}
    for k, v in flat.items():
        if "/" in k:
            a, b = k.split("/", 1)
            out.setdefault(a, {})[b] = v
        else:
            out[k] = v
    return out


def init(T: int, rec_flat):
    buf = {k: jnp.zeros((T,) + v.shape, jnp.float16 if (v.dtype == jnp.float32 and k.startswith(HALF)) else v.dtype) for k, v in rec_flat.items()}
    buf["ptr"], buf["size"] = jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32)
    return buf


def add(buf, rec_flat):
    ptr = buf["ptr"]
    out = {k: jax.lax.dynamic_update_index_in_dim(buf[k], rec_flat[k].astype(buf[k].dtype), ptr, 0) for k in rec_flat}
    return {**out, "ptr": (ptr + 1) % _T(buf), "size": jnp.minimum(buf["size"] + 1, _T(buf))}


def _T(buf):
    return buf["s_qpos"].shape[0]


def _rows(buf, age):
    return (buf["ptr"] - buf["size"] + age) % _T(buf)


def _uniform_int(key, n, B):
    return jnp.minimum((jax.random.uniform(key, (B,)) * n).astype(jnp.int32), n - 1)


def gather(buf, rows, envs):
    return {k: (v[rows, envs].astype(jnp.float32) if v.dtype == jnp.float16 else v[rows, envs]) for k, v in buf.items() if k not in ("ptr", "size")}


def sample_windows(buf, key, B: int, k: int):
    """k-step windows (B, k, ...) and their validity mask; the age range leaves room for the transition after the window.
    Transition j counts while transitions 0..j are finite and no transition before j ended its env's run of consecutive
    states."""
    kt, ke = jax.random.split(key)
    N = buf["s_qpos"].shape[1]
    t0 = _uniform_int(kt, buf["size"] - k, B)
    env = jax.random.randint(ke, (B,), 0, N)
    rows = _rows(buf, t0[:, None] + jnp.arange(k + 1))
    w = gather(buf, rows, env[:, None])
    nxt_q, nxt_v = w["s_qpos"][:, 1:], w["s_qvel"][:, 1:]
    jump = (jnp.abs(nxt_q - w["s1_qpos"][:, :-1]).max(-1) > 1e-5) | (jnp.abs(nxt_v - w["s1_qvel"][:, :-1]).max(-1) > 1e-5)
    cut = (w["done"][:, :-1] > 0) | jump
    w = {key_: v[:, :-1] for key_, v in w.items()}
    ok = jnp.cumprod(w["ok"], -1)
    w["mask"] = ok * jnp.cumprod(jnp.concatenate((jnp.ones((B, 1)), 1 - cut[:, :-1].astype(jnp.float32)), -1), -1)
    return w


def sample_starts(buf, key, B: int):
    """B starts uniform over the finite transitions (2B candidates, the finite ones first)."""
    ka, ke = jax.random.split(key)
    N = buf["s_qpos"].shape[1]
    age = _uniform_int(ka, buf["size"], 2 * B)
    env = jax.random.randint(ke, (2 * B,), 0, N)
    s = gather(buf, _rows(buf, age), env)
    idx = jnp.argsort(1 - s["ok"], stable=True)[:B]
    return jax.tree.map(lambda x: x[idx], s)


def sanitize(rec_flat):
    """Zero every non-finite float of a transition record and add ok = 1 if its states and context were finite."""
    fin = [jnp.isfinite(v).reshape(v.shape[0], -1).all(-1) for k, v in rec_flat.items() if jnp.issubdtype(v.dtype, jnp.floating)]
    ok = jnp.stack(fin, -1).all(-1)
    out = {k: (jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) if jnp.issubdtype(v.dtype, jnp.floating) else v) for k, v in rec_flat.items()}
    return {**out, "ok": ok.astype(jnp.float32)}
