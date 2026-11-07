"""Networks as pure functions on parameter pytrees.

- World model: XMimic's semi-implicit structured MLP (frame-local increments, drift weight 0.625 for 4 substeps), conditioned on
  the PD error of the step's joint target, on the terrain under the robot (pelvis height and the clearance of every sole
  collider point over the exact terrain surface, as in Terrain-Grounded Imagination) and on the dynamics context xi.
- Policy: an MLP 100 -> 512 -> LayerNorm -> ELU -> 256 -> ELU -> 128 -> ELU -> 29, orthogonal init (gain sqrt 2, last layer
  0.01): the action is Holosoma's raw action (joint target q0 + 0.25 a).
Matmuls run in bf16 with fp32 accumulation; with bf16 outputs (`mlp_apply(..., out_bf16=True)`, as torch autocast) the backward
pass runs on bf16 tensor cores too. Normalizers and state integration stay fp32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .rot import mat_to_6d, mm3, mv3, so3_exp, so3_log, tr3, yaw_mat

BF16 = jnp.bfloat16


def mlp_init(key, sizes, layer_norm=False):
    """torch.nn.Linear's default init (U(+-1/sqrt(fan_in))); weights stored (in, out); LayerNorm on every hidden layer if set."""
    layers = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        kw, kb = jax.random.split(k)
        bound = 1.0 / np.sqrt(i)
        layers.append({"w": jax.random.uniform(kw, (i, o), minval=-bound, maxval=bound), "b": jax.random.uniform(kb, (o,), minval=-bound, maxval=bound)})
    if layer_norm:
        for layer in layers[:-1]:
            layer["ln_g"], layer["ln_b"] = jnp.ones(layer["b"].shape[0]), jnp.zeros(layer["b"].shape[0])
    return layers


def mlp_apply(layers, x, act, ln_eps: float = 1e-5, out_bf16: bool = False):
    h = x
    out = BF16 if out_bf16 else jnp.float32
    for i, layer in enumerate(layers):
        h = jnp.dot(h.astype(BF16), layer["w"].astype(BF16), preferred_element_type=out).astype(jnp.float32) + layer["b"]
        if i == len(layers) - 1:
            return h
        if "ln_g" in layer:
            mu = h.mean(-1, keepdims=True)
            var = ((h - mu) ** 2).mean(-1, keepdims=True)
            h = (h - mu) * jax.lax.rsqrt(var + ln_eps) * layer["ln_g"] + layer["ln_b"]
        h = act(h)


def muon_mask(layers):
    return [{k: (k == "w" and 0 < i < len(layers) - 1) for k in layer} for i, layer in enumerate(layers)]


# ---------------------------------------------------------------- running normalizer
def rn_init(dim: int):
    return {"mean": jnp.zeros(dim), "var": jnp.ones(dim), "count": jnp.zeros(())}


def rn_update(st, x, max_count: float = 2e8, weights=None):
    """Merge the batch statistics (rows with weight 0 left out) until max_count samples, then freeze."""
    x = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
    wt = jnp.ones(x.shape[0]) if weights is None else weights.reshape(-1).astype(jnp.float32)
    n = jnp.maximum(wt.sum(), 1.0)
    m = (x * wt[:, None]).sum(0) / n
    v = (((x - m) ** 2) * wt[:, None]).sum(0) / n
    cnt = st["count"]
    tot = cnt + n
    delta = m - st["mean"]
    mean = st["mean"] + delta * n / tot
    var = (st["var"] * cnt + v * n + delta**2 * cnt * n / tot) / tot
    live = cnt < max_count
    return {"mean": jnp.where(live, mean, st["mean"]), "var": jnp.where(live, var, st["var"]), "count": jnp.where(live, tot, cnt)}


def rn_apply(st, x, eps: float = 1e-2):
    return (x - st["mean"]) / jnp.sqrt(st["var"] + eps**2)


def rn_std(st, eps: float = 1e-2):
    return jnp.sqrt(st["var"] + eps**2)


# ---------------------------------------------------------------- state features
def invariant_features(S, h):
    """Heading- and position-invariant state: pelvis height over the terrain, tilt (6-d rotation relative to the heading),
    body-frame velocities, joint positions and velocities."""
    p, R, v, w, q, qd = S
    Rt = tr3(R)
    return jnp.concatenate((h[..., None], mat_to_6d(mm3(tr3(yaw_mat(R)), R)), mv3(Rt, v), mv3(Rt, w), q, qd), -1)


# ---------------------------------------------------------------- world model
DRIFT_W = 0.625


def wm_init(key, in_dim: int, out_dim: int, hidden):
    layers = mlp_init(key, [in_dim, *hidden, out_dim], layer_norm=True)
    layers[-1]["w"] = jnp.zeros_like(layers[-1]["w"])
    layers[-1]["b"] = jnp.zeros_like(layers[-1]["b"])
    return {"net": layers}, {"in_norm": rn_init(in_dim), "out_norm": rn_init(out_dim), "inc_norm": rn_init(out_dim)}


def wm_inputs(S, h, soles, tgt, xi):
    return jnp.concatenate((invariant_features(S, h), soles, tgt - S[4], xi), -1)


def _drift(S, v1, w1, qd1, dt):
    p, R, v, w, q, qd = S
    a, b = dt * (1 - DRIFT_W), dt * DRIFT_W
    return mv3(tr3(yaw_mat(R)), a * v + b * v1), mv3(tr3(R), a * w + b * w1), a * qd + b * qd1


def increments(S, S1):
    p, R, v, w, q, qd = S
    p1, R1, v1, w1, q1, qd1 = S1
    Rt = tr3(R)
    return jnp.concatenate((mv3(tr3(yaw_mat(R)), p1 - p), so3_log(mm3(Rt, R1)), mv3(Rt, v1 - v), mv3(Rt, w1 - w), q1 - q, qd1 - qd), -1)


def _split(d, J):
    return jnp.split(d, np.cumsum([3, 3, 3, 3, J])[:5], -1)


def wm_targets(S, S1, dt):
    J = S[4].shape[-1]
    dp, dth, dv, dw, dq, dqd = _split(increments(S, S1), J)
    gp, gth, gq = _drift(S, S1[2], S1[3], S1[5], dt)
    return jnp.concatenate((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)


def wm_integrate(S, d, dt):
    p, R, v, w, q, qd = S
    dp, dth, dv, dw, dq, dqd = _split(d, q.shape[-1])
    v1, w1, qd1 = v + mv3(R, dv), w + mv3(R, dw), qd + dqd
    gp, gth, gq = _drift(S, v1, w1, qd1, dt)
    return p + mv3(yaw_mat(R), dp + gp), mm3(R, so3_exp(dth + gth)), v1, w1, q + dq + gq, qd1


def wm_step(params, bufs, S, h, soles, tgt, xi, dt, out_bf16=False):
    """Next state S = (p, R, v, w, q, qd): world-frame p, v and R, world-frame w."""
    y = mlp_apply(params["net"], rn_apply(bufs["in_norm"], wm_inputs(S, h, soles, tgt, xi)), jax.nn.silu, out_bf16=out_bf16)
    return wm_integrate(S, y * rn_std(bufs["out_norm"], 1e-4) + bufs["out_norm"]["mean"], dt)


def state_error(S_hat, S):
    p, R, v, w, q, qd = S
    ph, Rh, vh, wh, qh, qdh = S_hat
    Rt = tr3(R)
    return jnp.concatenate((mv3(tr3(yaw_mat(R)), ph - p), so3_log(mm3(Rt, Rh)), mv3(Rt, vh - v), mv3(Rt, wh - w), qh - q, qdh - qd), -1)


# ---------------------------------------------------------------- policy
def pi_init(key, obs_dim: int, act_dim: int):
    orth = jax.nn.initializers.orthogonal
    sizes = [obs_dim, 512, 256, 128, act_dim]
    gains = [np.sqrt(2), np.sqrt(2), np.sqrt(2), 0.01]
    layers = [{"w": orth(g)(k, (i, o)), "b": jnp.zeros(o)} for k, (i, o), g in zip(jax.random.split(key, 4), zip(sizes[:-1], sizes[1:]), gains)]
    layers[0]["ln_g"], layers[0]["ln_b"] = jnp.ones(512), jnp.zeros(512)
    return layers


def pi_apply(layers, o, out_bf16=False):
    return mlp_apply(layers, o, jax.nn.elu, ln_eps=1e-6, out_bf16=out_bf16)
