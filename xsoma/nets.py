"""Networks as pure functions on parameter pytrees: the world model and the policy.

A robot state is a tuple S = (p, R, v, w, q, qd): pelvis position and rotation, the pelvis's linear and angular velocity in the
world frame, and the joint positions and velocities.

- World model: XMimic's semi-implicit structured MLP. From a state and the PD target of the step it predicts the state one policy
  step later. Its input does not depend on where the robot stands or which way it faces: the state in the robot's own frames, the
  PD error of the step's joint target, the terrain under the robot (pelvis height and the clearance of every sole collider point
  over the exact terrain surface, as in Terrain-Grounded Imagination) and the dynamics context xi of the env. Its output is a set
  of changes in the robot's own frames: the velocities change by the predicted amount, and the position, rotation and joint angles
  advance by the drift of their velocities (weight 0.625 on the new velocity, for 4 substeps) plus a predicted correction.
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
    """Apply an MLP: every layer but the last is Linear, then LayerNorm if the layer has `ln_g`, then `act`; the last layer is
    Linear only. Matmuls take bf16 inputs and accumulate in fp32; `out_bf16` makes their outputs bf16 too (see above)."""
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
    """Which leaves train with Muon: the weight matrices of the layers between the first and the last, not the biases or the
    LayerNorm parameters."""
    return [{k: (k == "w" and 0 < i < len(layers) - 1) for k in layer} for i, layer in enumerate(layers)]


# ---------------------------------------------------------------- running normalizer
# mean, variance and sample count of all data seen so far, merged batch by batch (not trained by gradients)
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
    """Normalize x with the running mean and standard deviation; eps keeps the division stable for dimensions that hardly vary."""
    return (x - st["mean"]) / jnp.sqrt(st["var"] + eps**2)


def rn_std(st, eps: float = 1e-2):
    """The running standard deviation, with the same eps floor."""
    return jnp.sqrt(st["var"] + eps**2)


# ---------------------------------------------------------------- state features
def invariant_features(S, h):
    """Heading- and position-invariant state: pelvis height over the terrain, tilt (6-d rotation relative to the heading),
    body-frame velocities, joint positions and velocities."""
    p, R, v, w, q, qd = S
    Rt = tr3(R)
    return jnp.concatenate((h[..., None], mat_to_6d(mm3(tr3(yaw_mat(R)), R)), mv3(Rt, v), mv3(Rt, w), q, qd), -1)


# ---------------------------------------------------------------- world model
DRIFT_W = 0.625  # weight of the new velocity in the drift: (n + 1) / (2 n) for n = 4 semi-implicit physics substeps


def wm_init(key, in_dim: int, out_dim: int, hidden):
    """World-model parameters and normalizers. The last layer starts at zero, so the untrained network predicts the mean change
    of the data (the output normalizer's mean)."""
    layers = mlp_init(key, [in_dim, *hidden, out_dim], layer_norm=True)
    layers[-1]["w"] = jnp.zeros_like(layers[-1]["w"])
    layers[-1]["b"] = jnp.zeros_like(layers[-1]["b"])
    return {"net": layers}, {"in_norm": rn_init(in_dim), "out_norm": rn_init(out_dim), "inc_norm": rn_init(out_dim)}


def wm_inputs(S, h, soles, tgt, xi):
    """The network's input: the invariant state, the sole clearances, the PD error of the target (joint target minus q) and xi."""
    return jnp.concatenate((invariant_features(S, h), soles, tgt - S[4], xi), -1)


def _drift(S, v1, w1, qd1, dt):
    """The displacement over dt that the velocities alone explain, with weight DRIFT_W on the new velocity: position in the
    heading frame, rotation vector in the pelvis frame, joint angles."""
    p, R, v, w, q, qd = S
    a, b = dt * (1 - DRIFT_W), dt * DRIFT_W
    return mv3(tr3(yaw_mat(R)), a * v + b * v1), mv3(tr3(R), a * w + b * w1), a * qd + b * qd1


def increments(S, S1):
    """The change from S to S1 in the robot's own frames: position (heading frame), rotation (rotation vector in the pelvis
    frame), velocities (pelvis frame), q and qd."""
    p, R, v, w, q, qd = S
    p1, R1, v1, w1, q1, qd1 = S1
    Rt = tr3(R)
    return jnp.concatenate((mv3(tr3(yaw_mat(R)), p1 - p), so3_log(mm3(Rt, R1)), mv3(Rt, v1 - v), mv3(Rt, w1 - w), q1 - q, qd1 - qd), -1)


def _split(d, J):
    """Split a change vector of size 12 + 2 J into dp, dth, dv, dw, dq, dqd."""
    return jnp.split(d, np.cumsum([3, 3, 3, 3, J])[:5], -1)


def wm_targets(S, S1, dt):
    """What the network learns to predict: the increments from S to S1, with the drift of the new velocities taken off the
    position, rotation and joint angles."""
    J = S[4].shape[-1]
    dp, dth, dv, dw, dq, dqd = _split(increments(S, S1), J)
    gp, gth, gq = _drift(S, S1[2], S1[3], S1[5], dt)
    return jnp.concatenate((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)


def wm_integrate(S, d, dt):
    """The next state from the network's output d (in real units): the velocities add their predicted changes, then position,
    rotation and joint angles advance by the drift of the new velocities plus their predicted corrections."""
    p, R, v, w, q, qd = S
    dp, dth, dv, dw, dq, dqd = _split(d, q.shape[-1])
    v1, w1, qd1 = v + mv3(R, dv), w + mv3(R, dw), qd + dqd
    gp, gth, gq = _drift(S, v1, w1, qd1, dt)
    return p + mv3(yaw_mat(R), dp + gp), mm3(R, so3_exp(dth + gth)), v1, w1, q + dq + gq, qd1


def wm_step(params, bufs, S, h, soles, tgt, xi, dt, out_bf16=False):
    """The next state after dt, from the state S, the pelvis height h over the terrain, the sole clearances, the PD target tgt
    of the joints and the context xi. States are (p, R, v, w, q, qd) with p, v and w in the world frame. The network works on
    normalized inputs and outputs."""
    y = mlp_apply(params["net"], rn_apply(bufs["in_norm"], wm_inputs(S, h, soles, tgt, xi)), jax.nn.silu, out_bf16=out_bf16)
    return wm_integrate(S, y * rn_std(bufs["out_norm"], 1e-4) + bufs["out_norm"]["mean"], dt)


def state_error(S_hat, S):
    """The error of a predicted state S_hat against the true S in the frames of `increments`: position in the true heading
    frame, rotation as a rotation vector, velocities in the true pelvis frame, q and qd."""
    p, R, v, w, q, qd = S
    ph, Rh, vh, wh, qh, qdh = S_hat
    Rt = tr3(R)
    return jnp.concatenate((mv3(tr3(yaw_mat(R)), ph - p), so3_log(mm3(Rt, Rh)), mv3(Rt, vh - v), mv3(Rt, wh - w), qh - q, qdh - qd), -1)


# ---------------------------------------------------------------- policy
def pi_init(key, obs_dim: int, act_dim: int):
    """Policy parameters: layers of 512, 256 and 128 units, orthogonal weights, zero biases, a LayerNorm after the first layer."""
    orth = jax.nn.initializers.orthogonal
    sizes = [obs_dim, 512, 256, 128, act_dim]
    gains = [np.sqrt(2), np.sqrt(2), np.sqrt(2), 0.01]
    layers = [{"w": orth(g)(k, (i, o)), "b": jnp.zeros(o)} for k, (i, o), g in zip(jax.random.split(key, 4), zip(sizes[:-1], sizes[1:]), gains)]
    layers[0]["ln_g"], layers[0]["ln_b"] = jnp.ones(512), jnp.zeros(512)
    return layers


def pi_apply(layers, o, out_bf16=False):
    """The policy's raw action for the actor observation `o`: Holosoma's action, before its 0.25 scale and the default pose."""
    return mlp_apply(layers, o, jax.nn.elu, ln_eps=1e-6, out_bf16=out_bf16)
