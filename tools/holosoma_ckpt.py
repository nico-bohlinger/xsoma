"""Convert Holosoma G1 locomotion checkpoints (PPO model_*.pt, FastSAC model_*.pt) into plain arrays for xsoma/evaluate.py.

Runs in Holosoma's environment (torch): python tools/holosoma_ckpt.py <model.pt> [...] -> <model>.npz next to each checkpoint,
with the checkpoint's wall-clock time since its run's launch (file mtime minus <run>/t_start) when the run directory has one.
  PPO      actor mean: Linear 100-512-256-128-29 with ELU (agents/modules/ppo_modules.py act_inference); no observation normalizer
  FastSAC  (obs - mean) / (std + 0.01), Linear-LayerNorm-SiLU x 3 (512, 256, 128), action = tanh(fc_mu) * action_scale +
           action_bias (agents/fast_sac/fast_sac.py forward, fast_sac_agent.py get_inference_policy); the actor concatenates its
           observation terms in `actor_obs_keys` order, checked here to be the whole observation group in its own (alphabetical) order
"""

import os
import sys

import numpy as np
import torch

ACTOR_TERMS = ["actions", "base_ang_vel", "command_ang_vel", "command_lin_vel", "cos_phase", "dof_pos", "dof_vel", "projected_gravity", "sin_phase"]


def convert(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    if "actor_model_state_dict" in ck:  # PPO
        sd = ck["actor_model_state_dict"]
        assert ck.get("actor_obs_normalizer_state_dict") is None, "PPO run with an observation normalizer"
        out["kind"] = np.asarray("ppo")
        for i, m in enumerate((0, 2, 4, 6)):
            out[f"w{i}"] = sd[f"actor_module.module.{m}.weight"].numpy().T
            out[f"b{i}"] = sd[f"actor_module.module.{m}.bias"].numpy()
    else:  # FastSAC
        sd = ck["actor_state_dict"]
        keys = list(ck["args"].get("actor_obs_keys") or [])
        assert keys in ([], ["actor_obs"], ACTOR_TERMS), f"actor_obs_keys {keys}: not the whole observation in its own order"
        out["kind"] = np.asarray("fast_sac")
        for i, (lin, ln) in enumerate(((0, 1), (3, 4), (6, 7))):
            out[f"w{i}"], out[f"b{i}"] = sd[f"net.{lin}.weight"].numpy().T, sd[f"net.{lin}.bias"].numpy()
            out[f"g{i}"], out[f"h{i}"] = sd[f"net.{ln}.weight"].numpy(), sd[f"net.{ln}.bias"].numpy()
        out["w3"], out["b3"] = sd["fc_mu.0.weight"].numpy().T, sd["fc_mu.0.bias"].numpy()
        out["scale"], out["bias"] = sd["action_scale"].numpy(), sd["action_bias"].numpy()
        norm = ck["obs_normalizer_state"]
        out["mean"], out["std"] = norm["_mean"].numpy()[0], norm["_std"].numpy()[0]
        out["normalize"] = np.asarray(bool(ck["args"].get("obs_normalization", True)))
    out["iteration"] = np.asarray(int(ck.get("iteration", ck.get("iter", -1))))
    run = os.path.dirname(os.path.abspath(path))
    while run != "/" and not os.path.exists(os.path.join(run, "t_start")):  # the run directory of tools/run_holosoma.sh
        run = os.path.dirname(run)
    if run != "/":  # wall-clock time since the run's launch at which the checkpoint was written
        out["wall_s"] = np.asarray(os.path.getmtime(path) - float(open(os.path.join(run, "t_start")).read().split()[0]))
    dst = path[: -len(".pt")] + ".npz"
    np.savez(dst, **out)
    print(f"{path} -> {dst} ({out['kind']}, iteration {int(out['iteration'])})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
