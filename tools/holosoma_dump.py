"""Ground truth from Holosoma's own G1 29-DoF velocity-tracking environment, to check XSoma's JAX task term by term.

Holosoma = amazon-far/holosoma @ 79ba714, unmodified. The script builds Holosoma's environment the way train_agent.py does
(same CLI and config parser, same seeding, `LeggedRobotLocomotionManager`), calls `reset_all()` as PPO does before training,
then runs the policy steps of `--schedule` (a fixed action schedule) and records, per step t:

  pre_*    the state the action is chosen in (after step t-1's resets, command update, pushes; pre_*[0] = after reset_all)
  sub_*    every physics substep: MuJoCo state entering it and the torque in data.ctrl (Holosoma flushes its pending state
           writes, e.g. a push, before the first substep, so sub_qpos[t, 0] is exactly the state the physics starts from)
  post_*   the post-physics state before resets, when Holosoma computes terminations and rewards (raw reward terms, weights,
           total, termination terms, foot heights, body poses, contact forces)
  next_*   after the step: resets, command / curriculum / push updates and the observations returned by env.step
Everything is read from Holosoma's own attributes; recorders wrap a few methods on instances at runtime (the backend's mj_step,
the reward and termination term functions, reset_envs_idx, _push_robots, the actor observation noise) and call the originals
unchanged, so the run is Holosoma's. Holosoma's files are never written.

Default setup: Holosoma's classic CPU MuJoCo backend (`simulator:mujoco`, one env, documented as its eval/sim2sim backend) on
the plane terrain (`terrain:terrain-locomotion-plane`, a MuJoCo plane named floor, so foot heights = foot z); the PPO preset
`exp:g1-29dof`. Run it in Holosoma's environment (one CPU process, ~20 s), with a PPO checkpoint converted by
tools/holosoma_ckpt.py:

  WANDB_MODE=disabled python -u tools/holosoma_dump.py --out out/g1_ppo_plane --policy model_00700.npz -- \
    exp:g1-29dof simulator:mujoco terrain:terrain-locomotion-plane logger:disabled --logger.base-dir=out/hs \
    --training.num-envs=1 --training.seed=42

Everything after `--` is Holosoma's own train_agent.py CLI. Output: <out>.npz (arrays stacked over steps; Holosoma's own
tensors keep their float32, MuJoCo's arrays float64) and <out>.json (CLI, versions, orders, conventions, per-array docs, a
summary of resets / pushes). The full resolved config is stored in the npz as `config_json`.

Action schedule (`--schedule`, phases of <kind>:<steps>): `policy` = the mean action of a Holosoma checkpoint (`--policy`, an
.npz from tools/holosoma_ckpt.py; tests/data used PPO's model_00700) on the actor observation env.step returned, plus
optional N(0, policy_noise^2); `noise` = N(0, sigma^2) i.i.d. per joint (sigma 0.5); `zero`. Seeded numpy, independent of
Holosoma's RNGs. Open-loop noise alone cannot keep the G1 up (it falls within ~1 s), so pushes (250-500 steps into an
episode), the 500-step command resampling and time-outs need the policy; the default "policy:1050,noise:300,policy:150"
gives a long walking episode (pushes, resamples, a time-out if it survives), then repeated falls, then recovery walking.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True, help="output path without extension (writes <out>.npz and <out>.json)")
ap.add_argument("--schedule", default="policy:1050,noise:300,policy:150",
                help="comma-separated <kind>:<steps> phases; kinds: policy (--policy's mean action + N(0, policy_noise^2)), "
                "noise (N(0, sigma^2)), zero")
ap.add_argument("--policy", default=None, help="Holosoma checkpoint converted by tools/holosoma_ckpt.py (.npz, PPO or FastSAC)")
ap.add_argument("--policy-noise", type=float, default=0.0, help="std of i.i.d. noise added to the policy's mean action")
ap.add_argument("--sigma", type=float, default=0.5, help="std of the noise phases' i.i.d. actions")
ap.add_argument("--action-seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=2, help="torch CPU threads")
ap.add_argument("--mjb", action="store_true", help="also store the compiled MuJoCo model (after start-up randomization) as bytes")
ap.add_argument("holosoma_args", nargs=argparse.REMAINDER, help="-- followed by Holosoma's train_agent.py CLI")
args = ap.parse_args()
HOLO_ARGS = [a for a in args.holosoma_args if a != "--"] if args.holosoma_args else []
if not HOLO_ARGS:
    HOLO_ARGS = ["exp:g1-29dof", "simulator:mujoco", "terrain:terrain-locomotion-plane", "logger:disabled", "--training.num-envs=1"]

import torch  # noqa: E402
from loguru import logger  # noqa: E402

torch.set_num_threads(args.threads)
logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "WARNING"))

import mujoco  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.config_values.experiment import get_annotated_experiment_config  # noqa: E402
from holosoma.utils.common import seeding  # noqa: E402
from holosoma.utils.config_registry import parse_config  # noqa: E402
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402


def np32(x):
    return x.detach().cpu().numpy().astype(np.float32, copy=True)


def npi(x):
    return x.detach().cpu().numpy().astype(np.int64, copy=True)


# ------------------------------------------------------------------ Holosoma's environment, as train_agent.train builds it
t_start = time.time()
cfg = parse_config(get_annotated_experiment_config, args=HOLO_ARGS)
init_sim_imports(cfg)
seed = cfg.training.seed
seeding(seed, torch_deterministic=cfg.training.torch_deterministic)
device = "cpu" if not torch.cuda.is_available() else "cuda:0"
env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device=device)
assert env.num_envs == 1, "this dump records one environment (env 0); use --training.num-envs=1"
sim = env.simulator
backend = sim.backend
model, data = sim.root_model, sim.root_data
t_built = time.time()

# MuJoCo addressing (the robot's free joint, then its DoFs in Holosoma's config order)
QA, VA = sim.object_addrs["robot"]["qpos_addr"], sim.object_addrs["robot"]["qvel_addr"]
DOF_Q, DOF_V = list(sim.dof_qpos_addrs), list(sim.dof_qvel_addrs)
ACT = list(sim._dof_actuator_ids)
QIDX = np.r_[np.arange(QA, QA + 7), DOF_Q]
VIDX = np.r_[np.arange(VA, VA + 6), DOF_V]


def mj_state():
    return data.qpos[QIDX].copy(), data.qvel[VIDX].copy()


rm, tm, om, cm, am = env.reward_manager, env.termination_manager, env.observation_manager, env.command_manager, env.action_manager
cur = env.curriculum_manager
REWARD_TERMS = list(rm._term_names)
TERM_TERMS = list(tm._term_names)
assert not rm._term_instances and not tm._term_instances, "stateful reward / termination terms are not recorded"
action_term = next(iter(am._term_instances.values()))
gait = cm.get_state("locomotion_gait")
push_state = env.randomization_manager.get_state("push_randomizer_state")
penalty = cur.get_term("penalty_curriculum")
tracker = cur.get_term("average_episode_tracker")
terrain_state = env.terrain_manager.get_state("locomotion_terrain")

# ------------------------------------------------------------------ recorders (runtime wrappers that call Holosoma's originals)
REC = {"in_step": False, "sub": defaultdict(list), "post": None, "rew": {}, "term": {}, "noise": {}, "push": None}


def _wrap_backend_step():
    orig = backend.step

    def step():
        if REC["in_step"]:
            q, v = mj_state()
            REC["sub"]["qpos"].append(q)
            REC["sub"]["qvel"].append(v)
            REC["sub"]["ctrl"].append(data.ctrl[ACT].copy())
        orig()

    backend.step = step


def _wrap_terms(manager, key):
    """Record each term function's output (REC[key][term]) during env.step."""
    for name in list(manager._term_funcs):
        f = manager._term_funcs[name]

        def wrapped(env_, _f=f, _n=name, **kw):
            out = _f(env_, **kw)
            if REC["in_step"]:
                REC[key][_n] = out.detach().clone()
            return out

        manager._term_funcs[name] = wrapped


def _wrap_reward_compute():
    orig = rm.compute

    def compute(dt):
        if REC["in_step"]:
            REC["rew_weight"] = [float(c.weight) for c in rm._term_cfgs]
            REC["rew_dt"] = float(dt)
        return orig(dt)

    rm.compute = compute


def _capture_post_physics():
    """Post-physics state before resets: what Holosoma's terminations and rewards just saw."""
    q, v = mj_state()
    rs = sim.robot_root_states
    p = {
        "post_qpos": q,
        "post_qvel": v,
        "post_root_state": np32(rs[0]),
        "post_dof_pos": np32(sim.dof_pos[0]),
        "post_dof_vel": np32(sim.dof_vel[0]),
        "post_body_pos": np32(sim._rigid_body_pos[0]),
        "post_body_quat": np32(sim._rigid_body_rot[0]),
        "post_body_lin_vel": np32(sim._rigid_body_vel[0]),
        "post_body_ang_vel": np32(sim._rigid_body_ang_vel[0]),
        "post_contact_forces": np32(sim.contact_forces[0]),
        "post_ncon": np.int64(data.ncon),
        "post_feet_heights": np32(terrain_state.feet_heights[0]),
        "post_feet_ray_hits": np32(terrain_state._ray_hits_world_feet[0]),
        "post_base_height": np.float32(terrain_state.base_heights[0].item()),
        "post_episode_length": np.int64(env.episode_length_buf[0].item()),
        "post_command": np32(cm.commands[0]),
        "post_phase": np32(gait.phase[0]),
        "post_action": np32(am.action[0]),
        "post_prev_action": np32(am.prev_action[0]),
        "action_applied": np32(action_term._actions_after_delay[0]),
        "sub_torque": np32(action_term.torques_substep[0]),
        "reward": np.float32(env.rew_buf[0].item()),
        "reward_raw": np.asarray([REC["rew"][n][0].item() for n in REWARD_TERMS], np.float32),
        "reward_weight": np.asarray(REC["rew_weight"], np.float64),
        "reset": bool(env.reset_buf[0].item()),
        "time_out": bool(env.time_out_buf[0].item()),
    }
    for n in TERM_TERMS:
        p["term_" + n] = bool(REC["term"][n][0].item())
    REC["post"] = p


def _wrap_reset():
    orig = env.reset_envs_idx

    def reset_envs_idx(env_ids, *a, **kw):
        if REC["in_step"] and REC["post"] is None:
            _capture_post_physics()
        return orig(env_ids, *a, **kw)

    env.reset_envs_idx = reset_envs_idx


def _wrap_push():
    orig = env._push_robots

    def push_robots(env_ids):
        orig(env_ids)
        if REC["in_step"] and len(env_ids) > 0:
            REC["push"] = np32(env.push_robot_vel_buf[0])

    env._push_robots = push_robots


def _wrap_gait_step():
    """The command the gait's stand rule reads and the phase_dt the phase uses (after resets, before this step's command
    resampling, which also resamples the frequency)."""
    orig = gait.step

    def step():
        if REC["in_step"]:
            REC["gait_cmd"] = np32(cm.commands[0])
            REC["gait_phase_dt"] = np.float32(gait.phase_dt[0, 0].item())
        orig()

    gait.step = step


def _wrap_noise():
    for name, term in om._terms["actor_obs"].items():
        orig = term.generate_noise

        def gen(obs, generator, _o=orig, _n=name):
            out = _o(obs, generator)
            REC["noise"][_n] = (out - obs).detach().clone()
            return out

        term.generate_noise = gen


def task_state(prefix):
    """Holosoma's task state between steps (commands, gait, counters, randomized gains, curriculum, push schedule)."""
    return {
        prefix + "root_state": np32(sim.robot_root_states[0]),
        prefix + "dof_pos": np32(sim.dof_pos[0]),
        prefix + "dof_vel": np32(sim.dof_vel[0]),
        prefix + "command": np32(cm.commands[0]),
        prefix + "phase": np32(gait.phase[0]),
        prefix + "gait_freq": np.float32(gait.gait_freq[0, 0].item()),
        prefix + "phase_dt": np.float32(gait.phase_dt[0, 0].item()),
        prefix + "phase_offset": np32(gait.phase_offset[0]),
        prefix + "episode_length": np.int64(env.episode_length_buf[0].item()),
        prefix + "action_delay_idx": np.int64(env.action_delay_idx[0].item()),
        prefix + "action_queue": np32(action_term.action_queue[0]),
        prefix + "kp_scale": np32(action_term._kp_scale[0]),
        prefix + "kd_scale": np32(action_term._kd_scale[0]),
        prefix + "last_action": np32(am.action[0]),
        prefix + "penalty_scale": np.float64(penalty.current_scale),
        prefix + "avg_episode_length": np.float32(float(tracker.get_average())),
        prefix + "push_counter": np.int64(push_state.push_robot_counter[0].item()),
        prefix + "push_interval_s": np.float32(push_state.push_interval_s[0].item()),
        prefix + "push_interval_steps": np.int64(int((push_state.push_interval_s / env.dt).to(torch.int)[0].item())),
    }


# ------------------------------------------------------------------ static information (after start-up randomization)
body_names = list(sim.body_names)
dof_names = list(sim.dof_names)


def body_list(idx):
    return [body_names[int(i)] for i in idx.tolist()]


def obs_layout(group):
    """Terms in Holosoma's concatenation order (sorted names) with their slices, scales and noise (raw units)."""
    gcfg = om.cfg.groups[group]
    out, off = [], 0
    for name in sorted(gcfg.terms):
        t = gcfg.terms[name]
        dim = int(om._terms[group][name].shape(env)[0])
        out.append({"term": name, "start": off, "dim": dim, "scale": float(t.scale), "noise": float(t.noise) if gcfg.enable_noise else 0.0})
        off += dim
    return out, off


body_ids = list(sim.body_ids)
static = {
    "dof_names": np.asarray(dof_names),
    "body_names": np.asarray(body_names),
    "reward_terms": np.asarray(REWARD_TERMS),
    "feet_indices": npi(env.feet_indices),
    "feet_height_indices": npi(env.feet_height_indices),
    "termination_contact_indices": npi(env.termination_contact_indices),
    "default_dof_pos": np32(env.default_dof_pos[0]),
    "p_gains": np32(action_term.p_gains),
    "d_gains": np32(action_term.d_gains),
    "action_scales": np32(action_term.action_scales),
    "torque_limits": np32(env.torque_limits),
    "env_origin": np32(terrain_state.env_origins[0]),
    "mj_body_ids": np.asarray(body_ids, np.int64),
    "mj_body_mass": model.body_mass[body_ids].copy(),
    "mj_body_inertia": model.body_inertia[body_ids].copy(),
    "mj_body_ipos": model.body_ipos[body_ids].copy(),
    "mj_geom_names": np.asarray([model.geom(i).name for i in range(model.ngeom)]),
    "mj_geom_bodyid": model.geom_bodyid.copy(),
    "mj_geom_friction": model.geom_friction.copy(),
    "mj_actuator_ctrlrange": model.actuator_ctrlrange[ACT].copy(),
    "mj_pair_geoms": np.stack([model.pair_geom1, model.pair_geom2], -1) if model.npair else np.zeros((0, 2), np.int64),
    "mj_pair_friction": model.pair_friction.copy(),
    "mj_pair_solref": model.pair_solref.copy(),
    "mj_pair_solimp": model.pair_solimp.copy(),
    "config_json": np.asarray(json.dumps(cfg.to_serializable_dict(), indent=1)),
}
if args.mjb:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mjb") as f:
        mujoco.mj_saveModel(model, f.name, None)
        static["mjb"] = np.frombuffer(open(f.name, "rb").read(), np.uint8)

# ------------------------------------------------------------------ run
_wrap_backend_step()
_wrap_terms(rm, "rew")
_wrap_terms(tm, "term")
_wrap_reward_compute()
_wrap_reset()
_wrap_push()
_wrap_gait_step()
_wrap_noise()


def load_policy(path):
    """Mean action of a Holosoma actor from tools/holosoma_ckpt.py arrays: PPO (Linear-ELU MLP, no normalizer) or FastSAC
    (normalized input, Linear-LayerNorm-SiLU x 3, tanh(mu) * scale + bias)."""
    z = {k: torch.as_tensor(v) for k, v in np.load(path).items() if k not in ("kind",)}
    kind = str(np.load(path)["kind"])
    F = torch.nn.functional

    def act(o):
        x = torch.as_tensor(o, dtype=torch.float32)
        with torch.no_grad():
            if kind == "ppo":
                for i in range(3):
                    x = F.elu(x @ z[f"w{i}"] + z[f"b{i}"])
                return (x @ z["w3"] + z["b3"]).numpy()
            if bool(z["normalize"]):
                x = (x - z["mean"]) / (z["std"] + 0.01)
            for i in range(3):
                x = F.silu(F.layer_norm(x @ z[f"w{i}"] + z[f"b{i}"], (z[f"g{i}"].shape[0],), z[f"g{i}"], z[f"h{i}"]))
            return (torch.tanh(x @ z["w3"] + z["b3"]) * z["scale"] + z["bias"]).numpy()

    return act, kind, int(np.load(path)["iteration"])


KINDS = {"policy": 0, "noise": 1, "zero": 2}
schedule = []
for phase in args.schedule.split(","):
    kind, n = phase.split(":")
    assert kind in KINDS, f"unknown schedule kind {kind}"
    schedule += [kind] * int(n)
policy, policy_kind, policy_iter = load_policy(args.policy) if args.policy else (None, None, None)
assert policy is not None or "policy" not in schedule, "--schedule has policy phases: pass --policy"

obs0 = env.reset_all()
init = {"init_actor_obs": np32(obs0["actor_obs"][0]), "init_critic_obs": np32(obs0["critic_obs"][0])}
actor_layout, actor_dim = obs_layout("actor_obs")
critic_layout, critic_dim = obs_layout("critic_obs")
assert actor_dim == init["init_actor_obs"].shape[0] and critic_dim == init["init_critic_obs"].shape[0]
rng = np.random.RandomState(args.action_seed)
J = env.num_dof
rows = defaultdict(list)
actor_obs = init["init_actor_obs"][None]
t_loop = time.time()
for t, kind in enumerate(schedule):
    mean = policy(actor_obs).astype(np.float32) if kind == "policy" else np.zeros((1, J), np.float32)
    if kind == "policy":
        a = mean + args.policy_noise * rng.standard_normal((1, J))
    elif kind == "noise":
        a = args.sigma * rng.standard_normal((1, J))
    else:
        a = np.zeros((1, J))
    a = a.astype(np.float32)
    pre = task_state("pre_")
    REC.update(in_step=True, sub=defaultdict(list), post=None, rew={}, term={}, noise={}, push=None, gait_cmd=None, gait_phase_dt=None)
    obs, rew, reset, extras = env.step({"actions": torch.as_tensor(a, device=env.device)})
    REC["in_step"] = False
    assert REC["post"] is not None and len(REC["sub"]["qpos"]) == sim.simulator_config.sim.control_decimation_steps
    row = dict(pre)
    row["pre_qpos"], row["pre_qvel"] = REC["sub"]["qpos"][0], REC["sub"]["qvel"][0]
    row["action"] = a[0]
    row["action_kind"] = np.int64(KINDS[kind])
    row["policy_mean"] = mean[0]
    row["sub_qpos"] = np.stack(REC["sub"]["qpos"])
    row["sub_qvel"] = np.stack(REC["sub"]["qvel"])
    row["sub_ctrl"] = np.stack(REC["sub"]["ctrl"])
    row.update(REC["post"])
    row["actor_obs"], row["critic_obs"] = np32(obs["actor_obs"][0]), np32(obs["critic_obs"][0])
    noise = np.zeros(actor_dim, np.float32)
    for e in actor_layout:
        if e["term"] in REC["noise"]:
            noise[e["start"]:e["start"] + e["dim"]] = np32(REC["noise"][e["term"]][0])
    row["actor_noise"] = noise
    row.update(task_state("next_"))
    row["next_gait_command"] = REC["gait_cmd"]
    row["next_gait_phase_dt"] = REC["gait_phase_dt"]
    final = extras.get("final_observations", {}) if row["time_out"] else {}
    row["final_actor_obs"] = np32(final["actor_obs"][0]) if final else np.zeros(actor_dim, np.float32)
    row["final_critic_obs"] = np32(final["critic_obs"][0]) if final else np.zeros(critic_dim, np.float32)
    row["pushed"] = REC["push"] is not None
    row["push_vel"] = REC["push"] if REC["push"] is not None else np.zeros(2, np.float32)
    for k, val in row.items():
        rows[k].append(val)
    actor_obs = row["actor_obs"][None]
t_end = time.time()

arrays = {k: np.stack([np.asarray(v) for v in vals]) for k, vals in rows.items()}
arrays.update(init)
arrays.update(static)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez_compressed(args.out + ".npz", **arrays)

# ------------------------------------------------------------------ metadata
resets = np.nonzero(arrays["reset"])[0].tolist()
gcfg = cfg.simulator.config.sim
DOC = {
    "pre_qpos": "MuJoCo qpos the physics starts from (read at the first substep, after Holosoma flushed pending writes): free joint "
    "(x, y, z, qw, qx, qy, qz; world), then the 29 joints in dof_names order",
    "pre_qvel": "MuJoCo qvel likewise: free joint (vx, vy, vz world-frame velocity of the pelvis frame origin; wx, wy, wz in the "
    "pelvis BODY frame, MuJoCo's convention), then joint velocities",
    "pre_root_state": "Holosoma's robot_root_states before the step (float32): pos, quat xyzw, linear velocity world, angular velocity "
    "WORLD frame (Holosoma rotates MuJoCo's body-frame free-joint angular velocity to world on read)",
    "pre_dof_pos / pre_dof_vel": "Holosoma's dof_pos / dof_vel (dof_names order)",
    "pre_command": "[vx, vy, wz] body-frame command the policy saw (command_manager.commands)",
    "pre_phase": "gait phase (left, right) the policy saw; gait_freq Hz, phase_dt = 2 pi dt f, phase_offset (left, right)",
    "pre_episode_length": "episode_length_buf before the step (0 right after a reset)",
    "pre_action_delay_idx": "action delay in policy steps (0 or 1); pre_action_queue[0] newest clipped action, [1] previous",
    "pre_kp_scale / pre_kd_scale": "PD gain scales of the episode (multiply p_gains / d_gains)",
    "pre_last_action": "action_manager.action before the step = previous raw action (zero after a reset)",
    "pre_penalty_scale": "penalty curriculum scale (multiplies the penalty_curriculum-tagged weights at the step's reward)",
    "pre_avg_episode_length": "average episode length of the curriculum tracker",
    "pre_push_counter / pre_push_interval_s / pre_push_interval_steps": "push schedule: a push fires in the step whose incremented "
    "counter equals int(interval_s / dt)",
    "action": "raw action a_t given to env.step (float32)",
    "action_kind / policy_mean": "schedule phase of the step (0 policy, 1 noise, 2 zero) and the policy's mean action (zero "
    "outside policy phases); the policy reads actor_obs of the previous step (init_actor_obs at t = 0)",
    "action_applied": "after clip(+-100) and the delay queue: the PD law's action input during the step",
    "sub_qpos / sub_qvel": "(T, 4, .) MuJoCo state entering each of the 4 substeps (same layout as pre_qpos / pre_qvel)",
    "sub_ctrl": "(T, 4, 29) data.ctrl at each mj_step (dof order): Holosoma's PD torque clipped to torque_limits; MuJoCo then "
    "clamps it to mj_actuator_ctrlrange (hip roll 88 N m)",
    "sub_torque": "Holosoma's own torques_substep (float32 copy of the same torques)",
    "post_qpos / post_qvel": "MuJoCo state after the 4 substeps, before resets (rewards and terminations are computed on it)",
    "post_root_state / post_dof_pos / post_dof_vel": "Holosoma's served copies of it (float32; angular velocity world frame)",
    "post_body_pos / post_body_quat / post_body_lin_vel / post_body_ang_vel": "(T, nbody, .) Holosoma's _rigid_body_* in body_names "
    "order: world positions of the body frames, quaternions xyzw, world linear velocity of the body frame origin, world angular "
    "velocity (MuJoCo mj_objectVelocity, world orientation)",
    "post_contact_forces": "(T, nbody, 3) world-frame net contact force per body after the last substep (Holosoma's "
    "contact_forces; the contact termination reads termination_contact_indices of it)",
    "post_feet_heights": "terrain manager feet_heights (left, right): z of feet_height_indices bodies (foot contact points) minus the "
    "ray-cast terrain height; post_feet_ray_hits the hit points",
    "post_command / post_phase": "command and phase at reward time (= pre_*: the ones the policy saw)",
    "post_action / post_prev_action": "action manager's action and prev_action at reward time (a_t, a_{t-1})",
    "reward_raw": "(T, 10) unweighted reward terms in reward_terms order (Holosoma's term functions' outputs)",
    "reward_weight": "(T, 10) the weights used (config weight x penalty scale for tagged terms), without dt",
    "reward": "Holosoma's total reward rew_buf = sum_i reward_weight_i * dt * reward_raw_i",
    "term_contact / term_timeout": "termination terms (contact force > 1 N on a termination body; episode_length > 1000)",
    "reset / time_out": "reset_buf (contact or timeout) and time_out_buf of the step",
    "actor_obs / critic_obs": "observations returned by env.step (after resets, command update, pushes)",
    "actor_noise": "noise added to the actor observation in raw units before scaling: actor_obs = clip((raw + noise) * scale, "
    "+-100); critic_obs is noise free",
    "next_*": "task state after the step (same meaning as pre_*; next_*[t] = pre_*[t+1] except pre_qpos / pre_qvel, which are "
    "MuJoCo's state after Holosoma's pending writes are flushed; next_root_state already holds a push's velocity)",
    "next_gait_command / next_gait_phase_dt": "the command the gait step's stand rule read and the phase_dt it used (after this "
    "step's resets, before its command resampling: on a reset step the reset's draw, which the second draw in "
    "command_manager.step replaces; on a resampling step the old frequency, so next_phase = phase(next_episode_length, "
    "next_gait_phase_dt, next_phase_offset, next_gait_command) while next_phase_dt is already the new one)",
    "final_actor_obs / final_critic_obs": "Holosoma's terminal observations of the pre-reset state on time-out steps "
    "(extras['final_observations'], used for bootstrapping); zeros on other steps",
    "pushed / push_vel": "a push fired at the end of this step and the world xy root velocity it set",
    "init_actor_obs / init_critic_obs": "observations returned by reset_all (the policy's input for step 0)",
    "mj_*": "compiled MuJoCo model after Holosoma's start-up randomization (robot bodies in body_names order; all geoms; pairs)",
    "config_json": "Holosoma's resolved ExperimentConfig (to_serializable_dict, JSON text)",
}
meta = {
    "description": "Ground truth from Holosoma's G1 29-DoF velocity-tracking env (amazon-far/holosoma 79ba714, unmodified) for "
    "checking XSoma's task term by term; produced by xsoma/tools/holosoma_dump.py",
    "holosoma_commit": "79ba714",
    "holosoma_source": os.path.dirname(os.path.dirname(os.path.abspath(sys.modules["holosoma"].__file__))),
    "holosoma_cli": HOLO_ARGS,
    "command": " ".join([sys.executable, *sys.argv]),
    "host": platform.node(),
    "versions": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    },
    "seed": seed,
    "device": str(env.device),
    "num_envs": env.num_envs,
    "simulator": cfg.simulator._target_,
    "mujoco_backend": str(getattr(cfg.simulator.config, "mujoco_backend", "")),
    "terrain_mesh_type": str(cfg.terrain.terrain_term.mesh_type),
    "dt": float(env.dt),
    "sim_dt": float(env.sim_dt),
    "decimation": int(gcfg.control_decimation_steps),
    "max_episode_length": float(env.max_episode_length),
    "dof_names": dof_names,
    "body_names": body_names,
    "feet_bodies": body_list(env.feet_indices),
    "feet_height_bodies": body_list(env.feet_height_indices),
    "termination_bodies": body_list(env.termination_contact_indices),
    "reward_terms": [
        {"name": n, "weight": float(rm.cfg.terms[n].weight), "params": dict(rm.cfg.terms[n].params), "tags": list(rm.cfg.terms[n].tags)}
        for n in REWARD_TERMS
    ],
    "termination_terms": [{"name": n, "params": dict(tm.cfg.terms[n].params), "is_timeout": bool(tm.cfg.terms[n].is_timeout)} for n in TERM_TERMS],
    "penalty_curriculum": {k: getattr(penalty, k) for k in ("min_scale", "max_scale", "level_down_threshold", "level_up_threshold", "degree")},
    "curriculum_K": int(tracker.num_compute_average_epl),
    "observations": {"actor_obs": actor_layout, "critic_obs": critic_layout, "clip": float(om.cfg.clip_observations)},
    "mujoco": {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "nbody": int(model.nbody), "ngeom": int(model.ngeom),
        "npair": int(model.npair), "root_qpos_addr": int(QA), "root_qvel_addr": int(VA), "dof_qpos_addrs": [int(i) for i in DOF_Q],
        "dof_qvel_addrs": [int(i) for i in DOF_V], "dof_actuator_ids": [int(i) for i in ACT], "timestep": float(model.opt.timestep),
        "integrator": int(model.opt.integrator), "solver": int(model.opt.solver), "iterations": int(model.opt.iterations),
        "ls_iterations": int(model.opt.ls_iterations), "cone": int(model.opt.cone), "tolerance": float(model.opt.tolerance),
    },
    "action_schedule": {
        "schedule": args.schedule,
        "rule": "policy phases: the policy's mean action on the actor observation (with Holosoma's noise) + N(0, policy_noise^2); "
        "noise phases: N(0, sigma^2); i.i.d. per joint from numpy RandomState(action_seed)",
        "policy": os.path.abspath(args.policy) if args.policy else None, "policy_kind": policy_kind, "policy_iteration": policy_iter,
        "policy_noise": args.policy_noise, "sigma": args.sigma, "action_seed": args.action_seed,
    },
    "steps": len(schedule),
    "summary": {
        "resets_at": resets,
        "contact_terminations_at": np.nonzero(arrays["term_contact"])[0].tolist(),
        "timeouts_at": np.nonzero(arrays["term_timeout"])[0].tolist(),
        "pushes_at": np.nonzero(arrays["pushed"])[0].tolist(),
        "episode_lengths": [int(arrays["post_episode_length"][i]) for i in resets],
        "command_resamples_at": np.nonzero(arrays["next_episode_length"] % int(cfg.command.params["locomotion_command_resampling_time"] / env.dt) == 0)[0].tolist(),
        "command_changes_at": np.nonzero(np.any(arrays["next_command"] != arrays["pre_command"], -1))[0].tolist(),
        "stand_command_steps": int((np.abs(arrays["pre_command"]).sum(-1) == 0).sum()),
        "seconds": {"build": round(t_built - t_start, 1), "loop": round(t_end - t_loop, 1)},
    },
    "conventions": {
        "quaternions": "MuJoCo arrays (pre/sub/post_qpos) wxyz; Holosoma arrays (root_state, body_quat) xyzw",
        "angular_velocity": "qvel[3:6] pelvis body frame (MuJoCo); Holosoma root_state[10:13] and body_ang_vel world frame",
        "linear_velocity": "world frame, of the pelvis frame origin (MuJoCo free joint), also in Holosoma's root_state",
        "order": "joints in dof_names (Holosoma config) order; bodies in body_names order; t indexes policy steps after reset_all",
        "dtypes": "Holosoma tensors float32 (as in Holosoma), MuJoCo arrays float64",
    },
    "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items() if k not in ("config_json", "mjb")},
    "array_docs": DOC,
}
with open(args.out + ".json", "w") as f:
    json.dump(meta, f, indent=1)
print(f"wrote {args.out}.npz ({os.path.getsize(args.out + '.npz') / 1e6:.1f} MB) and {args.out}.json")
print(json.dumps(meta["summary"]))
