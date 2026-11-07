"""XSoma's task (xsoma/task.py) against Holosoma's own G1 velocity-tracking env, term by term on identical states.

Run: python tests/test_task_holosoma.py [dump.npz]   (or pytest; XSOMA_HOLOSOMA_DUMP overrides the default dump path)

The dump comes from tools/holosoma_dump.py: Holosoma 79ba714 unmodified, preset exp:g1-29dof, classic CPU MuJoCo backend, plane
terrain; Holosoma's PPO checkpoint walks a 1001-step episode (two pushes, a command resampling, a time-out), then random actions
make the robot fall five times, then the policy walks again. Every check prints max |XSoma - Holosoma| (Holosoma is float32):

  constants      DOF order, default pose, PD gains, effort limits, pose weights, reward weights and tags, feet / termination
                 bodies, observation layout, delay range
  kinematics     xsoma.kin forward kinematics of the frames the task reads vs Holosoma's body poses after each step
  observations   task.observations of the state after each step (the MuJoCo qpos / qvel the next step starts from, as XSoma's
                 simulator holds it), the last raw action, command and phase vs Holosoma's critic and actor (+ Holosoma's own
                 noise) observations, per block; the time-out step's final critic observation
  rewards        task.reward_terms on the post-physics state (Holosoma's foot heights; feet from XSoma's kinematics, and from
                 Holosoma's body poses) vs Holosoma's raw terms; task.reward with the step's penalty scale vs the total
  gait           task.gait_phase from the episode counter vs Holosoma's phase (stand override with the reset's first draw)
  curriculum     task.curriculum_step vs Holosoma's penalty scale and average episode length, per step and rolled out
  actions        sim.py's PD target (clip, delay) vs Holosoma's applied action; task.torques at every substep vs data.ctrl
  terminations   time-out from the counter; contact = force > 1 N on task.TERMINATION_BODIES (Holosoma's forces) vs its flags
  schedule       command resampling (every 500 steps, twice on resets), push timing / velocity, reset ranges
  physics        MuJoCo CPU: XSoma's scene (plane, priority sole contacts) with Holosoma's start-up randomization, 4 x 5 ms from
                 every recorded pre-step state with task.torques, vs Holosoma's post-physics state (MuJoCo versions may differ)
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from xsoma import scene, task  # noqa: E402
from xsoma.kin import Kinematics  # noqa: E402
from xsoma.rot import quat_to_mat  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.environ.get("XSOMA_HOLOSOMA_DUMP") or os.path.join(_HERE, "data", "holosoma_g1_ppo_plane.npz")
HOLO_NAME = {k: ("penalty_" + k if k in ("ang_vel_xy", "orientation", "action_rate", "close_feet_xy", "feet_ori") else k) for k in task.TERMS}
FRAMES = [f"robot_{b}" for b in task.FOOT_POINTS + task.FEET]  # foot contact points, ankle-roll links
_CACHE = {}


def data():
    if "z" not in _CACHE:
        _CACHE["z"] = dict(np.load(DUMP))
        with open(DUMP[: -len(".npz")] + ".json") as f:
            _CACHE["meta"] = json.load(f)
    return _CACHE["z"], _CACHE["meta"]


def f32(x):
    return jnp.asarray(np.asarray(x, np.float32))


def mx(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def model():
    if "m" not in _CACHE:
        _CACHE["m"] = scene.build()  # plane named floor, Holosoma's MJCF, priority sole contacts
        _CACHE["kin"] = Kinematics(_CACHE["m"], FRAMES)
    return _CACHE["m"], _CACHE["kin"]


def robot(qpos, qvel, frames=True):
    """task.Robot of MuJoCo free-joint states, as sim.py builds it: v world velocity of the pelvis origin, w MuJoCo's body-frame
    angular velocity; the foot frames from XSoma's forward kinematics."""
    qpos, qvel = f32(qpos), f32(qvel)
    p, R, q = qpos[:, :3], quat_to_mat(qpos[:, 3:7]), qpos[:, 7:]
    P = Rs = None
    if frames:
        P, Rs = model()[1].forward(p, R, q)
    return task.Robot(p, R, qvel[:, :3], qvel[:, 3:6], q, qvel[:, 6:], *((P[:, 0:2], P[:, 2:4], Rs[:, 2:4]) if frames else (None,) * 3))


def xyzw_mat(q):
    q = np.asarray(q, np.float32)
    return quat_to_mat(f32(np.concatenate([q[..., 3:4], q[..., :3]], -1)))


def report(title, rows, tol=None):
    print(f"\n{title}")
    for k, v in rows.items():
        flag = "" if tol is None or (v <= tol if isinstance(v, float) else True) else "   <-- above tolerance"
        print(f"  {k:34s} {v:.2e}{flag}" if isinstance(v, float) else f"  {k:34s} {v}")


# ------------------------------------------------------------------ checks
def test_constants():
    z, meta = data()
    names = [n.removesuffix("_joint") for n in meta["dof_names"]]
    assert tuple(names) == task.DOF, names
    rows = {
        "default pose": mx(task.DEFAULT_Q, z["default_dof_pos"]),
        "kp": mx(task.KP, z["p_gains"]),
        "kd": mx(task.KD, z["d_gains"]),
        "effort limits": mx(task.EFFORT, z["torque_limits"]),
        "action scale": mx(task.ACTION_SCALE, z["action_scales"]),
        "pose weights": mx(task.POSE_W, meta["reward_terms"][[r["name"] for r in meta["reward_terms"]].index("pose")]["params"]["pose_weights"]),
        "dt": abs(task.DT - meta["dt"]),
        "max episode length": abs(task.MAX_EPISODE - meta["max_episode_length"]),
    }
    weights = {r["name"]: r for r in meta["reward_terms"]}
    w = task.PPO_REWARD
    rows["reward weights"] = max(abs(getattr(w, k) - weights[HOLO_NAME[k]]["weight"]) for k in task.TERMS)
    tagged = sorted(k for k in task.TERMS if "penalty_curriculum" in weights[HOLO_NAME[k]]["tags"])
    assert tagged == sorted(task.PENALTIES), (tagged, task.PENALTIES)
    assert [weights[n]["params"] for n in ("tracking_lin_vel", "tracking_ang_vel")] == [{"tracking_sigma": 0.25}] * 2
    assert weights["feet_phase"]["params"] == {"swing_height": 0.09, "tracking_sigma": 0.008}
    assert weights["penalty_close_feet_xy"]["params"] == {"close_feet_threshold": 0.15}
    assert meta["feet_bodies"] == list(task.FEET) and meta["feet_height_bodies"] == list(task.FOOT_POINTS)
    assert sorted(meta["termination_bodies"]) == sorted(task.TERMINATION_BODIES), meta["termination_bodies"]
    assert meta["termination_terms"][0]["params"]["force_threshold"] == task.CONTACT_FORCE
    layout = [(e["term"], e["dim"]) for e in meta["observations"]["actor_obs"]]
    assert tuple(layout) == task.ACTOR_LAYOUT, layout
    pc = meta["penalty_curriculum"]
    c = task.PPO_CURRICULUM
    assert (pc["min_scale"], pc["max_scale"], pc["level_down_threshold"], pc["level_up_threshold"], pc["degree"]) == (c.min, c.max, c.down, c.up, c.degree)
    assert meta["observations"]["clip"] == task.OBS_CLIP
    report("constants (XSoma task.py vs Holosoma's config as loaded)", rows)
    assert max(rows.values()) < 1e-5


def test_kinematics():
    z, _ = data()
    m, kin = model()
    P, Rs = kin.forward(f32(z["post_qpos"][:, :3]), quat_to_mat(f32(z["post_qpos"][:, 3:7])), f32(z["post_qpos"][:, 7:]))
    idx = np.r_[z["feet_height_indices"], z["feet_indices"]]
    rows = {
        "foot contact points (m)": mx(P[:, 0:2], z["post_body_pos"][:, idx[:2]]),
        "ankle-roll links (m)": mx(P[:, 2:4], z["post_body_pos"][:, idx[2:]]),
        "ankle-roll rotations": mx(Rs[:, 2:4], xyzw_mat(z["post_body_quat"][:, idx[2:]])),
        "foot height = foot z on the plane": mx(P[:, 0:2, 2], z["post_feet_heights"]),
    }
    report("kinematics (xsoma.kin of the post-physics qpos vs Holosoma's _rigid_body_pos / _rot)", rows)
    assert max(rows.values()) < 1e-4


def _blocks(layout, a, b):
    return {f"{e['term']} [{e['start']}:{e['start'] + e['dim']}]": mx(a[:, e["start"]:e["start"] + e["dim"]], b[:, e["start"]:e["start"] + e["dim"]]) for e in layout}


def test_observations():
    z, meta = data()
    # the state each observation is taken in: after reset_all (t = 0), then after step t (= the next step's start state)
    rb = robot(z["pre_qpos"], z["pre_qvel"], frames=False)
    actor, critic = task.observations(rb, f32(z["pre_last_action"]), f32(z["pre_command"]), f32(z["pre_phase"]))
    critic_h = np.concatenate([z["init_critic_obs"][None], z["critic_obs"][:-1]])
    rows = _blocks(meta["observations"]["critic_obs"], np.asarray(critic), critic_h)
    report(f"critic observation (103), {len(critic_h)} states (reset_all + after each step)", rows)
    worst = max(rows.values())
    # actor: XSoma's noise-free actor + Holosoma's noise (raw units, scaled), for the states after each step
    scale = np.ones(task.ACTOR_DIM, np.float32)
    for e in meta["observations"]["actor_obs"]:
        scale[e["start"]:e["start"] + e["dim"]] = e["scale"]
    actor_x = np.clip(np.asarray(actor)[1:] + scale * z["actor_noise"][:-1], -task.OBS_CLIP, task.OBS_CLIP)
    rows = _blocks(meta["observations"]["actor_obs"], actor_x, z["actor_obs"][:-1])
    report("actor observation (100) with Holosoma's noise added", rows)
    worst = max(worst, max(rows.values()))
    # time-out bootstrap: the final critic observation of the pre-reset state (sim.py's critic_final)
    to = np.nonzero(z["time_out"])[0]
    if len(to):
        rb = robot(z["post_qpos"][to], z["post_qvel"][to], frames=False)
        _, cf = task.observations(rb, f32(z["post_action"][to]), f32(z["post_command"][to]), f32(z["post_phase"][to]))
        rows = {f"time-out steps {to.tolist()}": mx(cf, z["final_critic_obs"][to])}
        report("final critic observation (pre-reset state, for time-out bootstraps)", rows)
        worst = max(worst, max(rows.values()))
    assert worst < 1e-4, worst


def _reward_rows(z, rb):
    terms = task.reward_terms(rb, f32(z["post_feet_heights"]), f32(z["post_action"]), f32(z["post_prev_action"]), f32(z["post_command"]),
                              f32(z["post_phase"]))  # fmt: skip
    names = list(z["reward_terms"])
    rows = {}
    for k in task.TERMS:
        h = z["reward_raw"][:, names.index(HOLO_NAME[k])]
        rows[f"{k} (|r| <= {np.abs(h).max():.3g})"] = mx(terms[k], h)
    total = task.reward(terms, task.PPO_REWARD, f32(z["pre_penalty_scale"]))
    rows["total (weighted, x dt, penalty scale)"] = mx(total, z["reward"])
    close = np.asarray(terms["close_feet_xy"]) != z["reward_raw"][:, names.index("penalty_close_feet_xy")]
    return rows, int(close.sum())


def test_rewards():
    z, _ = data()
    rows, flips = _reward_rows(z, robot(z["post_qpos"], z["post_qvel"]))
    report(f"reward terms, post-physics state, feet from XSoma's kinematics ({len(z['reward'])} steps; close_feet flips: {flips})", rows)
    idx = np.r_[z["feet_height_indices"], z["feet_indices"]]
    rb = robot(z["post_qpos"], z["post_qvel"], frames=False)._replace(
        foot_p=f32(z["post_body_pos"][:, idx[:2]]), feet_p=f32(z["post_body_pos"][:, idx[2:]]), feet_R=xyzw_mat(z["post_body_quat"][:, idx[2:]]))
    rows2, flips2 = _reward_rows(z, rb)
    report(f"reward terms, feet from Holosoma's body poses (close_feet flips: {flips2})", rows2)
    worst = max(max(rows.values()), max(rows2.values()))
    assert flips == 0 and flips2 == 0 and worst < 1e-3, (worst, flips, flips2)


def test_gait():
    z, _ = data()
    reset = z["reset"]
    freq = np.where(reset, z["next_gait_freq"], z["pre_gait_freq"])  # the frequency before this step's resampling (irrelevant at t = 0)
    phase = task.gait_phase(jnp.asarray(z["next_episode_length"], jnp.int32), f32(freq), f32(z["next_phase_offset"]), f32(z["next_gait_command"]))
    d = np.asarray(phase, np.float64) - z["next_phase"]
    wrapped = np.abs(np.angle(np.exp(1j * d)))
    pdt = np.asarray(2 * math.pi * task.DT * f32(freq))
    rows = {
        "phase (wrapped)": float(wrapped.max()),
        "phase (raw)": float(np.abs(d).max()),
        "phase_dt = 2 pi dt f": mx(np.where(reset, z["next_gait_phase_dt"], pdt), z["next_gait_phase_dt"]),
        "phase_dt after resampling": mx(2 * math.pi * task.DT * z["next_gait_freq"].astype(np.float32), z["next_phase_dt"]),
        "stand steps (phase = pi)": int((np.abs(z["next_phase"] - np.pi) < 1e-6).all(-1).sum()),
    }
    sh = task.swing_height(f32(z["post_phase"]))
    rows["swing height range (m)"] = f"[{float(sh.min()):.4f}, {float(sh.max()):.4f}]"

    def shift(left):  # the right offset is the left one shifted by pi: both Holosoma's offsets and XSoma's samples obey it
        return jnp.fmod(jnp.asarray(left, jnp.float32) + 2 * math.pi, 2 * math.pi) - math.pi

    xs = np.asarray(task.sample_phase_offsets(jax.random.PRNGKey(0), 4096))
    rows["right offset rule, Holosoma"] = mx(shift(z["next_phase_offset"][:, 0]), z["next_phase_offset"][:, 1])
    rows["right offset rule, XSoma samples"] = mx(shift(xs[:, 0]), xs[:, 1])
    report("gait phase (task.gait_phase from the episode counter)", rows)
    assert rows["phase (wrapped)"] < 1e-4 and rows["right offset rule, Holosoma"] < 1e-6 and rows["right offset rule, XSoma samples"] < 1e-6


def test_curriculum():
    z, meta = data()
    c = task.PPO_CURRICULUM
    assert meta["curriculum_K"] == max(1, int(1000 * 1 / 4096))

    def one(scale, avg, length, done):
        return task.curriculum_step(c, scale, avg, length[None], done[None], 1)

    s1, a1 = jax.vmap(one)(f32(z["pre_penalty_scale"]), f32(z["pre_avg_episode_length"]), f32(z["post_episode_length"]), jnp.asarray(z["reset"]))

    def body(carry, x):
        s, a = one(*carry, *x)
        return (s, a), (s, a)

    _, (s2, a2) = jax.lax.scan(body, (f32(z["pre_penalty_scale"][0]), f32(z["pre_avg_episode_length"][0])),
                               (f32(z["post_episode_length"]), jnp.asarray(z["reset"])))  # fmt: skip
    rows = {
        "penalty scale, one step": mx(s1, z["next_penalty_scale"]),
        "average episode length, one step": mx(a1, z["next_avg_episode_length"]),
        f"penalty scale, rolled out {len(s2)} steps": mx(s2, z["next_penalty_scale"]),
        "average episode length, rolled out": mx(a2, z["next_avg_episode_length"]),
        "scale range": f"[{z['next_penalty_scale'].min():.5f}, {z['next_penalty_scale'].max():.5f}]",
    }
    report("penalty curriculum (task.curriculum_step, one env, K = 1)", rows)
    assert rows["penalty scale, one step"] < 1e-6 and rows["average episode length, one step"] < 1e-3


def test_actions():
    z, _ = data()
    a, a_last, delay = f32(z["action"]), f32(z["pre_last_action"]), jnp.asarray(z["pre_action_delay_idx"])
    target = jnp.clip(jnp.where(delay[:, None] == 0, a, a_last), -task.ACTION_CLIP, task.ACTION_CLIP)  # sim.py's PD target
    tau = task.torques(f32(z["sub_qpos"][..., 7:]), f32(z["sub_qvel"][..., 6:]), f32(z["action_applied"])[:, None], f32(z["pre_kp_scale"])[:, None],
                       f32(z["pre_kd_scale"])[:, None])  # fmt: skip
    rows = {
        f"PD target (clip, delay; delays {np.bincount(z['pre_action_delay_idx']).tolist()})": mx(target, z["action_applied"]),
        "task.torques vs data.ctrl (N m)": mx(tau, z["sub_ctrl"]),
        "saturated torques": int((np.abs(z["sub_ctrl"]) >= z["torque_limits"] - 1e-6).sum()),
    }
    report("actions and PD torques (4 substeps per step)", rows)
    assert rows[next(iter(rows))] == 0.0 and rows["task.torques vs data.ctrl (N m)"] < 1e-3


def test_terminations():
    z, _ = data()
    names = list(z["body_names"])
    idx = [names.index(b) for b in task.TERMINATION_BODIES]
    contact = (np.linalg.norm(z["post_contact_forces"][:, idx], axis=-1) > task.CONTACT_FORCE).any(-1)
    timeout = z["post_episode_length"] > task.MAX_EPISODE
    rows = {
        "contact flags (mismatches)": int((contact != z["term_contact"]).sum()),
        "time-out flags (mismatches)": int((timeout != z["term_timeout"]).sum()),
        "reset = contact | time-out (mismatches)": int(((contact | timeout) != z["reset"]).sum()),
        "contact terminations at": np.nonzero(z["term_contact"])[0].tolist(),
        "time-outs at": np.nonzero(z["term_timeout"])[0].tolist(),
    }
    report("terminations", rows)
    assert rows["contact flags (mismatches)"] == rows["time-out flags (mismatches)"] == rows["reset = contact | time-out (mismatches)"] == 0


def test_schedule():
    z, _ = data()
    reset, pushed, t1 = z["reset"], z["pushed"], z["next_episode_length"]
    changed = np.any(z["next_command"] != z["pre_command"], -1)
    due = t1 % task.RESAMPLE == 0
    pushes_expected = (np.where(reset, 0, z["pre_push_counter"]) + 1) == np.where(reset, z["next_push_interval_steps"], z["pre_push_interval_steps"])
    r = np.nonzero(reset)[0]
    r = r[r + 1 < len(reset)]  # reset steps whose reset state the next step starts from
    q0 = task.DEFAULT_Q
    nz = q0 != 0
    qr = z["pre_qpos"][r + 1] if len(r) else None
    rows = {
        "command changes off schedule": int((changed & ~due).sum()),
        "resampling steps (counter % 500 == 0)": np.nonzero(due)[0].tolist(),
        "commands within [-1, 1]": bool(np.abs(z["next_command"]).max() <= task.CMD_RANGE),
        "push timing mismatches": int((pushes_expected != pushed).sum()),
        "pushes at": np.nonzero(pushed)[0].tolist(),
        "push velocity = next root xy velocity": mx(z["next_root_state"][pushed, 7:9], z["push_vel"][pushed]),
        "push intervals (steps)": f"[{z['pre_push_interval_steps'].min()}, {z['pre_push_interval_steps'].max()}] within "
        f"[{int(task.PUSH_INTERVAL[0] / task.DT)}, {int(task.PUSH_INTERVAL[1] / task.DT)}]",
    }
    if qr is not None:
        s = qr[:, 7:][:, nz] / q0[nz]
        o = z["env_origin"]
        rows.update({
            "reset q / q0 within [0.5, 1.5]": bool((s >= task.SPAWN_Q[0] - 1e-6).all() and (s <= task.SPAWN_Q[1] + 1e-6).all()),
            "reset q where q0 = 0": mx(qr[:, 7:][:, ~nz], 0.0),
            "reset joint velocities": mx(z["pre_qvel"][r + 1][:, 6:], 0.0),
            "reset root z - 0.8 - origin z": mx(qr[:, 2], task.SPAWN_HEIGHT + o[2]),
            "reset |xy - origin| <= 1": bool((np.abs(qr[:, :2] - o[:2]) <= task.SPAWN_XY + 1e-6).all()),
            "reset quaternion identity": mx(qr[:, 3:7], [1, 0, 0, 0]),
            "reset root velocities within 0.5": bool((np.abs(z["pre_qvel"][r + 1][:, :6]) <= task.SPAWN_VEL + 1e-6).all()),
            "reset kp, kd within [0.9, 1.1]": bool(((z["next_kp_scale"][r] >= 0.9) & (z["next_kp_scale"][r] <= 1.1) & (z["next_kd_scale"][r] >= 0.9)
                                                    & (z["next_kd_scale"][r] <= 1.1)).all()),  # fmt: skip
            "reset gait frequency within [0.8, 1.2]": bool(((z["next_gait_freq"] >= 0.8) & (z["next_gait_freq"] <= 1.2)).all()),
            "reset: two command draws (reset's != step's)": int(np.any(z["next_gait_command"][r] != z["next_command"][r], -1).sum()),
            "reset: last action and delay queue zero": mx(np.concatenate([z["next_last_action"][r], z["next_action_queue"][r].reshape(len(r), -1)], -1), 0.0),
        })
    report("commands, pushes, resets", rows)
    assert rows["command changes off schedule"] == 0 and rows["push timing mismatches"] == 0 and rows["commands within [-1, 1]"]


def test_physics():
    """MuJoCo CPU replay of each step in XSoma's scene with Holosoma's start-up randomization (by name) and task.torques."""
    z, meta = data()
    m = scene.build()  # a fresh copy to randomize
    names = list(z["body_names"])
    for i, n in enumerate(names):
        b = m.body(f"robot_{n}").id
        m.body_mass[b], m.body_inertia[b], m.body_ipos[b] = z["mj_body_mass"][i], z["mj_body_inertia"][i], z["mj_body_ipos"][i]
    hg = {str(n): i for i, n in enumerate(z["mj_geom_names"])}
    soles = set(scene.sole_geoms(m))
    for g in range(m.ngeom):  # Holosoma writes one friction to every geom; its sole contacts are pairs (fixed 0.8): keep XSoma's 0.8
        if g not in soles and m.geom(g).name in hg:
            m.geom_friction[g, 0] = z["mj_geom_friction"][hg[m.geom(g).name], 0]
    d = mujoco.MjData(m)
    torques = jax.jit(task.torques)
    err_q, err_v = [], []
    for t in range(len(z["action"])):
        d.qpos[:], d.qvel[:] = z["pre_qpos"][t], z["pre_qvel"][t]
        args = (f32(z["action_applied"][t]), f32(z["pre_kp_scale"][t]), f32(z["pre_kd_scale"][t]))
        for _ in range(task.DECIMATION):
            d.ctrl[:] = np.asarray(torques(f32(d.qpos[7:]), f32(d.qvel[6:]), *args))
            mujoco.mj_step(m, d)
        err_q.append(np.abs(d.qpos - z["post_qpos"][t]).max())
        err_v.append(np.abs(d.qvel - z["post_qvel"][t]).max())
    err_q, err_v = np.asarray(err_q), np.asarray(err_v)
    rows = {
        "qpos after 4 substeps, median": float(np.median(err_q)),
        "qpos, 99th percentile": float(np.quantile(err_q, 0.99)),
        "qpos, max": float(err_q.max()),
        "qvel, median": float(np.median(err_v)),
        "qvel, 99th percentile": float(np.quantile(err_v, 0.99)),
        "qvel, max": float(err_v.max()),
        "steps with qvel error > 1e-3": int((err_v > 1e-3).sum()),
        "MuJoCo here / in Holosoma's dump": f"{mujoco.__version__} / {meta['versions']['mujoco']}",
    }
    report("physics: XSoma's scene on MuJoCo CPU vs Holosoma's classic MuJoCo, one policy step from each recorded state", rows)
    assert np.median(err_v) < 1e-3, rows


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DUMP = sys.argv[1]
    print(f"dump: {os.path.abspath(DUMP)}  (jax {jax.__version__} on {jax.devices()[0].platform}, mujoco {mujoco.__version__})")
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                failed.append(name)
                print(f"  FAILED {name}: {e}")
    print("\nall checks passed" if not failed else f"\nfailed: {failed}")
    sys.exit(1 if failed else 0)
