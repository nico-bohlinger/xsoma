"""Holosoma's G1 29-DoF velocity-tracking task (amazon-far/holosoma 79ba714, presets `g1_29dof` and `g1_29dof_fast_sac`)
as pure JAX functions of the robot's state, shared by the simulator (`sim.py`) and by imagination in the world model.

Every number below is Holosoma's; "spec N" cites section N of docs/holosoma_task.md, which cites Holosoma's source line by
line. Joint order is Holosoma's config order, which is also the MJCF's joint order, so `q = qpos[7:]`. The constants of the
task (PD gains, commands, rewards, penalty curriculum, randomization ranges, reset values) sit next to the functions that
use them.

A robot state is a `Robot`: pelvis position and rotation, the pelvis frame's world linear velocity (MuJoCo's free-joint
convention, spec 2.3), the body-frame angular velocity, joint positions and velocities, and the frames the task reads (foot
contact points and ankle-roll links). The simulator fills it from MuJoCo's kinematics, imagination from forward kinematics.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# ------------------------------------------------------------------ robot and control (spec 1.4-1.6)
DOF = ("left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
       "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
       "waist_yaw", "waist_roll", "waist_pitch",
       "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
       "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch",
       "right_wrist_yaw")  # fmt: skip
J = len(DOF)
LEG = np.asarray([-0.312, 0.0, 0.0, 0.669, -0.363, 0.0])
DEFAULT_Q = np.concatenate([LEG, LEG, np.zeros(3), [0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0], [0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]]).astype(np.float32)

# (kp, kd) of each motor type, from its armature I: kp = I w^2, kd = 2 zeta I w, with w = 2 pi 10 rad/s and zeta = 2 (spec 1.5)
_HIP_P = (40.179238471, 2.557889765)
_HIP_R = (99.098427777, 6.308801854)
_ANKLE = (28.501246196, 1.814445687)
_ARM = (14.250623098, 0.907222843)
_WRIST = (16.778327481, 1.068141502)
# per leg joint: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (the knee has the same gains as the hip roll)
_LEG_GAINS = [_HIP_P, _HIP_R, _HIP_P, _HIP_R, _ANKLE, _ANKLE]
_ARM_GAINS = [_ARM, _ARM, _ARM, _ARM, _ARM, _WRIST, _WRIST]
_GAINS = _LEG_GAINS * 2 + [_HIP_P, _ANKLE, _ANKLE] + _ARM_GAINS * 2  # waist yaw like hip_pitch, roll / pitch like ankles
KP = np.asarray([g[0] for g in _GAINS], np.float32)
KD = np.asarray([g[1] for g in _GAINS], np.float32)
# torque limits (N m) in Holosoma's config; the MJCF's motors clamp the hip roll to 88 on top of this (see sim.py)
EFFORT = np.asarray([88, 139, 88, 139, 50, 50] * 2 + [88, 50, 50] + [25, 25, 25, 25, 25, 5, 5] * 2, np.float32)
ACTION_SCALE, ACTION_CLIP = 0.25, 100.0  # joint target = DEFAULT_Q + ACTION_SCALE * action; the action is clipped first
SIM_DT, DECIMATION = 0.005, 4  # physics step, and physics steps per policy step
DT = SIM_DT * DECIMATION  # 50 Hz policy
MAX_EPISODE = 1000  # time-out once the step counter exceeds it: an un-terminated episode has 1001 policy steps (spec 1.6)

# bodies of the compiled scene (prefix "robot_"): frames the task reads and the bodies whose ground contact ends an episode
FOOT_POINTS = ("left_foot_contact_point", "right_foot_contact_point")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
TERMINATION_BODIES = ("pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "right_hip_pitch_link",
                      "right_hip_roll_link", "right_hip_yaw_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
                      "left_shoulder_yaw_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link")  # fmt: skip
CONTACT_FORCE = 1.0  # N: a net contact force above this on a termination body ends the episode (spec 7)


class Robot(NamedTuple):
    p: jax.Array  # (N, 3) pelvis position
    R: jax.Array  # (N, 3, 3) pelvis rotation (body to world)
    v: jax.Array  # (N, 3) world velocity of the pelvis frame origin
    w: jax.Array  # (N, 3) angular velocity in the pelvis frame
    q: jax.Array  # (N, J)
    qd: jax.Array  # (N, J)
    foot_p: jax.Array  # (N, 2, 3) world positions of the foot contact points (left, right)
    feet_p: jax.Array  # (N, 2, 3) world positions of the ankle-roll links
    feet_R: jax.Array  # (N, 2, 3, 3) rotations of the ankle-roll links


def torques(q, qd, a_applied, kp_scale, kd_scale):
    """PD torques of one 5 ms physics substep (spec 1.5): kp_scale * KP * (target - q) - kd_scale * KD * qd, clipped to the
    effort limits. The joint target is q0 + 0.25 a, with q0 the default pose."""
    tau = kp_scale * KP * (ACTION_SCALE * a_applied + DEFAULT_Q - q) - kd_scale * KD * qd
    return jnp.clip(tau, -EFFORT, EFFORT)


# ------------------------------------------------------------------ commands and gait (spec 5)
CMD_RANGE = 1.0  # vx, vy, wz ~ U[-1, 1] (body frame)
STAND_PROB = 0.2  # probability that a drawn command is zero in all three components (the robot should stand)
RESAMPLE = int(10.0 / DT)  # commands change every 10 s = 500 steps
GAIT_FREQ = (0.8, 1.2)  # Hz: 1 / gait_period (a period of 1 s), plus or minus a width of 0.2
STAND_PHASE = math.pi  # the gait phase of both feet while standing


def sample_commands(key, n):
    """Draw n commands [vx, vy, wz] and gait frequencies; every resample redraws both (spec 5)."""
    kc, ks, kf = jax.random.split(key, 3)
    cmd = jax.random.uniform(kc, (n, 3), minval=-CMD_RANGE, maxval=CMD_RANGE)
    cmd = jnp.where(jax.random.uniform(ks, (n, 1)) <= STAND_PROB, 0.0, cmd)
    return cmd, jax.random.uniform(kf, (n,), minval=GAIT_FREQ[0], maxval=GAIT_FREQ[1])


def sample_phase_offsets(key, n):
    """Left offset U[-pi, pi], right offset the left one shifted by pi (spec 5)."""
    left = jax.random.uniform(key, (n,), minval=-math.pi, maxval=math.pi)
    return jnp.stack([left, jnp.fmod(left + 2 * math.pi, 2 * math.pi) - math.pi], -1)


def gait_phase(episode_step, freq, offsets, cmd):
    """Gait phase phi of each foot from the episode counter t: fmod(t * 2 pi dt f + offset + pi, 2 pi) - pi. Both feet are held
    at pi (standing) when the command is zero."""
    phase_dt = 2 * math.pi * DT * freq
    phi = jnp.fmod(episode_step[:, None] * phase_dt[:, None] + offsets + math.pi, 2 * math.pi) - math.pi
    stand = (jnp.linalg.norm(cmd[:, :2], axis=-1) < 0.01) & (jnp.abs(cmd[:, 2]) < 0.01)
    return jnp.where(stand[:, None], STAND_PHASE, phi)


def swing_height(phi, height: float = 0.09):
    """Desired foot height: cubic Bezier rise to 0.09 m at phi = 0 and back to 0 at +-pi (spec 6)."""
    x = (phi + math.pi) / (2 * math.pi)

    def bezier(t):
        return t**3 + 3 * t**2 * (1 - t)

    return jnp.where(x <= 0.5, height * bezier(2 * x), height - height * bezier(2 * x - 1))


# ------------------------------------------------------------------ observations (spec 3)
GRAVITY = np.asarray([0.0, 0.0, -1.0], np.float32)
ACTOR_DIM, CRITIC_DIM = 100, 103
# the terms of the actor observation in order, with their sizes (Holosoma sorts its observation terms alphabetically)
ACTOR_LAYOUT = (("actions", J), ("base_ang_vel", 3), ("command_ang_vel", 1), ("command_lin_vel", 2), ("cos_phase", 2), ("dof_pos", J),
                ("dof_vel", J), ("projected_gravity", 3), ("sin_phase", 2))  # fmt: skip
OBS_CLIP = 100.0


def projected_gravity(R):
    """The direction of gravity in the pelvis frame: R^T (0, 0, -1)."""
    return jnp.einsum("...ji,j->...i", R, GRAVITY)


def observations(robot: Robot, a_last, cmd, phase, key=None):
    """(actor, critic) observations. Terms in alphabetical order; uniform noise (actor only, raw units) before scaling.

    actor: actions, base_ang_vel x0.25, command_ang_vel, command_lin_vel, cos_phase, dof_pos (+-0.01), dof_vel x0.05 (+-0.1),
    projected_gravity, sin_phase; critic: the same without noise plus base_lin_vel x2.0 after base_ang_vel.
    """
    dq = robot.q - DEFAULT_Q
    qd = robot.qd
    if key is not None:
        k1, k2 = jax.random.split(key)
        dq_n = dq + jax.random.uniform(k1, dq.shape, minval=-0.01, maxval=0.01)
        qd_n = qd + jax.random.uniform(k2, qd.shape, minval=-0.1, maxval=0.1)
    else:
        dq_n, qd_n = dq, qd
    g = projected_gravity(robot.R)
    v_b = jnp.einsum("nji,nj->ni", robot.R, robot.v)
    common = lambda dq_, qd_: (jnp.cos(phase), dq_, 0.05 * qd_, g, jnp.sin(phase))  # noqa: E731
    head = (a_last, 0.25 * robot.w)
    cmds = (cmd[:, 2:3], cmd[:, :2])
    actor = jnp.concatenate(head + cmds + common(dq_n, qd_n), -1)
    critic = jnp.concatenate(head + (2.0 * v_b,) + cmds + common(dq, qd), -1)
    return jnp.clip(actor, -OBS_CLIP, OBS_CLIP), jnp.clip(critic, -OBS_CLIP, OBS_CLIP)


# ------------------------------------------------------------------ rewards (spec 6)
# weight of each joint's squared offset from the default pose, in DOF order: the six leg joints twice, then 3 waist, 14 arm joints
POSE_W = np.asarray([0.01, 1.0, 5.0, 0.01, 5.0, 5.0] * 2 + [50.0] * 3 + [50.0] * 14, np.float32)
# the terms whose weights the penalty curriculum scales
PENALTIES = ("ang_vel_xy", "orientation", "action_rate", "pose", "close_feet_xy", "feet_ori")


@dataclasses.dataclass(frozen=True)
class RewardCfg:
    """Weights of r = sum_i w_i dt r_i (no clipping); the penalty terms' weights are scaled by the penalty curriculum."""

    tracking_lin_vel: float = 2.0
    tracking_ang_vel: float = 1.5
    ang_vel_xy: float = -1.0
    orientation: float = -10.0
    action_rate: float = -2.0
    feet_phase: float = 5.0
    pose: float = -0.5
    close_feet_xy: float = -10.0
    feet_ori: float = -5.0
    alive: float = 1.0


PPO_REWARD = RewardCfg()
FAST_SAC_REWARD = RewardCfg(alive=10.0)  # the presets differ only in the alive weight


def reward_terms(robot: Robot, foot_heights, a, a_prev, cmd, phase):
    """Unweighted terms of one step. foot_heights (N, 2): foot contact points over the terrain; phase / cmd: the ones the
    policy observed when it chose `a` (spec 6, timing)."""
    v_b = jnp.einsum("nji,nj->ni", robot.R, robot.v)  # velocity in the pelvis frame
    g = projected_gravity(robot.R)
    yaw = jnp.arctan2(robot.R[:, 1, 0], robot.R[:, 0, 0])
    dx = robot.feet_p[:, 0, 0] - robot.feet_p[:, 1, 0]
    dy = robot.feet_p[:, 0, 1] - robot.feet_p[:, 1, 1]
    lateral = jnp.abs(jnp.cos(yaw) * dy - jnp.sin(yaw) * dx)  # the feet's sideways distance, in the pelvis heading's frame
    g_feet = jnp.einsum("nfji,j->nfi", robot.feet_R, GRAVITY)  # gravity in each foot's frame (zero x and y when flat)
    err_h = jnp.sum(jnp.square(foot_heights - swing_height(phase)), -1)  # squared foot height error against the gait reference
    return {
        "tracking_lin_vel": jnp.exp(-jnp.sum(jnp.square(cmd[:, :2] - v_b[:, :2]), -1) / 0.25),
        "tracking_ang_vel": jnp.exp(-jnp.square(cmd[:, 2] - robot.w[:, 2]) / 0.25),
        "ang_vel_xy": jnp.sum(jnp.square(robot.w[:, :2]), -1),
        "orientation": jnp.sum(jnp.square(g[:, :2]), -1),
        "action_rate": jnp.sum(jnp.square(a_prev - a), -1),
        "feet_phase": jnp.exp(-err_h / 0.008),
        "pose": jnp.sum(POSE_W * jnp.square(robot.q - DEFAULT_Q), -1),
        "close_feet_xy": (lateral < 0.15).astype(jnp.float32),
        "feet_ori": jnp.sum(jnp.linalg.norm(g_feet[..., :2], axis=-1), -1),
        "alive": jnp.ones(robot.q.shape[0]),
    }


TERMS = tuple(f.name for f in dataclasses.fields(RewardCfg))  # the reward terms, in the order of every per-term array


def weighted(terms, cfg: RewardCfg, penalty_scale):
    """The summands w_i dt r_i of the reward (Holosoma logs their episode sums as Episode/rew_<term>)."""
    w = dataclasses.asdict(cfg)
    return {k: DT * w[k] * (penalty_scale if k in PENALTIES else 1.0) * terms[k] for k in TERMS}


def reward(terms, cfg: RewardCfg, penalty_scale):
    """The reward of one step: the sum of the weighted terms."""
    return sum(weighted(terms, cfg, penalty_scale).values())


# ------------------------------------------------------------------ curriculum (spec 10)
@dataclasses.dataclass(frozen=True)
class CurriculumCfg:
    """Penalty scale s, shared by all envs, updated once per policy step from the average episode length: s *= 1 - degree
    below 150 steps, s *= 1 + degree above 750, clipped to [min, max]. The average is an EMA over finished episodes with
    weight min(n / K, 1), K = 1000 * num_envs / 4096."""

    initial: float = 0.1
    min: float = 0.0
    max: float = 1.0
    degree: float = 0.00025
    down: float = 150.0
    up: float = 750.0


PPO_CURRICULUM = CurriculumCfg()
FAST_SAC_CURRICULUM = CurriculumCfg(initial=0.5, min=0.5, degree=0.001)


def curriculum_step(cfg: CurriculumCfg, scale, avg_len, finished_lengths, finished, num_envs: int):
    """Average-episode tracker on this step's finished episodes, then the per-step penalty-scale update."""
    k = max(1, int(1000 * num_envs / 4096))
    n = finished.sum()
    w = jnp.minimum(n / k, 1.0)
    mean = jnp.sum(jnp.where(finished, finished_lengths, 0.0)) / jnp.maximum(n, 1)
    avg_len = jnp.where(n > 0, avg_len * (1 - w) + mean * w, avg_len)
    scale = jnp.where(avg_len < cfg.down, scale * (1 - cfg.degree), jnp.where(avg_len > cfg.up, scale * (1 + cfg.degree), scale))
    return jnp.clip(scale, cfg.min, cfg.max), avg_len


# ------------------------------------------------------------------ domain randomization (spec 9)
# "setup only": drawn once per env when the simulator is built; the others are redrawn at every reset or on a schedule
MASS_BODIES = ("pelvis", "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link", "left_knee_link",
               "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link", "right_knee_link")  # fmt: skip
MASS_SCALE = (0.9, 1.2)  # the masses of MASS_BODIES times U, inertia scaled by the same ratio (setup only)
TORSO_ADDED_MASS = (-1.0, 3.0)  # kg on torso_link (setup only)
TORSO_COM = 0.05  # +-m per axis on torso_link (setup only)
FRICTION = (0.5, 1.25)  # range of the friction coefficient; each env gets one of 64 equally spaced values (setup only)
FRICTION_BUCKETS = 64
GAIN_SCALE = (0.9, 1.1)  # kp and kd per (env, joint), every reset
MAX_DELAY = 1  # action delay in policy steps, U{0, 1} per env at every reset
PUSH_INTERVAL = (5.0, 10.0)  # s between pushes, per env; a push overwrites the root's world (vx, vy) with U[-1, 1] each
PUSH_VEL = 1.0

# ------------------------------------------------------------------ resets (spec 8)
SPAWN_HEIGHT = 0.8  # pelvis height above the tile origin (the terrain at the spawn point is not looked up)
SPAWN_XY = 1.0  # U[-1, 1] m offset from the tile origin
SPAWN_VEL = 0.5  # all six root velocities U[-0.5, 0.5] (world frame)
SPAWN_Q = (0.5, 1.5)  # q = q0 * U, joints with q0 = 0 start at 0; qd = 0
