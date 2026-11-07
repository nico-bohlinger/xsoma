# Holosoma's G1 29-DoF velocity-tracking task, line by line

This is the specification XSoma's task (`xsoma/task.py`, `xsoma/sim.py`, `xsoma/terrain.py`, `xsoma/scene.py`) was written
from: the `g1_29dof` (PPO) and `g1_29dof_fast_sac` (FastSAC) locomotion presets of
[Holosoma](https://github.com/amazon-far/holosoma) at commit `79ba714`, with every number cited to the line of Holosoma's source
it comes from. The code cites it as "spec N" (section N). `tests/test_task_holosoma.py` checks the result against Holosoma's
own environment.

## Citation conventions

- `H/...` = `src/holosoma/holosoma/...` in Holosoma's repository at `79ba714` (the Python package).
- `R/...` = the repository root (README, CI, scripts, inference package).
- `EXT:` = evidence from outside Holosoma's code: (a) small read-only MuJoCo probes (MuJoCo 3.12; Holosoma pins
  `MUJOCO_VERSION=3.10.0`, `R/scripts/versions.sh:6`), summarized in Appendix A; (b) the sources of MuJoCo Warp and MJX 3.12;
  (c) Isaac Lab / PhysX defaults that Holosoma does not vendor (flagged as such).
- Both presets differ only in `algo`, `curriculum`, `reward` and `nightly`; everything else is shared
  (`H/config_values/loco/g1/experiment.py:18-56`).

---

## 0. Per-step pipeline (order matters for equivalence)

One environment (policy) step, `BaseTask.step` (`H/envs/base_task/base_task.py:485-491`):

1. `action_manager.process_actions(a)`: `prev_action <- action`, `action <- a` (raw policy output)
   (`H/managers/action/manager.py:176-178`); the joint term clips to ±100 and applies the action delay
   (`H/managers/action/terms/joint_control.py:110-154`).
2. Physics loop, `control_decimation_steps = 4` times (`base_task.py:497-504`): `apply_actions()` recomputes
   PD torques from the *current* `q, q̇` and writes them (`joint_control.py:156-168`), then one physics step.
3. Post-physics (`base_task.py:510-551`), in this order:
   `episode_length_buf += 1` (515) → `terrain_manager.update_heights()` (518, via
   `H/envs/locomotion/locomotion_manager.py:146-147`) → terminations (520) → rewards (521) →
   terminal observations for **time-outs only** (524-529) → `reset_envs_idx(reset ids)` (531, called
   **every step, even with an empty id list**) → forward-kinematics refresh of reset/pushed envs
   (533-535, `locomotion_manager.py:137-140`) → `_update_tasks_callback` = `command_manager.step()`,
   `curriculum_manager.step()`, `randomization_manager.step()` (pushes) (539, `base_task.py:590-593`) →
   `observation_manager.step()` (541).
4. Reset order inside `reset_envs_idx` (`base_task.py:278-323`): observation reset → state reset
   (`_reset_buffers_callback`, `_reset_robot_states_callback`, extras; 325-348) → `dr_episode_count += 1`
   then randomization reset (300-304) → action reset (306-307) → command reset (309-310) → curriculum
   reset (312-313) → termination reset (315-316).

Consequences used below: rewards at step *t* see the gait phase and commands computed at the end of
step *t−1*; commands/gait/pushes are updated after the reset block; the penalty curriculum's `reset()`
runs every control step.

---

## 1. Robot and model

### 1.1 Asset files per backend

| Backend | File actually loaded | Citation |
|---|---|---|
| Isaac Sim (preset default) | `H/data/robots/g1/g1_29dof.urdf` converted to USD at start-up (`usd_file=None`, `force_usd_conversion=True`) | `H/config_values/robot.py:539-540`; `H/simulator/isaacsim/isaacsim_articulation_cfg.py:174-201` |
| MuJoCo classic / MJWarp | `H/data/robots/g1/g1_29dof.xml` (MJCF) via `MjSpec.from_file`, attached with prefix `robot_` | `H/config_values/robot.py:541`; `H/simulator/mujoco/scene_manager.py:337-389` |

- URDF provenance: "G1 model from https://github.com/WhoKnowsssss/tracking_implementation" (`g1_29dof.urdf:3`).
  MJCF provenance: Unitree MJCF from the `qiayuanl/unitree` docker image plus added actuators (`g1_29dof.xml:1-10`).
- `main_mesh_collision_halfspherehand.urdf` is **not** used by the locomotion presets; it is only referenced
  by `g1_29dof_w_object` (`H/config_values/robot.py:1110-1121`) for the whole-body-tracking-with-object
  presets (`H/config_values/wbt/g1/experiment.py:154-172`); it uses mesh colliders for hips/torso and
  half-sphere hand meshes (`main_mesh_collision_halfspherehand.urdf:93-96,537-550,867-876`).
- Isaac Sim URDF import options: `merge_fixed_joints=collapse_fixed_joints=True`,
  `replace_cylinders_with_capsules=True`, joint drives with zero stiffness/damping and
  `target_type="none"` (`isaacsim_articulation_cfg.py:182-201`; `robot.py:534-535`). The two
  `*_foot_contact_point` joints carry `dont_collapse="true"` so those links survive the merge
  (`g1_29dof.urdf:259,472`); Holosoma asserts the resulting body list equals the config body list
  (`H/simulator/isaacsim/isaacsim.py:519-520`).
- Unused legacy asset fields (IsaacGym era): `armature=0.001`, `thickness=0.01`,
  `default_dof_drive_mode=3`, `flip_visual_attachments`, `contact_pairs_multiplier=16`
  (`robot.py:488,536-544`) are not read by the Isaac Sim or MuJoCo backends (only `replace_cylinder_with_capsule`,
  `dof_armature_list`, `dof_joint_friction_list` are, `isaacsim_articulation_cfg.py:187,222-235`).

### 1.2 Bodies, joints, ordering

- 32 bodies, 29 actuated DoF (`robot.py:16-18`); body order (`robot.py:309-342`): pelvis, left leg
  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll, foot_contact_point), right leg (same),
  waist_yaw, waist_roll, torso, left arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw), right arm.
- DoF order (`robot.py:28-58`), index: 0-5 left hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll;
  6-11 right leg (same); 12-14 waist_yaw, waist_roll, waist_pitch; 15-21 left shoulder_pitch,
  shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw; 22-28 right arm (same).
  Both backends re-index to this config order (`isaacsim.py:476-481`;
  `H/simulator/base_simulator/base_simulator.py:155` docstring "in robot_config.dof_names order").
- Body index sets (substring matching over the simulator body list, `locomotion_manager.py:89-115`):
  `feet_indices` = bodies containing `ankle_roll_link` → [left_ankle_roll_link, right_ankle_roll_link];
  `feet_height_indices` = `foot_contact_point` bodies; termination bodies = names containing
  `pelvis`, `shoulder`, `hip` → pelvis + 6 hip links + 6 shoulder links (`robot.py:24-27,343`).
- `left/right_foot_contact_point`: massless-ish (1 g) frame 0.037 m below the ankle-roll origin
  (`g1_29dof.xml:198-200,249-251`; `g1_29dof.urdf:243-263`). With the default pose on flat ground this
  point sits 1.7 mm below the lowest sole capsule point (EXT probe, Appendix A).
- Total robot mass in the MJCF: 33.343 kg; pelvis 3.813 kg, torso_link 7.818 kg (EXT probe). The URDF
  splits the torso into torso_link 6.78 kg + head_link 1.036 kg + small fixed links that Isaac Sim merges
  (`g1_29dof.urdf:526-618`).

### 1.3 Collision geometry: URDF (Isaac Sim) vs MJCF (MuJoCo/MJWarp)

All colliders are primitives in both files; no collision meshes in either.

| Body | URDF collider (→ Isaac Sim) | MJCF collider (→ MuJoCo) | Same? |
|---|---|---|---|
| pelvis | sphere r 0.07 at (0,0,−0.08) (`urdf:22-27`) | sphere r 0.07 at (0,0,−0.08) (`xml:151`) | yes |
| hip_roll_link (thigh) | cylinder r 0.05, L 0.2, centre (0.02,0,−0.1), axis z (`urdf:89-94, 302-307`) | capsule r 0.05 fromto (0.02,0,0)→(0.02,0,−0.2) (`xml:166, 217`) | yes after cylinder→capsule |
| knee_link (shin) | cylinder r **0.05**, L **0.2**, centre (0.02,0,−0.1) (`urdf:141-146, 354-359`) | capsule r **0.04** fromto (0.02,0,0)→(0.02,0,−0.25) (`xml:177, 228`) | **no** |
| ankle_roll_link (sole) | 7 cylinders along x at z −0.025: y=±0.026 r 0.01 x∈[0.05,0.10]; y=±0.018 r **0.008** x∈[−0.044,0.123]; y=±0.01 r 0.01 x∈[−0.052,0.13]; y=0 r 0.01 x∈[−0.054,0.132] (`urdf:193-234, 406-447`) | 7 capsules, same segments (`xml:189-195, 240-246`), but **left** foot2/foot6 r **0.01** (`xml:190,194`) while **right** foot2/foot6 r 0.008 (`xml:241,245`) | right yes; **left differs** |
| torso_link | 3 cylinders along y (r 0.073/0.07/0.065 at z 0.22/0.13/0.06) + head cylinder r 0.068 at z 0.415 (`urdf:541-564`) | 4 capsules with identical segments (`xml:275-278`) | yes |
| shoulder_yaw_link | cylinder r 0.035, z∈[−0.08,0.05] (`urdf:740-745, 944-949`) | capsule r 0.035 fromto (0,0,−0.08)→(0,0,0.05) (`xml:300, 350`) | yes |
| elbow_link | cylinder r 0.035, x∈[−0.01,0.12] at z −0.01 (`urdf:769-774, 973-978`) | capsule, same (`xml:306, 356`) | yes |
| wrist_yaw_link (hand) | cylinder r 0.05, x∈[0.05,0.10] (`urdf:854-859, 1058-1063`) | capsule, same (`xml:324, 374`) | yes |
| all other links | none | none (only visual meshes, class `visual` contype 0) (`xml:17,78-80`) | yes |

- "yes after cylinder→capsule" assumes the Isaac Sim URDF importer replaces a cylinder of length L by a
  capsule with cylindrical part L (total L+2r), which matches MJCF `fromto` capsules; this importer
  convention is external to the checkout and should be confirmed on a converted USD.
- MJCF sole geoms default class: `class="collision"` → group 3, contype 1, conaffinity 1
  (`xml:81-86`); the `foot` box default (`xml:83-85`) is not used by any geom.

Self-collision:
- Isaac Sim: `enabled_self_collisions = enable_self_collisions = False` (`isaacsim_articulation_cfg.py:166-171`;
  `robot.py:543`). Envs are collision-filtered from each other, all collide with `/World/ground`
  (`isaacsim.py:323-325`; `H/simulator/isaacsim/scene_fixtures.py:114,134`).
- MuJoCo: every robot geom with contype=1/conaffinity=1 gets conaffinity 2 when self-collisions are off;
  terrain gets contype 2/conaffinity 1, so robot↔terrain only (`scene_manager.py:188-193, 840-888`; EXT
  probe confirms all 29 robot collision geoms compile with contype 1/conaffinity 2). MJWarp uses each world
  separately, so robots never touch each other.

### 1.4 Joint limits, effort limits, armature, damping, friction

Per-joint lists (config order):

| quantity | values | citation |
|---|---|---|
| pos lower | −2.5307, −0.5236, −2.7576, −0.087267, −0.87267, −0.2618 (L leg); −2.5307, −2.9671, −2.7576, −0.087267, −0.87267, −0.2618 (R leg); −2.618, −0.52, −0.52; −3.0892, −1.5882, −2.618, −1.0472, −1.972222054, −1.61443, −1.61443 (L arm); −3.0892, −2.2515, −2.618, −1.0472, −1.972222054, −1.61443, −1.61443 (R arm) | `robot.py:123-153` |
| pos upper | 2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618; 2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618; 2.618, 0.52, 0.52; 2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.61443, 1.61443; 2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.61443, 1.61443 | `robot.py:154-184` |
| velocity limit (rad/s) | legs 32, 20, 32, 20, 37, 37 (×2); waist 32, 37, 37; arms 37, 37, 37, 37, 37, 22, 22 (×2) | `robot.py:185-215` |
| effort limit (N·m) | legs 88, 139, 88, 139, 50, 50 (×2); waist 88, 50, 50; arms 25, 25, 25, 25, 25, 5, 5 (×2) | `robot.py:216-246` |
| armature (kg·m²) | legs 0.010177520, 0.025101925, 0.010177520, 0.025101925, 0.007219450, 0.007219450; waist 0.010177520, 0.007219450, 0.007219450; arms 0.003609725 ×5, 0.00425, 0.00425 | `robot.py:247-277` (Isaac Sim, `isaacsim_articulation_cfg.py:234`); identical MJCF values (`xml:23-75`) |
| joint friction | all 0.0 (Isaac Sim) | `robot.py:278-308`, `isaacsim_articulation_cfg.py:235` |
| soft pos limit factor | 0.95 (only used by the unused `limits_dof_pos` reward) | `H/config_types/robot.py:193`; `H/simulator/shared/dof_limits.py:31-68` |

Backend-specific joint/actuator physics (not in the shared config):
- MuJoCo only: every joint has `frictionloss=0.1` and `solimplimit="0.97 0.995 0.001"` (MJCF default class,
  `xml:19`); `dof_damping=0` (EXT probe). Isaac Sim: friction 0, no damping (zero-gain drive).
- MuJoCo only: 8 limited fixed tendons couple ankle pitch/roll per foot (`xml:407-440`):
  `−0.2618·roll + 0.2618·pitch ≤ 0.13708`, `0.2618·roll + 0.2618·pitch ≤ 0.13708`,
  `−0.14967·roll − 0.2618·pitch ≤ 0.22846`, `0.14967·roll − 0.2618·pitch ≤ 0.22846`
  (solreflimit 0.003 1, solimplimit 0.99 0.999 0.0005 0 2). There is no such coupling in the URDF/Isaac Sim.
- MuJoCo only: motor actuators have `ctrlrange` that MuJoCo clamps (`ctrllimited=True`, EXT probe):
  hip_pitch/hip_roll/hip_yaw ±88, knee ±139, ankles ±50, waist_yaw ±88, waist roll/pitch ±50, arms ±25,
  wrist pitch/yaw ±5 (`xml:442-476`). **hip_roll is ±88 here although the config/URDF/joint
  `actuatorfrcrange` say ±139** (`xml:26,444,451`; `robot.py:218`; `urdf:101`). Torques are written to
  `data.ctrl` (`H/simulator/mujoco/mujoco.py:816-856`), so on MuJoCo/MJWarp hip roll saturates at 88 N·m.
- Isaac Sim: one `IdealPDActuatorCfg` over all 29 DoF with zero stiffness/damping, per-joint
  `effort_limit` and `velocity_limit` from the config lists (`isaacsim_articulation_cfg.py:225-237`);
  torques go in via `set_joint_effort_target` (`isaacsim.py:739-746`). Whether PhysX also enforces the
  URDF joint `velocity` limits as max joint velocity depends on the Isaac Lab URDF importer (external, unverified).
- Rigid-body solver knobs (Isaac Sim only): linear/angular damping 0, max lin/ang velocity 1000
  (`robot.py:548-555`), `max_depenetration_velocity=1.0` (`H/simulator/isaacsim/converters.py:117-127`).

### 1.5 Default pose, PD gains, action mapping

Default joint angles (`robot.py:350-380`): hip_pitch −0.312, knee 0.669, ankle_pitch −0.363 (both legs);
left shoulder_pitch 0.2, shoulder_roll +0.2, elbow 0.6; right shoulder_pitch 0.2, shoulder_roll −0.2,
elbow 0.6; all other joints 0. Initial root: pos (0, 0, 0.8), quat xyzw (0,0,0,1), zero velocities
(`robot.py:345-349`). With this pose the soles touch flat ground at pelvis height 0.757 m (EXT probe).

PD gains (substring-matched per joint, `joint_control.py:300-324`; values `robot.py:489-526`). They equal
`kp = I_arm·ω²`, `kd = 2·ζ·I_arm·ω` with ω = 2π·10 rad/s, ζ = 2 (comments `STIFFNESS_7520_14` etc.):

| joints | kp | kd |
|---|---|---|
| hip_pitch, hip_yaw, waist_yaw | 40.179238471 | 2.557889765 |
| hip_roll, knee | 99.098427777 | 6.308801854 |
| ankle_pitch, ankle_roll, waist_roll, waist_pitch | 28.501246196 | 1.814445687 |
| shoulder_pitch/roll/yaw, elbow, wrist_roll | 14.250623098 | 0.907222843 |
| wrist_pitch, wrist_yaw | 16.778327481 | 1.068141502 |

Action → torque (`joint_control.py:110-214`; config `robot.py:489-531`):
- `a_clip = clip(a, −100, 100)` (`clip_actions=True`, `action_clip_value=100`).
- `a_d` = `a_clip` or the previous step's `a_clip` (action delay, §4).
- At each of the 4 physics sub-steps, with current `q, q̇`:
  `τ = kp_scale ⊙ kp ⊙ (0.25·a_d + q0 − q) − kd_scale ⊙ kd ⊙ q̇`, then `τ = clip(τ, ±effort_limit)`
  (`control_type="P"`, `action_scale=0.25`, `clip_torques=True`; `joint_control.py:180-212`). `effort_limit`
  is the config list (`H/simulator/shared/dof_limits.py:62`), not the MJCF ctrlrange.
- `action_scales_by_effort_limit_over_p_gain=False` (default, `H/config_types/robot.py:74`), so the scale
  is 0.25 for every joint (`joint_control.py:326-337`).
- The action term config's own `scale=1.0, clip=None` (`H/config_values/loco/g1/action.py:7-12`) is never
  read by `JointPositionActionTerm`.
- Torque RFI is disabled (§9), so the RFI branch (`joint_control.py:203-208`) is inactive.

### 1.6 Timing

- Physics: `fps=200` → `sim_dt = 0.005 s`; `control_decimation=4` → policy dt 0.02 s (50 Hz); `substeps=1`
  (`H/config_values/simulator.py:20-22, 45-47, 71-73`); `dt = decimation·sim_dt` (`base_task.py:142-144`).
- Episode: `max_episode_length_s = 20.0` (`H/config_types/simulator.py:138`) →
  `max_episode_length = ceil(20/0.02) = 1000` (`base_task.py:145-146`); time-out fires when
  `episode_length_buf > 1000` (`H/managers/termination/terms/common.py:10-12`), i.e. an un-terminated
  episode has **1001** policy steps.

---

## 2. Simulators

### 2.1 Supported backends and preset default

- Presets: both use `simulator.isaacsim` (`H/config_values/loco/g1/experiment.py:22,42`).
  Registered alternatives: `mujoco` (classic CPU, `MujocoBackend.CLASSIC`) and `mjwarp`
  (`MujocoBackend.WARP`) (`H/config_values/simulator.py:37-86`).
- MuJoCo classic is single-env only (`H/simulator/mujoco/backends/classic_backend.py:34,63`) and is
  documented as evaluation/sim2sim only; MJWarp is the GPU training backend, "in beta"
  (`R/src/holosoma/README.md:5-7,41-43`). The README requires `--terrain.terrain-term.scale-factor=0.5`
  for PPO on MJWarp ("at full scale the policy's action distribution goes non-finite within the first few
  rollouts") (`R/src/holosoma/README.md:53-59,81`); nightly MJWarp cells use scale-factor 0.5 for both
  G1 presets (`R/.github/workflows/nightly-training.yaml:43-52`).

### 2.2 Isaac Sim configuration (Isaac Sim 5.1.0 / Isaac Lab v2.3.2, `R/scripts/versions.sh:3-4`)

- `SimulationCfg(dt=1/200, render_interval=4, physx=PhysxCfg(solver_type=1 (TGS),
  max_position_iteration_count=8, max_velocity_iteration_count=4, bounce_threshold_velocity=0.5,
  gpu_max_rigid_patch_count=10·2^15))` (`isaacsim.py:100-111`; values `H/config_values/simulator.py:20-30`).
- Articulation root: `solver_position_iteration_count=8`, `solver_velocity_iteration_count=4`
  ("(4, 0) -> (8, 4) necessary for reproducing FAR-tracking-implementation") (`isaacsim_articulation_cfg.py:166-171`).
- Global default physics material (applies to robot shapes, since the robot's `link_physics` has no
  `isaacsim` sub-config and the URDF importer authors no material): static 1.0, dynamic 1.0,
  restitution 0.0 (`isaacsim.py:114-118`; `isaacsim.py:255-259`; `H/simulator/isaacsim/object_spawner.py:156-171`;
  "the URDF importer authors no physics material", `object_spawner.py:177-179`).
  Combine modes are left at Isaac Lab's `RigidBodyMaterialCfg` default ("average"; external).
- Terrain (trimesh): `create_prim_from_mesh` with material static/dynamic 1.0, restitution 0.0 and
  default combine mode (average) (`scene_fixtures.py:119-135`; `H/config_values/terrain.py:40-42`). Only the
  plane terrain uses `friction_combine_mode="multiply"` (`scene_fixtures.py:97-110`).
- Contact sensor on every robot link: `update_period=0.0` (every physics step), `track_air_time=False`,
  `force_threshold=10.0` (unused by this task) (`isaacsim_articulation_cfg.py:246-264`).
  `contact_forces` = `net_forces_w` re-ordered to config body order (`isaacsim.py:722,734-737`) = forces of
  the **last** sub-step.
- A height-scanner RayCaster is attached to the pelvis for non-flat terrain (`scene_fixtures.py:82-93`)
  but no locomotion term reads it; terrain heights come from Warp ray casts (§11).
- Root state = Isaac Lab `root_state_w` (`isaacsim.py:703-707`): link-frame pose, and (Isaac Lab
  convention, external) the root **centre-of-mass** velocity. Writes use the same convention.
- `env_spacing=20`, `replicate_physics=True` (`H/config_types/scene.py:353,367`); irrelevant for
  placement because the task writes world positions from terrain origins (§8).

### 2.3 MuJoCo classic and MJWarp configuration

- World options: gravity (0,0,−9.81), `timestep = 1/fps = 0.005` (`scene_manager.py:90-100`;
  `H/simulator/mujoco/native_config.py:196-211`). All native options default to `None` = "preserve"
  (`H/config_types/mujoco.py:105-140`), so the compiled model uses MuJoCo defaults: Euler integrator,
  Newton solver, 100 iterations, 50 line-search iterations, pyramidal cone, tolerance 1e-8, impratio 1,
  no noslip, jacobian auto (EXT probe). The `physx` block of the `mujoco`/`mjwarp` presets
  (`H/config_values/simulator.py:48-53,74-79`) is not read by MuJoCo.
- MJWarp: `mjw.put_model` (`H/simulator/mujoco/backends/warp_backend.py:244`); contact pool 96 per env,
  `njmax = max(6·96, 4·nv) = 576` per world (nv = 35) (`H/simulator/mujoco/backends/native_config.py:83-87`);
  per-world expanded model fields: `body_pos, body_quat, geom_friction, geom_solref, geom_solimp,
  body_mass, body_inertia, body_ipos, dof_damping, cam_pos` (`warp_backend.py:313-331`) — **no `pair_*`
  field is expanded**. One CUDA graph per sub-step: `mjw.step` + `mjw.contact_force` + per-body accumulation
  (`warp_backend.py:334-361`).
- Torques: `data.ctrl[actuator(dof)] = τ` (`mujoco.py:841-856`), then MuJoCo clamps to `ctrlrange`.
- Root state: freejoint `qpos/qvel`; linear velocity is the world-frame velocity of the pelvis **frame
  origin**; angular velocity converted body→world on read and world→body on write
  (`mujoco.py:922-946, 801-807`).
- Contact forces: per-body net contact force from `mj_contactForce` (classic) or `mjw.contact_force`
  (Warp), rotated to world and accumulated with Newton's third law
  (`classic_backend.py:242-293`; `warp_backend.py:146-168`), evaluated after the last sub-step.
- Terrain geom: for `mesh_type="trimesh"` MuJoCo builds an **hfield** from the raw heightfield, not the
  trimesh ("Use heightfield to reduce penetrations") (`scene_manager.py:173-177, 262-335`); geom name =
  `TerrainTermCfg.name = "floor"` (`H/config_types/terrain.py:209-210`; `scene_manager.py:319`), friction
  [0.7, 0.005, 0.001] ("Ignore terrain config"), solimp [0.99, 0.99, 0.01, 0.5, 2], solref [0.001, 1]
  (`scene_manager.py:327-334`), contype 2/conaffinity 1 (`scene_manager.py:188-193`). The terrain config's
  static/dynamic friction 1.0 is ignored by MuJoCo.

### 2.4 Explicit foot–floor contact pairs and friction DR per backend

The MJCF declares 14 explicit pairs, one per sole capsule:
`<pair geom1="left_footK_collision" geom2="floor" solref="0.01 1" friction="0.8 0.8"/>`
(`xml:388-405`).

How the pairs resolve in Holosoma's MuJoCo scene:
- The terrain is added before the robot so that robot XML contact references can resolve
  ("Always add robot after terrain, in case it references ground/floor, etc for contacts",
  `mujoco.py:340-343`), and the terrain geom is named `floor` (`H/config_types/terrain.py:209-210`).
  Pairs are not rebuilt or rewritten anywhere (no `pair` handling in `H/simulator/mujoco/`).
- EXT probe (Appendix A, replicating `scene_manager.add_terrain`/`add_robot` with prefix `robot_`): the
  compiled model has `npair = 14`, pairs `floor ↔ robot_left_foot1_collision` …, i.e. the robot-side names
  get the prefix while `floor` resolves to the unprefixed terrain geom. Compiled pair parameters:
  `pair_friction = [0.8, 0.8, 0.005, 0.0001, 0.0001]`, `pair_solref = [0.01, 1]`,
  `pair_solimp = [0.9, 0.95, 0.001, 0.5, 2]` (MuJoCo default), `pair_dim = 3`, margin 0, gap 0.
  With a plane floor each touching capsule yields exactly one contact (hfield: two), all with the pair
  parameters; no duplicate dynamic (contype/conaffinity) contact is generated for these geoms although
  their masks would allow one (observed; the compiled `pair_signature` is 7 = world × left-ankle body).

What each backend's friction randomization writes (`randomize_friction_startup`,
`H/managers/randomization/terms/locomotion.py:751-841`):

| Backend | Field written | Reaches sole–ground contact? | Effective sole friction |
|---|---|---|---|
| Isaac Sim | `root_physx_view.set_material_properties` for **all robot shapes**: static `μ_s`, dynamic `μ_d`, restitution untouched (`H/simulator/isaacsim/events.py:259-352`, called at `locomotion.py:796-819`) | yes (no pairs in PhysX) | PhysX combine of robot material with terrain μ=1.0 under "average" (external default): `(μ_robot + 1.0)/2` |
| MuJoCo classic | `model.geom_friction[:, 0]` (sliding only) for **every geom in the model**, robot and `floor` alike, one value per env (`randomize_field(..., ranges={0: friction_range}, operation="abs", shared_across_entities=True)`, `locomotion.py:821-836`; `H/simulator/mujoco/backends/randomization.py:333-334, 416-419, 438-439`) | **no** — sole contacts use `pair_friction` | fixed 0.8 (DR has no effect on the soles) |
| MJWarp | same `geom_friction` write on the per-world expanded array (`warp_backend.py:313-331`; `randomization.py:161-171`) | **no** — MJWarp contact params use `pair_friction[worldid % n, pairid]` whenever `pairid > −1` (EXT: `mujoco_warp/_src/collision_core.py:335-340`, pair ids assigned in `io.py:604-613`); `pair_friction` is not expanded or written | fixed 0.8 |

- EXT probe: after setting `geom_friction[:,0] = 0.3` for all geoms, every sole contact still reports
  friction 0.8 (Appendix A). **Friction DR is silently ineffective on the soles on both MuJoCo backends.**
  It does act on all non-pair contacts (pelvis, thighs, shins, torso, arms touching the terrain), whose
  friction is `max(geom1, geom2)` = the per-env DR value because the floor geom was overwritten too.
- Non-pair MuJoCo contact parameters (EXT probe, pelvis on floor, before DR): friction
  [1.0, 1.0, 0.005, 0.001, 0.001] (element-wise max of robot default [1, 0.005, 0.0001] and floor
  [0.7, 0.005, 0.001]), solref [0.0105, 1] (solmix average of 0.02 and 0.001), solimp
  [0.945, 0.97, 0.0055, 0.5, 2]. MuJoCo's default `refsafe` keeps time constants ≥ 2·timestep = 0.01.
- Restitution: not randomized on any backend (Isaac Sim channel is `None`, `locomotion.py:814`; MuJoCo has
  no restitution). solref/solimp: no DR term writes them (MJWarp expands `geom_solref/solimp` but nothing
  uses them in this preset).
- Isaac Sim sampling detail: with 64 buckets, static and dynamic channels use differently permuted copies of
  the same 64 quantiles but the same bucket index per env, then `μ_d = min(μ_s, μ_d)`
  (`events.py:336-349`). So `μ_s ~ U[0.5,1.25]` (64-atom staircase) and `μ_d` is the minimum of two
  (nearly independent) such draws. MuJoCo has a single friction coefficient.
- Other DR terms are field writes that do reach the dynamics on all backends: masses/inertia
  (`events.py:68-139`; `randomization.py:174-189`), torso CoM (`events.py:142-188`; `body_ipos`), PD-gain
  scales and delay (pure Python, backend-independent).

---

## 3. Observations

Config: `g1_29dof_loco_single_wolinvel` (`H/config_values/loco/g1/observation.py:5-117`). "single" =
history length 1; "wolinvel" = **without base linear velocity in the actor**; only the critic gets
`base_lin_vel` (scale 2.0) (`observation.py:64-68`).

Processing per term (`H/managers/observation/base.py:60-135`):
1. raw value; 2. if the group has `enable_noise`: `obs += U(−noise, +noise)` element-wise, drawn from a
dedicated `torch.Generator` seeded `initial_seed + 0x0B5E` (`base.py:72-75`; `H/managers/observation/manager.py:46-49`);
3. `obs *= scale` (noise is applied **before** scaling); 4. per-term clip (none configured);
5. global clip ±100 (`H/config_types/observation.py:60-61`; `base.py:99-106`).
Groups concatenate terms in **alphabetical order of term name**, not config order
(`manager.py:183-194`). History length 1 for both groups → no stacking (`observation.py:10,62`).

Actor (`enable_noise=True`, `observation.py:7-58`), 100 dims:

| idx | term | dim | definition | scale | noise (raw units) |
|---|---|---|---|---|---|
| 0-28 | actions | 29 | last raw policy action `action_manager.action` (unclipped, unscaled) | 1.0 | 0 |
| 29-31 | base_ang_vel | 3 | `R_base^T ω_root,world` | 0.25 | 0 |
| 32 | command_ang_vel | 1 | yaw-rate command | 1.0 | 0 |
| 33-34 | command_lin_vel | 2 | (vx, vy) command | 1.0 | 0 |
| 35-36 | cos_phase | 2 | cos(φ_left), cos(φ_right) | 1.0 | 0 |
| 37-65 | dof_pos | 29 | `q − q0` | 1.0 | ±0.01 |
| 66-94 | dof_vel | 29 | `q̇` | 0.05 | ±0.1 |
| 95-97 | projected_gravity | 3 | `R_base^T (0,0,−1)` | 1.0 | 0 |
| 98-99 | sin_phase | 2 | sin(φ_left), sin(φ_right) | 1.0 | 0 |

Critic (`enable_noise=False`, `observation.py:59-115`), 103 dims: same terms plus `base_lin_vel`
(`R_base^T v_root,world`, scale 2.0) inserted alphabetically after `base_ang_vel`: actions 0-28,
base_ang_vel 29-31, base_lin_vel 32-34, command_ang_vel 35, command_lin_vel 36-37, cos_phase 38-39,
dof_pos 40-68, dof_vel 69-97, projected_gravity 98-100, sin_phase 101-102.

Definitions: `H/managers/observation/terms/locomotion.py:25-179` (quaternions xyzw, `quat_rotate_inverse`;
gravity axis (0,0,−1) at 25-27). Base frame = root body (pelvis). Phase dims are 2
(`H/managers/command/terms/locomotion.py:143-146`; inference config `R/src/holosoma_inference/holosoma_inference/config/config_values/observation.py:36-58`).
Root linear velocity semantics differ by backend (COM on Isaac Sim, frame origin on MuJoCo, §2).

Terminal observations: for time-out envs only, `step_terminal` computes the full observation (with actor
noise) of the pre-reset state and returns it as `extras["final_observations"]` (`base_task.py:524-544`).

---

## 4. Actions

- Policy output 29 dims, consumed in config DoF order. Raw actions are stored (`action_manager.action`);
  `prev_action` keeps the previous step's raw action (`H/managers/action/manager.py:176-178`).
- Clip ±100 (`joint_control.py:122-129`). Scale 0.25 and default offset inside the PD law (§1.5).
- Delay (enabled, `H/config_values/loco/g1/randomization.py:15-21`): per-env integer
  `action_delay_idx ∈ {0, 1}` (inclusive, uniform), drawn at setup and redrawn at every reset
  (`H/managers/randomization/terms/locomotion.py:269-288, 443-488`). A queue of length `max_delay+1 = 2`
  shifts once per **policy step** (`joint_control.py:86-89, 142-154`); the applied action is
  `queue[idx]` → 0 or 20 ms delay. The queue is zeroed on reset (`locomotion.py:480-481`;
  `joint_control.py:224-229`), so a delayed env applies a zero action (= default pose target) on its
  first step.
- No filtering/smoothing; PD torques recomputed every 5 ms sub-step with the held target.
- After reset, `action` and `prev_action` are zeroed (`manager.py:204-210`).
- FastSAC only: the actor emits `tanh(μ)·b`, with per-joint `b_i = max(|lower_i − q0_i|, |upper_i − q0_i|) / 0.25`
  so that ±1 reaches the farther joint limit (`H/agents/fast_sac/fast_sac_agent.py:114-151, 260-261`); the
  env then applies its own ±100 clip and 0.25 scale as above. PPO samples an unbounded Gaussian.

---

## 5. Commands

Config `g1_29dof_command` (`H/config_values/loco/g1/command.py:5-38`). Two stateful terms share one
instance each across setup/reset/step (`H/managers/command/manager.py:205-230`); registration order is
`locomotion_gait` then `locomotion_command`, which is also the call order at reset and step
(`manager.py:156-174`).

`LocomotionCommand` (`H/managers/command/terms/locomotion.py:13-118`):
- 3-dim command `[vx, vy, ωz]` (`command_dim=3`, line 24), body frame.
- Resample: `vx ~ U[−1,1]`, `vy ~ U[−1,1]`, `ωz ~ U[−1,1]` independently (lines 80-97; ranges `command.py:20-25`).
  Then, with probability `stand_prob = 0.2` (`torch.rand <= 0.2`), all three set to 0 (lines 108-111).
  Each resample also resamples the gait frequency (lines 99-106).
- When: at every reset (lines 38-46) and in `step()` for every env with
  `episode_length_buf % int(10.0/0.02) == 0`, i.e. every 500 steps = 10 s (lines 48-66; resampling time
  `command.py:6-8`). Because `step()` runs after the reset block, freshly reset envs (buffer 0) are
  resampled **twice** in the same step; the second draw is what the first observation shows.
- No heading control: the `heading: [−3.14, 3.14]` range (`command.py:24`) is never read. No curriculum on
  command ranges.
- No resampling during evaluation (`line 50`).

`LocomotionGait` (`H/managers/command/terms/locomotion.py:121-225`):
- Params: `gait_period=1.0` → mean frequency 1 Hz; `gait_period_randomization_width=0.2`
  (`command.py:12-15`); `randomize_phase=True`, `stand_phase_value=π` (defaults, lines 129-130).
- Frequency per env `f ~ U[0.8, 1.2]` Hz (mean ± width; lines 185-190), `phase_dt = 2π·0.02·f` (192).
- Offsets at reset: `off_L ~ U[−π, π]`, `off_R = fmod(off_L + 2π, 2π) − π` (= off_L shifted by π)
  (lines 209-213); eval: (0, −π) (206-208).
- Each step (after the reset block): `φ = fmod(episode_length_buf·phase_dt + off + π, 2π) − π`
  (lines 158-159); if `‖cmd_xy‖ < 0.01` and `|ωz_cmd| < 0.01`, both phases are set to π (165-172).
  Since gait steps before command, the stand override uses the commands from before this step's
  resample, and a frequency resample makes φ jump (φ is recomputed from the episode counter, not
  integrated).

---

## 6. Rewards

Manager (`H/managers/reward/manager.py:126-184`): `r = Σ_i w_i · dt · r_i(s)` with `dt = 0.02`
(`base_task.py:585-588`); terms evaluated in config order; zero-weight terms skipped (62-63);
`only_positive_rewards=False` → no clipping of the total (`H/config_values/loco/g1/reward.py:6,100`).
Rewards are computed on the post-physics state **before** resets (`base_task.py:520-531`). Terms tagged
`penalty_curriculum` have their weight multiplied by the curriculum scale `s` (§10). No termination
penalty term is configured.

| term | weight PPO / FastSAC | tag | formula `r_i` (citation in `H/managers/reward/terms/locomotion.py`) |
|---|---|---|---|
| tracking_lin_vel | 2.0 / 2.0 | – | `exp(−‖c_xy − v^b_xy‖² / 0.25)`, `v^b = R_base^T v_root` (143-157) |
| tracking_ang_vel | 1.5 / 1.5 | – | `exp(−(c_ωz − ω^b_z)² / 0.25)` (160-175) |
| penalty_ang_vel_xy | −1.0 / −1.0 | yes | `ω^b_x² + ω^b_y²` (178-188) |
| penalty_orientation | −10.0 / −10.0 | yes | `g^b_x² + g^b_y²`, `g^b = R_base^T(0,0,−1)` (79-89) |
| penalty_action_rate | −2.0 / −2.0 | yes | `‖a_{t−1} − a_t‖²` on raw actions; `a_{t−1}=0` after reset (65-76) |
| feet_phase | 5.0 / 5.0 | – | `exp(−[(h_L − ĥ(φ_L))² + (h_R − ĥ(φ_R))²] / 0.008)`, swing_height 0.09 (256-288) |
| pose | −0.5 / −0.5 | yes | `Σ_j w_j (q_j − q0_j)²` (291-319) |
| penalty_close_feet_xy | −10.0 / −10.0 | yes | `1[ d_lat < 0.15 ]` (191-215) |
| penalty_feet_ori | −5.0 / −5.0 | yes | `‖g^L_xy‖ + ‖g^R_xy‖`, `g^F = R_foot^T(0,0,−1)` (92-109) |
| alive | **1.0 / 10.0** | – | `1` (338-347) |

Weights/params: `H/config_values/loco/g1/reward.py:8-95` (PPO) and `102-189` (FastSAC); the two configs
differ only in `alive` (`reward.py:91-95` vs `185-189`).

Details:
- "sigma" is a plain divisor: `exp(−err/σ)` with σ = 0.25 or 0.008, not `2σ²`.
- `ĥ(φ)` (cubic Bézier, `locomotion.py:29-40`): `x = (φ+π)/(2π)`, `B(t) = t³ + 3t²(1−t)`;
  `ĥ = 0.09·B(2x)` for `x ≤ 0.5`, else `ĥ = 0.09 − 0.09·B(2x−1)`. Peak 0.09 m at φ = 0, zero at φ = ±π
  (stand phase π gives ĥ = 0).
- `h_L, h_R` = foot heights from the terrain manager: z of `*_foot_contact_point` minus the terrain height
  found by a Warp ray cast straight down from 0.25 m above that point
  (`H/managers/terrain/terms/locomotion.py:187-194`), updated every step before rewards (§0). Index 0 is
  left (body order), matching `phase[:, 0]`.
- `pose` weights per DoF in config order (`reward.py:45-75`): legs [0.01, 1.0, 5.0, 0.01, 5.0, 5.0]
  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll) ×2; waist and all arm joints 50.0.
- `penalty_close_feet_xy`: feet = `ankle_roll_link` origins; `yaw = atan2` of the base x-axis in world;
  `d_lat = |cos(yaw)(y_L − y_R) − sin(yaw)(x_L − x_R)|`, threshold 0.15 m (`reward.py:79-84`).
- `penalty_feet_ori` uses the `ankle_roll_link` orientations (`_rigid_body_rot`, xyzw).
- Timing: `a_t`/`a_{t−1}` are this step's and last step's raw actions; φ used here was computed at the end
  of the previous step (the phase the policy observed when choosing `a_t`).
- Logged `Episode/rew_<term>` = mean over finished episodes of `Σ_t w·dt·r / 20 s`
  (`reward/manager.py:229-239`; `H/agents/modules/logging_utils.py:247-261`); a perfect full-length
  lin-vel episode logs 2.0·0.02·1001/20 ≈ 2.0.

---

## 7. Terminations

Config `g1_29dof_termination` (`H/config_values/loco/g1/termination.py:5-19`), manager
`H/managers/termination/manager.py:82-111`:
- `contact`: `any_b ‖F_b‖ > 1.0 N` over the 13 termination bodies (pelvis, 6 hip links, 6 shoulder links)
  (`H/managers/termination/terms/locomotion.py:21-31`). `F_b` is the net contact force of the last physics
  sub-step (§2). Of those bodies only pelvis (sphere), hip_roll_link (thigh) and shoulder_yaw_link carry
  colliders, so the episode ends when the pelvis, a thigh or an upper arm touches the ground (self
  collisions are off).
- `timeout`: `episode_length_buf > max_episode_length (1000)`, `is_timeout=True`
  (`termination.py:14-17`; `common.py:10-12`).
- `reset_buf = contact | timeout`; `time_out_buf = timeout` (`base_task.py:601-610`); `extras["time_outs"]`
  (`base_task.py:480`).
- Time-outs are truncations in both algorithms: PPO adds `γ·V(final critic obs)` to the reward of the
  time-out step (`H/agents/ppo/ppo.py:413-431`); FastSAC stores the true final observation and bootstraps
  when `truncation | ~done` (`fast_sac_agent.py:426-441, 726-741`).
- Not configured (exist but unused): gravity tilt, base height, DoF position/velocity/torque limits
  (`H/managers/termination/terms/locomotion.py:34-83`).

---

## 8. Resets

Order: §0 item 4. State written (`H/envs/locomotion/locomotion_manager.py:164-343` plus DR):
- Joint positions: `_reset_dofs` sets `q = q0 ⊙ U(0.5, 1.5)` with the global torch RNG
  (`locomotion_manager.py:269-286`), then the reset DR term `randomize_dof_state` overwrites it with
  `q = q0 ⊙ s`, `s ~ U[0.5, 1.5]` per (env, joint), bias 0, `q̇ = 0` (`joint_vel_range [0,0]`)
  (`H/config_values/loco/g1/randomization.py:83-91`; `H/managers/randomization/terms/locomotion.py:491-526`).
  Joints with `q0 = 0` reset exactly to 0; no clipping to joint limits.
- Root position: `(0, 0, 0.8) + env_origin + (δx, δy, 0)`, `δx, δy ~ U[−1, 1]` m
  (`xy_offset_range=1.0`); terrain height is **not** re-queried (`query_terrain_height=False`), so z stays
  `0.8 + origin_z` (`locomotion_manager.py:302-338`; `H/config_types/terrain.py:61-107`).
- Root orientation: identity (xyzw (0,0,0,1)); no yaw randomization (`robot.py:347`; `locomotion_manager.py:303`).
- Root velocities: all 6 components `~ U[−0.5, 0.5]` (m/s, rad/s, world frame) (`locomotion_manager.py:340-343`).
- Env origins: fixed for the whole run, one tile per env, chosen at construction (§11): random row, column
  `floor(env_id / (N/20))` (`H/simulator/shared/terrain.py:150-157`;
  `H/managers/terrain/terms/locomotion.py:116-128`).
- Buffers: `episode_length_buf = 0` (`locomotion_manager.py:190`); writes are flushed by a forward-kinematics
  pass in the same step (`locomotion_manager.py:137-140`).
- Spawn clearance: soles touch flat ground at pelvis z ≈ 0.757 in the default pose (EXT), so robots drop
  ≥ 4 cm (more on rough tiles, whose surface lies 2.5-7 cm below 0 while `origin_z` is the max height of
  the central 1 m², `terrain.py:282-287`); the leg-angle scaling changes this.
- Start of training: `reset_all` resets every env and executes one zero-action step (`base_task.py:265-276`;
  `locomotion_manager.py:160-162`). PPO then sets `episode_length_buf ~ randint[0, 1000)`
  (`init_at_random_ep_len=True`, `H/agents/ppo/ppo.py:361-364`); FastSAC does not.
- Evaluation overrides (not training): all envs at tile (0,0), no xy offset, 100000 s episodes
  (`H/config_types/experiment.py:101-110, 189-222`).

---

## 9. Domain randomization

Config `g1_29dof_randomization` (`H/config_values/loco/g1/randomization.py:5-104`). All DR draws (except
pushes' velocity and `_reset_dofs`) go through the keyed sampler: `value = hash(seed=42, term name, stage,
env id, per-env episode counter, coords)` (SplitMix64; `H/utils/sampler.py:187-214, 297-432`;
`base_task.py:131-132, 300-304`). `[lo, hi]` = continuous uniform (`sampler.py:89-99`); `draw_int` is
inclusive.

| term | when | range / value | applied to | citation |
|---|---|---|---|---|
| link mass scale | setup only | ×U[0.9, 1.2], independent per body | pelvis, L/R hip_yaw, hip_roll, hip_pitch, knee links (9 bodies; `robot.py:382-392`); inertia ×same ratio | `randomization.py:46-54`; `terms/locomotion.py:615-748` |
| added base mass | setup only | +U[−1.0, 3.0] kg | **torso_link** (`torso_name`), inertia ×ratio | same |
| friction | setup only | U[0.5, 1.25], 64 quantile buckets, one value per env | Isaac Sim: all robot shapes (static; dynamic = min of two draws); MuJoCo/MJWarp: `geom_friction[:,0]` of all geoms — **not the sole pairs** (§2.4) | `randomization.py:55-61`; `terms/locomotion.py:751-841` |
| base CoM offset | setup only | +U[−0.05, 0.05] m on x, y, z independently | **torso_link** CoM (`body_ipos` / PhysX coms) | `randomization.py:62-68`; `terms/locomotion.py:529-612` |
| PD gain scales | every reset (+ allocated at setup) | `kp_scale, kd_scale ~ U[0.9, 1.1]` per (env, joint), independent | multiply kp/kd in the PD law | `randomization.py:36-45`; `terms/locomotion.py:158-266`; `joint_control.py:186-190` |
| action delay | setup + every reset | integer {0, 1} policy steps | delay queue (§4) | `randomization.py:15-21, 80-82`; `terms/locomotion.py:269-288, 443-488` |
| pushes | step | every `int(U[5,10] s / 0.02)` = 250-500 steps per env; sets world root `(v_x, v_y)` **:=** `U[−1,1]` m/s each (overwrite, not add; global torch RNG) | root linear velocity xy | `randomization.py:7-14, 71-73, 77-79, 96-103`; `terms/locomotion.py:46-155, 330-357, 921-948`; `locomotion_manager.py:247-265` |
| joint init state | every reset | `q = q0·U[0.5,1.5]`, `q̇ = 0` | §8 | `randomization.py:83-91` |
| torque RFI | – | disabled (`enabled: False`, `rfi_lim 0.1`; `enable_rfi_lim: False`) | – | `randomization.py:22-28, 43` |
| default-pose bias | – | disabled (`[-0.01, 0.01]`, `enabled: False`) | – | `randomization.py:29-35` |

Push mechanics: counter +1 every step (after resets), zeroed at reset and after a push; push when
`counter == int(interval_s / dt)`; a new interval is drawn at reset and after each push
(`terms/locomotion.py:91-97, 123-129, 141-149, 941-948`). The write lands at the next physics step
(`H/simulator/base_simulator/state.py:1-16`), but the same step's critic observation already reads the
overwritten velocity because observations are computed after `randomization_manager.step()` (§0). No
pushes during evaluation (`terms/locomotion.py:938`). Velocity semantics follow §2 (COM on Isaac Sim).

Not randomized: terrain friction as an independent quantity (Isaac Sim keeps the terrain at μ = 1.0; on
MuJoCo the floor's `geom_friction` is overwritten with the same per-env value as the robot), restitution,
motor strength beyond kp/kd, observation latency, joint friction/damping/armature, gravity, IMU bias.

---

## 10. Curriculum

Configs: `g1_29dof_curriculum` (PPO, `H/config_values/loco/g1/curriculum.py:5-30`) and
`g1_29dof_curriculum_fast_sac` (`curriculum.py:32-57`). Two class terms in this order:
`average_episode_tracker`, `penalty_curriculum` (`H/managers/curriculum/manager.py:70-86,135-153`).
No terrain curriculum (terrain levels are fixed, §11), no command-range curriculum.

Average episode length tracker (`H/managers/curriculum/terms/locomotion.py:16-108`):
- On each reset with `n > 0` finished episodes: `avg ← avg·(1 − w) + mean(lengths)·w`,
  `w = min(n / K, 1)`, `K = max(1, int(1000·num_envs/4096))` = 1000 for 4096 envs (lines 22-24, 71-81).
  Lengths are `episode_length_buf` at reset (control steps, includes PPO's random initial offset)
  (`base_task.py:293`). Initial `avg = 0` (line 25).
- The manager-level `params={"num_compute_average_epl": 1000}` (`curriculum.py:6-8,33-35`) is never passed to
  the term (it reads its own empty params); the default 1000 gives the same number.

Penalty curriculum (`terms/locomotion.py:111-243`):
- At setup: collects reward terms tagged `penalty_curriculum` (ang_vel_xy, orientation, action_rate, pose,
  close_feet_xy, feet_ori) and multiplies their weights by `initial_scale` (lines 138-172).
- `reset(env_ids)` — called by `reset_envs_idx` **every control step, even with no resets**, and it does not
  check for an empty id list (lines 174-193; `base_task.py:312-313,531`):
  `if avg < 150: s ← s·(1 − degree)` ; `elif avg > 750: s ← s·(1 + degree)`; `s ← clip(s, min, max)`;
  weights ← original·s. `step()` only re-clamps (231-243).
- So the scale changes geometrically **per policy step** (all envs share one scale), not per episode.

| | PPO | FastSAC |
|---|---|---|
| initial_scale | 0.1 | 0.5 |
| min / max | 0.0 / 1.0 | 0.5 / 1.0 |
| level_down / level_up thresholds | 150 / 750 steps (3 s / 15 s) | 150 / 750 |
| degree per step | 0.00025 | 0.001 |
| steps from initial to 1.0 (avg > 750) | ln 10 / ln 1.00025 ≈ 9.2k steps ≈ 384 PPO iterations | ln 2 / ln 1.001 ≈ 693 steps = 693 FastSAC iterations |

(`curriculum.py:14-26, 41-53`.) PPO's `min_scale = 0` lets the penalties shrink toward 0 while episodes are
short; FastSAC's penalties never drop below half weight.

---

## 11. Terrain

Config `terrain_locomotion_mix` (`H/config_values/terrain.py:31-58`): `mesh_type=TRIMESH`,
`horizontal_scale=0.1 m`, `vertical_scale=0.005 m`, `border_size=40 m`, 8×8 m tiles, `num_rows=10`
(difficulty axis), `num_cols=20`, proportions flat 0.2 / rough 0.6 / low_obstacles 0.2
(`smooth_slope`, `rough_slope` = 0 and filtered out), `max_slope=0.3`, `slope_treshold=0.75`,
friction 1.0/1.0, restitution 0. Defaults: `scale_factor=1.0`, `name="floor"`, spawn as §8
(`H/config_types/terrain.py:110-210`).

Generation (`H/simulator/shared/terrain.py:95-141, 208-287`), using the global NumPy RNG seeded with the
training seed (`H/train_agent.py:230-234`; `H/utils/common.py:96-117`):
- Height field `int16[1600, 2400]` (x = 10·80 + 2·400 rows, y = 20·80 + 2·400 cols), border flat at 0.
- Each of the 200 tiles independently: type by `np.random.choice` with the proportions; difficulty
  `d ∈ {0.5, 0.75, 0.9}` uniformly (slopes would use the row index, but none are enabled) (lines 216-227).
- flat: 0 (289-299).
- rough: per 0.1 m pixel `z ~ U[−0.025 − 2·0.025·d/0.9, −0.025)` m, i.e. entirely below 0, down to
  −0.075 m (d=0.9), then truncated toward zero to 5 mm units when copied into the int16 map
  (lines 301-317; 276) → heights −0.025 … −0.070 m.
- low_obstacles: flat tile with 30 random 0.8×0.8 m (8×8 px) square pits (may overlap) of depth
  `0.03·d/0.9` → stored −3/−4/−6 units = −0.015/−0.020/−0.030 m after int16 truncation (lines 446-471;
  computed in Appendix A).
- Tile origin = tile centre `((i+0.5)·8, (j+0.5)·8)`, z = max height of the central 1×1 m (lines 278-287).
- Trimesh: `convert_heightfield_to_trimesh` (two triangles per cell, vertices at `k·0.1 − 40`)
  (`H/utils/terrain_utils.py:214-323`; `terrain.py:136-141`). The vertical-wall correction triggers for
  neighbour differences > 0.75·0.1/0.005 = 15 units; this mix never exceeds 14, so the trimesh is the
  plain heightfield interpolation (rough/flat tile boundaries are 2.5-7 cm ramps over 0.1 m).
- No terrain curriculum (`terrain_move_up/down_ratio` are never read, `H/config_types/terrain.py:152-156`).

Representation per backend:
- Isaac Sim: the trimesh as a static triangle-mesh collider at `/World/ground` (`scene_fixtures.py:119-135`).
- MuJoCo/MJWarp: an **hfield** built from the int16 map: `nrow = 2400` (y), `ncol = 1600` (x),
  `size = [80, 120, height_range, ~1e-9]`, centred at (40, 80) (`scene_manager.py:262-335`). MuJoCo spaces
  hfield samples `2·size/(n−1)` apart, so sample spacing is 160/1599 = 0.100063 m in x and
  240/2399 = 0.100042 m in y instead of 0.1 m (EXT probe, Appendix A: toy step ramp lands at [0.55, 0.66]
  vs trimesh [0.5, 0.6]). The physical MuJoCo ground is therefore shifted relative to the trimesh by
  `k·6.3e-5` m (x) / `k·4.2e-5` m (y) at sample k: 2.5-7.5 cm (x) and 1.7-8.3 cm (y) over the robot tiles,
  ~10 cm at the far edges. Foot heights (Warp ray casts on the trimesh) are computed against this shifted
  surface on MuJoCo/MJWarp.

Terrain heights in observations/rewards/terminations:
- Observations: none (no height scan in either group).
- Rewards: `feet_phase` uses `feet_heights` relative to the trimesh (§6). `base_heights` (interquartile
  mean over a 5×5 grid, ±0.2 m, around the yaw-rotated base, ray starts +0.25 m) is computed every step
  (`H/managers/terrain/terms/locomotion.py:130-185`) but only the unused `base_height` reward reads it.
- Terminations: none use terrain height. Spawn does not query terrain height.
- Ray misses return `inf` hit positions → foot height −∞ → `feet_phase` → 0
  (`H/utils/warp_utils.py:55-86`); only possible off the mesh.

---

## 12. Training baselines

Shared: `num_envs = 4096`, seed 42, headless (`H/config_types/experiment.py:59-73`); presets override
neither. Symmetry: mirrored copies (joint map + sign flips `robot.py:431-487`; base_ang_vel → (−x, y, −z),
projected_gravity/base_lin_vel/command_lin_vel → y negated, command_ang_vel negated, sin/cos phase
channels swapped) (`H/agents/modules/augmentation_utils.py:9-19, 242-304, 306-400, 453-534`).

PPO (`H/config_values/algo.py:23-70`; preset sets `num_learning_iterations=25000`, `use_symmetry=True`,
`H/config_values/loco/g1/experiment.py:21`):
- 24 steps/env per iteration (98,304 transitions), 8 epochs × 4 minibatches (24,576 samples; ×2 with
  symmetry), one fixed permutation reused across epochs (`H/agents/modules/data_utils.py:107-129`).
- clip 0.2, value clip 0.2 (clipped value loss), value coef 1.0, entropy coef 0.01, γ 0.99, λ 0.95,
  advantages standardized per rollout (`ppo.py:460-469, 515-619`; `H/agents/modules/advantages.py:8-42`).
- Separate AdamW optimizers (weight decay 0.001) for actor and critic, both start at 1e-5; adaptive LR per
  minibatch: KL > 0.02 → lr/1.5, KL < 0.005 → lr·1.5, clamped to [1e-5, 1e-2] (`ppo.py:227-232, 641-652`).
  Grad-norm clip 1.0 per network (`ppo.py:498-499`).
- Actor/critic MLPs [512, 256, 128], ELU, linear output (`algo.py:54-67`; `H/agents/modules/modules.py:173-194`);
  state-independent std parameter initialised to 0.8 (`H/agents/modules/ppo_modules.py:30`).
- Symmetry: surrogate/value losses on original + mirrored samples; actor symmetry loss coef 1.0 (MSE between
  mean action on mirrored obs and mirrored mean action); critic symmetry coef 0.0 (`ppo.py:526-608`).
- No observation normalization (`empirical_normalization=False`); `init_at_random_ep_len=True`;
  checkpoints every 100 iterations.
- Budget: 25,000 iterations × 98,304 ≈ 2.46·10⁹ env steps.

FastSAC (`H/config_values/algo.py:72-119`; preset sets `num_learning_iterations=50000`, `use_symmetry=True`,
`experiment.py:41`):
- One env step per iteration (4096 transitions); replay 1024 per env (4.19M); global batch 8192
  (= 2 per env); 8 critic updates per iteration, actor updated when `i % 4 == 1` (2 per iteration);
  target soft update τ = 0.125 after each critic update; learning starts after step 10
  (`fast_sac_agent.py:764-808`).
- γ 0.97, 1-step returns; distributional critic: 2 Q-nets, 101 atoms on [−20, 20]; lr 3e-4 (actor, critic,
  α); α init 0.001 with autotune, target entropy `−29·0.0 = 0`; AdamW weight decay 0.001; no grad clipping
  (`max_grad_norm=0`); bf16 AMP; `torch.compile`.
- Actor: obs → 512 → 256 → 128 (Linear + LayerNorm + SiLU each) → mean and log-std heads; log-std squashed
  into [−5, 0]; tanh-squashed actions scaled per joint (§4) (`H/agents/fast_sac/fast_sac.py:64-99`).
  Critic: (obs, act) → 768 → 384 → 192 → 101 (LayerNorm + SiLU) (`fast_sac.py:232-260`).
- Running mean/std observation normalization for actor and critic inputs (eps 0.01), updated from
  replay samples (`H/agents/fast_sac/fast_sac_utils.py:246-290`; `fast_sac_agent.py:251-257, 603-606`).
- Symmetry = data augmentation of sampled batches (`fast_sac_agent.py:560-597`).
- Budget: 50,000 × 4096 ≈ 2.05·10⁸ env steps; saves every 1000 steps.

Documented results / hardware:
- Nightly thresholds on the mean of the last 100 logged values
  (`R/tests/nightly/nightly.py:209-237`): PPO (5000 iterations): `Episode/rew_tracking_ang_vel ≥ 0.7`,
  `Episode/rew_tracking_lin_vel ≥ 0.55`; FastSAC (50,000 iterations): ≥ 0.8 and ≥ 0.95
  (`H/config_values/loco/g1/experiment.py:32-35, 52-55`). Maxima are ≈ 1.5 and ≈ 2.0 (§6).
- Nightly runs on AWS CodeBuild L4 GPU runners (1× and 4× GPU cells), Isaac Sim cells for both presets,
  MJWarp cells with terrain scale-factor 0.5, job timeout 1000 min (`R/.github/workflows/nightly-training.yaml:18-63`).
- No wall-clock training times or expected-return curves are documented in the README or docs
  (searched `R/README.md`, `R/src/holosoma/README.md`, `R/docs/`).
- README examples: FastSAC on Isaac Sim (`R/README.md:57-66`); MJWarp PPO needs scale-factor 0.5;
  T1 does not train on MJWarp (`R/src/holosoma/README.md:41-82`).

---

## 13. Decisions a JAX/MJX reimplementation must make (ambiguous, backend-specific, surprising)

1. **Which reference backend.** Presets default to Isaac Sim (URDF + PhysX TGS). Holosoma's own MuJoCo
   path differs in contact model, friction semantics, actuator limits, joint friction and ankle coupling.
   An MJX port that loads `g1_29dof.xml` reproduces Holosoma-on-MJWarp, not the Isaac Sim preset.
2. **Sole friction.** Friction DR never reaches the 14 explicit sole–floor `<pair>`s on MuJoCo/MJWarp
   (fixed μ = 0.8, solref 0.01). To randomize sole friction in MJX either write `pair_friction[:, 0:2]`
   per env or delete the pairs. On Isaac Sim the sole friction is `(μ_robot + 1.0)/2` with static ≠
   dynamic. Decide which to match.
3. **Terrain geom name.** If the MJCF pairs are kept, the terrain geom must be named `floor` or the model
   does not compile (EXT: loading `g1_29dof.xml` alone fails with "geom 'floor' not found in collision 0").
   MJX supports pairs and hfield–capsule/sphere collisions
   (EXT: `mjx/_src/collision_driver.py:92-95, 125-196, 274-282`).
4. **hfield spacing.** Holosoma's MuJoCo hfield is stretched by n/(n−1) relative to the trimesh used for
   foot heights (cm-level misalignment). Use one consistent representation (e.g. hfield
   `size_x = (ncol−1)·0.1/2` with foot heights sampled from the same hfield).
5. **Collision geometry mismatches between files:** shin capsule (URDF r 0.05, L 0.2 vs MJCF r 0.04,
   L 0.25) and left-sole foot2/foot6 radius (URDF/right 0.008 vs MJCF left 0.01, which also makes the MJCF
   left/right asymmetric under symmetry augmentation).
6. **Actuator limits.** Config clips hip_roll at 139 N·m but MuJoCo's `ctrlrange` clamps it at 88 N·m.
   Isaac Sim may cap joint speed at the URDF velocity limits; MuJoCo does not.
7. **MuJoCo-only physics:** `frictionloss = 0.1` on all joints and the 8 ankle tendon limits.
8. **Root velocity reference point:** Isaac Sim root linear velocity is the pelvis COM velocity (COM
   0.076 m below the origin); MuJoCo uses the frame origin. Affects `tracking_lin_vel`, critic
   `base_lin_vel`, reset and push velocities.
9. **Contact termination** uses the net force of the last sub-step only (not max over the 4 sub-steps)
   with a 1 N threshold on 13 bodies, 3 of which (pelvis, thighs, upper arms) have colliders.
10. **Penalty curriculum cadence:** the scale updates once per policy step (because `reset()` runs with
    empty id lists), shared by all envs; replicate that rather than a per-episode update.
11. **Command/gait quirks:** commands resampled twice on reset steps; gait frequency resampled on every
    command resample with a phase jump; stand override evaluated on pre-resample commands; heading range
    ignored; stand probability uses `rand <= 0.2`.
12. **Timing quirks:** episodes last 1001 steps (`> 1000`); rewards use the phase from the previous
    update; pushes overwrite xy velocity and are visible to the same step's critic observation; the
    first action after a reset is zero for envs with delay 1.
13. **Observation details:** alphabetical concatenation; uniform noise before scaling; raw unclipped
    actions observed; no observation history; noise only on dof_pos (±0.01) and dof_vel (±0.1 raw).
14. **DR semantics:** friction quantized to 64 quantiles and drawn once per env at startup; masses/CoM
    startup-only; kp/kd per joint per episode; base mass and CoM applied to **torso_link**, not pelvis;
    no yaw randomization at reset; joint reset scaling multiplies the default (zero-default joints never
    move). The keyed SplitMix64 sampler is reproducible but bit-exact RNG equivalence is not needed.
15. **Spawn height** ignores local terrain (rough tiles sit below the tile-centre max), so robots start
    with a few-cm drop; MJX may need the same `0.8 + origin_z` rule to reproduce early-episode statistics.
16. **PPO on mixed terrain needs care in MuJoCo:** Holosoma itself halves the terrain scale for PPO on MJWarp.
17. **Latent issues that are inert in this preset:** `LeggedRobotLocomotionManager.reset_all` re-creates
    `default_dof_pos` (`locomotion_manager.py:51-62,160-162`), which would discard a setup-time pose bias if
    `setup_dof_pos_bias` were enabled; the curriculum manager's `num_compute_average_epl` param is unused.
18. **Unknowns that need a run of the real stack:** Isaac Lab combine-mode default and cylinder→capsule
    length convention; PhysX enforcement of URDF velocity limits; contact offsets; behaviour of MJWarp at
    Holosoma's pinned commit `ecaef88` (local evidence is from `mujoco_warp 3.12.0`).

---

## Appendix A. Read-only MuJoCo probes (EXT)

Run with MuJoCo 3.12.0 on the CPU. Core of the pair probe, mirroring `scene_manager.add_terrain` + `add_robot`:

```python
world = mujoco.MjSpec(); world.copy_during_attach = True; world.option.timestep = 0.005
floor = world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                                 friction=[0.7, 0.005, 0.001], solimp=[0.99, 0.99, 0.01, 0.5, 2], solref=[0.001, 1])
floor.contype, floor.conaffinity = 2, 1
robot = mujoco.MjSpec.from_file("H/data/robots/g1/g1_29dof.xml")
for b in robot.bodies:
    for g in b.geoms:
        if g.contype == 1 and g.conaffinity == 1:
            g.conaffinity = 2                      # self-collisions off
world.attach(robot, frame=world.worldbody.add_frame(), prefix="robot_")
m = world.compile()
```

Results: `npair 14`; pair 0 = (`floor`, `robot_left_foot1_collision`), friction
[0.8, 0.8, 0.005, 0.0001, 0.0001], solref [0.01, 1], solimp [0.9, 0.95, 0.001, 0.5, 2], dim 3;
`pair_signature` 7 (world × left ankle body); options integrator 0 (Euler), solver 2 (Newton),
iterations 100, ls_iterations 50, cone 0 (pyramidal), tolerance 1e-8; hip_roll actuator ctrlrange
[−88, 88] with `jnt_actfrcrange` [−139, 139]; frictionloss 0.1; 8 limited tendons; robot mass 33.343 kg;
soles touch at pelvis z 0.757 (default pose); default-pose contacts: 8 (plane) or 16 (hfield), one/two per
touching capsule, all with pair parameters; after `geom_friction[:, 0] = 0.3` all sole contacts still have
friction 0.8. Non-pair pelvis contact: friction [1, 1, 0.005, 0.001, 0.001], solref [0.0105, 1], solimp
[0.945, 0.97, 0.0055, 0.5, 2]. Toy hfield with Holosoma's sizing (n = 11 samples, size = n·h/2): step
ramp at x ∈ [0.55, 0.66] vs trimesh [0.5, 0.6] → spacing n·h/(n−1). Terrain quantization:
low-obstacle pit depths after int16 truncation −0.015/−0.020/−0.030 m; `20/0.02 = 1000.0`;
`int(10/0.02) = 500`.
