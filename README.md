# XSoma

**Humanoid velocity tracking on rough terrain in about three minutes on one GPU.** XSoma trains the Unitree G1 task of
[Holosoma](https://github.com/amazon-far/holosoma) (its rewards, terminations, commands, randomization and terrain, checked
against Holosoma's own code) with a learned world model and backpropagation through imagined rollouts, all in JAX. On one
H100 it passes Holosoma's own tracking bar 3.2 minutes after launch, 3.3× sooner than Holosoma's FastSAC, and ends with a
38 % lower velocity error; Holosoma's PPO does not pass the bar within an hour.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/time_to_target_dark.png">
  <img alt="Time to Holosoma's FastSAC tracking bar on one H100: XSoma 3.2 min, Holosoma FastSAC 10.6 min, Holosoma PPO not reached in 60 min" src="figures/time_to_target.png">
</picture>

## What XSoma is

[Holosoma](https://github.com/amazon-far/holosoma) (Amazon FAR) trains a G1 to follow velocity commands (forward, sideways,
turning) on randomized rough terrain with PPO or FastSAC. XSoma keeps Holosoma's task and replaces the learner:

- **The task, reimplemented and checked.** [`xsoma/task.py`](xsoma/task.py) and [`xsoma/sim.py`](xsoma/sim.py) are Holosoma's
  `g1_29dof` task as JAX functions, simulated with MuJoCo Warp through MJX, with Holosoma's MJCF and solver settings. Against
  Holosoma's own environment ([`tests/test_task_holosoma.py`](tests/test_task_holosoma.py): 1500 recorded steps with pushes,
  command changes, a time-out and falls), every observation matches to 1.4·10⁻⁶, every reward term to float32 precision,
  and terminations, gait clock, curriculum, delays, torques and resets exactly. [`docs/holosoma_task.md`](docs/holosoma_task.md)
  cites Holosoma's source for every number.
- **The learner.** XSoma learns a world model of the robot from replayed transitions and improves the policy by
  backpropagating Holosoma's reward through 48 imagined steps (about one second), the recipe of SuperMimic and
  [XMimic](https://github.com/nico-bohlinger/xmimic). Imagination does not need a learned reward or critic: the task's own
  observation, reward and termination functions run on the imagined states, with forward kinematics and an exact lookup of
  the terrain under every foot.
- **One GPU, compiled.** Collection, world-model fits and policy updates are three XLA programs, compiled in parallel
  before training starts (about 70 s, or 20 to 40 s from the compilation cache).

<p align="center"><img src="figures/xsoma.gif" width="720" alt="XSoma: 3.3x faster than FastSAC to Holosoma's tracking bar, with the G1 walking over rough ground and pits"></p>
<p align="center"><sub>The G1 driven by an XSoma policy 2.7 minutes after launch on one H100 (seed 0's target checkpoint), walking
over rough ground and a field of pits on Holosoma's terrain, replayed in MuJoCo.</sub></p>

## Results

Every method on one H100, from launch, on Holosoma's task at terrain scale 0.5 (the setting Holosoma's README uses for
MJWarp), evaluated the same way:

| method | runs | time to target (min) | speedup | env steps to target | after 20 min: velocity error, falls | at the end: velocity error, falls |
|---|---|---|---|---|---|---|
| XSoma | 5 | 3.2 (2.7–3.9) | 3.3× | 5 M | 0.33 m/s, 0.6 % | 0.29 m/s, 0.2 % (61 min) |
| Holosoma FastSAC | 4 | 10.6 (9.5–11.7) | 1.0× | 41 M | 0.48 m/s, 0.2 % | 0.47 m/s, 0.2 % (50 min) |
| Holosoma PPO | 4 | not within 60 | – | – | 0.72 m/s, 58.7 % | 0.64 m/s, 38.0 % (59 min) |

- **The target** is Holosoma's own bar for its FastSAC preset: its nightly test passes when the velocity-tracking term
  `Episode/rew_tracking_lin_vel` reaches 0.95 and the yaw-rate term 0.8. Here both are measured on deterministic evaluation
  episodes and must hold for three consecutive checkpoints.
- **Evaluation.** Every checkpoint drives 2048 robots for one 20-s episode each, with the policy's mean action, Holosoma's
  randomization, pushes and command sampling, on the same terrain and the same episodes for every method. Holosoma's
  checkpoints are converted to plain arrays and run in the same simulator.
- **Time** is wall-clock time since launch, including start-up and compilation.
- **Fewer simulated steps.** XSoma collects 28,000 env steps per second and spends the rest of each iteration learning;
  FastSAC collects 71,000 and PPO 100,000. XSoma reaches the target after 5 million env steps, FastSAC after 41 million.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/learning_curves_dark.png">
  <img alt="Learning curves over 60 minutes: XSoma's velocity tracking rises and its velocity error falls fastest and end best, FastSAC plateaus, PPO drops after 12 minutes" src="figures/learning_curves.png">
</picture>

## What we found on the way

These concern anyone training on MuJoCo Warp (MJWarp) or with Holosoma's MuJoCo model, with or without XSoma:

1. **MuJoCo Warp's heightfield collision breaks down far from the field's centre.** Holosoma's MuJoCo backends build the
   whole terrain as one heightfield (160 × 240 m at full scale). In float32, robots far from its centre blow up: under
   violent random actions, 511 of 512 robots on the corner tiles reach non-finite or impossible velocities within 20 s,
   against 14 on the central tiles, and under normal actions 10 of 512 on the corner tiles and none in the centre
   ([`tools/stress_sim.py`](tools/stress_sim.py)). This matches the failure Holosoma's README reports for PPO on MJWarp at
   full terrain scale (we saw it crash in its first rollout), and why it recommends half scale there. XSoma builds one
   heightfield per tile (plus four boxes for the flat border): the same surface, and the corner tiles become as stable as
   the central ones (12 of 512 robots under violent actions, none under normal ones). With it XSoma also trains at full
   scale.
2. **Explicit contact pairs ignore friction randomization.** The G1 MJCF binds its 14 sole capsules to the floor with
   explicit `<pair>`s, which carry their own friction (0.8). Holosoma's MuJoCo backends randomize `geom_friction`, which
   pairs never read, so the feet always have friction 0.8 there. XSoma turns the pairs into priority contacts with the same
   parameters (identical contacts and solver output, [`tests/test_scene.py`](tests/test_scene.py)), so that the soles'
   friction can be randomized as Holosoma intends.
3. **Two smaller differences between the MuJoCo path and the config.** The hip-roll motors are clamped at 88 N·m by the
   MJCF's `ctrlrange` (the config says 139). The heightfield is stretched by n/(n−1) against the mesh the foot-height
   reward reads, a centimetre-level misalignment; XSoma's physics and rewards use one surface.

## How it works

An iteration takes about 147 ms on an H100:

1. **Collect** one step of 4096 robots into a replay buffer that holds the last 200 steps (0.8 M transitions): half of
   them act with the policy's action, the other half add Gaussian exploration noise.
2. **Fit the world model** three times on 8-step windows of real transitions, open loop. It is an MLP (3 × 512) that
   predicts the robot's next state (pelvis pose and velocity, joint positions and velocities) from the current one, the
   PD error of the joint targets, the terrain under the robot (pelvis height and the clearance of 28 points on the soles)
   and the robot's randomized dynamics (PD gains, masses, friction: 72 numbers).
3. **Update the policy** twice: from 2048 replayed states, roll the policy out 48 steps (one second) in the world model and
   ascend the discounted sum of Holosoma's rewards along it. Every imagined state goes through Holosoma's own reward and
   termination functions, so their gradients reach the policy through the world model. A fall (a termination body below
   the terrain) ends an imagined rollout; its gradient comes from a sigmoid of the clearance.

The policy network, its inputs and its outputs are Holosoma's (the 100-number actor observation in, joint targets out), so
an XSoma policy runs wherever a Holosoma policy runs. The evaluated policy is an average of recent policies.

What mattered most:

- **More policy updates per environment step.** A policy update costs about 49 ms on an H100, one step of 4096 robots
  plus the world-model fits about as much. Two updates per step learn fastest in wall-clock time: five minutes after
  launch the velocity error is 0.34 to 0.36 m/s, against 0.45 to 0.49 with one update per one or two steps, and one update
  per four steps collapsed in 2 of 4 runs. Three or four updates per step reach the target at the same time as two.
- **Holosoma's FastSAC reward preset.** The two presets differ only in the reward for staying alive (10 vs 1 per second).
  Within a one-second horizon, PPO's 1 hardly outweighs the tracking terms, and XSoma learns to balance much later. All
  methods are scored on the same tracking terms.
- **No critic.** Bootstrapping the return with a learned value function made tracking slower (Holosoma's tracking metric
  0.79 against 1.31 after 17 minutes); the exact one-second return is enough here.
- **Some robots collect without exploration noise.** When every collected action is noisy, the policy comes to rely on the
  noise: after about 45 minutes of training, its noise-free actions let the robot sink into a crouch within seconds of
  some commands, and by 60 minutes 32 to 61 % of the evaluation episodes end in a fall, while the noisy training episodes
  still look fine. Letting half of the robots act with the policy's mean action puts that behaviour into the world
  model's data: every run stays stable through the hour (at most 0.3 % falls at the last checkpoint), and early learning gets
  faster too. Half works better than a quarter (target at 3.2 against 3.8 minutes, 5 runs each) and than halving the noise
  of every robot (3.7 minutes, 2 runs), which also keep the runs stable.

## How it gets fast

- **Three compiled programs.** Collection, world-model fits and policy updates are XLA programs with donated buffers; the
  host only dispatches them and fetches statistics every 50 iterations. They compile in parallel threads before training
  (about 70 s, or 20 to 40 s when the persistent cache has them). The MuJoCo model and the terrain are program arguments, so
  every seed reuses the cached programs.
- **The simulator stays on the GPU.** MuJoCo Warp through MJX runs Holosoma's whole task step (four 5-ms physics substeps
  with the PD law, terminations, rewards, resets, pushes, observations) at 250,000 env steps per second on an H100
  ([`tools/bench_sim.py`](tools/bench_sim.py)). The contact-force termination is decoded from MuJoCo
  Warp's contact pool, so no extra pass computes contact sensors.
- **bf16 where PyTorch would use it.** Matmuls in imagination output bf16, so their backward passes run on bf16 tensor
  cores; cuBLAS replaces XLA's Triton GEMMs, which are slower for these sizes on Hopper.
- **The task once, for both.** The simulator and imagination call the same JAX functions for kinematics, terrain heights,
  observations, rewards and terminations ([`tools/check_heads.py`](tools/check_heads.py) checks that imagination's rewards
  equal the simulator's on real transitions).

<details>
<summary><b>What did not help</b></summary>

<br>

- A critic: as a bootstrap at the horizon or as λ-returns, it slowed tracking or destabilized training.
- Fewer policy updates per environment step: slower in wall-clock time and less stable (see above).
- A left-right symmetry loss (Holosoma's PPO uses one), an extra survival bonus in imagination, lagging the penalty
  curriculum in imagination, clipping exploding trajectories' gradients, rejecting policy steps that lowered the imagined
  return, a decaying learning rate, collecting with the averaged policy: within noise or worse.
- Holosoma's pushes inside imagination (1 or 3 % of imagined steps), against the remaining falls, which all come 0.5 to 2 s
  after a push: no fewer falls, and slower tracking.
- A shorter horizon (32 steps, 0.64 s): faster updates, but the policy did not learn to avoid falls.
- Smaller imagined batches (1024): barely faster per update (the update is bound by kernel latency, not arithmetic) and a
  temporary collapse in one run. Larger ones (4096): slower per update, no faster learning.
- Longer unrolling of the imagination loop than 4: no faster, much longer compilation.

</details>

## How to run

Linux with an NVIDIA GPU and CUDA 12, Python 3.11. The published numbers are from H100s.

```bash
pip install -r requirements.txt
python train.py --log_dir runs/xsoma        # 20 minutes of training (the published runs: --minutes 60), then every
                                            # checkpoint is evaluated into runs/xsoma/eval.jsonl
```

The defaults are the published configuration; [`xsoma/train.py`](xsoma/train.py) lists every option. Evaluate any run or
checkpoint, XSoma's or Holosoma's:

```bash
python -m xsoma.evaluate runs/xsoma         # 2048 robots, one 20-s episode each, per checkpoint
```

Holosoma's baselines, in Holosoma's own environment (its MJWarp setup):

```bash
HOLOSOMA=~/holosoma bash tools/run_holosoma.sh runs/fast_sac 60 g1-29dof-fast-sac
HOLOSOMA=~/holosoma bash tools/run_holosoma.sh runs/ppo 60 g1-29dof --training.seed=1
python tools/holosoma_ckpt.py runs/ppo/hs/*/*/model_*.pt    # their checkpoints as plain arrays (needs torch)
python -m xsoma.evaluate runs/ppo                            # back in XSoma's environment
```

Checks and figures:

```bash
JAX_PLATFORMS=cpu python tests/test_task_holosoma.py   # XSoma's task vs Holosoma's environment (recorded in tests/data)
JAX_PLATFORMS=cpu python tests/test_scene.py           # priority contacts vs explicit pairs, tiles vs one heightfield
python tools/check_sim.py                              # kinematics, terrain heights, contact forces vs MuJoCo (GPU)
python tools/check_heads.py                            # imagination's reward and termination vs the simulator's (GPU)
python tools/stress_sim.py                             # physics failures under hard random actions, per terrain layout (GPU)
python tools/bench_sim.py                              # simulator throughput (GPU)
python tools/make_figures.py                           # the figures and the table above, from results/
bash tools/download_meshes.sh && python tools/rollout.py ckpt=runs/xsoma/ckpt/0019.npz && python tools/render_gif.py rollout.npz
```

## Faithfulness and caveats

- **Which Holosoma.** Holosoma's presets default to Isaac Sim; XSoma reproduces Holosoma on MuJoCo (its MJCF, solver
  settings and MuJoCo backends), and both baselines ran Holosoma's MJWarp backend at terrain scale 0.5, as its README
  recommends for MJWarp. Isaac Sim differs in contact model, friction and actuator limits.
- **Where XSoma's simulator differs from Holosoma's MJWarp** (see [What we found on the way](#what-we-found-on-the-way)):
  one heightfield per tile instead of one for the whole terrain, no n/(n−1) stretch, and in training the soles' friction is
  randomized as Holosoma's config intends (Holosoma's MJWarp keeps it at 0.8). Evaluation keeps the soles at 0.8 for every
  method, the setting both baselines trained in. A step that times out and ends in a fall counts as a fall.
- **One evaluation for all.** Every checkpoint of every method is scored by the same evaluator in XSoma's simulator, on the
  same 2048 episodes. Holosoma's own logged metrics come from its noisy training episodes and are lower.
- **Wall-clock time from launch**, start-up and compilation included: XSoma 27 to 42 s with its compilation cache (about
  70 s without), Holosoma 16 to 60 s (Warp kernel and Inductor caches warm). Every run had a GPU of its own; four runs shared a
  node.
- **Budgets.** FastSAC ran its whole schedule (50,000 iterations, about 49 minutes). Holosoma's PPO preset trains 25,000
  iterations (about 7 hours at this speed); it ran for one hour here. PPO peaks after about 12 minutes. Then its penalty
  curriculum, which scales all penalty terms up while episodes are long, raises their weight from about 0.05 to 1, and in
  every run its tracking and its share of upright episodes drop for the rest of the hour.
- **Falls.** After 20 minutes 0.6 % of XSoma's evaluation episodes end in a fall, against 0.2 % for FastSAC; at the end of
  training both are at 0.2 %. Earlier XSoma runs fell mostly 0.5 to 2 s after one of Holosoma's pushes.
- **Seeds and checkpoints.** XSoma and FastSAC save a checkpoint about once a minute, PPO every 1.7 minutes, so times to
  the target come in steps of a minute: three of the five XSoma runs pass at their second checkpoint (2.7 min), two at
  their third (3.9 min). Compare means; XSoma has 5 runs, FastSAC and PPO 4 each.

## Repository

```
train.py           entry point (every option: xsoma/train.py)
xsoma/             task.py (Holosoma's task in JAX), sim.py (MuJoCo Warp simulator), scene.py, terrain.py, kin.py,
                   train.py (world model and imagination), nets.py, buffer.py, optim.py, rot.py, evaluate.py
assets/            Holosoma's G1 MJCF (meshes: tools/download_meshes.sh)
docs/              holosoma_task.md: Holosoma's task, line by line
results/           per-run evaluation files behind every number and figure
figures/           figures and the GIF
tools/             run_holosoma.sh, holosoma_ckpt.py, holosoma_dump.py (baselines and the test data), check_sim.py,
                   check_heads.py, stress_sim.py, bench_sim.py (checks and speed), make_figures.py, rollout.py,
                   render_gif.py, download_meshes.sh (figures and the GIF)
tests/             equivalence tests against Holosoma and MuJoCo (CPU)
```

## Acknowledgements

- **[Holosoma](https://github.com/amazon-far/holosoma)** (Amazon FAR): the task, the robot model and both baselines that
  XSoma reimplements and compares against; its FastSAC recipe is described in *Learning Sim-to-Real Humanoid Locomotion in
  15 Minutes* (Seo et al., [arXiv:2512.01996](https://arxiv.org/abs/2512.01996)).
- **[XMimic](https://github.com/nico-bohlinger/xmimic)**, and through it [BlinkMimic](https://github.com/fuyichen0725/blinkmimic)
  by Fuyi Chen, [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) by Huy-Quang-Dao and
  [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) by Chong Zhang (ETH Zurich): the world model, the
  policy-learning recipe and much of the training code.
- **SuperMimic** (Liao et al., arXiv:2610.28480) for the method, and **Terrain-Grounded Imagination** (Heyrman and Hutter,
  [code](https://github.com/leggedrobotics/terrain-grounded-imagination)) for grounding imagination in exact terrain lookups.
  XSoma is an independent work, not affiliated with or endorsed by these authors.
- [MuJoCo](https://mujoco.org), [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
  [NVIDIA Warp](https://github.com/NVIDIA/warp) and [JAX](https://github.com/jax-ml/jax).

Licenses and notices of the code and model XSoma derives from: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{bohlinger2026xsoma,
  title  = {XSoma: Learning Humanoid Velocity Tracking on Rough Terrain in Minutes},
  author = {Bohlinger, Nico and Peters, Jan},
  year   = {2026},
  url    = {https://github.com/nico-bohlinger/xsoma}
}
```

## License

XSoma is released under the [MIT License](LICENSE). The parts derived from Holosoma (the task reimplementation and the
bundled G1 MJCF) are under the Apache License 2.0, and the G1 model derives from Unitree's BSD-3-Clause description; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
