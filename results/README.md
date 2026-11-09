# Results

Every number and figure in the README comes from the files here (`python tools/make_figures.py`).

```
results/<method>/<run>/eval.jsonl        one record per checkpoint (xsoma/evaluate.py)
results/xsoma*/<run>/log.jsonl           XSoma's training log (every 50 iterations)
results/xsoma*/<run>/config.json         the run's full configuration
results/<fast_sac|ppo>*/<run>/train_log.csv  the training time Holosoma logs ("Total time") per logged iteration
```

Time is training time, start-up and one-time compilation excluded, as in XMimic. XSoma compiles before its training clock
starts, and `log.jsonl` records both its clocks. Holosoma compiles inside its training loop, so each checkpoint's time is the
logged training time at its iteration, minus the extra time of the first logged interval, in which the compilation happens
(about 12 s for FastSAC, 0.3 s for PPO). `tools/make_figures.py` does both (`training_seconds`).

Methods, each trained for at least 15 minutes with a checkpoint about every 15 s, so that every learning curve covers the
15 minutes shown: `xsoma` (this repository's defaults on one GPU), `xsoma_2gpu`, `xsoma_3gpu` and `xsoma_4gpu` (`python
train.py --layout AL` on two to four GPUs), `fast_sac` and `ppo` (Holosoma `79ba714`, presets `g1_29dof_fast_sac` and
`g1_29dof` on its MJWarp backend at terrain scale 0.5, 4096 envs, unmodified, run as `tools/run_holosoma.sh` does, with a
checkpoint every 250 and every 15 iterations; their checkpoints converted with `tools/holosoma_ckpt.py`). Two sets of longer
runs answer what lies beyond 15 minutes: `fast_sac_full` (FastSAC's whole 50,000-iteration schedule, a checkpoint every 1000
iterations: its tracking at the end) and `ppo_full` (PPO's first hour, a checkpoint every 100 iterations: it does not reach
the target within it). Runs are named by seed. All runs: NVIDIA H100s (95 GB) on the same cluster; the 1-GPU runs four per
node, the 2-GPU runs two per node, the 3- and 4-GPU runs one per node (a 3-GPU run next to an evaluation on the fourth GPU).
The multi-GPU and Holosoma runs were evaluated after training, in a separate step with the evaluator's defaults.
FastSAC has eight runs (seeds 42 and 1 to 7), as its time to the target varies from run to run even with the same seed
(8.77 to 10.82 minutes here); an earlier set of four 14-minute runs with the same settings, not published as it ends
before the figures do, took 8.64 minutes (8.21 to 8.92).

An evaluation record (`xsoma/evaluate.py`; 2048 robots, one 20-s episode each, the policy's mean action, training's
randomization with the soles' friction at 0.8, terrain scale 0.5, penalties at full weight, the same seed for every checkpoint):

| key | meaning |
|---|---|
| `wall_s` | the checkpoint's wall-clock time since the run's launch (s), start-up and compilation included (not used for the comparison) |
| `steps` | env steps (XSoma) or iterations (Holosoma) at the checkpoint |
| `rew_<term>` | Holosoma's `Episode/rew_<term>`: the episode's sum of the weighted term, divided by 20 s, averaged over episodes |
| `len` | mean episode length (policy steps, at most 1001) |
| `success` | share of episodes without a fall (a contact on the pelvis, hips or shoulders) |
| `err_v`, `err_w` | mean velocity-tracking errors while upright: ‖commanded − actual planar velocity‖ (m/s), |commanded − actual yaw rate| (rad/s) |
