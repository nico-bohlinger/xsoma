# Results

Every number and figure in the README comes from the files here (`python tools/make_figures.py`).

```
results/<method>/<run>/eval.jsonl    one record per checkpoint (xsoma/evaluate.py)
results/xsoma/<run>/log.jsonl        XSoma's training log (every 50 iterations)
results/xsoma/<run>/config.json      the run's full configuration
```

Methods: `xsoma` (this repository's defaults), `fast_sac` and `ppo` (Holosoma `79ba714`, presets `g1_29dof_fast_sac` and
`g1_29dof` on its MJWarp backend at terrain scale 0.5, 4096 envs, unmodified, run with `tools/run_holosoma.sh`; their
checkpoints converted with `tools/holosoma_ckpt.py`). Runs are named by seed. All runs: one NVIDIA H100 (95 GB) each, on the
same cluster, four runs per node.

An evaluation record (`xsoma/evaluate.py`; 2048 robots, one 20-s episode each, the policy's mean action, training's
randomization with the soles' friction at 0.8, terrain scale 0.5, penalties at full weight, the same seed for every checkpoint):

| key | meaning |
|---|---|
| `wall_s` | the checkpoint's wall-clock time since the run's launch (s), start-up and compilation included |
| `steps` | env steps (XSoma) or iterations (Holosoma) at the checkpoint |
| `rew_<term>` | Holosoma's `Episode/rew_<term>`: the episode's sum of the weighted term, divided by 20 s, averaged over episodes |
| `len` | mean episode length (policy steps, at most 1001) |
| `success` | share of episodes without a fall (a contact on the pelvis, hips or shoulders) |
| `err_v`, `err_w` | mean velocity-tracking errors while upright: ‖commanded − actual planar velocity‖ (m/s), |commanded − actual yaw rate| (rad/s) |
