#!/bin/bash
# Train a Holosoma baseline as in the published comparison and record its launch time, so that every checkpoint gets its
# wall-clock time since launch (tools/holosoma_ckpt.py reads <run>/t_start). Run it in Holosoma's environment (its MuJoCo Warp
# setup, https://github.com/amazon-far/holosoma at 79ba714) with HOLOSOMA pointing at the checkout:
#   bash tools/run_holosoma.sh <run dir> <minutes> <g1-29dof | g1-29dof-fast-sac> [extra Holosoma flags, e.g. --training.seed=1]
# The run stops after <minutes> (SIGTERM, which Holosoma handles); checkpoints land in <run>/hs/. Then, still in Holosoma's
# environment: python tools/holosoma_ckpt.py <run>/hs/*/*/model_*.pt, and in XSoma's: python -m xsoma.evaluate <run>.
set -uo pipefail
RUN=$1; MIN=$2; PRESET=$3; shift 3
: "${HOLOSOMA:?set HOLOSOMA to Holosoma's checkout}"
mkdir -p "$RUN"
date +%s.%N > "$RUN/t_start"
WANDB_MODE=disabled timeout -k 60 "${MIN}m" python -u "$HOLOSOMA/src/holosoma/holosoma/train_agent.py" "exp:$PRESET" simulator:mjwarp \
  logger:disabled --logger.base-dir="$RUN/hs" --training.num-envs=4096 --terrain.terrain-term.scale-factor=0.5 "$@" > "$RUN/stdout.log" 2>&1
echo $? > "$RUN/exit"
date +%s.%N > "$RUN/t_end"
