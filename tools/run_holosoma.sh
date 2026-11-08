#!/bin/bash
# Train a Holosoma baseline as in the published comparison. Besides the checkpoints it keeps the launch time (<run>/t_start)
# and <run>/train_log.csv, the training time Holosoma logs per iteration ("Total time"), from which tools/make_figures.py
# takes each checkpoint's training time. Run it in Holosoma's environment (its MuJoCo Warp setup,
# https://github.com/amazon-far/holosoma at 79ba714) with HOLOSOMA pointing at the checkout:
#   bash tools/run_holosoma.sh <run dir> <minutes> <g1-29dof | g1-29dof-fast-sac> [extra Holosoma flags, e.g. --training.seed=1]
# The run stops after <minutes> (SIGTERM, which Holosoma handles); checkpoints land in <run>/hs/. Then, still in Holosoma's
# environment: python tools/holosoma_ckpt.py <run>/hs/*/*/model_*.pt, and in XSoma's: python -m xsoma.evaluate <run>.
set -uo pipefail
RUN=$1; MIN=$2; PRESET=$3; shift 3
: "${HOLOSOMA:?set HOLOSOMA to the path of the Holosoma checkout}"
mkdir -p "$RUN"
date +%s.%N > "$RUN/t_start"
WANDB_MODE=disabled timeout -k 60 "${MIN}m" python -u "$HOLOSOMA/src/holosoma/holosoma/train_agent.py" "exp:$PRESET" simulator:mjwarp \
  logger:disabled --logger.base-dir="$RUN/hs" --training.num-envs=4096 --terrain.terrain-term.scale-factor=0.5 "$@" > "$RUN/stdout.log" 2>&1
echo $? > "$RUN/exit"
date +%s.%N > "$RUN/t_end"
{ echo "iteration,total_s"; sed 's/\x1b\[[0-9;]*m//g' "$RUN/stdout.log" | awk '
  /Learning iteration [0-9]+\// { match($0, /iteration [0-9]+/); it = substr($0, RSTART + 10, RLENGTH - 10) }
  /Total time:/ { match($0, /Total time: *[0-9.]+/); t = substr($0, RSTART, RLENGTH); sub(/Total time: */, "", t); print it "," t }'; } > "$RUN/train_log.csv"
