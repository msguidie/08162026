#!/usr/bin/env bash
# The two-hour 1v1 sprint, turnkey, on an INTERACTIVE 4-GPU node:
#
#     qsub -I -l select=1:ngpus=4 -l walltime=2:00:00 -q normal -P 13004357
#     cd ~/scratch/splendor && bash scripts/train_1v1_2h.sh
#
# Trains for MINUTES of self-play (default 100, so startup, the final
# evaluation and the checkpoint fit in a 2 h allocation), prints the trainer's
# log live and keeps a copy in runs/$NAME/train.log.  Interrupted, or out of
# walltime?  Run it again: it resumes from runs/$NAME/trainer_state.pt.
#
# When it ends, the model for the worker is runs/$NAME/ind2.pt.
#
#   NAME=other MINUTES=60 bash scripts/train_1v1_2h.sh   # override either
set -uo pipefail

NAME=${NAME:-sprint1v1}
MINUTES=${MINUTES:-100}
CONFIG=splendor_ai/configs/nscc_2h_1v1.yaml
ENV_PREFIX=${ENV_PREFIX:-$HOME/scratch/conda-envs/splendor}

cd "$(dirname "$0")/.." || exit 2

# --- environment (left alone if it is already active) --------------------
if [ "${CONDA_PREFIX:-}" != "$ENV_PREFIX" ]; then
    if ! command -v conda >/dev/null 2>&1; then
        module load anaconda 2>/dev/null || module load miniforge3 2>/dev/null || true
    fi
    eval "$(conda shell.bash hook)" || exit 2
    conda activate "$ENV_PREFIX" || exit 2
fi

# --- cores, counted BEFORE threads are pinned ----------------------------
# GNU nproc honours OMP_NUM_THREADS: once that is 1, `nproc` says 1 -- which is
# exactly how the first long run started with 4 actors instead of ~120.  Ask
# the affinity mask instead, and leave 8 cores for the learner, the three
# inference servers, the evaluator and the OS.
CORES=$(python -c 'import os; print(len(os.sched_getaffinity(0)))')
ACTORS=$((CORES - 8))
[ "$ACTORS" -lt 4 ] && ACTORS=4

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2

RUN_DIR=runs/$NAME
LOG=$RUN_DIR/train.log
mkdir -p "$RUN_DIR"
export TORCHINDUCTOR_CACHE_DIR=$PWD/$RUN_DIR/.torch_cache
export TRITON_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR/triton

RESUME=""
[ -f "$RUN_DIR/trainer_state.pt" ] && RESUME="--resume $RUN_DIR"

{
    echo "=== $(date)  host $(hostname)  cores $CORES -> $ACTORS actors x 128 games" \
         " budget ${MINUTES} min  ${RESUME:-fresh run}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} 2>&1 | tee -a "$LOG"

# max_seconds is the run's whole budget (restored on resume); the LR schedule
# is stretched with it, ~240 learner steps per minute at the measured rate.
python -m splendor_ai.selfplay.train \
    --config "$CONFIG" $RESUME \
    --set run_dir="$RUN_DIR" \
    --set selfplay.actors="$ACTORS" \
    --set max_seconds=$((MINUTES * 60)) \
    --set learner.cosine_steps=$((MINUTES * 240)) \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

if [ -f "$RUN_DIR/weights/latest.pt" ]; then
    cp -f "$RUN_DIR/weights/latest.pt" "$RUN_DIR/ind2.pt"
    echo "=== model for the worker: $PWD/$RUN_DIR/ind2.pt  (goes in the worker's models/ folder)" \
        | tee -a "$LOG"
fi
echo "=== trainer exit $rc  $(date)" | tee -a "$LOG"
exit "$rc"
