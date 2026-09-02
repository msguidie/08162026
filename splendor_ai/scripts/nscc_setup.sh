#!/usr/bin/env bash
# =============================================================================
# splendor_ai — one-time environment setup on NSCC ASPIRE 2A.
#
# WHERE TO RUN THIS: on the **login node**, from the repository root:
#
#     bash splendor_ai/scripts/nscc_setup.sh
#
# It is not a PBS job.  Login nodes have no GPU and a short CPU-time limit, so
# this script only creates the conda environment, installs the wheels and runs
# the fast test suite — anything heavy belongs in a job (`nscc_train.pbs`,
# `nscc_eval.pbs`).  It is safe to re-run: an existing environment is reused.
#
# What you get afterwards:
#   * a conda environment named $ENV_NAME (default "splendor") with torch+CUDA
#   * a green `pytest splendor_ai/tests -q`
#   * optionally the Node cross-validation gate (docs/PLAN.md §4), if a Node
#     module is available on the login node — it is skipped, not failed, if not
# =============================================================================

set -euo pipefail          # -e stop at the first error, -u refuse unset vars,
                           # -o pipefail let a failure inside a pipe count

# --- where are we -----------------------------------------------------------
# ${BASH_SOURCE[0]} is this file; two levels up is the repository root, so the
# script works no matter which directory you call it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-splendor}"          # override: ENV_NAME=foo bash ...
PY_VERSION="${PY_VERSION:-3.11}"          # the version the code is tested on
REQUIREMENTS="splendor_ai/requirements.txt"

echo "=== splendor_ai setup ======================================="
echo "repo        : ${REPO_ROOT}"
echo "environment : ${ENV_NAME} (python ${PY_VERSION})"
echo

# --- 1. module system -------------------------------------------------------
# NSCC uses environment modules: `module load X` puts X on $PATH for this
# shell.  In a non-interactive shell the `module` function sometimes is not
# defined yet, so source the profile script first if we can find it.
if ! command -v module >/dev/null 2>&1; then
    for candidate in /etc/profile.d/modules.sh /usr/share/Modules/init/bash \
                     /opt/pbs/etc/profile.d/modules.sh; do
        # shellcheck disable=SC1090
        [ -r "${candidate}" ] && . "${candidate}" && break
    done
fi

# `|| true` on every module load: a missing module must not kill the script,
# because a working conda/python may already be on $PATH.
if command -v module >/dev/null 2>&1; then
    module load anaconda 2>/dev/null || module load anaconda3 2>/dev/null \
        || module load miniforge3 2>/dev/null || module load miniforge 2>/dev/null \
        || echo "note: no anaconda/miniforge module — using conda from \$PATH"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: no 'conda' on PATH.  Run 'module avail' and load the" >&2
    echo "       anaconda/miniforge module your account provides, then" >&2
    echo "       re-run this script." >&2
    exit 1
fi

# `conda activate` needs conda's shell hook; `conda run` would work too but
# hides output.  This is the documented non-interactive way.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
. "${CONDA_BASE}/etc/profile.d/conda.sh"

# --- 2. the environment -----------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "--- conda environment '${ENV_NAME}' already exists — reusing it"
else
    echo "--- creating conda environment '${ENV_NAME}'"
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi
conda activate "${ENV_NAME}"
echo "python      : $(python -V 2>&1) at $(command -v python)"

# --- 3. wheels --------------------------------------------------------------
# torch on PyPI ships the CUDA runtime on Linux (manylinux wheels bundle
# cuDNN/cuBLAS), so a plain `pip install torch` is the GPU build — do NOT add
# the CPU index URL here, that is only for a machine without a GPU.  The
# compute nodes need no CUDA module for this: the wheel carries its own
# runtime and only needs the host NVIDIA driver.
echo "--- installing ${REQUIREMENTS}"
python -m pip install --upgrade pip
python -m pip install -r "${REQUIREMENTS}"

# --- 4. sanity checks -------------------------------------------------------
# `torch.cuda.is_available()` is False on a login node (no GPU) — that is
# expected and not an error; the same wheel will see 4 A100s inside a job.
echo "--- versions"
python - <<'PY'
import numpy, torch, yaml
print(f"  numpy {numpy.__version__}")
print(f"  torch {torch.__version__}  cuda build {torch.version.cuda}")
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()} "
      f"(False on a login node is normal)")
print(f"  pyyaml {yaml.__version__}")
try:
    import tensorboard
    print(f"  tensorboard {tensorboard.__version__}")
except Exception:
    print("  tensorboard not installed (optional)")
PY

echo "--- import check"
python -c "
from splendor_ai.rules import engine as E
from splendor_ai import encode, anchors, arena, export
s = E.new_game(4, 'TEAM', 'ADJACENT')
print(f'  engine ok, OBS_DIM={encode.OBS_DIM}, anchors={list(anchors.ANCHOR_LADDER)}')
"

# --- 5. the test suite ------------------------------------------------------
# The slow go/no-go gates (G2's MCTS@400-vs-greedy, the 100k-state symmetry
# sweep) are opt-in via environment variables so this stays a couple of
# minutes on a login node.  Run them once inside a job if you change search.
echo "--- pytest splendor_ai/tests -q"
python -m pytest splendor_ai/tests -q

# --- 6. Node cross-validation (optional) ------------------------------------
# The bit-exactness gate against server/gameLogic.js needs Node.  If no Node
# module is available on this login node, skip it: it is a *rules* gate and
# has already been passed on the workstation; nothing about training depends
# on Node being present here.
echo "--- Node cross-validation (optional)"
if command -v module >/dev/null 2>&1; then
    module load nodejs 2>/dev/null || module load node 2>/dev/null || true
fi
if command -v node >/dev/null 2>&1; then
    echo "    node $(node --version) found — running a 200-game pass"
    # PY= overrides the Makefile's workstation interpreter with this env's one.
    make -C splendor_ai/validation GAMES=200 PY="$(command -v python)" all
else
    echo "    no 'node' on PATH — skipping (this is fine on NSCC)"
fi

# --- 7. what to do next -----------------------------------------------------
cat <<EOF

=== setup complete ==========================================
Environment '${ENV_NAME}' is ready.  Next steps:

  1. First throughput job (gate G4) — a short run to measure sims/s:
       qsub -l walltime=01:00:00 -v CHAIN=0,MAX_CHAIN=0 \\
            splendor_ai/scripts/nscc_train.pbs

  2. Chained training (each job resumes the previous one and re-submits
     itself until you 'touch STOP'):
       qsub splendor_ai/scripts/nscc_train.pbs

  3. Evaluation of the latest checkpoint against the anchor ladder:
       qsub splendor_ai/scripts/nscc_eval.pbs

  Watch a job:      qstat -answ1 \$USER
  Stop the chain:   touch ${REPO_ROOT}/STOP
  Logs:             runs/nscc/logs/
See splendor_ai/README.md, section "Evaluation & export".
EOF
