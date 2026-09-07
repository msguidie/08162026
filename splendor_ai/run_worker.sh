#!/usr/bin/env bash
# ============================================================
#  Splendor AI deployment worker — Linux / macOS launcher.
#  The POSIX twin of run_worker.bat: activate the virtualenv, point the
#  worker at splendor_ai/.env, restart it unless it stopped cleanly (0) or
#  the configuration is wrong (2).
#
#  Usage:  ./run_worker.sh            # normal
#          ./run_worker.sh --once     # offline self-test, no restart loop
# ============================================================
set -uo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PROJECT/.." && pwd)"

PY=""
for candidate in "$REPO/.venv/bin/python" "$PROJECT/.venv/bin/python" \
                 "${VIRTUAL_ENV:-}/bin/python"; do
  if [ -x "$candidate" ]; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
    echo "[run_worker] no .venv found, falling back to $PY"
  else
    echo "[run_worker] no Python found.  python3 -m venv .venv && \\"
    echo "             .venv/bin/pip install -r splendor_ai/requirements-worker.txt"
    exit 2
  fi
fi

if [ ! -f "$PROJECT/.env" ]; then
  echo "[run_worker] $PROJECT/.env is missing — copy .env.example to .env first."
  exit 2
fi

# The package lives at <repo>/splendor_ai/splendor_ai and is imported as
# `splendor_ai` from the repository root.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export SPLENDOR_WORKER_ENV="$PROJECT/.env"
cd "$REPO"

echo "[run_worker] python: $PY"
echo "[run_worker] config: $SPLENDOR_WORKER_ENV"

case " $* " in
  *" --once "*|*" --print-config "*|*" --help "*|*" -h "*)
    exec "$PY" -m splendor_ai.worker.worker "$@" ;;
esac

trap 'echo "[run_worker] interrupted"; exit 0' INT TERM

while true; do
  "$PY" -m splendor_ai.worker.worker "$@"
  code=$?
  case "$code" in
    0) echo "[run_worker] worker stopped cleanly."; break ;;
    2) echo "[run_worker] configuration error — fix .env and start again."; break ;;
    *) echo "[run_worker] worker exited with code $code — restarting in 5 s."
       sleep 5 ;;
  esac
done
