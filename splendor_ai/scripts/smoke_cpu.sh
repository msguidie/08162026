#!/usr/bin/env bash
# G3 smoke gate (docs/AI_DESIGN.md §2): the whole self-play trainer on CPU in
# ~20-25 minutes.  Run it from the repository root:
#
#     bash splendor_ai/scripts/smoke_cpu.sh [RUN_DIR] [-- extra --set args]
#
# It prints the per-generation learning curve at the end.  Targets:
#   NetBot   >= 80% vs RandomBot
#   SearchBot@48 > 55% vs GreedyBot
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${1:-runs/smoke_cpu}"
shift || true
if [[ "${1:-}" == "--" ]]; then shift; fi

# The single most common way a "correct" pipeline runs at 20% speed with no
# error message (judges.md): every process must be single-threaded here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

PY="${PYTHON:-python3}"
if [[ -x /home/user/venv-splendor/bin/python ]]; then
  PY="${PYTHON:-/home/user/venv-splendor/bin/python}"
fi

cd "$REPO_ROOT"
echo "[smoke] repo=$REPO_ROOT python=$PY run_dir=$RUN_DIR"
"$PY" -m splendor_ai.selfplay.train \
  --config splendor_ai/configs/smoke_cpu.yaml \
  --set "run_dir=$RUN_DIR" "$@"

echo
echo "[smoke] learning curve ($RUN_DIR/metrics.jsonl)"
"$PY" - "$RUN_DIR/metrics.jsonl" <<'PYEOF'
import json, sys

rows = [json.loads(line) for line in open(sys.argv[1])]
learner = [r for r in rows if r["kind"] == "learner"]
evals = [r for r in rows if r["kind"] == "eval"]
gens = {r["generation"]: r for r in rows if r["kind"] == "generation"}


def near(step):
    best = None
    for r in learner:
        if best is None or abs(r["step"] - step) < abs(best["step"] - step):
            best = r
    return best or {}


print(f"{'gen':>4} {'step':>7} {'games':>7} {'net/rand':>9} {'net/greedy':>11} "
      f"{'search/greedy':>14} {'loss':>7} {'v_mse':>7} {'v_ev':>6} {'top1':>6}")
summary_games = ([r for r in rows if r["kind"] == "summary"] or [{}])[-1].get(
    "games_done", 0)
for e in evals:
    g = e.get("generation", 0)
    m = near(e.get("step", 0))
    games = gens.get(g, {}).get("games_done") or (summary_games if e.get("final")
                                                  else 0)
    print(f"{g:>4} {e.get('step', 0):>7} {games:>7} "
          f"{e.get('net_vs_random', float('nan')):>9.3f} "
          f"{e.get('net_vs_greedy', float('nan')):>11.3f} "
          f"{e.get('search_vs_greedy', float('nan')):>14.3f} "
          f"{m.get('total', float('nan')):>7.3f} {m.get('value_mse', float('nan')):>7.3f} "
          f"{m.get('value_explained_variance', float('nan')):>6.3f} "
          f"{m.get('policy_top1_agreement', float('nan')):>6.3f}")

summary = [r for r in rows if r["kind"] == "summary"]
if summary:
    s = summary[-1]
    tp = s.get("throughput", {})
    print(f"\n[smoke] {s['games_done']} games, {s['generations']} generations, "
          f"{s['steps']} steps, {s['actor_restarts']} actor restarts")
    print(f"[smoke] sims/s {tp.get('sims_per_s', 0):.0f}  "
          f"moves/s {tp.get('moves_per_s', 0):.1f}  "
          f"games/s {tp.get('games_per_s', 0):.2f}  "
          f"records/s {tp.get('records_per_s', 0):.0f}  "
          f"steps/s {tp.get('steps_per_s', 0):.2f}")
    last = [e for e in evals if e.get("net_vs_random") is not None]
    if last:
        e = last[-1]
        ok_random = e.get("net_vs_random", 0) >= 0.80
        ok_search = (e.get("search_vs_greedy") or 0) > 0.55
        print(f"[smoke] G3 gate: NetBot vs random {e.get('net_vs_random'):.3f} "
              f"({'PASS' if ok_random else 'FAIL'}), SearchBot vs greedy "
              f"{e.get('search_vs_greedy', float('nan')):.3f} "
              f"({'PASS' if ok_search else 'FAIL'})")
PYEOF
