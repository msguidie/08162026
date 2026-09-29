#!/usr/bin/env bash
# How a self-play run is doing.  Usage:
#
#     scripts/splendor_progress.sh            # the long run (runs/nscc0)
#     scripts/splendor_progress.sh sprint2h   # the 1v1 sprint
#
# Three things decide whether a run is healthy, and only one of them is about
# playing strength:
#
#   net-vs-greedy   THE quality signal.  Paired games (same seeds, seats
#                   swapped) against a frozen opponent, so 60 games already
#                   separate two nets.  Should pass 0.8 within a few
#                   generations and then climb slowly.  Flat or falling across
#                   SEVERAL generations is the self-play failure mode.
#   sims/s          throughput.  The whole node should do >=400k; under 200k
#                   the actors are starved and nothing else matters.
#   trunc / stuck   degeneracy.  `trunc` climbing means the players are
#                   stalling instead of buying -- a bad equilibrium.
#
# The learner's loss is deliberately NOT summarised: in self-play the data
# distribution moves with the player, so the loss can rise while the player
# gets stronger.  `value_explained_variance`, `policy_top1_agreement` and
# `policy_entropy` vs `target_entropy` are the usable learner diagnostics and
# are printed below instead.
set -u
NAME=${1:-nscc0}
RUN=${RUN_DIR:-$HOME/scratch/splendor/runs/$NAME}

echo "=== queue"
qstat -answ1 "$USER" 2>/dev/null | tail -20

echo
echo "=== tail of the PBS log"
ls -t "$HOME"/scratch/splendor/*.o* 2>/dev/null | head -2 | xargs -r tail -n 12

echo
echo "=== $RUN"
python3 - "$RUN/metrics.jsonl" <<'PY'
import json, os, sys

path = sys.argv[1]
if not os.path.exists(path):
    print("no metrics yet:", path)
    raise SystemExit

last, evals = {}, []
for line in open(path):
    try:
        row = json.loads(line)
    except Exception:
        continue
    last[row.get("kind")] = row
    if row.get("kind") == "eval":
        evals.append(row)


def f(v, d=2):
    return format(v, f".{d}f") if isinstance(v, (int, float)) else "--"


p = last.get("progress", {})
print(f"self-play   {f(p.get('lifetime_s', 0) / 3600)} h total"
      f"   (this slot {f(p.get('elapsed_s', 0) / 3600)} h)")
print(f"games       {p.get('games_done', 0):,}"
      f"   generation {p.get('generation', 0)}"
      f"   learner steps {p.get('steps', 0):,}")
print(f"throughput  {f(p.get('sims_per_s', 0), 0)} sims/s"
      f"   {f(p.get('games_per_s', 0), 1)} games/s"
      f"   {f(p.get('steps_per_s', 0))} steps/s")
print(f"buffer      {p.get('buffer', 0):,} records"
      f"   window {p.get('window_retained', '?')}/{p.get('window', '?')} generations")
print(f"degeneracy  stuck {f(p.get('stuck_rate'), 3)}"
      f"   truncated {f(p.get('truncation_rate'), 3)}"
      f"   restarts {p.get('actor_restarts', 0)}")
print(f"modes       {p.get('mode_games', {})}")

learner = last.get("learner", {})
if learner:
    print(f"\nlearner     value-explained-variance "
          f"{f(learner.get('value_explained_variance'))}"
          f"   top1-agreement {f(learner.get('policy_top1_agreement'))}")
    print(f"            policy entropy {f(learner.get('policy_entropy'))}"
          f" vs target {f(learner.get('target_entropy'))}"
          f"   sample reuse {f(learner.get('reuse'), 1)}")

print("\nevaluations (paired win rate, frozen opponents)")
for e in evals[-8:]:
    print(f"  gen {e.get('generation'):>4}  vs random {f(e.get('net_vs_random'))}"
          f"   vs greedy {f(e.get('net_vs_greedy'))}"
          f"   search vs greedy {f(e.get('search_vs_greedy'))}")
if not evals:
    print("  (none yet — the first one lands at generation 5)")

if "summary" in last:
    print("\nfinished:", json.dumps(last["summary"])[:400])
PY
