#!/usr/bin/env bash
# How a self-play run is doing.  Usage:
#
#     scripts/splendor_progress.sh               # snapshot, the long run
#     scripts/splendor_progress.sh sprint2h      # snapshot, the 1v1 sprint
#     scripts/splendor_progress.sh -f            # snapshot, then FOLLOW live
#     scripts/splendor_progress.sh -f sprint2h
#
# `-f` follows metrics.jsonl, which the trainer writes line-buffered to scratch,
# so it works on a job that is ALREADY running -- unlike the PBS stdout file,
# which PBS only copies back to the submit directory when the job ends.
# Ctrl-C stops watching; it does not touch the job.
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
FOLLOW=0
if [ "${1:-}" = "-f" ] || [ "${1:-}" = "--follow" ]; then FOLLOW=1; shift; fi
NAME=${1:-nscc0}
RUN=${RUN_DIR:-$HOME/scratch/splendor/runs/$NAME}

echo "=== queue"
qstat -u "${USER:-$(id -un)}" 2>&1 | tail -20

echo
echo "=== this slot's live log"
SLOT_LOG=$(ls -t "$RUN"/logs/slot.*.log 2>/dev/null | head -1)
if [ -n "$SLOT_LOG" ]; then
    echo "$SLOT_LOG"
    tail -n 12 "$SLOT_LOG"
else
    echo "(none yet — written once the job starts running)"
fi

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

[ "$FOLLOW" = "1" ] || exit 0

echo
echo "=== following $RUN/metrics.jsonl  (Ctrl-C stops watching, not the job)"
while [ ! -f "$RUN/metrics.jsonl" ]; do sleep 5; done
tail -n 0 -F "$RUN/metrics.jsonl" | python3 -u - <<'FOLLOWPY'
import json, sys


def f(v, d=2):
    return format(v, f".{d}f") if isinstance(v, (int, float)) else "--"


def clock(seconds):
    s = int(seconds or 0)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


for line in sys.stdin:
    try:
        r = json.loads(line)
    except Exception:
        continue
    kind, t = r.get("kind"), clock(r.get("t"))
    if kind == "progress":
        print(f"{t} gen {r.get('generation'):>3} games {r.get('games_done', 0):>8,}"
              f"  {f(r.get('sims_per_s'), 0):>8} sims/s"
              f"  {f(r.get('games_per_s'), 1):>6} games/s"
              f"  buffer {r.get('buffer', 0):>10,}"
              f"  stuck {f(r.get('stuck_rate'), 3)}"
              f"  trunc {f(r.get('truncation_rate'), 3)}")
    elif kind == "eval":
        print(f"{t} EVAL gen {r.get('generation')}"
              f"   vs random {f(r.get('net_vs_random'))}"
              f"   vs greedy {f(r.get('net_vs_greedy'))}"
              f"   search vs greedy {f(r.get('search_vs_greedy'))}")
    elif kind == "generation":
        print(f"{t} GENERATION {r.get('generation')} closed"
              f"   {r.get('samples_in_generation', 0):,} samples"
              f"   window {r.get('window_retained')}/{r.get('window')}")
    elif kind == "learner":
        print(f"{t}   learner step {r.get('step', 0):>7,}"
              f"   value-ev {f(r.get('value_explained_variance'))}"
              f"   top1 {f(r.get('policy_top1_agreement'))}"
              f"   entropy {f(r.get('policy_entropy'))}/{f(r.get('target_entropy'))}"
              f"   reuse {f(r.get('reuse'), 1)}")
    elif kind in ("actor_restart", "replay_window_truncated", "eval_skipped"):
        print(f"{t} ! {kind}: {json.dumps(r)[:200]}")
    elif kind == "summary":
        print(f"{t} SUMMARY {json.dumps(r)[:400]}")
FOLLOWPY
