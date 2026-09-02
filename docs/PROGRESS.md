# Progress log

Legend: ⬜ todo · 🟨 in progress · ✅ done · ❌ blocked

## Phase A — Research
- ✅ Read the whole codebase; rule differences documented in docs/PLAN.md §0
- ✅ Read chlligence/PPO-splendorReinforcementLearning end to end
- ✅ Workflow: profiled 11 repos/papers → two independent judges (both recommend multiplayer AlphaZero + PIMC + search at inference)
- 🟨 Second round: multiplayer-alphazero, RinascimentoFramework, LightZero, literature (running; informs details only)
- ✅ Decision record: docs/AI_DESIGN.md (binding interfaces, gates); evidence in docs/research/

## Phase B — Replay feature
- ✅ server/replayRecorder.js, replayEngine.js, replayStore.js, replayGithub.js + index.js hooks + REST (82 tests incl. socket.io e2e in all modes)
- ✅ src/replay/* (browser + viewer), LoginScreen button, store/App wiring (build clean; mock-server screenshots in docs/screenshots)
- ⬜ Node end-to-end test (scripted games in all modes → replay reconstructable)
- ✅ Playwright screenshots (mobile + desktop) against the mock and the real server (5 recorded games, all modes)

## Phase C — Python engine + cross-validation
- ✅ splendor_ai/rules (cards, engine, actions, view) + 116 pytest; encode.py deferred to Phase D
- ✅ validation: 16,000 games / 1.27M steps in all modes, 0 mismatches (two-way legal-set check), 73k steps/s

## Phase D — Training system (multiplayer AlphaZero, see docs/AI_DESIGN.md)
- 🟨 D1 encoder (OBS v1), colour-symmetry, terminal values, network + checkpoint gate
- 🟨 D1 search: determinization, PUCT/Gumbel MCTS with per-seat value vectors, evaluators, bots (random/greedy/MCTS anchor)
- ⬜ D2 self-play system: actors, inference servers, replay window, learner, train orchestrator, configs, bootstrap
- ⬜ D2 arena (paired seeds, seat rotation, Bradley–Terry), export
- ⬜ G3 CPU smoke run in this sandbox (beats random/greedy)
- ⬜ PBS scripts + requirements

## Phase E — Deployment
- ✅ server/aiBridge.js + aiFallback.js + lobby AI seats + WaitingRoom UI (126 server tests incl. bot e2e in 2p/1v2/2v2 and worker-crash fallback)
- ⬜ splendor_ai/worker (Windows 3060) + run_worker.bat
- ⬜ local end-to-end: AI plays 2p / 1v2 / 2v2 against scripted humans

## Phase F — Docs & delivery
- ⬜ requirements.txt, requirements-worker.txt, PBS scripts, README, Chinese delivery notes
- ⬜ Independent review + adversarial verification workflow
- ⬜ Commit & push
