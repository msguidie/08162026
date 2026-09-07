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
- ✅ D1 encoder (OBS v1 = 860), C5 symmetry (exact on 100k states), terminal values, 12.5M-param net + checkpoint gate
- ✅ D1 search: PIMC PUCT/Gumbel MCTS, evaluators, scheduler, bots; G2 passed (greedy ≥96% vs random; MCTS@400 81% vs greedy)
- ✅ D2 self-play system: actors, inference servers, replay window, learner, train orchestrator, configs, bootstrap, partial PPO fallback (55 tests)
- ✅ D2 arena (paired seeds, seat rotation, Bradley–Terry, 54 tests), export bundle, NSCC PBS scripts + setup
- ✅ G3 CPU smoke run (21 min, 2p): NetBot 0.81–0.83 vs random, SearchBot@48 0.80–0.92 vs greedy; found+fixed forced-playout bug at low sims
- ✅ PBS scripts + requirements (nscc_setup.sh, nscc_train.pbs self-chaining, nscc_eval.pbs)

## Phase E — Deployment
- ✅ server/aiBridge.js + aiFallback.js + lobby AI seats + WaitingRoom UI (126 server tests incl. bot e2e in 2p/1v2/2v2 and worker-crash fallback)
- ✅ splendor_ai/worker (Windows 3060) + run_worker.bat (30 real-server games, all modes, every bot move from the worker)
- ✅ local end-to-end: AI plays 2p / 1v2 (solo+duo) / 2v2 / 3p against scripted humans; lobby UI screenshots QA'd

## Phase F — Docs & delivery
- ✅ requirements.txt, requirements-worker.txt, PBS scripts, README sections; ✅ docs/DEPLOY_zh.md A–D
- ✅ Independent review + adversarial verification: 23 findings → 12 confirmed → all fixed (replay id validation, aiAvailable re-broadcast, in-flight request invalidation, orphaned-noble handling, standings/winners/serialization, docs); trainer review: 29 findings, confirmed critical/high/medium being fixed (final checkpoint before eval, global curriculum counter, per-instance seeds, queue drain, PPO guard, retention, graceful SIGINT)
- ⬜ Commit & push
