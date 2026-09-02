# Progress log

Legend: ⬜ todo · 🟨 in progress · ✅ done · ❌ blocked

## Phase A — Research
- ✅ Read the whole codebase; rule differences documented in docs/PLAN.md §0
- ✅ Read chlligence/PPO-splendorReinforcementLearning end to end
- 🟨 Workflow: profile 11 repos/papers → two independent judges (running)
- ⬜ Second round: multiplayer-alphazero, RinascimentoFramework, literature via search snippets → final algorithm decision
- ⬜ Write decision record into docs/AI_DESIGN.md

## Phase B — Replay feature
- ✅ server/replayRecorder.js, replayEngine.js, replayStore.js, replayGithub.js + index.js hooks + REST (82 tests incl. socket.io e2e in all modes)
- ✅ src/replay/* (browser + viewer), LoginScreen button, store/App wiring (build clean; mock-server screenshots in docs/screenshots)
- ⬜ Node end-to-end test (scripted games in all modes → replay reconstructable)
- ✅ Playwright screenshots (mobile + desktop) against the mock; ⬜ against the real server

## Phase C — Python engine + cross-validation
- ✅ splendor_ai/rules (cards, engine, actions, view) + 116 pytest; encode.py deferred to Phase D
- ✅ validation: 16,000 games / 1.27M steps in all modes, 0 mismatches (two-way legal-set check), 73k steps/s

## Phase D — Training system
- ⬜ model / algo / league / DDP train / eval / export
- ⬜ CPU smoke run beats random

## Phase E — Deployment
- ⬜ server/aiBridge.js + lobby AI seats + WaitingRoom UI
- ⬜ splendor_ai/worker (Windows 3060) + run_worker.bat
- ⬜ local end-to-end: AI plays 2p / 1v2 / 2v2 against scripted humans

## Phase F — Docs & delivery
- ⬜ requirements.txt, requirements-worker.txt, PBS scripts, README, Chinese delivery notes
- ⬜ Independent review + adversarial verification workflow
- ⬜ Commit & push
