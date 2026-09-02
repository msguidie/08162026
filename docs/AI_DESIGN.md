# AI design record (binding contract for `splendor_ai/`)

Decision date: 2026-09-02. Inputs: 15 code/paper profiles + two independent judge reports
(`docs/research/` has the extracted evidence). This file is the contract every implementation stream follows.

## 0. Decision

**Multiplayer AlphaZero**: one mode-conditioned policy/value network with a **per-seat value vector**, trained by
**PUCT self-play with per-simulation determinization (PIMC)** of hidden information, playout-cap randomization,
forced playouts + policy-target pruning, and **search at deployment**. Masked PPO is kept only as a fallback learner
(same actors/encoder/network/arena; different target producer). Why (evidence in `docs/research/judges.md`):

1. The user-supplied MaskablePPO project plateaued after 51.8M steps on an easier 2-player variant
   (its own `pool_index.json`: win rate vs previous generation oscillating 0.36–0.64 for the last 35 generations).
2. Search on fixed weights is the largest measured effect in the corpus (+590 Elo; 150-sim search beat its own
   DQN champion 29–1). Our deployment budget is seconds per move on a dedicated GPU.
3. AlphaZero needs only the exact terminal outcome, which the validated engine already computes for all modes.
   PPO needs hand-tuned shaping re-derived for INDIVIDUAL / 1v2 / 2v2.
4. A per-seat value vector makes 3p/4p, 1v2 (asymmetric thresholds) and 2v2 (shared team entry) one mechanism.
5. The closest working reference (cestpasphoto/alpha-zero-general, MIT) is AlphaZero-shaped and matches our token
   economy, noble table and selection-blocking 10-token cap.

**Engine**: keep the validated pure-Python engine (`splendor_ai/rules`, 0 mismatches over 16k games, 73k steps/s).
No Numba/C++ rewrite unless the throughput gate (§7) fails.

**One network for all modes** (2p, 3p, 4p individual, 1v2, 2v2), conditioned on mode/seat/role/threshold features.
Per-mode fine-tuning is an optional last step shipped only if it wins the arena (≥55% over ≥2,000 paired games).
`--modes` lets the user train a single-mode specialist with the same code if they prefer.

**No DDP by default**: a ~13M-parameter MLP learner is >15× oversupplied by one A100. Layout: GPU0 learner,
GPU1–3 batched inference servers, 56 CPU actor processes. The learner is DDP-ready (`WORLD_SIZE>1` branch) for
future >40M-parameter nets.

## 1. Fixed interfaces (do not change without updating every consumer)

### 1.1 Action space (unchanged, from `splendor_ai/rules/actions.py`)
65 actions: 0–9 take-3-distinct, 10–19 take-2-distinct, 20–24 take-1, 25–29 take-2-same, 30–41 reserve board
(`30+tier0*4+slot`), 42–44 reserve deck, 45–56 buy board, 57–59 buy reserved slot, 60–64 choose tile (index into
`state.tiles`). No pass. `ACTION_RESIGN=-1`, `ACTION_TIMEOUT=-2` are out-of-band.

### 1.2 Terminal value vector — `splendor_ai/values.py`
`terminal_values(state) -> np.float32[4]` in ABSOLUTE seat order (entries ≥ n are 0), only valid when
`state.phase == 'GAME_OVER'`:
- INDIVIDUAL: rank-linear over all seats, `z = 1 - 2*(rank-1)/(n-1)` (2p: ±1), ties share the mean rank; ranking
  by (score desc, cards asc) exactly as the server; resigned seats are ranked last (score 0) → -1.
- ONE_V_TWO: +1 to the winning side's seat(s), -1 to the other side, 0 to all on a tie (`winningTeamIds` has 2 ids)
  or when nobody qualifies; FORFEIT → forfeiting side -1.
- TEAM: same per team; FORFEIT → forfeiting team -1.
`seat_relative(z, seat) = np.roll(z, -seat)` gives the network target (index 0 = acting seat).
`z_valid_mask(n) = [1]*n + [0]*(4-n)`.

Stuck seat (no legal move): the actor calls `engine.resign(state, seat)` (mirrors the server, which has no pass) and
the game continues/ends per the engine. Truncation at `max_plies` (default 400): score by current standings via the
same ranking, value-target weight 0.3, policy targets full weight. Both are distinct labelled outcomes in stats.

### 1.3 Observation encoder — `splendor_ai/encode.py`
`OBS_VERSION = 1`, `OBS_DIM` a module constant. `encode(state, seat, out=None) -> np.float32[OBS_DIM]` reads ONLY the
information set of `seat` (`rules/view.public_view`): never `state.decks` order, never another seat's deck-reserved
card id. Content-addressed cards (no card-id embedding). All features in [-1, 1].
Layout (seat-relative: player block j is seat `(seat + j) % n`; blocks j ≥ n zero with `present=0`):
- Card block (23): cost/7 (5), reward one-hot (5), points/5, tier one-hot (3), per-colour shortfall after MY
  discounts and tokens /7 (5), gold needed /5, affordable-now, turns_to_buy=max(ceil(sum(shortfall)/3),max(shortfall))/6, present.
  12 board slots (tier-major, left→right as the action indices) ×23; own reserved 3×23; other seats' reserved
  3 seats × 3 slots × 25 (23 + known + deck_reserved; an unknown card is zero except tier one-hot, present, deck_reserved).
- Player blocks 4 × 28: gems/supply-max (6), discounts/7 (5), score/15, cards/20, reserved/3, tiles/3, resigned,
  present, is_self, is_teammate, is_solo_role, seat-offset one-hot (4), excess over own side's threshold /15.
- Tile slots 5 × 18: requirement/4 (5), my per-colour card shortfall /4 (5), my total shortfall /12, present,
  qualifies-now, per-other-seat min total shortfall /12 (3), reserved for future (1) — zero.
- Public deck composition 48: per tier, counts of unseen cards by (reward colour × points bucket {0, 1–2, 3+}) /8 (45)
  + per-tier remaining/deck size (3). "Unseen" = not on board, not in any tableau, not publicly reserved, not my own
  reserved; other seats' deck-reserved cards remain in the unseen pool.
- Global 40: supply/max (6), mode one-hot (3), team layout one-hot (2), num_players one-hot (3), turn/100,
  final-round flag, final-round triggered-by seat-offset one-hot (4), plies until the turn returns to the round leader /4,
  revocable flag (TEAM), my side threshold progress, other side threshold progress, pending-tile-choice flag,
  my absolute seat one-hot (4), padding to 40.
Also: `encode_batch(states, seats, out)`, and the symmetry helpers below.

### 1.4 Colour symmetry — `splendor_ai/symmetry.py`
The card and tile tables are closed under the 5 cyclic colour rotations (`addCycle`). `rotate_state(state, k)`
returns an equivalent state with colour c → (c+k)%5 everywhere (gems, discounts, card ids via
`cards.rotate_id(id, k)`, tile ids via `cards.rotate_tile_id`, decks, reserved). `action_perm(k) -> int[65]`
satisfies `legal_mask(rotate_state(s,k)) == legal_mask(s)[action_perm(k)]` and the same for policy targets; only
indices 0–29 move (slot-based actions are invariant). Verified by a test over ≥100k random states for k in 0..4:
`encode(rotate_state(s,k), seat)` must equal the feature-permuted `encode(s, seat)` (feature permutation given by
`feature_perm(k)`), and `legal_mask` must permute exactly. Augmentation and the test-time ensemble use these.

### 1.5 Network — `splendor_ai/model.py`
`SplendorNet(cfg)`; forward `(obs[B,OBS_DIM] float, mask[B,65] bool) -> dict(logits[B,65] (masked to -1e9 before
softmax, inside the module), value[B,4] tanh, score[B,4], stuck[B,4])`. Trunk: LayerNorm → Linear(OBS_DIM, width) →
`blocks` × pre-LN residual [LN → Linear → GELU → Linear] → LN. Defaults width=768, blocks=10 (~13M params);
smoke config width=128, blocks=2. Checkpoint format: `{"obs_version", "action_version": 1, "cfg", "state_dict",
"step", "generation", "meta"}`; loading refuses an `obs_version` mismatch with a RuntimeError naming it.
Loss = CE(policy_target, log_softmax(masked logits)) + 1.0·masked-MSE(value, z) + 0.15·MSE(score) + 0.15·BCE(stuck).

### 1.6 Search — `splendor_ai/search/`
- `determinize.py`: `determinize(state, seat, rng) -> GameState`: unseen multiset per tier = all cards minus
  visible/tableaus/public reserves/my reserves; assign each other seat's deck-reserved slot a random unseen card of
  its tier; shuffle the remainder into the decks respecting `deck_counts`. Deterministic given `rng`.
- `mcts.py`: `MCTS(cfg, rng)` with a tree keyed by action path (open loop), nodes in flat arrays; per simulation:
  determinize at the root (K universes cycled by simulation index), descend with PUCT
  `Q + c_puct·P·sqrt(N_parent)/(1+N)` (FPU: parent Q minus `fpu_reduction`), `same_player` flag per edge (CHOOSE_TILE
  is a same-seat sub-decision), per-seat value vectors rotated on backup (max^n: each node maximises its own acting
  seat's entry), Dirichlet noise at the root (alpha = 10/num_legal, eps 0.25) when `noise=True`, forced playouts
  `n_forced = sqrt(k·P·N)` (k=2) with policy-target pruning, optional Gumbel root selection (`root="gumbel"`,
  sequential halving over Gumbel top-m, completed-Q policy target) behind a flag, optional anti-clairvoyance PUCT
  penalty on deck reserves. Stuck leaves: resign path → terminal. Terminal leaves use `terminal_values`.
  API for batching across games: `leaf = tree.select_leaf(root_state, seat)` returns `(obs, mask, token)` or None
  when the simulation completed without a network call (terminal); `tree.backup(token, priors, values)`.
  `tree.result() -> SearchResult(visits[65], policy_target[65], root_value[4], chosen_action, stats)`.
  `Scheduler` runs G trees in lockstep with one `Evaluator.evaluate(obs[B], mask[B]) -> (priors[B,65], values[B,4])`
  call per step (no virtual loss needed).
- `evaluators.py`: `NetEvaluator` (torch, batch), `UniformRolloutEvaluator` (NN-free: uniform prior, greedy/random
  rollout value), used for the anchor bot and bootstrap.
- Tests: 2p antisymmetry (`value[0] ≈ -value[1]` in symmetric positions), seat relabelling consistency,
  search@400 with the NN-free evaluator beats the greedy bot ≥75% in 2p over ≥100 paired games.

### 1.7 Bots & arena — `splendor_ai/bots.py`, `splendor_ai/arena.py`
Bots implement `act(state, seat, rng) -> action`: `RandomBot`, `GreedyBot` (1-ply: buy max points affordable →
take gems reducing the cheapest attractive card's shortfall → reserve best → stuck→resign; also used as the server
fallback design), `MctsBot(evaluator, sims)`, `NetBot` (policy argmax, no search), `SearchBot` (net + MCTS).
Arena: paired seeds with full seat rotation (2p: swap; n>2: all cyclic rotations of the seating), per-mode results,
Bradley–Terry/BayesElo fit over the whole result matrix (pinned anchors: random=0, greedy, mcts-anchor), 95% CIs,
win-as-seat-k table, STALE/deadlock bucket, JSON + markdown report.

### 1.8 Self-play system — `splendor_ai/selfplay/`
- `config.py`: dataclasses + YAML (`configs/smoke_cpu.yaml`, `configs/nscc_4xa100.yaml`).
- `sample.py`: record = compact state bytes (`engine.GameState.to_bytes()/from_bytes()`), seat, sparse policy target
  (idx, prob), z[4], z_weight, aux targets, generation; the learner re-encodes on the fly.
- `actor.py`: process running G games in lockstep (PCR: full search `sims_full` with noise, recorded, prob 0.25;
  fast `sims_fast`, no noise, not recorded), temperature 1 for the first 12 plies then argmax, mode sampled from the
  mixture, opponent sampling per seat for mixed games (latest / uniform historical / pinned anchor / greedy),
  stuck→resign, truncation, C5 augmentation on write (5 rotations), sends records to the replay writer.
- `inference_server.py`: one per GPU; shared-memory request slots per actor; coalesces up to `max_batch` or `max_wait_ms`;
  bf16 `torch.inference_mode`; reloads weights when the versioned file changes; refuses obs_version mismatch.
- `replay.py`: in-memory rolling generational window (4 → 20 generations), uniform sampling, checkpointable.
- `learner.py`: batch 4096, AdamW lr 2e-4 warmup 2k → cosine 2e-5, wd 1e-4, clip 1.0, bf16 autocast, replay ratio
  target 3–6×, publishes `weights/latest.pt` atomically every N steps, checkpoints model+optimizer+replay+RNG every 10 min.
- `train.py`: orchestrator (spawns servers/actors/learner; resumable; `--smoke` runs everything on CPU in minutes).
- `bootstrap.py` (optional): NN-free MCTS teacher → supervised warm start.
- Curriculum: games 0–300k 100% 2p at reduced sims; then mixture 25% ind2 / 10% ind3 / 15% ind4 / 25% ovt / 25% team.
- Instrumentation (per generation, JSONL + TensorBoard): sims/s, moves/s, games/s, per-mode game length, stuck rate,
  truncation rate, policy entropy, value MSE/explained variance, aux accuracies, search-vs-policy argmax disagreement,
  arena Elo vs fixed anchors.

### 1.9 Deployment worker — `splendor_ai/worker/`
`worker.py`: python-socketio client per `docs/AI_BRIDGE.md`; `adapter.py`: hydrate `GameState` from the server payload
(re-derive discounts/scores from cards and reject inconsistencies; rebuild `reserved_public` from `knownReserved`);
search with a wall-clock budget (soft 1.5 s, hard 2.5 s, anytime), K=16 universes, C5 root ensemble, 1-ply stuck
filter, re-validate the chosen action with `legal_mask` on the fresh state, fallback ladder (GPU→CPU search→policy
argmax→greedy→NONE), per-move structured log, reconnection. `run_worker.bat` / `.env.example` for Windows.

## 2. Go/no-go gates (recorded in docs/PROGRESS.md)
- G1 encoder: rotation equivariance exact on 100k states; ≥100k encodes/s/core.
- G2 search: NN-free MCTS@400 ≥75% vs greedy (2p); greedy ≥95% vs random in every mode.
- G3 smoke (this sandbox, CPU): `train.py --smoke` (2p, 16-wide net... width 128) reaches ≥80% vs random and
  >55% vs greedy with search within ~20 minutes wall-clock; checkpoint/resume works; obs_version gate fires.
- G4 throughput (NSCC, first job): ≥400k sims/s node-wide, ≥250k evals/s per inference GPU, clean resume.
- G5 learning (NSCC): generation 10 ≥85% vs the MCTS anchor in 2p with monotone anchored Elo.
- G6 deployment: ≥500 games through a local Node server with zero rejected actions/timeouts.

## 3. Licensing
Copy code only from MIT sources (cestpasphoto/alpha-zero-general with upstream attribution, RinascimentoFramework,
roeey777/Splendor-AI) with notices retained; everything else re-implemented from the described ideas.

## 4. Addendum after research round 2 (docs/research/profiles_round2.md)
- **Warning evidence**: seal256/splendor (the one inspectable AlphaZero-style Splendor attempt) could not train a useful
  value head and only reached stable self-play at a 5-point win condition. Mitigations adopted: auxiliary score/stuck
  heads from day one; blended value target (root Q + outcome); optional **score-utility** in search
  (`Q_search = value + score_utility_weight · predicted_score_margin`, default 0 → A/B); a **reduced-threshold
  smoke matrix** (INDIVIDUAL win threshold override 5/8/15, engine config only, never used for deployment) so a
  non-learning full game is diagnosed against a working short game; the **PPO fallback learner** is a real module
  (`selfplay/ppo_learner.py`: search-free actors, per-seat returns tracked by decision points, margin-scaled terminal
  rewards, additive -inf masking) sharing actors/encoder/network/arena — switched by `learner: az|ppo` in config.
- KataGo defaults adopted: Dirichlet alpha = 10.83 / num_legal, PCR 25% full (recorded) / 75% cheap (not recorded),
  noise pruning of the policy target, uncertainty-agnostic uniform sampling.
- Gumbel root selection (m=16, sequential halving, completed-Q targets, mctx constants: gumbel_scale 1.0,
  value_scale 0.1, maxvisit_init 50) is a first-class option, not an afterthought; A/B against PUCT in the smoke matrix.
- Per-seat value vector semantics follow petosa/multiplayer-alphazero: the full vector is backed up unchanged in
  absolute seat order; each node consumes only its mover's entry (max^n). No sign flips. 1v2 vectors are not zero-sum.
- Chance is handled by per-simulation determinization on a shared action-path tree (equivalent to capping chance
  branching at 1); never divide backed-up values by the number of chance children.
- Opponent pool: PFSP weights (1 - winrate)^0.5 with pinned anchors; for 1v2 track solo-seat and duo-seat win rates
  separately (an agent can be exploitable in one role while fine on average).
- Evaluation ladder: NN-free MCTS anchor at 40/160/640 sims as an absolute, monotone strength curve.
