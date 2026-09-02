# Cross-language validation harness

The hard go/no-go gate before any RL training: prove that the Python rules
engine in `splendor_ai/rules/` is **bit-exact** with the authoritative Node
engine `server/gameLogic.js` — same legal moves, same resulting state, same
action-result payloads, in every game mode.

Three pieces:

| file | role |
| --- | --- |
| `gen_trajectories.js` | plays random games inside the **real** `gameLogic.js` and dumps every step |
| `replay_check.py` | replays that dump through the Python engine and asserts equality |
| `probe_state.js` | materialises an **arbitrary** position in `gameLogic.js` for the pytest suite |

`package.json` here only sets `"type": "commonjs"` — the repository root
declares `"type": "module"`, and these scripts are CommonJS like `server/`.

---

## Running it

```bash
cd splendor_ai/validation

make all                    # 2000 games x 6 configurations, then check + bench
make GAMES=200 all          # quick pass (~30 s)
make stress                 # extra coverage batches (see below)
make check                  # check whatever is already in ./data
make bench                  # Python engine throughput only
make clean
```

Overridable variables: `GAMES` (per configuration), `SEED`, `OUT` (default
`data`), `PY` (default `/home/user/venv-splendor/bin/python`), `NODE`.

Raw equivalents:

```bash
node gen_trajectories.js --out data/ind2.jsonl.gz --games 2000 \
     --mode INDIVIDUAL --players 2 --seed 2
node gen_trajectories.js --out data/ovt.jsonl.gz  --games 2000 --mode ONE_V_TWO --seed 5
node gen_trajectories.js --out data/team_opp.jsonl.gz --games 2000 \
     --mode TEAM --layout OPPOSITE --seed 7

/home/user/venv-splendor/bin/python replay_check.py data --bench 300000
```

`replay_check.py` exits non-zero and prints the first mismatch with the full
pre-action position, the action, and a field-by-field diff.

### Configurations covered

`INDIVIDUAL` with 2 / 3 / 4 seats, `ONE_V_TWO` (3 seats, solo on seat 0),
`TEAM` with `ADJACENT` (seats `0,0,1,1`) and `OPPOSITE` (seats `0,1,0,1`) —
the same seat→team maps `server/index.js` builds in `startGame()`.

### Stress batches (`make stress`)

Uniform random play rarely reaches some positions, so four extra batches steer
exploration (legality is still decided black-box, so the assertions are
unchanged — only the sampling distribution moves):

* `--t1-bias 0.85` — prefer cheap 0-point tier-1 buys, which grows tableaus
  without growing scores and therefore reaches three-noble positions.
* `--orphan-hunt` — right after a `CHOOSE_TILE`, prefer a non-buy action, so a
  still-qualifying second noble produces the *orphaned pending choice* state
  (`_pendingTileChoice` set while `turnAction === null`).
* `--chaos-frac 1.0 --resign-p 0.03 --timeout-p 0.02` — resign/timeout on
  nearly every game, covering the forfeit and elimination paths.

---

## What `gen_trajectories.js` does

* Installs a seeded `mulberry32` PRNG as `Math.random` **before** requiring
  `gameLogic.js`, so shuffles and first-player picks are reproducible.
* For every step it enumerates candidate actions and decides acceptance
  **black-box**: each candidate runs on a structural copy of the live state and
  is kept only if `processAction` returns `ok`.  Nothing about legality is
  re-derived, so the dump is an independent oracle.
  Candidates: all **55** colour multisets of size 1–3 (`TAKE_GEMS_CONFIRMED`),
  12 board reserves + 3 deck reserves (`ENTER_RESERVE` then `RESERVE_*`,
  accepted only if *both* messages succeed), 12 board buys + up to 3 reserved
  buys, and `CHOOSE_TILE` for every revealed tile.
* Picks uniformly among the accepted set (unless a bias flag is set).
* ~30 % of gem takes go through the incremental desktop path
  (`SELECT_GEM` one colour at a time, in a **random order**), asserting that
  the take completes exactly on the last colour and yields the same multiset.
* ~2 % of steps in a "chaos" game inject a random resign, ~1 % a timeout
  (`processResign` on the current seat, exactly as `eliminateTimedOutPlayer`
  does).  30 % of games are chaos games by default (`--chaos-frac`), keeping
  enough clean games to exercise the score-based endings.
* ~2 % of steps brute-force **every ordering of every colour multiset** and
  throw if acceptance or the resulting selection ever depends on the order —
  this is what licenses the canonical sorted ordering behind each take index.
* If the server accepts nothing at all (10 tokens + 3 reserved + nothing
  affordable), the seat is resigned and the step is tagged `stuck-resign`; the
  checker requires `is_stuck(state)` to be true there.
* Emits per game a `game` record (setup), one `step` record per completed turn
  action, and an `end` record that also carries the full **replay-format JSON**
  of `docs/REPLAY_FORMAT.md` §1.

### Output format

JSONL, gzipped when the path ends in `.gz`.  A `step` record carries the
compact action, the accepted set, the post-action snapshot and the result
payload; compact codes map 1:1 onto protocol messages:

| code | protocol |
| --- | --- |
| `["G",[colors]]` | `[{type:"TAKE_GEMS_CONFIRMED",colors}]` |
| `["R",cardId]` | `[{type:"ENTER_RESERVE"},{type:"RESERVE_CARD",cardId}]` |
| `["RD",tier]` | `[{type:"ENTER_RESERVE"},{type:"RESERVE_FROM_DECK",tier}]` |
| `["B",id,"b"\|"r"]` | `[{type:"BUY_CARD",cardId:id,source:"board"\|"reserved"}]` |
| `["N",tileId]` | `[{type:"CHOOSE_TILE",tileId}]` |
| `["X"]` / `["T"]` | `processResign` / `eliminateTimedOutPlayer` |

---

## What `replay_check.py` asserts, per step

1. **Legality, both directions.** The Python legal action set must equal the
   set the Node server accepted.  Compact codes are mapped to action indices
   with `from_replay_code`; gem takes therefore compare as colour multisets,
   because the take index *is* the canonical multiset.
2. **Full post-state equality**: board card ids per tier in order, deck counts
   and deck tops, gem supply, revealed tiles, per-player gems / cards /
   reserved / tiles / score, `currentPlayerIndex`, `roundStartPlayer`,
   `turnNumber`, `phase`, `finalRoundTriggeredBy`, `resignedPlayers`,
   `gameResult`, `_pendingTileChoice` and `turnAction`.
3. **Action result payload**: `selected`, `gemsReturned`, `goldTaken`, `tier`,
   `cardId`, `source`, `tileId` and the consumed `tileClaimed`.
4. Per game: final state, the full remaining decks, `calculateRatingChanges`,
   the winner lists — **and** a from-scratch rebuild of the stored
   replay-format JSON that must land on the identical final state (so real
   `replays/**.json` files are provably consumable by the same code path).

## Real replay files

`E.replay(json)` takes a stored replay file (`GET /api/replays/:id/raw`)
verbatim and returns the final `GameState`; the checker runs it on every
generated game.

## Last full run

16,000 games / 1,274,214 steps across all six configurations plus the four
stress batches — **0 mismatches**.  See the "Rules engine & validation"
section of `../README.md` for the numbers.
