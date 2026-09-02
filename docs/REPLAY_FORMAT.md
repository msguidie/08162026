# Replay format & API contract (v1)

This contract is shared by: `server/replayRecorder.js`, `server/replayEngine.js`, `server/replayGithub.js`,
the client (`src/replay/*`) and the Python validator (`splendor_ai/validation/*`). Do not change it without
updating all four.

## 1. Stored replay file — `replays/YYYY/MM/<id>.json` (~1.5–3 KB)

```jsonc
{
  "v": 1,
  "id": "game-1725280000000-ab12",      // room id
  "t": 1725280000000,                   // game start (ms)
  "e": 1725281800000,                   // game end (ms)
  "mode": "INDIVIDUAL",                 // "INDIVIDUAL" | "TEAM" | "ONE_V_TWO"
  "layout": null,                       // null | "ADJACENT" | "OPPOSITE" (TEAM only)
  "n": 3,                               // number of players
  "clock": true,                        // time control was enabled
  "players": [                          // index == playerIndex (seat order)
    { "u": "alice", "a": 7, "team": 0, "ai": false }   // "team" omitted in INDIVIDUAL; "ai" true for bot seats
  ],
  "first": 1,                           // initial currentPlayerIndex (== roundStartPlayer)
  "setup": {
    "board": [[4 card ids], [4 ids], [4 ids]],   // tier1, tier2, tier3 face-up, left→right
    "decks": [[ids...], [ids...], [ids...]],      // remaining deck arrays in SERVER ORDER: pop() takes the LAST element
    "tiles": [tile ids]                           // revealed nobles, in order
  },
  "actions": [                          // one entry per COMPLETED turn action, in order
    [0, "G", [0, 1, 2]],                //  G  take gems: colors list (1–3 entries; same color twice for take-2-same)
    [1, "R", 37],                       //  R  reserve face-up card by cardId (gold is recomputed on replay)
    [2, "RD", 2],                       //  RD reserve from deck top, tier 1..3
    [0, "B", 12, "b"],                  //  B  buy cardId from "b" board | "r" own reserved
    [0, "N", 4],                        //  N  choose noble tileId (only when a choice was pending)
    [1, "X"],                           //  X  resign
    [2, "T"]                            //  T  timed out (server auto-resign)
  ],
  "result": {
    "scores": [15, 9, 7], "cards": [11, 8, 6], "resigned": [],
    "winners": [0],                     // playerIndex list (INDIVIDUAL: top rank incl. ties) or null
    "winningTeamIds": null,             // team modes: [0] | [1] | [0,1] draw | []
    "reason": "SCORE",                  // "SCORE" | "FORFEIT" | null
    "rating": [5, 3, 1]                 // rating deltas as applied by the server
  }
}
```

Card ids are the deterministic ids 0..89 from `server/gameLogic.js` (`ALL_CARDS`), tile ids 0..9 (`ALL_BONUS_TILES`).
Anything derivable is NOT stored (gold taken on reserve, auto-claimed nobles, deck refills, scores per turn).

`replays/index.json`:
```jsonc
{ "v": 1, "games": [ { "id": "...", "t": 1725280000000, "e": 1725281800000, "mode": "INDIVIDUAL", "n": 3,
  "players": ["alice", "bob", "carol"], "ai": [false,false,true], "winners": [0], "winningTeamIds": null, "turns": 57 } ] }
```
Newest first. Written with GitHub Contents API (read sha → PUT; on 409/422 conflict re-read and retry up to 3×).

## 2. Recording semantics (server)

- `begin(room)` right after `createInitialGameState` in `startGame()`.
- After every successful `processAction` (`game_action` handler), map the returned `ActionResult`:
  - `SELECT_GEM` with `payload.completed !== false` → `G payload.selected`
  - `TAKE_GEMS_CONFIRMED` → `G payload.selected`
  - `RESERVE_CARD` → `R payload.cardId`; `RESERVE_FROM_DECK` → `RD payload.tier`
  - `BUY_CARD` → `B cardId, source === 'board' ? 'b' : 'r'`
  - `CHOOSE_TILE` → `N payload.tileId`
  - `ENTER_RESERVE`, `CANCEL_GEMS`, incomplete `SELECT_GEM` → not recorded
- `resign` handler → `X`; `eliminateTimedOutPlayer` → `T`.
- When `room.gameState.phase === 'GAME_OVER'` (after `broadcastProcessedAction`, so `room.ratingChanges` exists) → `finish(room)`:
  build JSON, keep in memory ring (last 100), push to GitHub if configured (async, never blocks the game).
- `quit_room`, inactivity abandon, TTL cleanup → recording discarded.

## 3. Reconstruction (server, `replayEngine.reconstruct(json) → frames`)

1. `state = createInitialGameState(players, { gameMode, teamLayout, unlimitedTime: true, firstPlayerIndex: first })`
2. Overwrite `state.board`, `state.decks` (card objects looked up by id), `state.deckCounts`, `state.bonusTiles`,
   `state.currentPlayerIndex = state.roundStartPlayer = first`, `state.timeControl = null`.
3. Frame 0 = `{ i: 0, turn: 0, actor: null, action: null, result: null, state: clientView(state), pendingTileChoice: null }`.
4. For each action, call the SAME `processAction`/`processResign` the live server uses:
   - `G` → `TAKE_GEMS_CONFIRMED {colors}`
   - `R` → `ENTER_RESERVE` then `RESERVE_CARD {cardId}` (result type `RESERVE_CARD`, payload `{cardId, tier, fromDeck:false, goldTaken}`)
   - `RD` → `ENTER_RESERVE` then `RESERVE_FROM_DECK {tier}` (result type `RESERVE_FROM_DECK`, payload `{tier, fromDeck:true, goldTaken, cardId}`)
   - `B` → `BUY_CARD {cardId, source}`; `N` → `CHOOSE_TILE {tileId}`
   - `X` → `processResign` (result `{type:'RESIGN', payload:{resignedPlayerIndex}}`)
   - `T` → `processResign` (result `{type:'TIMEOUT', payload:{timedOutPlayerIndex}}`)
   After each: move `state._tileClaimed` into `result.tileClaimed` (and delete it), record
   `pendingTileChoice = state._pendingTileChoice || null`, push a frame whose `state` is `clientView(state)`
   (decks stripped, ALL reserved cards visible — the client hides per perspective).
5. Any error → throw `ReplayCorruptError(actionIndex, message)`.

## 4. REST API (server)

- `GET /api/replays?limit=50&offset=0` → `{ games: [index entries], total, source: "github" | "memory" }`
- `GET /api/replays/:id` → `{ id, meta: { t, e, mode, layout, n, clock, first, result,
  players: [{ username, avatarSeed, teamId?, isAI }] }, frames: [...] }` — 404 if unknown, 422 if corrupt.
- `GET /api/replays/:id/raw` → the stored JSON (used by the Python validator).
- `GET /api/replays/status` → `{ github: bool, memory: n }`.

Env vars: `REPLAY_GITHUB_TOKEN`, `REPLAY_GITHUB_REPO` (owner/name), `REPLAY_GITHUB_BRANCH` (default `main`),
`REPLAY_GITHUB_DIR` (default `replays`). Without a token the feature works in-memory only.

## 5. Client behaviour

- Entry: "Replays" button on the login screen (both the pre-login card and the logged-in card).
- `REPLAY_BROWSER` lists games from `/api/replays`; `REPLAY_VIEWER` plays `/api/replays/:id`.
- Viewer = read-only clone of the in-game layout (same components/classes), random default perspective,
  controls: play/pause, step ◀ ▶, speed 0.5×/1×/2×/4× (base 2000 ms per frame), scrub, perspective, exit.
- Animations are driven exactly like `GameBoard`: gem deltas from `result.payload.selected` / `gemsReturned` /
  `goldTaken`, opponent card `+1` from `BUY_CARD`, noble toast when the perspective player's tile count grows,
  framer-motion `layout` on market cards and noble tiles.
