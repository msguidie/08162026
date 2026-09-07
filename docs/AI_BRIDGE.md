# AI bridge contract (v1): Render server ⇄ local GPU worker ⇄ lobby

Shared by `server/aiBridge.js`, `server/aiFallback.js`, lobby handlers in `server/index.js`,
`src/components/WaitingRoom.tsx` and `splendor_ai/worker/*`. Free-tier friendly: the worker opens an
OUTBOUND socket.io connection to the Render server (no port forwarding, no tunnels).

## 1. Worker ⇄ server (socket.io, worker is a client)

Env on Render: `AI_WORKER_SECRET` (if unset, all AI features are hidden/disabled).
Env on the worker: `SERVER_URL`, `AI_WORKER_SECRET`, `MODEL_DIR`, `DEVICE` (cuda|cpu), `SEARCH_SIMS`, `TIME_BUDGET_MS`.

| direction | event | payload | ack |
|---|---|---|---|
| worker→server | `ai_worker_register` | `{ secret, name, version, modes: ["INDIVIDUAL","ONE_V_TWO","TEAM"] }` | `{ ok: true }` or `{ error }` |
| server→worker | `ai_move_request` | `{ requestId, roomId, playerIndex, kind: "MOVE" \| "TILE", deadlineMs, state, knownReserved, pendingTileChoice }` | — |
| worker→server | `ai_move_response` | `{ requestId, action, info?: { policy?: number, value?: number, sims?: number, ms?: number } }` | `{ ok }` or `{ error }` |
| server→worker | `ai_move_cancel` | `{ requestId }` (game ended, room gone, deadline passed, or the position moved on) | — |

`state` = `clientView(room.gameState)` for that seat with OTHER seats' reserved cards replaced by
`{ id: <cardId or -1>, tier, hidden: true, known: <bool> }`; `knownReserved` = list of card ids that were reserved
from the board (public knowledge) so the worker can fill them in; own reserved cards are full objects.
`deadlineMs` is an absolute epoch ms; the server applies a fallback move if no valid response arrives by then
(default budget 15 s; typical worker reply < 2 s).

`action` (worker → server) is one of:
```jsonc
{ "type": "TAKE_GEMS", "colors": [0, 1, 2] }          // 1–3 colours, same colour twice for take-2-same
{ "type": "RESERVE_CARD", "cardId": 37 }
{ "type": "RESERVE_FROM_DECK", "tier": 2 }
{ "type": "BUY_CARD", "cardId": 12, "source": "board" | "reserved" }
{ "type": "CHOOSE_TILE", "tileId": 4 }                 // only for kind == "TILE"
{ "type": "RESIGN" }
{ "type": "NONE" }                                     // no legal move (server resigns the bot)
```
The server translates to the existing protocol (`TAKE_GEMS_CONFIRMED`; `ENTER_RESERVE` + `RESERVE_CARD`/`RESERVE_FROM_DECK`;
`BUY_CARD`; `CHOOSE_TILE`) and applies it through the SAME code path as a human `game_action`
(`applyGameAction(room, playerIndex, action)` extracted from the socket handler with identical behaviour:
timer update, consumeTurnTime, increment, broadcastProcessedAction, replay recording). If the worker's action is
rejected by `processAction`, the server logs it and uses the fallback.

Only one worker is active at a time (a new registration replaces the old one). Disconnect → `aiAvailable=false`;
games in progress continue with the fallback policy for bot seats. Both edges of `available` (first worker
registers / last worker disconnects) re-broadcast the lobby, so `aiAvailable` is never stale for members already
sitting in it.

## 2. Fallback policy (server, `aiFallback.js`)
Deterministic greedy used when no worker is connected, the worker times out, or returns an invalid action:
1. buy the affordable card with the most points (ties: reserved before board, then cheapest);
2. otherwise take up to 3 distinct colours that most reduce the deficit of the cheapest attractive board card
   (validated black-box by trying `TAKE_GEMS_CONFIRMED` candidates on a `structuredClone`);
3. otherwise reserve the best board card (if reserved < 3);
4. otherwise `NONE` → the bot resigns.
Tile choice fallback: the first pending tile.

## 3. Lobby integration (server `index.js` + `WaitingRoom.tsx`)

Lobby state gains `aiAvailable: boolean`; each lobby player gains `isAI: boolean`.
Bot accounts are ordinary accounts named `Bot Alpha`, `Bot Beta`, `Bot Gamma`, `Bot Delta` (auto-created, rated like humans).

| event | payload | rule |
|---|---|---|
| `lobby_add_ai` | `{}` → ack `{ ok }` / `{ error }` | requires `aiAvailable`, a human in the lobby, lobby size < 6 (team lobbies: < required seats); bot joins `ready=true`, `wantsFirst=false` |
| `lobby_remove_ai` | `{ username }` | any lobby member may remove any bot |
| `select_team_seat` | `{ teamId, seatIndex, forUsername? }` | `forUsername` allowed only when it names a bot in this lobby |

Bots never toggle ready (always ready), never volunteer to go first, count toward `set_team_mode` player counts,
and are removed when the last human leaves. `checkAutoStart` unchanged otherwise.

Game start: `playerSockets[i] = { socketId: "ai:<n>", username, playerIndex, isAI: true }` — emits to a
non-existent socket id are harmless no-ops; `isPlayerConnected` stays true so clocks behave as for a present human.

Turn driver (`aiBridge.maybeAct(room)`): called after `game_start` emission and after every `broadcastProcessedAction`.
If `phase === 'PLAYING'`, the current seat is a bot, and no request is in flight for the room → wait 600 ms
(UX pacing) → send `ai_move_request` → apply the response → the normal broadcast triggers the next `maybeAct`.
Guard against re-entrancy and stale responses (room deleted, game over, turn changed) via `requestId`.

- **Request kind.** `TILE` only when a noble choice is *actionable*: `_pendingTileChoice` non-empty **and**
  `turnAction.type === 'BUY'`, which is what `processAction` requires to accept `CHOOSE_TILE`. A pending choice
  left behind by a gem take or a reserve (docs/KNOWN_ISSUES.md §1) can never be answered, so the bridge asks for a
  `MOVE`, the seat plays on, and it is not treated as a stalled turn — a bot is never resigned over it.
- **Superseded requests.** If `currentPlayerIndex` or `turnNumber` changes while a request is in flight (a human
  resigns, a timeout eliminates a seat, the bot seat itself is resigned), the next `maybeAct` cancels it
  (`ai_move_cancel`), refuses any late answer for that `requestId`, and issues a fresh request for the seat that is
  really to move.

## 4. Replays
Bot seats are recorded like any other seat with `ai: true` in `players[]` of the replay file.

## 5. REST
`GET /api/ai/status` → `{ enabled: bool, available: bool, name?: string, modes?: string[] }`.
