// Dev-only mock of the replay REST API (docs/REPLAY_FORMAT.md §4).
//
// It plays random legal games through the real server/gameLogic.js, records them in the
// stored-file shape of §1, and rebuilds frames exactly like §3 — so the UI can be developed
// and screenshotted before server/replayEngine.js lands. Nothing here is imported by the app.
//
//   node scripts/dev/mockReplayServer.mjs [--port 10011] [--games 8]

import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const serverDir = join(here, '..', '..', 'server');
const require = createRequire(join(serverDir, 'index.js'));

const express = require('express');
const cors = require('cors');
const { Server: SocketServer } = require('socket.io');
const http = require('node:http');

const logic = require(join(serverDir, 'gameLogic.js'));
const {
  createInitialGameState, clientView, processAction, processResign, calculateRatingChanges,
} = logic;

const args = process.argv.slice(2);
const argValue = (name, fallback) => {
  const i = args.indexOf(name);
  return i === -1 ? fallback : args[i + 1];
};
const PORT = Number(argValue('--port', 10011));
const GAME_COUNT = Number(argValue('--games', 8));

// ── Card / tile catalogues, by id ──────────────────────────────────────────

const CARD_BY_ID = new Map();
const TILE_BY_ID = new Map();

function fillCatalogues() {
  if (Array.isArray(logic.ALL_CARDS) && Array.isArray(logic.ALL_BONUS_TILES)) {
    for (const card of logic.ALL_CARDS) CARD_BY_ID.set(card.id, card);
    for (const tile of logic.ALL_BONUS_TILES) TILE_BY_ID.set(tile.id, tile);
    return;
  }
  // Fallback: harvest them from fresh game states (board + decks = every card).
  for (let attempt = 0; attempt < 200 && TILE_BY_ID.size < 10; attempt++) {
    const state = createInitialGameState(
      [{ username: 'a', avatarSeed: 1 }, { username: 'b', avatarSeed: 2 }], { unlimitedTime: true });
    for (const row of state.board) for (const card of row) CARD_BY_ID.set(card.id, card);
    for (const deck of state.decks) for (const card of deck) CARD_BY_ID.set(card.id, card);
    for (const tile of state.bonusTiles) TILE_BY_ID.set(tile.id, tile);
  }
}
fillCatalogues();

const cardById = id => structuredClone(CARD_BY_ID.get(id));
const tileById = id => structuredClone(TILE_BY_ID.get(id));

// ── Random legal play ──────────────────────────────────────────────────────

const pick = list => list[Math.floor(Math.random() * list.length)];
const shuffled = list => {
  const copy = [...list];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
};

function discountOf(player) {
  const d = [0, 0, 0, 0, 0];
  for (const card of player.cards) d[card.reward]++;
  return d;
}

function affordable(player, card) {
  const discount = discountOf(player);
  let gold = 0;
  for (let i = 0; i < 5; i++) {
    const need = Math.max(0, card.cost[i] - discount[i]);
    if (player.gems[i] < need) gold += need - player.gems[i];
  }
  return gold <= player.gems[5];
}

function listBuys(state, seat) {
  const player = state.players[seat];
  const buys = [];
  for (const row of state.board) {
    for (const card of row) if (affordable(player, card)) buys.push({ cardId: card.id, source: 'board', points: card.points });
  }
  for (const card of player.reserved) {
    if (affordable(player, card)) buys.push({ cardId: card.id, source: 'reserved', points: card.points });
  }
  return buys;
}

function listGemSets(state) {
  const available = [0, 1, 2, 3, 4].filter(color => state.gems[color] > 0);
  if (available.length === 0) return [];
  const sets = [];
  const three = shuffled(available).slice(0, 3);
  if (three.length === 3) sets.push(three);
  const stacked = available.filter(color => state.gems[color] >= state.config.take2MinStack);
  if (stacked.length) { const color = pick(stacked); sets.push([color, color]); }
  const two = shuffled(available).slice(0, 2);
  if (two.length === 2) sets.push(two);
  sets.push([pick(available)]);
  return sets;
}

/** Applies one random legal turn; returns the recorded action tuple, or null when stuck. */
function playTurn(state, seat) {
  const player = state.players[seat];
  const buys = listBuys(state, seat);
  const gemSets = listGemSets(state);
  const reserveTargets = [];
  if (player.reserved.length < state.config.maxReserved) {
    for (const row of state.board) for (const card of row) reserveTargets.push({ kind: 'card', cardId: card.id });
    state.deckCounts.forEach((count, tierIdx) => {
      if (count > 0) reserveTargets.push({ kind: 'deck', tier: tierIdx + 1 });
    });
  }

  const roll = Math.random();
  const order = [];
  if (buys.length && roll < 0.62) order.push('buy');
  if (roll < 0.93) order.push('gems', 'buy', 'reserve');
  else order.push('reserve', 'gems', 'buy');

  for (const kind of order) {
    if (kind === 'buy' && buys.length) {
      const ranked = [...buys].sort((a, b) => b.points - a.points);
      const choice = Math.random() < 0.6 ? ranked[0] : pick(buys);
      const res = processAction(state, seat, { type: 'BUY_CARD', cardId: choice.cardId, source: choice.source });
      if (!res.error) return [seat, 'B', choice.cardId, choice.source === 'board' ? 'b' : 'r'];
    }
    if (kind === 'gems') {
      for (const colors of gemSets) {
        const res = processAction(state, seat, { type: 'TAKE_GEMS_CONFIRMED', colors });
        if (!res.error) return [seat, 'G', colors];
      }
    }
    if (kind === 'reserve' && reserveTargets.length) {
      const enter = processAction(state, seat, { type: 'ENTER_RESERVE' });
      if (enter.error) continue;
      const deckTargets = reserveTargets.filter(t => t.kind === 'deck');
      const cardTargets = reserveTargets.filter(t => t.kind === 'card');
      const target = (deckTargets.length && (Math.random() < 0.3 || !cardTargets.length))
        ? pick(deckTargets) : pick(cardTargets.length ? cardTargets : deckTargets);
      if (target.kind === 'deck') {
        const res = processAction(state, seat, { type: 'RESERVE_FROM_DECK', tier: target.tier });
        if (!res.error) return [seat, 'RD', target.tier];
      } else {
        const res = processAction(state, seat, { type: 'RESERVE_CARD', cardId: target.cardId });
        if (!res.error) return [seat, 'R', target.cardId];
      }
      // Reserve mode already took the gold; fall through only if nothing worked.
      const fallback = state.board.flat()[0];
      if (fallback) {
        const res = processAction(state, seat, { type: 'RESERVE_CARD', cardId: fallback.id });
        if (!res.error) return [seat, 'R', fallback.id];
      }
      return null;
    }
  }
  return null;
}

/**
 * Plays one whole game and returns the stored replay JSON of docs/REPLAY_FORMAT.md §1.
 * `ending` forces the last moves: 'resign' → X, 'timeout' → T.
 */
function recordRandomGame({ id, mode, layout, players, startedAt, clock, ending = null, endAfter = 30 }) {
  const state = createInitialGameState(players, {
    gameMode: mode, teamLayout: layout, unlimitedTime: true,
  });
  const first = state.currentPlayerIndex;
  const setup = {
    board: state.board.map(row => row.map(card => card.id)),
    decks: state.decks.map(deck => deck.map(card => card.id)),
    tiles: state.bonusTiles.map(tile => tile.id),
  };

  const actions = [];
  let guard = 0;
  while (state.phase === 'PLAYING' && guard++ < 1500) {
    const seat = state.currentPlayerIndex;

    if (state._pendingTileChoice?.length) {
      const tileId = pick(state._pendingTileChoice);
      const res = processAction(state, seat, { type: 'CHOOSE_TILE', tileId });
      if (res.error) throw new Error(`CHOOSE_TILE failed: ${res.error}`);
      actions.push([seat, 'N', tileId]);
      continue;
    }

    if (ending && actions.length >= endAfter) {
      processResign(state, seat);
      actions.push([seat, ending === 'timeout' ? 'T' : 'X']);
      continue;
    }

    const recorded = playTurn(state, seat);
    if (recorded) { actions.push(recorded); continue; }
    // No legal move (10 gems, 3 reserved, nothing affordable) — the server has no pass.
    processResign(state, seat);
    actions.push([seat, 'X']);
  }

  const endedAt = startedAt + 60_000 + actions.length * 12_000;
  return {
    v: 1,
    id,
    t: startedAt,
    e: endedAt,
    mode,
    layout: layout ?? null,
    n: players.length,
    clock,
    players: players.map(p => ({
      u: p.username,
      a: p.avatarSeed,
      ...(mode === 'INDIVIDUAL' ? {} : { team: p.teamId }),
      ai: !!p.ai,
    })),
    first,
    setup,
    actions,
    result: buildResult(state),
  };
}

function buildResult(state) {
  const scores = state.players.map(p => p.score);
  const cards = state.players.map(p => p.cards.length);
  const resigned = [...(state.resignedPlayers || [])];
  let winners = null;
  let winningTeamIds = null;

  if (state.gameMode === 'INDIVIDUAL') {
    const ranked = state.players
      .map((p, i) => ({ i, score: p.score, cardCount: p.cards.length }))
      .sort((a, b) => b.score - a.score || a.cardCount - b.cardCount);
    const best = ranked[0];
    winners = ranked
      .filter(r => r.score === best.score && r.cardCount === best.cardCount)
      .map(r => r.i);
  } else {
    winningTeamIds = state.gameResult?.winningTeamIds ?? [];
    winners = state.players
      .map((p, i) => (winningTeamIds.includes(p.teamId) ? i : -1))
      .filter(i => i >= 0);
  }

  return {
    scores, cards, resigned, winners, winningTeamIds,
    reason: state.gameResult?.reason ?? (resigned.length ? 'FORFEIT' : 'SCORE'),
    rating: calculateRatingChanges(state.players, state),
  };
}

// ── Reconstruction (docs/REPLAY_FORMAT.md §3) ─────────────────────────────

class ReplayCorruptError extends Error {
  constructor(actionIndex, message) {
    super(`Replay corrupt at action ${actionIndex}: ${message}`);
    this.actionIndex = actionIndex;
  }
}

function snapshot(state) {
  // clientView is a shallow copy — frames must not alias the live, mutating state.
  return structuredClone(clientView(state));
}

function reconstruct(json) {
  const players = json.players.map(p => ({
    username: p.u, avatarSeed: p.a, ...(p.team === undefined ? {} : { teamId: p.team }),
  }));
  const state = createInitialGameState(players, {
    gameMode: json.mode,
    teamLayout: json.layout,
    unlimitedTime: true,
    firstPlayerIndex: json.first,
  });
  state.board = json.setup.board.map(ids => ids.map(cardById));
  state.decks = json.setup.decks.map(ids => ids.map(cardById));
  state.deckCounts = state.decks.map(deck => deck.length);
  state.bonusTiles = json.setup.tiles.map(tileById);
  state.currentPlayerIndex = json.first;
  state.roundStartPlayer = json.first;
  state.timeControl = null;

  const frames = [{
    i: 0, turn: 0, actor: null, action: null, result: null,
    state: snapshot(state), pendingTileChoice: null,
  }];

  json.actions.forEach((entry, index) => {
    const [seat, code, arg, arg2] = entry;
    let result;

    try {
      if (code === 'G') {
        result = expect(processAction(state, seat, { type: 'TAKE_GEMS_CONFIRMED', colors: arg }), index);
      } else if (code === 'R') {
        const enter = expect(processAction(state, seat, { type: 'ENTER_RESERVE' }), index);
        result = expect(processAction(state, seat, { type: 'RESERVE_CARD', cardId: arg }), index);
        result.payload.goldTaken = !!enter.payload.goldTaken;
      } else if (code === 'RD') {
        const enter = expect(processAction(state, seat, { type: 'ENTER_RESERVE' }), index);
        result = expect(processAction(state, seat, { type: 'RESERVE_FROM_DECK', tier: arg }), index);
        result.payload.goldTaken = !!enter.payload.goldTaken;
        const reserved = state.players[seat].reserved;
        result.payload.cardId = reserved[reserved.length - 1]?.id;
      } else if (code === 'B') {
        result = expect(processAction(state, seat, {
          type: 'BUY_CARD', cardId: arg, source: arg2 === 'r' ? 'reserved' : 'board',
        }), index);
      } else if (code === 'N') {
        result = expect(processAction(state, seat, { type: 'CHOOSE_TILE', tileId: arg }), index);
      } else if (code === 'X' || code === 'T') {
        processResign(state, seat);
        result = code === 'X'
          ? { type: 'RESIGN', actingPlayer: seat, payload: { resignedPlayerIndex: seat } }
          : { type: 'TIMEOUT', actingPlayer: seat, payload: { timedOutPlayerIndex: seat } };
      } else {
        throw new ReplayCorruptError(index, `unknown action code ${code}`);
      }
    } catch (err) {
      if (err instanceof ReplayCorruptError) throw err;
      throw new ReplayCorruptError(index, err.message);
    }

    if (state._tileClaimed) {
      result.tileClaimed = state._tileClaimed;
      delete state._tileClaimed;
    }
    const pendingTileChoice = state._pendingTileChoice || null;

    frames.push({
      i: index + 1,
      turn: state.turnNumber,
      actor: seat,
      action: entry,
      result: structuredClone(result),
      state: snapshot(state),
      pendingTileChoice,
    });
  });

  return frames;
}

function expect(res, index) {
  if (!res || res.error) throw new ReplayCorruptError(index, res?.error ?? 'no result');
  return res.result;
}

// ── Fixture games ──────────────────────────────────────────────────────────

const NAMES = ['alice', 'bob', 'carol', 'dave', 'erin', 'frank', 'grace', 'heidi', 'ivan', 'judy'];
let seedCounter = 3;

function seat(username, teamId, ai = false) {
  return { username, avatarSeed: (seedCounter += 7) % 97, ...(teamId === undefined ? {} : { teamId }), ai };
}

function buildFixtures(count) {
  const specs = [
    { mode: 'INDIVIDUAL', layout: null, seats: [seat('alice'), seat('bob')], clock: false },
    { mode: 'INDIVIDUAL', layout: null, seats: [seat('carol'), seat('dave'), seat('erin')], clock: true },
    { mode: 'INDIVIDUAL', layout: null, seats: [seat('frank'), seat('grace'), seat('heidi'), seat('ivan')], clock: true },
    { mode: 'ONE_V_TWO', layout: null, seats: [seat('judy', 0), seat('alice', 1), seat('bob', 1)], clock: true },
    // Prefer a game that hits a multi-noble choice so the "choosing a noble…" caption is exercised.
    { mode: 'TEAM', layout: 'ADJACENT', seats: [seat('carol', 0), seat('dave', 0), seat('erin', 1), seat('frank', 1)], clock: true, preferNoble: true },
    { mode: 'TEAM', layout: 'OPPOSITE', seats: [seat('grace', 0), seat('heidi', 1), seat('ivan', 0), seat('judy', 1)], clock: false },
    { mode: 'INDIVIDUAL', layout: null, seats: [seat('alice'), seat('carol'), seat('ivan', undefined, true)], clock: true, ending: 'resign', endAfter: 22 },
    { mode: 'INDIVIDUAL', layout: null, seats: [seat('bob'), seat('dave'), seat('grace'), seat('judy')], clock: true, ending: 'timeout', endAfter: 26 },
  ];

  const games = [];
  const now = Date.now();
  for (let i = 0; i < Math.min(count, specs.length); i++) {
    const spec = specs[i];
    // Newest first: game 0 is the most recent.
    const startedAt = now - (i * 3.5 + 1) * 3600_000;
    let json = null;
    for (let attempt = 0; attempt < 12 && !json; attempt++) {
      try {
        const candidate = recordRandomGame({
          id: `game-${startedAt}-${Math.random().toString(36).slice(2, 6)}`,
          mode: spec.mode, layout: spec.layout, players: spec.seats,
          startedAt, clock: spec.clock, ending: spec.ending ?? null, endAfter: spec.endAfter ?? 30,
        });
        // Prove the round trip before serving it.
        reconstruct(candidate);
        const wantsNoble = spec.preferNoble && attempt < 8;
        if (wantsNoble && !candidate.actions.some(a => a[1] === 'N')) continue;
        json = candidate;
      } catch (err) {
        console.warn(`  retrying ${spec.mode} game (${err.message})`);
      }
    }
    if (json) games.push(json);
  }
  return games;
}

console.log(`Generating ${GAME_COUNT} random legal games through server/gameLogic.js ...`);
const STORE = buildFixtures(GAME_COUNT);
const FRAME_CACHE = new Map();

for (const game of STORE) {
  const frames = reconstruct(game);
  FRAME_CACHE.set(game.id, frames);
  console.log(`  ${game.id}  ${game.mode.padEnd(11)} n=${game.n}  ${game.actions.length} actions  ${frames.length} frames  reason=${game.result.reason}`);
}

const indexEntry = game => ({
  id: game.id,
  t: game.t,
  e: game.e,
  mode: game.mode,
  layout: game.layout ?? null,
  n: game.n,
  players: game.players.map(p => p.u),
  ai: game.players.map(p => !!p.ai),
  teams: game.players.map(p => p.team ?? null),
  winners: game.result.winners,
  winningTeamIds: game.result.winningTeamIds,
  turns: game.actions.length,
});

const metaOf = game => ({
  t: game.t, e: game.e, mode: game.mode, layout: game.layout,
  n: game.n, clock: game.clock, first: game.first, result: game.result,
  players: game.players.map(p => ({
    username: p.u, avatarSeed: p.a, ...(p.team === undefined ? {} : { teamId: p.team }), isAI: !!p.ai,
  })),
});

// ── HTTP + socket.io (the login screen expects both) ───────────────────────

const app = express();
app.use(cors({ origin: '*' }));
app.use(express.json());

app.get('/health', (_req, res) => res.json({ status: 'healthy', timestamp: Date.now(), mock: true }));
app.get('/api/accounts', (_req, res) => res.json(
  NAMES.slice(0, 4).map((username, i) => ({
    username, rating: 1000 + i * 37, gamesPlayed: 12 + i, wins: 4 + i, avatarSeed: 11 + i * 5, created: Date.now(),
  })),
));

app.get('/api/replays/status', (_req, res) => res.json({ github: false, memory: STORE.length }));

app.get('/api/replays', (req, res) => {
  const limit = Math.min(Number(req.query.limit) || 50, 200);
  const offset = Number(req.query.offset) || 0;
  const games = STORE.map(indexEntry).slice(offset, offset + limit);
  res.json({ games, total: STORE.length, source: 'memory' });
});

app.get('/api/replays/:id/raw', (req, res) => {
  const game = STORE.find(g => g.id === req.params.id);
  if (!game) return res.status(404).json({ error: 'Replay not found' });
  res.json(game);
});

app.get('/api/replays/:id', (req, res) => {
  const game = STORE.find(g => g.id === req.params.id);
  if (!game) return res.status(404).json({ error: 'Replay not found' });
  try {
    const frames = FRAME_CACHE.get(game.id) ?? reconstruct(game);
    FRAME_CACHE.set(game.id, frames);
    res.json({ id: game.id, meta: metaOf(game), frames });
  } catch (err) {
    res.status(422).json({ error: err.message, actionIndex: err.actionIndex });
  }
});

const server = http.createServer(app);
const io = new SocketServer(server, { cors: { origin: '*' } });
io.on('connection', socket => {
  socket.on('ping', () => socket.emit('pong'));
  socket.on('login', ({ username }, cb) => cb?.({
    success: true,
    account: { username, rating: 1000, gamesPlayed: 0, wins: 0, avatarSeed: 21, created: Date.now() },
  }));
  socket.on('enter_lobby', cb => cb?.({ action: 'ok', lobbyState: {
    players: [], teamMode: false, teamFormat: null, teamLayout: 'ADJACENT',
    teamSeats: [[null, null], [null, null]], unlimitedTime: false,
  } }));
});

server.listen(PORT, () => {
  console.log(`Mock replay server on http://localhost:${PORT} (${STORE.length} games)`);
});
