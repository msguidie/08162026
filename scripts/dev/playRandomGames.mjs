// Dev-only: drives the REAL server (server/index.js) over socket.io exactly the way the
// browser client does, playing random legal games to GAME_OVER so the replay REST API has
// real recordings to serve. Nothing here is imported by the app.
//
//   PORT=10012 node server/index.js &
//   node scripts/dev/playRandomGames.mjs --server http://localhost:10012
//
// Flags:
//   --server <url>   base URL of a running server (default http://localhost:10012)
//   --games <n>      how many of the configurations below to play (default: all of them)
//   --seed <n>       PRNG seed for the random policy (default 0x9e3779b9)
//   --quiet          only print the final replay-id table
//
// The play policy is the one from server/test/replay.e2e.js: probe candidate actions until
// the ack comes back without an error, answer tile_choice_required, CANCEL_GEMS after a
// failed SELECT_GEM probe, and resign a seat that has no legal move left.

import { io } from 'socket.io-client';

// ── args ───────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const argValue = (name, fallback) => {
  const i = args.indexOf(name);
  return i === -1 || i + 1 >= args.length ? fallback : args[i + 1];
};
const BASE_URL = String(argValue('--server', 'http://localhost:10012')).replace(/\/$/, '');
const QUIET = args.includes('--quiet');

const ACK_TIMEOUT_MS = 15000;
const SYNC_TIMEOUT_MS = 15000;
const GAME_TIMEOUT_MS = 5 * 60 * 1000;

const log = (...parts) => { if (!QUIET) console.log(...parts); };

// ── deterministic pseudo-random policy (same generator as the e2e suite) ───
let seed = Number(argValue('--seed', 0x9e3779b9)) | 0;
function random() {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function shuffled(list) {
  const copy = [...list];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

// ── infrastructure ─────────────────────────────────────────────────────────
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

async function waitFor(condition, message, timeout = SYNC_TIMEOUT_MS) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (condition()) return true;
    await sleep(3);
  }
  throw new Error(`timed out waiting for ${message}`);
}

async function api(path, options) {
  const response = await fetch(`${BASE_URL}${path}`, options);
  const body = await response.json().catch(() => null);
  return { status: response.status, body };
}

async function waitForServer() {
  const deadline = Date.now() + 30000;
  for (;;) {
    try {
      const health = await api('/health');
      if (health.status === 200) return;
    } catch { /* not listening yet */ }
    if (Date.now() > deadline) throw new Error(`no server at ${BASE_URL}`);
    await sleep(200);
  }
}

// ── socket client, mirroring src/store/gameStore.ts ────────────────────────

function emitAck(client, event, payload) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`ack timeout: ${event}`)), ACK_TIMEOUT_MS);
    const done = response => {
      clearTimeout(timer);
      resolve(response || {});
    };
    // `enter_lobby` takes the acknowledgement as its only argument.
    if (payload === undefined) client.socket.emit(event, done);
    else client.socket.emit(event, payload, done);
  });
}

function action(client, payload) {
  return emitAck(client, 'game_action', { roomId: client.roomId, action: payload });
}

const runTag = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`;
function freshName(prefix, seat) {
  return `${prefix}${seat}-${runTag}`;
}

async function connect(username) {
  const created = await api('/api/accounts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username }),
  });
  assert(created.status === 200, `could not create account ${username}: ${JSON.stringify(created.body)}`);

  const socket = io(BASE_URL, { transports: ['websocket'], forceNew: true, reconnection: false });
  const client = {
    username,
    socket,
    playerIndex: null,
    roomId: null,
    state: null,
    pendingTile: null,
    finalState: null,
  };
  socket.on('game_start', data => {
    client.roomId = data.roomId;
    client.playerIndex = data.playerIndex;
    client.state = data.gameState;
  });
  socket.on('game_state_update', state => {
    client.state = state;
    if (state.phase === 'GAME_OVER') client.finalState = state;
  });
  socket.on('tile_choice_required', ({ tileIds }) => { client.pendingTile = tileIds; });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`connect timeout for ${username}`)), ACK_TIMEOUT_MS);
    socket.on('connect', () => { clearTimeout(timer); resolve(); });
    socket.on('connect_error', err => { clearTimeout(timer); reject(err); });
  });
  const login = await emitAck(client, 'login', { username });
  assert(login.success, `login failed for ${username}: ${JSON.stringify(login)}`);
  return client;
}

function disconnectAll(clients) {
  for (const client of clients) {
    try { client.socket.disconnect(); } catch { /* already gone */ }
  }
}

// ── turn driving ───────────────────────────────────────────────────────────

function turnMoved(client, fromIndex) {
  return !client.state
    || client.state.phase !== 'PLAYING'
    || client.state.currentPlayerIndex !== fromIndex
    || !!client.pendingTile;
}

function gemCandidates() {
  const out = [];
  for (let a = 0; a < 5; a++) {
    out.push([a, a]);
    out.push([a]);
    for (let b = 0; b < 5; b++) {
      if (b === a) continue;
      out.push([a, b]);
      for (let c = 0; c < 5; c++) {
        if (c === a || c === b) continue;
        out.push([a, b, c]);
      }
    }
  }
  return out;
}

async function tryBuy(client) {
  const from = client.playerIndex;
  const me = client.state.players[from];
  const candidates = [
    ...shuffled(me.reserved.filter(card => card.id >= 0)).map(card => ({ cardId: card.id, source: 'reserved' })),
    ...shuffled(client.state.board.flat()).map(card => ({ cardId: card.id, source: 'board' })),
  ];
  for (const candidate of candidates) {
    const res = await action(client, { type: 'BUY_CARD', ...candidate });
    if (!res.error) return true;
    if (turnMoved(client, from)) return true;
  }
  return turnMoved(client, from);
}

// Mobile flow: one TAKE_GEMS_CONFIRMED with the full selection.
async function tryGemsMobile(client) {
  const from = client.playerIndex;
  for (const colors of shuffled(gemCandidates())) {
    const res = await action(client, { type: 'TAKE_GEMS_CONFIRMED', colors });
    if (!res.error) return true;
    if (turnMoved(client, from)) return true;
  }
  return turnMoved(client, from);
}

// Desktop flow: SELECT_GEM one colour at a time until the server completes it.
async function tryGemsDesktop(client) {
  const from = client.playerIndex;
  for (let step = 0; step < 4; step++) {
    let accepted = false;
    for (const color of shuffled([0, 1, 2, 3, 4])) {
      const res = await action(client, { type: 'SELECT_GEM', color });
      if (!res.error) { accepted = true; break; }
    }
    if (!accepted) break;
    if (turnMoved(client, from)) return true;
  }
  // A rejected SELECT_GEM leaves an empty TAKE_GEMS turn action behind; clear it.
  await action(client, { type: 'CANCEL_GEMS' });
  return turnMoved(client, from);
}

async function tryReserve(client) {
  const from = client.playerIndex;
  const state = client.state;
  const me = state.players[from];
  if (me.reserved.length >= state.config.maxReserved) return false;
  const boardCards = state.board.flat();
  const openDecks = [1, 2, 3].filter(tier => state.deckCounts[tier - 1] > 0);
  if (boardCards.length === 0 && openDecks.length === 0) return false;

  const entered = await action(client, { type: 'ENTER_RESERVE' });
  if (entered.error) return false;

  const fromDeck = openDecks.length > 0 && (boardCards.length === 0 || random() < 0.35);
  if (fromDeck) {
    for (const tier of shuffled(openDecks)) {
      const res = await action(client, { type: 'RESERVE_FROM_DECK', tier });
      if (!res.error) return true;
    }
  }
  for (const card of shuffled(boardCards)) {
    const res = await action(client, { type: 'RESERVE_CARD', cardId: card.id });
    if (!res.error) return true;
  }
  for (const tier of shuffled(openDecks)) {
    const res = await action(client, { type: 'RESERVE_FROM_DECK', tier });
    if (!res.error) return true;
  }
  return turnMoved(client, from);
}

async function chooseTile(client) {
  const tileIds = client.pendingTile || [];
  client.pendingTile = null;
  for (const tileId of shuffled(tileIds)) {
    const res = await action(client, { type: 'CHOOSE_TILE', tileId });
    if (!res.error) return true;
  }
  return false;
}

async function takeTurn(client) {
  if (client.pendingTile) return chooseTile(client);
  if (!client.state || client.state.phase !== 'PLAYING') return true;

  const from = client.playerIndex;

  // Occasionally start a selection and cancel it (neither is recorded).
  // A rejected SELECT_GEM also opens an empty TAKE_GEMS turn action, so the
  // cancel has to run either way or every later action is refused.
  if (random() < 0.12) {
    await action(client, { type: 'SELECT_GEM', color: Math.floor(random() * 5) });
    if (turnMoved(client, from)) return true;
    await action(client, { type: 'CANCEL_GEMS' });
  }

  const order = random() < 0.55
    ? ['buy', 'gems', 'reserve']
    : shuffled(['buy', 'gems', 'reserve']);
  for (const kind of order) {
    if (kind === 'buy' && await tryBuy(client)) return true;
    if (kind === 'gems') {
      const useDesktopFlow = random() < 0.5;
      const ok = useDesktopFlow ? await tryGemsDesktop(client) : await tryGemsMobile(client);
      if (ok) return true;
    }
    if (kind === 'reserve' && await tryReserve(client)) return true;
  }
  return turnMoved(client, from);
}

async function syncClients(clients, actor) {
  await waitFor(
    () => clients.every(client => client.state
      && client.state.phase === actor.state.phase
      && client.state.turnNumber === actor.state.turnNumber
      && client.state.currentPlayerIndex === actor.state.currentPlayerIndex),
    'all clients to see the same turn',
  );
}

async function playToEnd(seats, options = {}) {
  const cap = options.maxActions || 2000;
  const deadline = Date.now() + GAME_TIMEOUT_MS;
  let steps = 0;
  let stuck = 0;
  while (steps < cap && Date.now() < deadline) {
    const reference = seats.find(client => client.state);
    if (!reference || reference.state.phase !== 'PLAYING') break;
    const index = reference.state.currentPlayerIndex;
    const actor = seats[index];
    assert(actor, `no client for seat ${index}`);
    steps++;
    const acted = await takeTurn(actor);
    if (!acted) {
      // A seat with no legal action resigns, exactly like a stuck player:
      // 10 tokens + 3 reserved + nothing affordable is a real dead end and
      // these rules have no pass action.
      stuck++;
      const seat = actor.state.players[actor.playerIndex];
      log(`    seat ${actor.playerIndex} is stuck: gems ${seat.gems.join('/')}, `
        + `reserved ${seat.reserved.length}, cards ${seat.cards.length} — resigning`);
      actor.socket.emit('resign', { roomId: actor.roomId });
      await waitFor(() => actor.state.resignedPlayers.includes(actor.playerIndex)
        || actor.state.phase === 'GAME_OVER', 'resignation to register');
    }
    await syncClients(seats, actor);
  }
  await waitFor(() => seats.every(client => client.state && client.state.phase === 'GAME_OVER'),
    'the game to reach GAME_OVER');
  return { steps, stuck };
}

// ── lobby setup ────────────────────────────────────────────────────────────

async function enterLobby(clients) {
  for (const client of clients) {
    const res = await emitAck(client, 'enter_lobby');
    assert(res.action === 'lobby', `${client.username} could not enter the lobby: ${JSON.stringify(res)}`);
  }
}

async function readyUp(clients) {
  const started = clients.map(() => false);
  clients.forEach((client, i) => { client.socket.once('game_start', () => { started[i] = true; }); });
  for (const client of clients) client.socket.emit('lobby_ready');
  await waitFor(() => started.every(Boolean), 'game_start for every seat');
  await waitFor(() => clients.every(client => client.state && client.state.phase === 'PLAYING'), 'the first state');
}

/** Clients indexed by the seat the server gave them (team seating reorders players). */
function bySeat(clients) {
  const seats = [];
  for (const client of clients) {
    assert(Number.isInteger(client.playerIndex), `${client.username} never received game_start`);
    seats[client.playerIndex] = client;
  }
  assert(seats.length === clients.length && seats.every(Boolean), 'every seat is filled');
  return seats;
}

async function ok(client, event, payload) {
  const res = await emitAck(client, event, payload);
  assert(res.ok === true, `${event} rejected: ${JSON.stringify(res)}`);
}

// ── game configurations ────────────────────────────────────────────────────

/**
 * Each entry connects its own fresh accounts, configures the shared lobby, plays a random
 * game to GAME_OVER and returns { roomId, mode, ... } for the summary table.
 */
const CONFIGS = [
  {
    label: '2-player INDIVIDUAL',
    prefix: 'rnd-ind2-p',
    async setup() { /* the default lobby is already 2-player individual */ },
  },
  {
    label: '3-player ONE_V_TWO',
    prefix: 'rnd-1v2-p',
    async setup(clients) {
      await ok(clients[0], 'set_team_mode', { enabled: true });
      await ok(clients[0], 'set_unlimited_time', { enabled: true });
      // ONE_V_TWO seating order is [0,0], [1,0], [1,1].
      await ok(clients[0], 'select_team_seat', { teamId: 0, seatIndex: 0 });
      await ok(clients[1], 'select_team_seat', { teamId: 1, seatIndex: 0 });
      await ok(clients[2], 'select_team_seat', { teamId: 1, seatIndex: 1 });
    },
    verify(seats) {
      assert(seats[0].state.gameMode === 'ONE_V_TWO', 'gameMode is ONE_V_TWO');
      assert(seats[0].state.players.map(p => p.teamId).join(',') === '0,1,1', '1v2 team ids');
    },
  },
  {
    label: '4-player TEAM (ADJACENT)',
    prefix: 'rnd-2v2-p',
    async setup(clients) {
      await ok(clients[0], 'set_team_mode', { enabled: true });
      await ok(clients[0], 'set_team_layout', { layout: 'ADJACENT' });
      await ok(clients[0], 'set_unlimited_time', { enabled: true });
      // ADJACENT seating order is [0,0], [0,1], [1,0], [1,1].
      await ok(clients[0], 'select_team_seat', { teamId: 0, seatIndex: 0 });
      await ok(clients[1], 'select_team_seat', { teamId: 0, seatIndex: 1 });
      await ok(clients[2], 'select_team_seat', { teamId: 1, seatIndex: 0 });
      await ok(clients[3], 'select_team_seat', { teamId: 1, seatIndex: 1 });
    },
    verify(seats) {
      assert(seats[0].state.gameMode === 'TEAM', 'gameMode is TEAM');
      assert(seats[0].state.teamLayout === 'ADJACENT', 'ADJACENT layout');
      assert(seats[0].state.players.map(p => p.teamId).join(',') === '0,0,1,1', '2v2 adjacent team ids');
    },
  },
  {
    label: '3-player INDIVIDUAL',
    prefix: 'rnd-ind3-p',
    async setup(clients) {
      // Keep the clock on here so a `clock: true` recording is covered too.
      await ok(clients[0], 'set_unlimited_time', { enabled: false });
    },
  },
  {
    label: '2-player INDIVIDUAL ended by resignation',
    prefix: 'rnd-quit-p',
    async setup() { /* default lobby */ },
    // Play a handful of turns, then seat 0 resigns — in a 2-player game that ends it.
    async play(seats) {
      for (let i = 0; i < 6; i++) {
        const reference = seats[0].state;
        if (reference.phase !== 'PLAYING') break;
        const actor = seats[reference.currentPlayerIndex];
        await takeTurn(actor);
        await syncClients(seats, actor);
      }
      const quitter = seats[0];
      if (seats[0].state.phase === 'PLAYING') {
        quitter.socket.emit('resign', { roomId: quitter.roomId });
      }
      await waitFor(() => seats.every(client => client.state && client.state.phase === 'GAME_OVER'),
        'the game to end on resignation');
      return { steps: 6, stuck: 0, resignedBy: 0 };
    },
  },
];

const PLAYER_COUNT = { 'rnd-ind2-p': 2, 'rnd-1v2-p': 3, 'rnd-2v2-p': 4, 'rnd-ind3-p': 3, 'rnd-quit-p': 2 };

async function playConfig(config, ordinal) {
  const count = PLAYER_COUNT[config.prefix];
  log(`\n▶ game ${ordinal}: ${config.label}`);
  const clients = [];
  try {
    for (let seat = 0; seat < count; seat++) clients.push(await connect(freshName(config.prefix, seat)));
    await enterLobby(clients);
    await config.setup(clients);
    await readyUp(clients);

    const seats = bySeat(clients);
    config.verify?.(seats);
    const roomId = seats[0].roomId;
    const played = config.play ? await config.play(seats) : await playToEnd(seats);
    const final = seats.find(client => client.finalState)?.finalState
      ?? seats.find(client => client.state?.phase === 'GAME_OVER')?.state;
    assert(final, 'a GAME_OVER state was broadcast');

    log(`    room ${roomId} — ${played.steps} turns`
      + `${played.stuck ? `, ${played.stuck} stuck seat(s)` : ''}`
      + `, scores ${final.players.map(p => p.score).join('/')}`
      + `, resigned [${final.resignedPlayers.join(',')}]`
      + `, reason ${final.gameResult?.reason ?? '—'}`);

    return {
      roomId,
      label: config.label,
      mode: final.gameMode,
      layout: final.teamLayout ?? null,
      n: final.numPlayers,
      players: seats.map(client => client.username),
      scores: final.players.map(p => p.score),
      cards: final.players.map(p => p.cards.length),
      resigned: [...final.resignedPlayers],
      gameResult: final.gameResult ?? null,
      turns: played.steps,
    };
  } finally {
    disconnectAll(clients);
    // Give the server a moment to drain the lobby before the next configuration.
    await sleep(250);
  }
}

// ── main ───────────────────────────────────────────────────────────────────

await waitForServer();
const status = await api('/api/replays/status');
log(`server ${BASE_URL} — replay store: github=${status.body?.github}, memory=${status.body?.memory}`);

const requested = Number(argValue('--games', CONFIGS.length));
const plan = [];
for (let i = 0; i < requested; i++) plan.push(CONFIGS[i % CONFIGS.length]);

const results = [];
for (const [i, config] of plan.entries()) {
  results.push(await playConfig(config, i + 1));
}

// ── report ─────────────────────────────────────────────────────────────────
const list = await api('/api/replays?limit=200');
assert(list.status === 200, `GET /api/replays failed with ${list.status}`);
const listed = new Map((list.body?.games ?? []).map(game => [game.id, game]));

console.log(`\n${results.length} game(s) played; /api/replays reports `
  + `${list.body?.total} total from ${list.body?.source}\n`);
console.log('replay id                       mode        n  turns  winners        players');
console.log('─'.repeat(100));
let missing = 0;
for (const result of results) {
  const entry = listed.get(result.roomId);
  if (!entry) {
    missing++;
    console.log(`${result.roomId.padEnd(31)} MISSING FROM /api/replays (${result.label})`);
    continue;
  }
  const winners = entry.winningTeamIds
    ? `team ${JSON.stringify(entry.winningTeamIds)}`
    : `seats ${JSON.stringify(entry.winners)}`;
  console.log(`${entry.id.padEnd(31)} ${String(entry.mode).padEnd(11)} ${entry.n}  ${String(entry.turns).padStart(5)}  `
    + `${winners.padEnd(14)} ${entry.players.join(', ')}`);
}

console.log('\nids:');
for (const result of results) console.log(result.roomId);

if (missing) {
  console.error(`\n${missing} recorded game(s) never reached /api/replays`);
  process.exit(1);
}
process.exit(0);
