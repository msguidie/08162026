#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════
// Human driver for the deployment-worker end-to-end test.
//
// Boots N socket.io clients that behave like impatient humans (random legal
// moves), fills the remaining seats with server-side bots (`lobby_add_ai`,
// answered by splendor_ai/worker over the AI bridge) and plays whole games to
// GAME_OVER. The logic is the one `server/test/ai.e2e.js` and
// `scripts/dev/playRandomGames.mjs` use; this file only adds a CLI and a
// machine-readable summary so pytest can assert on it.
//
//   node humanDriver.mjs --url http://127.0.0.1:10000 --scenario ind2 --games 3
//
// Scenarios (bots are always the seats a human does not take):
//   ind2      1 human + 1 bot, INDIVIDUAL
//   ind3      1 human + 2 bots, INDIVIDUAL
//   ovt-solo  2 humans + 1 bot, ONE_V_TWO, bot on the solo side
//   ovt-duo   2 humans + 1 bot, ONE_V_TWO, bot in the duo
//   team-2v2  2 humans + 2 bots, TEAM, both bots on team 1
//
// stdout carries `##GAME##{json}` per game and `##RESULT##{json}` at the end;
// everything else is human-readable progress.
// ═══════════════════════════════════════════════════════════

import { io } from 'socket.io-client';

// ── CLI ────────────────────────────────────────────────────

function parseArgs(argv) {
  const out = { url: 'http://127.0.0.1:10000', scenario: 'ind2', games: 1, tag: 'e2e', seed: 1 };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    const name = key.slice(2);
    const value = argv[i + 1];
    if (name === 'games' || name === 'seed') out[name] = Number.parseInt(value, 10);
    else out[name] = value;
    i += 1;
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));

const SCENARIOS = {
  ind2: { humans: 1, bots: 1, team: null },
  ind3: { humans: 1, bots: 2, team: null },
  'ovt-solo': { humans: 2, bots: 1, team: 'ONE_V_TWO', botSeats: [[0, 0]], humanSeats: [[1, 0], [1, 1]] },
  'ovt-duo': { humans: 2, bots: 1, team: 'ONE_V_TWO', botSeats: [[1, 1]], humanSeats: [[0, 0], [1, 0]] },
  'team-2v2': { humans: 2, bots: 2, team: 'TWO_V_TWO', botSeats: [[1, 0], [1, 1]], humanSeats: [[0, 0], [0, 1]] },
};

const scenario = SCENARIOS[args.scenario];
if (!scenario) {
  console.error(`unknown scenario ${args.scenario}; known: ${Object.keys(SCENARIOS).join(', ')}`);
  process.exit(2);
}

const ACK_TIMEOUT_MS = 15000;
const SYNC_TIMEOUT_MS = 20000;
const BOT_TURN_TIMEOUT_MS = 30000;
const GAME_TIMEOUT_MS = 300000;

// ── deterministic pseudo-randomness (mulberry-ish, as in ai.e2e.js) ──

let seed = (args.seed || 1) * 0x9e3779b9;
function random() {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function shuffled(list) {
  const copy = [...list];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function waitFor(condition, message, timeout = SYNC_TIMEOUT_MS) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await condition()) return true;
    await sleep(3);
  }
  throw new Error(`timed out waiting for ${message}`);
}

async function api(path, options) {
  const response = await fetch(`${args.url}${path}`, options);
  const body = await response.json().catch(() => null);
  return { status: response.status, body };
}

// ── client ─────────────────────────────────────────────────

function emitAck(client, event, payload) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`ack timeout: ${event}`)), ACK_TIMEOUT_MS);
    const done = response => { clearTimeout(timer); resolve(response || {}); };
    if (payload === undefined) client.socket.emit(event, done);
    else client.socket.emit(event, payload, done);
  });
}

const action = (client, payload) => emitAck(client, 'game_action', { roomId: client.roomId, action: payload });

async function connect(username) {
  const socket = io(args.url, { transports: ['websocket'], forceNew: true, reconnection: false });
  const client = { username, socket, playerIndex: null, roomId: null, state: null, lobby: null, pendingTile: null };
  socket.on('lobby_update', lobby => { client.lobby = lobby; });
  socket.on('game_start', data => {
    client.roomId = data.roomId;
    client.playerIndex = data.playerIndex;
    client.state = data.gameState;
  });
  socket.on('game_state_update', state => { client.state = state; });
  socket.on('tile_choice_required', ({ tileIds }) => { client.pendingTile = tileIds; });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`connect timeout for ${username}`)), ACK_TIMEOUT_MS);
    socket.on('connect', () => { clearTimeout(timer); resolve(); });
    socket.on('connect_error', err => { clearTimeout(timer); reject(err); });
  });
  await api('/api/accounts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username }),
  });
  const login = await emitAck(client, 'login', { username });
  if (!login.success) throw new Error(`login failed for ${username}: ${JSON.stringify(login)}`);
  return client;
}

// ── human turn policy: a random legal move ─────────────────

function turnMoved(client, fromIndex) {
  return !client.state
    || client.state.phase !== 'PLAYING'
    || client.state.currentPlayerIndex !== fromIndex
    || !!client.pendingTile;
}

function gemCandidates() {
  const out = [];
  for (let a = 0; a < 5; a += 1) {
    out.push([a, a]);
    out.push([a]);
    for (let b = 0; b < 5; b += 1) {
      if (b === a) continue;
      out.push([a, b]);
      for (let c = 0; c < 5; c += 1) {
        if (c === a || c === b) continue;
        out.push([a, b, c]);
      }
    }
  }
  return out;
}

async function tryBuy(client) {
  const from = client.state.currentPlayerIndex;
  const me = client.state.players[client.playerIndex];
  const candidates = [
    ...shuffled(me.reserved.filter(card => card.id >= 0)).map(card => ({ cardId: card.id, source: 'reserved' })),
    ...shuffled(client.state.board.flat()).map(card => ({ cardId: card.id, source: 'board' })),
  ];
  for (const candidate of candidates) {
    const res = await action(client, { type: 'BUY_CARD', ...candidate });
    if (!res.error) return true;
  }
  return turnMoved(client, from);
}

async function tryGems(client) {
  const from = client.state.currentPlayerIndex;
  for (const colors of shuffled(gemCandidates())) {
    const res = await action(client, { type: 'TAKE_GEMS_CONFIRMED', colors });
    if (!res.error) return true;
  }
  return turnMoved(client, from);
}

async function tryReserve(client) {
  const from = client.state.currentPlayerIndex;
  const state = client.state;
  const me = state.players[client.playerIndex];
  if (me.reserved.length >= state.config.maxReserved) return false;
  const boardCards = state.board.flat();
  const openDecks = [1, 2, 3].filter(tier => state.deckCounts[tier - 1] > 0);
  if (boardCards.length === 0 && openDecks.length === 0) return false;
  const entered = await action(client, { type: 'ENTER_RESERVE' });
  if (entered.error) return false;
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
  const from = client.state.currentPlayerIndex;
  for (const kind of shuffled(['buy', 'gems', 'reserve'])) {
    if (kind === 'buy' && await tryBuy(client)) return true;
    if (kind === 'gems' && await tryGems(client)) return true;
    if (kind === 'reserve' && await tryReserve(client)) return true;
  }
  return turnMoved(client, from);
}

async function syncHumans(clients, actor) {
  await waitFor(() => clients.every(client => client.state
    && client.state.phase === actor.state.phase
    && client.state.turnNumber === actor.state.turnNumber
    && client.state.currentPlayerIndex === actor.state.currentPlayerIndex),
  'all human clients to see the same turn');
}

// ── one game ───────────────────────────────────────────────

async function playGame(clients, botNames) {
  const reference = clients[0];
  const deadline = Date.now() + GAME_TIMEOUT_MS;
  const botSeats = reference.state.players
    .map((player, index) => (botNames.includes(player.username) ? index : -1))
    .filter(index => index >= 0);
  let humanTurns = 0;
  let botTurns = 0;
  let humanResigns = 0;
  let slowestBotMs = 0;
  const botLatencies = [];

  while (Date.now() < deadline) {
    if (!reference.state || reference.state.phase !== 'PLAYING') break;
    const seat = reference.state.currentPlayerIndex;
    const actor = clients.find(client => client.playerIndex === seat);

    if (!actor) {
      const turnBefore = reference.state.turnNumber;
      const started = Date.now();
      await waitFor(() => !reference.state
        || reference.state.phase !== 'PLAYING'
        || reference.state.turnNumber !== turnBefore
        || reference.state.currentPlayerIndex !== seat,
      `bot seat ${seat} to play`, BOT_TURN_TIMEOUT_MS);
      const elapsed = Date.now() - started;
      slowestBotMs = Math.max(slowestBotMs, elapsed);
      botLatencies.push(elapsed);
      botTurns += 1;
      continue;
    }

    humanTurns += 1;
    const acted = await takeTurn(actor);
    if (!acted) {
      humanResigns += 1;
      actor.socket.emit('resign', { roomId: actor.roomId });
      await waitFor(() => actor.state.resignedPlayers.includes(actor.playerIndex)
        || actor.state.phase === 'GAME_OVER', 'the resignation to register');
    }
    await syncHumans(clients, actor);
  }

  await waitFor(() => clients.every(client => client.state && client.state.phase === 'GAME_OVER'),
    'the game to reach GAME_OVER', 30000);

  return {
    roomId: reference.roomId,
    scenario: args.scenario,
    mode: reference.state.gameMode,
    numPlayers: reference.state.numPlayers,
    botSeats,
    botNames,
    humanTurns,
    botTurns,
    humanResigns,
    slowestBotMs,
    botLatencies,
    turnNumber: reference.state.turnNumber,
    scores: reference.state.players.map(player => player.score),
    resigned: reference.state.resignedPlayers,
    phase: reference.state.phase,
  };
}

// ── lobby setup ────────────────────────────────────────────

async function setupAndPlay(index) {
  const clients = [];
  for (let i = 0; i < scenario.humans; i += 1) {
    clients.push(await connect(`w-${args.tag}-${args.scenario}-${index}-${i}`));
  }
  for (const client of clients) {
    const res = await emitAck(client, 'enter_lobby');
    if (res.action !== 'lobby') throw new Error(`${client.username} could not enter the lobby: ${JSON.stringify(res)}`);
    client.lobby = res.lobbyState;
  }
  if (!clients[0].lobby.aiAvailable) throw new Error('the server reports no AI worker');

  const botNames = [];
  for (let i = 0; i < scenario.bots; i += 1) {
    const res = await emitAck(clients[i % clients.length], 'lobby_add_ai', {});
    if (!res.ok) throw new Error(`lobby_add_ai failed: ${JSON.stringify(res)}`);
    botNames.push(res.username);
  }
  await waitFor(() => clients[0].lobby?.players.length === scenario.humans + scenario.bots,
    'every seat to be in the lobby');

  if (scenario.team) {
    const teamRes = await emitAck(clients[0], 'set_team_mode', { enabled: true });
    if (!teamRes.ok) throw new Error(`set_team_mode failed: ${JSON.stringify(teamRes)}`);
    if (scenario.team === 'TWO_V_TWO') {
      await emitAck(clients[0], 'set_team_layout', { layout: 'ADJACENT' });
    }
  }
  if (scenario.humans + scenario.bots >= 3) {
    const timeRes = await emitAck(clients[0], 'set_unlimited_time', { enabled: true });
    if (!timeRes.ok) throw new Error(`set_unlimited_time failed: ${JSON.stringify(timeRes)}`);
  }
  if (scenario.team) {
    for (const [i, client] of clients.entries()) {
      const [teamId, seatIndex] = scenario.humanSeats[i];
      const res = await emitAck(client, 'select_team_seat', { teamId, seatIndex });
      if (!res.ok) throw new Error(`select_team_seat failed for ${client.username}: ${JSON.stringify(res)}`);
    }
    for (const [i, botName] of botNames.entries()) {
      const [teamId, seatIndex] = scenario.botSeats[i];
      const res = await emitAck(clients[0], 'select_team_seat', { teamId, seatIndex, forUsername: botName });
      if (!res.ok) throw new Error(`select_team_seat failed for ${botName}: ${JSON.stringify(res)}`);
    }
  }

  const started = clients.map(() => false);
  clients.forEach((client, i) => { client.socket.once('game_start', () => { started[i] = true; }); });
  for (const client of clients) client.socket.emit('lobby_ready');
  await waitFor(() => started.every(Boolean), 'game_start for every human seat');
  await waitFor(() => clients.every(client => client.state && client.state.phase === 'PLAYING'), 'the first state');

  const summary = await playGame(clients, botNames);
  for (const client of clients) client.socket.disconnect();
  await sleep(120);
  return summary;
}

// ── main ───────────────────────────────────────────────────

async function main() {
  const status = await api('/api/ai/status');
  if (!status.body?.available) throw new Error(`no worker registered: ${JSON.stringify(status.body)}`);
  console.log(`[driver] worker "${status.body.name}" is live; scenario ${args.scenario} x ${args.games}`);

  const games = [];
  for (let i = 0; i < args.games; i += 1) {
    const summary = await setupAndPlay(i);
    games.push(summary);
    console.log(`##GAME##${JSON.stringify(summary)}`);
    console.log(`[driver] ${args.scenario} game ${i + 1}/${args.games}: `
      + `${summary.humanTurns} human turns, ${summary.botTurns} bot turns, `
      + `slowest bot reply ${summary.slowestBotMs} ms, scores ${summary.scores.join('/')}`);
  }
  console.log(`##RESULT##${JSON.stringify({ scenario: args.scenario, games })}`);
}

main().then(() => process.exit(0)).catch(err => {
  console.error(`[driver] FAILED: ${err?.stack || err}`);
  process.exit(1);
});
