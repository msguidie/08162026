// End-to-end: boots the real server as a CHILD PROCESS with AI_WORKER_SECRET
// set, connects scripts/dev/mockAiWorker.mjs to it, and plays real games over
// socket.io where some seats are bots driven by the server.
//
// A child process is used (rather than require('../index.js') like
// replay.e2e.js) because AI_WORKER_SECRET has to differ per server: one
// server with AI enabled, one without.

const net = require('net');
const path = require('path');
const { spawn } = require('child_process');
const { io } = require('socket.io-client');
const { suite, test, assert, assertEqual } = require('./harness');

const ROOT = path.resolve(__dirname, '../..');
const SERVER_ENTRY = path.join(ROOT, 'server', 'index.js');
const WORKER_ENTRY = path.join(ROOT, 'scripts', 'dev', 'mockAiWorker.mjs');

const SECRET = 'test';
const ACK_TIMEOUT_MS = 8000;
const SYNC_TIMEOUT_MS = 8000;
const AI_DEADLINE_MS = 15000;        // docs/AI_BRIDGE.md §1 budget
const BOT_TURN_TIMEOUT_MS = 20000;
const GAME_TIMEOUT_MS = 180000;

const children = [];

// ── deterministic pseudo-random policy (same generator as replay.e2e.js) ──
let seed = 0x1234abcd;
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

// ── infrastructure ──

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.on('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitFor(condition, message, timeout = SYNC_TIMEOUT_MS) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await condition()) return true;
    await sleep(3);
  }
  throw new Error(`timed out waiting for ${message}`);
}

function track(child, label) {
  const lines = [];
  child.log = lines;
  const record = chunk => {
    for (const line of String(chunk).split('\n')) {
      if (line.trim()) lines.push(line);
    }
    while (lines.length > 200) lines.shift();
  };
  child.stdout?.on('data', record);
  child.stderr?.on('data', record);
  child.on('exit', code => { if (code) lines.push(`${label} exited with code ${code}`); });
  children.push(child);
  return child;
}

function stop(child) {
  if (child && !child.killed) child.kill('SIGTERM');
}

function stopAll() {
  for (const child of children) stop(child);
  children.length = 0;
}
process.on('exit', stopAll);

async function api(baseUrl, path_, options) {
  const response = await fetch(`${baseUrl}${path_}`, options);
  const body = await response.json().catch(() => null);
  return { status: response.status, body };
}

async function startServer({ secret, moveDelayMs = 10 } = {}) {
  const port = await freePort();
  const env = { ...process.env, PORT: String(port), AI_MOVE_DELAY_MS: String(moveDelayMs) };
  delete env.REPLAY_GITHUB_TOKEN;
  delete env.REPLAY_GITHUB_REPO;
  delete env.RENDER_EXTERNAL_URL;
  if (secret) env.AI_WORKER_SECRET = secret;
  else delete env.AI_WORKER_SECRET;

  const child = track(spawn(process.execPath, [SERVER_ENTRY], { env, stdio: ['ignore', 'pipe', 'pipe'] }), 'server');
  const baseUrl = `http://127.0.0.1:${port}`;
  await waitFor(async () => {
    if (child.exitCode !== null) throw new Error(`server exited: ${child.log.join('\n')}`);
    try {
      return (await api(baseUrl, '/health')).status === 200;
    } catch (err) {
      return false;
    }
  }, 'the server to listen', 15000);
  return { baseUrl, port, child };
}

async function startWorker(baseUrl, { secret = SECRET, name = 'mock-worker' } = {}) {
  const child = track(spawn(process.execPath, [WORKER_ENTRY], {
    env: { ...process.env, SERVER_URL: baseUrl, AI_WORKER_SECRET: secret, MOCK_AI_NAME: name },
    stdio: ['ignore', 'pipe', 'pipe'],
  }), 'worker');
  await waitFor(async () => (await api(baseUrl, '/api/ai/status')).body?.available === true,
    'the mock worker to register', 15000);
  return child;
}

// ── socket client, mirroring src/store/gameStore.ts ──

function emitAck(client, event, payload) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`ack timeout: ${event}`)), ACK_TIMEOUT_MS);
    const done = response => { clearTimeout(timer); resolve(response || {}); };
    if (payload === undefined) client.socket.emit(event, done);
    else client.socket.emit(event, payload, done);
  });
}

function action(client, payload) {
  return emitAck(client, 'game_action', { roomId: client.roomId, action: payload });
}

async function connect(baseUrl, username) {
  const socket = io(baseUrl, { transports: ['websocket'], forceNew: true, reconnection: false });
  const client = {
    username, socket, baseUrl,
    playerIndex: null, roomId: null, state: null, lobby: null,
    lastResult: null, pendingTile: null, finalState: null,
  };
  socket.on('lobby_update', lobbyState => { client.lobby = lobbyState; });
  socket.on('game_start', data => {
    client.roomId = data.roomId;
    client.playerIndex = data.playerIndex;
    client.state = data.gameState;
  });
  socket.on('game_state_update', state => {
    client.state = state;
    if (state.phase === 'GAME_OVER') client.finalState = state;
  });
  socket.on('action_result', result => { client.lastResult = result; });
  socket.on('tile_choice_required', ({ tileIds }) => { client.pendingTile = tileIds; });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`connect timeout for ${username}`)), ACK_TIMEOUT_MS);
    socket.on('connect', () => { clearTimeout(timer); resolve(); });
    socket.on('connect_error', err => { clearTimeout(timer); reject(err); });
  });
  await api(baseUrl, '/api/accounts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username }),
  });
  const login = await emitAck(client, 'login', { username });
  assert(login.success, `login failed for ${username}: ${JSON.stringify(login)}`);
  return client;
}

function disconnectAll(clients) {
  for (const client of clients) client.socket.disconnect();
}

// ── turn driving (human seats play random legal moves) ──

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

/**
 * Plays until GAME_OVER. Human seats play random legal moves; bot seats are
 * driven by the server and only waited for.
 */
async function playMixedGame(clients, options = {}) {
  const reference = clients[0];
  const deadline = Date.now() + (options.timeout || GAME_TIMEOUT_MS);
  let humanTurns = 0;
  let botTurns = 0;
  let stuck = 0;
  let slowestBotMs = 0;

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
      assert(elapsed < AI_DEADLINE_MS, `bot seat ${seat} took ${elapsed} ms (deadline ${AI_DEADLINE_MS} ms)`);
      botTurns++;
      if (options.onBotTurn) await options.onBotTurn(botTurns);
      continue;
    }

    humanTurns++;
    const acted = await takeTurn(actor);
    if (!acted) {
      // A seat with no legal action resigns, exactly like in replay.e2e.js.
      stuck++;
      actor.socket.emit('resign', { roomId: actor.roomId });
      await waitFor(() => actor.state.resignedPlayers.includes(actor.playerIndex)
        || actor.state.phase === 'GAME_OVER', 'the resignation to register');
    }
    await syncHumans(clients, actor);
  }

  await waitFor(() => clients.every(client => client.state && client.state.phase === 'GAME_OVER'),
    'the game to reach GAME_OVER');
  return { humanTurns, botTurns, stuck, slowestBotMs };
}

// ── lobby helpers ──

async function enterLobby(clients) {
  for (const client of clients) {
    const res = await emitAck(client, 'enter_lobby');
    assertEqual(res.action, 'lobby', `${client.username} could not enter the lobby`);
    client.lobby = res.lobbyState;
  }
}

async function addBot(client, expectedName) {
  const res = await emitAck(client, 'lobby_add_ai', {});
  assertEqual(res.ok, true, `lobby_add_ai failed: ${JSON.stringify(res)}`);
  if (expectedName) assertEqual(res.username, expectedName);
  await waitFor(() => client.lobby?.players.some(p => p.username === res.username), 'the bot to appear in the lobby');
  return res.username;
}

async function readyUp(clients) {
  const started = clients.map(() => false);
  clients.forEach((client, i) => { client.socket.once('game_start', () => { started[i] = true; }); });
  for (const client of clients) client.socket.emit('lobby_ready');
  await waitFor(() => started.every(Boolean), 'game_start for every human seat');
  await waitFor(() => clients.every(client => client.state && client.state.phase === 'PLAYING'), 'the first state');
}

async function replayFor(baseUrl, roomId) {
  const raw = await api(baseUrl, `/api/replays/${roomId}/raw`);
  assertEqual(raw.status, 200, `no replay stored for ${roomId}`);
  const list = await api(baseUrl, '/api/replays?limit=200');
  const entry = list.body.games.find(game => game.id === roomId);
  assert(entry, `${roomId} missing from /api/replays`);
  return { raw: raw.body, entry };
}

function seatNames(state) {
  return state.players.map(player => player.username);
}

// ── suites ──

async function run() {
  suite('ai e2e — server with AI_WORKER_SECRET');

  const aiServer = await startServer({ secret: SECRET });
  const { baseUrl } = aiServer;
  let worker = await startWorker(baseUrl);

  await test('GET /api/ai/status reports the registered worker', async () => {
    const status = await api(baseUrl, '/api/ai/status');
    assertEqual(status.status, 200);
    assertEqual(status.body.enabled, true);
    assertEqual(status.body.available, true);
    assertEqual(status.body.name, 'mock-worker');
    assertEqual(status.body.modes, ['INDIVIDUAL', 'ONE_V_TWO', 'TEAM']);
    const root = await api(baseUrl, '/');
    assertEqual(root.body.status, 'running', 'existing routes are untouched');
  });

  await test('a worker with the wrong secret is refused', async () => {
    const client = await connect(baseUrl, 'ai-e2e-imposter');
    const res = await emitAck(client, 'ai_worker_register', { secret: 'wrong', name: 'imposter' });
    assertEqual(res, { error: 'Invalid worker secret' });
    assertEqual((await api(baseUrl, '/api/ai/status')).body.name, 'mock-worker', 'the real worker still holds');
    client.socket.disconnect();
  });

  suite('ai e2e — 1 human + 1 bot (INDIVIDUAL)');

  await test('the lobby exposes aiAvailable and an always-ready bot', async () => {
    const human = await connect(baseUrl, 'ai-e2e-h1');
    await enterLobby([human]);
    assertEqual(human.lobby.aiAvailable, true, 'aiAvailable');
    assertEqual(human.lobby.players[0].isAI, false, 'humans are not bots');

    const botName = await addBot(human, 'Bot Alpha');
    const bot = human.lobby.players.find(player => player.username === botName);
    assertEqual(bot.isAI, true);
    assertEqual(bot.ready, true, 'bots are always ready');
    assertEqual(bot.wantsFirst, false, 'bots never volunteer to go first');

    // lobby_ready never reaches a bot, and a second add takes the next name.
    const second = await addBot(human, 'Bot Beta');
    assertEqual(human.lobby.players.length, 3);
    const removed = await emitAck(human, 'lobby_remove_ai', { username: second });
    assertEqual(removed.ok, true);
    await waitFor(() => human.lobby.players.length === 2, 'the bot to be removed');
    assertEqual((await emitAck(human, 'lobby_remove_ai', { username: 'nobody' })).error,
      'That AI player is not in this lobby');

    human.socket.disconnect();
    await sleep(80);
    const after = await api(baseUrl, '/');
    assertEqual(after.body.lobby, 0, 'the bot leaves with the last human');
  });

  let firstBotRoom = null;
  await test('plays a full 2-player game against the bot and records ai: true', async () => {
    const human = await connect(baseUrl, 'ai-e2e-duel');
    await enterLobby([human]);
    const botName = await addBot(human, 'Bot Alpha');
    await readyUp([human]);

    assertEqual(human.state.numPlayers, 2);
    assertEqual(seatNames(human.state), ['ai-e2e-duel', botName], 'seat order follows the lobby');
    const played = await playMixedGame([human]);
    assert(played.botTurns > 0, 'the bot actually played');
    firstBotRoom = human.roomId;

    const { raw, entry } = await replayFor(baseUrl, human.roomId);
    assertEqual(raw.players.map(p => p.u), ['ai-e2e-duel', botName]);
    assertEqual(raw.players.map(p => p.ai), [false, true], 'the bot seat is recorded with ai: true');
    assertEqual(entry.ai, [false, true], 'and in the replay index');
    assert(raw.actions.length > 0, 'actions recorded');

    const accounts = (await api(baseUrl, '/api/accounts')).body;
    const botAccount = accounts.find(account => account.username === botName);
    assert(botAccount, 'the bot account was auto-created');
    assert(botAccount.gamesPlayed >= 1, 'bots are rated like humans');

    console.log(`        ${played.humanTurns} human turns, ${played.botTurns} bot turns, `
      + `slowest bot reply ${played.slowestBotMs} ms`);
    human.socket.disconnect();
  });

  suite('ai e2e — 2 humans + 1 bot (ONE_V_TWO)');

  for (const variant of [
    { name: 'solo', botSeat: [0, 0], humanSeats: [[1, 0], [1, 1]] },
    { name: 'duo', botSeat: [1, 1], humanSeats: [[0, 0], [1, 0]] },
  ]) {
    await test(`plays a 1v2 game with the bot as the ${variant.name} side`, async () => {
      const clients = [
        await connect(baseUrl, `ai-e2e-1v2-${variant.name}-a`),
        await connect(baseUrl, `ai-e2e-1v2-${variant.name}-b`),
      ];
      await enterLobby(clients);
      const botName = await addBot(clients[0], 'Bot Alpha');
      assertEqual((await emitAck(clients[0], 'set_team_mode', { enabled: true })).ok, true);
      assertEqual(clients[0].lobby.teamFormat, 'ONE_V_TWO', 'the bot counts toward the player count');
      assertEqual((await emitAck(clients[0], 'set_unlimited_time', { enabled: true })).ok, true);

      assertEqual((await emitAck(clients[0], 'select_team_seat',
        { teamId: variant.humanSeats[0][0], seatIndex: variant.humanSeats[0][1] })).ok, true);
      assertEqual((await emitAck(clients[1], 'select_team_seat',
        { teamId: variant.humanSeats[1][0], seatIndex: variant.humanSeats[1][1] })).ok, true);
      // Bots cannot click: a human seats them with forUsername.
      assertEqual((await emitAck(clients[0], 'select_team_seat',
        { teamId: variant.botSeat[0], seatIndex: variant.botSeat[1], forUsername: botName })).ok, true);
      assertEqual((await emitAck(clients[0], 'select_team_seat',
        { teamId: 0, seatIndex: 0, forUsername: clients[1].username })).error,
      'Only AI players can be seated by someone else', 'forUsername is bots-only');

      await readyUp(clients);
      assertEqual(clients[0].state.gameMode, 'ONE_V_TWO');
      const botIndex = seatNames(clients[0].state).indexOf(botName);
      assertEqual(botIndex, variant.name === 'solo' ? 0 : 2, 'the bot took the requested seat');
      assertEqual(clients[0].state.currentPlayerIndex, 0, 'the solo seat opens');

      const played = await playMixedGame(clients);
      assert(played.botTurns > 0, 'the bot played');
      const { raw } = await replayFor(baseUrl, clients[0].roomId);
      assertEqual(raw.mode, 'ONE_V_TWO');
      assertEqual(raw.players[botIndex].ai, true);
      assertEqual(raw.players.filter(p => p.ai).length, 1);
      console.log(`        ${played.humanTurns} human turns, ${played.botTurns} bot turns`);
      disconnectAll(clients);
      await sleep(60);
    });
  }

  suite('ai e2e — 2 humans + 2 bots (TWO_V_TWO)');

  for (const variant of [
    { name: 'same team', seats: { bots: [[1, 0], [1, 1]], humans: [[0, 0], [0, 1]] } },
    { name: 'opposing teams', seats: { bots: [[0, 1], [1, 1]], humans: [[0, 0], [1, 0]] } },
  ]) {
    await test(`plays a 2v2 game with the bots on the ${variant.name}`, async () => {
      const tag = variant.name.replace(/\s+/g, '-');
      const clients = [
        await connect(baseUrl, `ai-e2e-2v2-${tag}-a`),
        await connect(baseUrl, `ai-e2e-2v2-${tag}-b`),
      ];
      await enterLobby(clients);
      const botNames = [await addBot(clients[0], 'Bot Alpha'), await addBot(clients[1], 'Bot Beta')];
      assertEqual((await emitAck(clients[0], 'set_team_mode', { enabled: true })).ok, true);
      assertEqual(clients[0].lobby.teamFormat, 'TWO_V_TWO');
      assertEqual((await emitAck(clients[0], 'set_unlimited_time', { enabled: true })).ok, true);
      assertEqual((await emitAck(clients[0], 'lobby_add_ai', {})).error, 'This lobby is full',
        'team lobbies cap at the required seat count');

      for (const [index, client] of clients.entries()) {
        const [teamId, seatIndex] = variant.seats.humans[index];
        assertEqual((await emitAck(client, 'select_team_seat', { teamId, seatIndex })).ok, true);
      }
      for (const [index, botName] of botNames.entries()) {
        const [teamId, seatIndex] = variant.seats.bots[index];
        assertEqual((await emitAck(clients[0], 'select_team_seat', { teamId, seatIndex, forUsername: botName })).ok, true);
      }

      await readyUp(clients);
      assertEqual(clients[0].state.gameMode, 'TEAM');
      assertEqual(clients[0].state.numPlayers, 4);
      const names = seatNames(clients[0].state);
      const botTeams = botNames.map(name => clients[0].state.players[names.indexOf(name)].teamId);
      assertEqual(botTeams, variant.seats.bots.map(([teamId]) => teamId), 'bot teams');

      const played = await playMixedGame(clients);
      assert(played.botTurns > 0, 'the bots played');
      const { raw } = await replayFor(baseUrl, clients[0].roomId);
      assertEqual(raw.mode, 'TEAM');
      assertEqual(raw.players.filter(p => p.ai).length, 2, 'two bot seats recorded');
      assertEqual(raw.players.map(p => p.ai), names.map(name => botNames.includes(name)));
      console.log(`        ${played.humanTurns} human turns, ${played.botTurns} bot turns`);
      disconnectAll(clients);
      await sleep(60);
    });
  }

  suite('ai e2e — worker disconnect');

  await test('the fallback finishes the game after the worker goes away', async () => {
    const human = await connect(baseUrl, 'ai-e2e-fallback');
    await enterLobby([human]);
    const botName = await addBot(human, 'Bot Alpha');
    await readyUp([human]);

    let killedAfter = 0;
    const played = await playMixedGame([human], {
      onBotTurn: async count => {
        if (count !== 3 || killedAfter) return;
        killedAfter = count;
        stop(worker);
        worker = null;
        await waitFor(async () => (await api(baseUrl, '/api/ai/status')).body.available === false,
          'the server to notice the worker is gone', 10000);
      },
    });

    assertEqual(killedAfter, 3, 'the worker was killed mid-game');
    assert(played.botTurns > 3, 'the bot kept playing without a worker');
    assertEqual((await api(baseUrl, '/api/ai/status')).body, { enabled: true, available: false });
    const { raw } = await replayFor(baseUrl, human.roomId);
    assertEqual(raw.players.map(p => p.ai), [false, true]);
    assertEqual(raw.result.scores.length, 2);
    console.log(`        ${played.botTurns} bot turns, ${played.botTurns - killedAfter} of them on the fallback`);
    human.socket.disconnect();
    await sleep(60);
  });

  await test('a lobby cannot add a bot while no worker is connected', async () => {
    const human = await connect(baseUrl, 'ai-e2e-noworker');
    await enterLobby([human]);
    assertEqual(human.lobby.aiAvailable, false, 'aiAvailable follows the worker');
    assertEqual((await emitAck(human, 'lobby_add_ai', {})).error, 'No AI worker is connected right now');
    human.socket.disconnect();
    await sleep(60);
    // …and a fresh worker restores the feature.
    worker = await startWorker(baseUrl, { name: 'mock-worker-2' });
    assertEqual((await api(baseUrl, '/api/ai/status')).body.name, 'mock-worker-2');
  });

  suite('ai e2e — server without AI_WORKER_SECRET');

  await test('nothing AI-related is exposed', async () => {
    const plain = await startServer({ secret: null });
    const status = await api(plain.baseUrl, '/api/ai/status');
    assertEqual(status.status, 200);
    assertEqual(status.body, { enabled: false, available: false });

    const human = await connect(plain.baseUrl, 'plain-human');
    await enterLobby([human]);
    assertEqual(human.lobby.aiAvailable, false, 'aiAvailable is false');
    assertEqual((await emitAck(human, 'lobby_add_ai', {})).error, 'AI is not enabled on this server');
    assertEqual((await emitAck(human, 'ai_worker_register', { secret: SECRET, name: 'w' })).error,
      'AI is not enabled on this server');
    assertEqual((await api(plain.baseUrl, '/api/ai/status')).body, { enabled: false, available: false });

    const accounts = (await api(plain.baseUrl, '/api/accounts')).body;
    assertEqual(accounts.filter(account => account.username.startsWith('Bot ')), [], 'no bot accounts');
    assertEqual(human.lobby.players.length, 1, 'the lobby is unchanged');

    // A normal two-human lobby still behaves exactly as before.
    const other = await connect(plain.baseUrl, 'plain-human-2');
    await enterLobby([other]);
    await readyUp([human, other]);
    assertEqual(human.state.phase, 'PLAYING');
    assertEqual(human.state.numPlayers, 2);
    disconnectAll([human, other]);
    stop(plain.child);
  });

  stopAll();
  assert(firstBotRoom, 'the first bot game was played');
}

module.exports = { run };
