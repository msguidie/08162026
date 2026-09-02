// End-to-end: boots the real server (require('../index.js')) on a free port
// with no GitHub configuration, then drives complete games over socket.io
// exactly the way the browser client does, and checks the replay REST API.

const net = require('net');
const { io } = require('socket.io-client');
const { suite, test, assert, assertEqual } = require('./harness');

const ACTION_CODES = ['G', 'R', 'RD', 'B', 'N', 'X', 'T'];
const ACK_TIMEOUT_MS = 8000;
const SYNC_TIMEOUT_MS = 8000;

let baseUrl = null;

// ── deterministic pseudo-random policy ──
let seed = 0x9e3779b9;
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
    if (condition()) return true;
    await sleep(3);
  }
  throw new Error(`timed out waiting for ${message}`);
}

async function api(path, options) {
  const response = await fetch(`${baseUrl}${path}`, options);
  const body = await response.json().catch(() => null);
  return { status: response.status, body };
}

async function startServer() {
  const port = await freePort();
  process.env.PORT = String(port);
  delete process.env.REPLAY_GITHUB_TOKEN;
  delete process.env.REPLAY_GITHUB_REPO;
  delete process.env.RENDER_EXTERNAL_URL;
  require('../index.js');
  baseUrl = `http://127.0.0.1:${port}`;
  await waitFor(async () => true, 'boot', 10);
  const deadline = Date.now() + 10000;
  for (;;) {
    try {
      const health = await api('/health');
      if (health.status === 200) return port;
    } catch (err) { /* not listening yet */ }
    if (Date.now() > deadline) throw new Error('server did not start');
    await sleep(20);
  }
}

// ── socket client, mirroring src/store/gameStore.ts ──

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

async function connect(username) {
  const socket = io(baseUrl, { transports: ['websocket'], forceNew: true, reconnection: false });
  const client = {
    username,
    socket,
    playerIndex: null,
    roomId: null,
    state: null,
    lastResult: null,
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
  socket.on('action_result', result => { client.lastResult = result; });
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
  assert(login.success, `login failed for ${username}: ${JSON.stringify(login)}`);
  return client;
}

function disconnectAll(clients) {
  for (const client of clients) client.socket.disconnect();
}

// ── turn driving ──

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
  }
  return turnMoved(client, from);
}

// Mobile flow: one TAKE_GEMS_CONFIRMED with the full selection.
async function tryGemsMobile(client) {
  const from = client.playerIndex;
  for (const colors of shuffled(gemCandidates())) {
    const res = await action(client, { type: 'TAKE_GEMS_CONFIRMED', colors });
    if (!res.error) return true;
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

async function playToEnd(clients, options = {}) {
  const cap = options.maxActions || 2000;
  let steps = 0;
  let stuck = 0;
  while (steps < cap) {
    const reference = clients.find(client => client.state);
    if (!reference || reference.state.phase !== 'PLAYING') break;
    const index = reference.state.currentPlayerIndex;
    const actor = clients[index];
    assert(actor, `no client for seat ${index}`);
    steps++;
    const acted = await takeTurn(actor);
    if (!acted) {
      // A seat with no legal action resigns, exactly like a stuck player.
      // (10 tokens + 3 reserved + nothing affordable is a real dead end in
      // these rules — there is no pass action.)
      stuck++;
      const seat = actor.state.players[actor.playerIndex];
      console.log(`        seat ${actor.playerIndex} is stuck: gems ${seat.gems.join('/')}, `
        + `reserved ${seat.reserved.length}, cards ${seat.cards.length}`);
      actor.socket.emit('resign', { roomId: actor.roomId });
      await waitFor(() => actor.state.resignedPlayers.includes(actor.playerIndex)
        || actor.state.phase === 'GAME_OVER', 'resignation to register');
    }
    await syncClients(clients, actor);
  }
  await waitFor(() => clients.every(client => client.state && client.state.phase === 'GAME_OVER'),
    'the game to reach GAME_OVER');
  return { steps, stuck };
}

// ── lobby setup ──

async function enterLobby(clients) {
  for (const client of clients) {
    const res = await emitAck(client, 'enter_lobby');
    assertEqual(res.action, 'lobby', `${client.username} could not enter the lobby`);
  }
}

async function readyUp(clients) {
  const started = clients.map(() => false);
  clients.forEach((client, i) => { client.socket.once('game_start', () => { started[i] = true; }); });
  for (const client of clients) client.socket.emit('lobby_ready');
  await waitFor(() => started.every(Boolean), 'game_start for every seat');
  await waitFor(() => clients.every(client => client.state && client.state.phase === 'PLAYING'), 'the first state');
}

// ── replay assertions ──

function assertRawShape(raw, expected) {
  assertEqual(Object.keys(raw).sort(),
    ['actions', 'clock', 'e', 'first', 'id', 'layout', 'mode', 'n', 'players', 'result', 'setup', 't', 'v'],
    'stored replay keys');
  assertEqual(raw.v, 1);
  assertEqual(raw.id, expected.roomId);
  assertEqual(raw.mode, expected.mode);
  assertEqual(raw.layout, expected.layout ?? null);
  assertEqual(raw.n, expected.n);
  assert(raw.t > 0 && raw.e >= raw.t, 'timestamps');
  assertEqual(raw.players.length, expected.n);
  for (const player of raw.players) {
    assertEqual(Object.keys(player).sort(),
      expected.mode === 'INDIVIDUAL' ? ['a', 'ai', 'u'] : ['a', 'ai', 'team', 'u'], 'player entry keys');
    assertEqual(player.ai, false);
  }
  assert(Number.isInteger(raw.first) && raw.first >= 0 && raw.first < raw.n, 'first player');
  assertEqual(raw.setup.board.map(row => row.length), [4, 4, 4], 'four face-up cards per tier');
  assertEqual(raw.setup.decks.map(deck => deck.length), [36, 26, 16], 'remaining decks');
  assertEqual(raw.setup.tiles.length, expected.n + 1, 'revealed nobles');
  const allIds = [...raw.setup.board.flat(), ...raw.setup.decks.flat()];
  assertEqual(allIds.length, 90, 'every card is accounted for');
  assertEqual(new Set(allIds).size, 90, 'card ids are unique');

  for (const entry of raw.actions) {
    assert(Array.isArray(entry), 'action entries are arrays');
    assert(Number.isInteger(entry[0]) && entry[0] < raw.n, `bad actor in ${JSON.stringify(entry)}`);
    assert(ACTION_CODES.includes(entry[1]), `unknown action code ${entry[1]}`);
    if (entry[1] === 'G') {
      assert(Array.isArray(entry[2]) && entry[2].length >= 1 && entry[2].length <= 3, 'gem payload');
      assert(entry[2].every(color => color >= 0 && color <= 4), 'gem colours');
    }
    if (entry[1] === 'B') assert(entry[3] === 'b' || entry[3] === 'r', 'buy source');
    if (entry[1] === 'RD') assert([1, 2, 3].includes(entry[2]), 'reserve tier');
  }

  assertEqual(Object.keys(raw.result).sort(),
    ['cards', 'rating', 'reason', 'resigned', 'scores', 'winners', 'winningTeamIds'], 'result keys');
  assertEqual(raw.result.scores.length, expected.n);
  assertEqual(raw.result.rating.length, expected.n);
}

async function assertReplayMatches(clients, expected) {
  const roomId = expected.roomId;
  const live = clients.find(client => client.finalState).finalState;

  const list = await api('/api/replays?limit=200');
  assertEqual(list.status, 200);
  assertEqual(list.body.source, 'memory');
  const entry = list.body.games.find(game => game.id === roomId);
  assert(entry, `${roomId} is missing from /api/replays`);
  assertEqual(entry.mode, expected.mode);
  assertEqual(entry.n, expected.n);
  assertEqual(entry.players, clients.map(client => client.username));
  assertEqual(entry.ai, clients.map(() => false));
  assert(list.body.total >= list.body.games.length, 'total');

  const rawResponse = await api(`/api/replays/${roomId}/raw`);
  assertEqual(rawResponse.status, 200);
  const raw = rawResponse.body;
  assertRawShape(raw, expected);
  assertEqual(entry.turns, raw.actions.length, 'index turn count');
  assertEqual(raw.result.scores, live.players.map(p => p.score), 'stored scores');
  assertEqual(raw.result.cards, live.players.map(p => p.cards.length), 'stored card counts');
  assertEqual(raw.result.resigned, [...live.resignedPlayers].sort((a, b) => a - b), 'stored resignations');
  if (expected.mode === 'INDIVIDUAL') {
    assertEqual(raw.result.winningTeamIds, null);
    assert(Array.isArray(raw.result.winners), 'individual winners');
  } else {
    assertEqual(raw.result.winners, null);
    assertEqual(raw.result.winningTeamIds, live.gameResult.winningTeamIds, 'team winners');
    assertEqual(raw.result.reason, live.gameResult.reason);
  }

  const framesResponse = await api(`/api/replays/${roomId}`);
  assertEqual(framesResponse.status, 200);
  const { id, meta, frames } = framesResponse.body;
  assertEqual(id, roomId);
  assertEqual(frames.length, raw.actions.length + 1, 'frames === actions + 1');
  assertEqual(meta.mode, expected.mode);
  assertEqual(meta.n, expected.n);
  assertEqual(meta.first, raw.first);
  assertEqual(meta.players.map(p => p.username), clients.map(client => client.username));
  assertEqual(meta.result, raw.result);

  assertEqual(frames[0].i, 0);
  assertEqual(frames[0].action, null);
  assertEqual(frames[0].result, null);
  for (let i = 1; i < frames.length; i++) {
    assert(frames[i].state.turnNumber >= frames[i - 1].state.turnNumber, `turnNumber fell at frame ${i}`);
    assertEqual(frames[i].i, i);
    assert(frames[i].state.decks === undefined, 'frames must not leak the decks');
    assertEqual(frames[i].state.timeControl, null, 'replays have no clock');
  }

  const last = frames[frames.length - 1];
  assertEqual(last.state.phase, 'GAME_OVER', 'replay ends in GAME_OVER');
  assertEqual(last.state.players.map(p => p.score), live.players.map(p => p.score), 'final scores');
  assertEqual(last.state.players.map(p => p.cards.length), live.players.map(p => p.cards.length), 'final card counts');
  assertEqual(last.state.players.map(p => p.bonusTiles.length), live.players.map(p => p.bonusTiles.length), 'final nobles');
  assertEqual(last.state.resignedPlayers, live.resignedPlayers, 'final resignations');
  assertEqual(last.state.gameResult, live.gameResult ?? null, 'final game result');
  return raw;
}

// ── suites ──

async function run() {
  suite('replay e2e — server boot');

  const port = await startServer();

  await test('the replay routes are registered and report an in-memory store', async () => {
    const status = await api('/api/replays/status');
    assertEqual(status.status, 200);
    assertEqual(status.body.github, false, 'no GitHub configuration in tests');
    assert(Number.isInteger(status.body.memory), 'memory count');
    const root = await api('/');
    assertEqual(root.status, 200, 'existing routes still work');
    assertEqual(root.body.status, 'running');
    console.log(`        server listening on ${port}`);
  });

  await test('an unknown replay id is a 404', async () => {
    assertEqual((await api('/api/replays/game-does-not-exist')).status, 404);
    assertEqual((await api('/api/replays/game-does-not-exist/raw')).status, 404);
  });

  suite('replay e2e — 2 player INDIVIDUAL');

  let individualClients = null;
  await test('plays a full random game and records it', async () => {
    individualClients = [await connect('e2e-alice'), await connect('e2e-bob')];
    await enterLobby(individualClients);
    await readyUp(individualClients);
    const played = await playToEnd(individualClients);
    const roomId = individualClients[0].roomId;
    const raw = await assertReplayMatches(individualClients, {
      roomId, mode: 'INDIVIDUAL', n: 2, layout: null,
    });
    console.log(`        ${played.steps} turns, ${raw.actions.length} actions, ended by ${raw.result.reason}`
      + `${played.stuck ? ` (${played.stuck} stuck seat)` : ''}`);
    disconnectAll(individualClients);
  });

  suite('replay e2e — 3 player ONE_V_TWO');

  await test('plays a full 1v2 game and records it', async () => {
    const clients = [await connect('e2e-solo'), await connect('e2e-duo1'), await connect('e2e-duo2')];
    await enterLobby(clients);
    assertEqual((await emitAck(clients[0], 'set_team_mode', { enabled: true })).ok, true);
    assertEqual((await emitAck(clients[0], 'set_unlimited_time', { enabled: true })).ok, true);
    assertEqual((await emitAck(clients[0], 'select_team_seat', { teamId: 0, seatIndex: 0 })).ok, true);
    assertEqual((await emitAck(clients[1], 'select_team_seat', { teamId: 1, seatIndex: 0 })).ok, true);
    assertEqual((await emitAck(clients[2], 'select_team_seat', { teamId: 1, seatIndex: 1 })).ok, true);
    await readyUp(clients);

    assertEqual(clients[0].state.gameMode, 'ONE_V_TWO');
    assertEqual(clients[0].state.currentPlayerIndex, 0, 'the solo seat opens');
    const played = await playToEnd(clients);
    const raw = await assertReplayMatches(clients, {
      roomId: clients[0].roomId, mode: 'ONE_V_TWO', n: 3, layout: null,
    });
    assertEqual(raw.first, 0);
    assertEqual(raw.clock, false, 'unlimited time was selected');
    assertEqual(raw.players.map(p => p.team), [0, 1, 1]);
    console.log(`        ${played.steps} turns, ${raw.actions.length} actions, ended by ${raw.result.reason}`
      + `${played.stuck ? ` (${played.stuck} stuck seat)` : ''}`);
    disconnectAll(clients);
  });

  suite('replay e2e — 4 player TEAM (OPPOSITE)');

  await test('plays a full 2v2 game and records it', async () => {
    const clients = [
      await connect('e2e-t1'), await connect('e2e-t2'), await connect('e2e-t3'), await connect('e2e-t4'),
    ];
    await enterLobby(clients);
    assertEqual((await emitAck(clients[0], 'set_team_mode', { enabled: true })).ok, true);
    assertEqual((await emitAck(clients[0], 'set_team_layout', { layout: 'OPPOSITE' })).ok, true);
    assertEqual((await emitAck(clients[0], 'set_unlimited_time', { enabled: true })).ok, true);
    // OPPOSITE seating order is [0,0], [1,0], [0,1], [1,1].
    assertEqual((await emitAck(clients[0], 'select_team_seat', { teamId: 0, seatIndex: 0 })).ok, true);
    assertEqual((await emitAck(clients[1], 'select_team_seat', { teamId: 1, seatIndex: 0 })).ok, true);
    assertEqual((await emitAck(clients[2], 'select_team_seat', { teamId: 0, seatIndex: 1 })).ok, true);
    assertEqual((await emitAck(clients[3], 'select_team_seat', { teamId: 1, seatIndex: 1 })).ok, true);
    await readyUp(clients);

    assertEqual(clients[0].state.gameMode, 'TEAM');
    assertEqual(clients[0].state.teamLayout, 'OPPOSITE');
    assertEqual(clients[0].state.players.map(p => p.teamId), [0, 1, 0, 1]);
    const played = await playToEnd(clients);
    const raw = await assertReplayMatches(clients, {
      roomId: clients[0].roomId, mode: 'TEAM', n: 4, layout: 'OPPOSITE',
    });
    assertEqual(raw.players.map(p => p.team), [0, 1, 0, 1]);
    console.log(`        ${played.steps} turns, ${raw.actions.length} actions, ended by ${raw.result.reason}`
      + `${played.stuck ? ` (${played.stuck} stuck seat)` : ''}`);
    disconnectAll(clients);
  });

  suite('replay e2e — game ended by resignation');

  await test('records the resignation and finishes the replay', async () => {
    const clients = [await connect('e2e-quitter'), await connect('e2e-stayer')];
    await enterLobby(clients);
    await readyUp(clients);

    for (let i = 0; i < 6; i++) {
      const reference = clients[0].state;
      if (reference.phase !== 'PLAYING') break;
      await takeTurn(clients[reference.currentPlayerIndex]);
      await syncClients(clients, clients[reference.currentPlayerIndex]);
    }

    const quitter = clients[0];
    quitter.socket.emit('resign', { roomId: quitter.roomId });
    await waitFor(() => clients.every(client => client.state && client.state.phase === 'GAME_OVER'),
      'the game to end on resignation');

    const raw = await assertReplayMatches(clients, {
      roomId: quitter.roomId, mode: 'INDIVIDUAL', n: 2, layout: null,
    });
    assertEqual(raw.actions[raw.actions.length - 1], [0, 'X'], 'the resignation is the last action');
    assertEqual(raw.result.reason, 'FORFEIT');
    assertEqual(raw.result.resigned, [0]);
    assertEqual(raw.result.winners, [1]);
    disconnectAll(clients);
  });

  suite('replay e2e — discarding');

  await test('quit_room discards the recording', async () => {
    const clients = [await connect('e2e-x1'), await connect('e2e-x2')];
    await enterLobby(clients);
    await readyUp(clients);
    const roomId = clients[0].roomId;
    await takeTurn(clients[clients[0].state.currentPlayerIndex]);

    clients[0].socket.emit('quit_room', { roomId });
    await sleep(150);
    assertEqual((await api(`/api/replays/${roomId}/raw`)).status, 404, 'a quit game is not stored');
    const list = await api('/api/replays?limit=200');
    assert(!list.body.games.some(game => game.id === roomId), 'a quit game is not listed');
    disconnectAll(clients);
  });

  suite('replay e2e — pagination');

  await test('limit and offset page through the newest-first list', async () => {
    const all = await api('/api/replays?limit=200');
    assert(all.body.games.length >= 4, 'the finished games are listed');
    const timestamps = all.body.games.map(game => game.t);
    for (let i = 1; i < timestamps.length; i++) {
      assert(timestamps[i] <= timestamps[i - 1], 'the list is newest first');
    }
    const firstPage = await api('/api/replays?limit=2&offset=0');
    const secondPage = await api('/api/replays?limit=2&offset=2');
    assertEqual(firstPage.body.games.length, 2);
    assertEqual(firstPage.body.games.map(g => g.id), all.body.games.slice(0, 2).map(g => g.id));
    assertEqual(secondPage.body.games.map(g => g.id), all.body.games.slice(2, 4).map(g => g.id));
    assertEqual(firstPage.body.total, all.body.total);
  });
}

module.exports = { run };
