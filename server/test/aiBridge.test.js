// Unit tests for the AI bridge (docs/AI_BRIDGE.md §1/§3) with a fake socket
// and a fake room. `applyGameAction` is stubbed with the real rules engine so
// the turn really advances, but nothing is broadcast.

const { suite, test, assert, assertEqual } = require('./harness');
const { createInitialGameState, processAction, ALL_CARDS, ALL_BONUS_TILES } = require('../gameLogic');
const aiBridge = require('../aiBridge');

const SECRET = 'unit-test-secret';

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function fakeSocket(id = 'worker-socket') {
  const handlers = new Map();
  return {
    id,
    sent: [],
    on(event, fn) { handlers.set(event, fn); },
    emit(event, payload) { this.sent.push({ event, payload }); },
    trigger(event, ...args) {
      const fn = handlers.get(event);
      if (!fn) throw new Error(`no handler registered for ${event}`);
      return fn(...args);
    },
    lastOf(event) {
      const found = [...this.sent].reverse().find(message => message.event === event);
      return found ? found.payload : null;
    },
    countOf(event) { return this.sent.filter(message => message.event === event).length; },
  };
}

// A 2-player room whose seat 1 is a bot and is to move.
function makeWorld() {
  const gameState = createInitialGameState(
    [{ username: 'human', avatarSeed: 1 }, { username: 'Bot Alpha', avatarSeed: 2 }],
    { gameMode: 'INDIVIDUAL', unlimitedTime: true },
  );
  gameState.currentPlayerIndex = 1;
  gameState.roundStartPlayer = 0;
  const room = {
    id: 'room-unit',
    gameState,
    playerSockets: [
      { socketId: 'human-socket', username: 'human', playerIndex: 0 },
      { socketId: 'ai:1', username: 'Bot Alpha', playerIndex: 1, isAI: true },
    ],
  };
  const applied = [];
  const resigned = [];
  aiBridge.init({
    getRoom: id => (id === room.id ? room : null),
    applyGameAction: (target, playerIndex, action) => {
      const result = processAction(target.gameState, playerIndex, action);
      if (result.error) return { error: result.error };
      applied.push({ playerIndex, action });
      return { ok: true };
    },
    resignPlayer: (target, playerIndex) => {
      resigned.push(playerIndex);
      target.gameState.phase = 'GAME_OVER';
      return { ok: true };
    },
  });
  return { room, applied, resigned };
}

function freshBridge({ enabled = true, delayMs = 1, deadlineMs = 5000 } = {}) {
  if (enabled) process.env.AI_WORKER_SECRET = SECRET;
  else delete process.env.AI_WORKER_SECRET;
  aiBridge.reset();
  aiBridge.configure({ delayMs, deadlineMs });
}

function registerWorker(name = 'unit-worker', id = 'worker-socket') {
  const socket = fakeSocket(id);
  aiBridge.attach(socket);
  let ack = null;
  socket.trigger('ai_worker_register', {
    secret: SECRET, name, version: '1.0.0', modes: ['INDIVIDUAL', 'ONE_V_TWO', 'TEAM'],
  }, response => { ack = response; });
  return { socket, ack };
}

async function run() {
  suite('aiBridge — worker registration');

  await test('rejects a bad secret and accepts the configured one', async () => {
    freshBridge();
    const socket = fakeSocket();
    aiBridge.attach(socket);

    let ack = null;
    socket.trigger('ai_worker_register', { secret: 'nope', name: 'imposter' }, response => { ack = response; });
    assertEqual(ack, { error: 'Invalid worker secret' }, 'bad secret');
    assertEqual(aiBridge.isAvailable(), false, 'still unavailable');

    socket.trigger('ai_worker_register', {
      secret: SECRET, name: 'gpu-1', version: '2.0.0', modes: ['INDIVIDUAL'],
    }, response => { ack = response; });
    assertEqual(ack, { ok: true }, 'accepted');
    assertEqual(aiBridge.status(), {
      enabled: true, available: true, name: 'gpu-1', modes: ['INDIVIDUAL'],
    }, 'status');
  });

  await test('refuses registration when AI_WORKER_SECRET is unset', async () => {
    freshBridge({ enabled: false });
    const socket = fakeSocket();
    aiBridge.attach(socket);
    let ack = null;
    socket.trigger('ai_worker_register', { secret: 'anything' }, response => { ack = response; });
    assertEqual(ack, { error: 'AI is not enabled on this server' });
    assertEqual(aiBridge.status(), { enabled: false, available: false });
    assertEqual(aiBridge.isEnabled(), false);
  });

  await test('a new registration replaces the previous worker', async () => {
    freshBridge();
    const first = registerWorker('gpu-1', 'socket-1');
    const second = registerWorker('gpu-2', 'socket-2');
    assertEqual(second.ack, { ok: true });
    assertEqual(aiBridge.status().name, 'gpu-2', 'the newest worker wins');

    // The replaced socket disconnecting must not clear the active worker.
    first.socket.trigger('disconnect');
    assertEqual(aiBridge.isAvailable(), true, 'still available');
    second.socket.trigger('disconnect');
    assertEqual(aiBridge.isAvailable(), false, 'disconnect clears the worker');
    assertEqual(aiBridge.status(), { enabled: true, available: false });
  });

  suite('aiBridge — turn driver');

  await test('asks the worker and applies the answer through applyGameAction', async () => {
    freshBridge();
    const { room, applied } = makeWorld();
    const { socket } = registerWorker();

    aiBridge.maybeAct(room);
    assertEqual(socket.countOf('ai_move_request'), 0, 'the UX delay comes first');
    await sleep(25);

    const request = socket.lastOf('ai_move_request');
    assert(request, 'a request was sent');
    assertEqual(request.roomId, room.id);
    assertEqual(request.playerIndex, 1);
    assertEqual(request.kind, 'MOVE');
    assertEqual(request.pendingTileChoice, null);
    assert(request.deadlineMs > Date.now(), 'the deadline is absolute epoch ms');
    assert(request.state.decks === undefined, 'the observation never leaks the decks');

    let ack = null;
    socket.trigger('ai_move_response', {
      requestId: request.requestId, action: { type: 'TAKE_GEMS', colors: [0, 1, 2] },
    }, response => { ack = response; });

    assertEqual(ack, { ok: true });
    assertEqual(applied.length, 1, 'exactly one action applied');
    assertEqual(applied[0], { playerIndex: 1, action: { type: 'TAKE_GEMS_CONFIRMED', colors: [0, 1, 2] } });
    assertEqual(room.gameState.currentPlayerIndex, 0, 'the turn moved on');
    assertEqual(aiBridge._inFlightCount(), 0, 'the request was released');
  });

  await test('hides other seats reserved cards and marks the public ones', async () => {
    freshBridge();
    const { room } = makeWorld();
    const [fromBoard, fromDeck] = ALL_CARDS.filter(card => card.tier === 2).slice(0, 2);
    room.gameState.players[0].reserved = [fromBoard, fromDeck];
    room.gameState.players[1].reserved = [ALL_CARDS.find(card => card.tier === 3)];
    // Only a card reserved FROM THE BOARD is public knowledge.
    aiBridge.onActionResult(room, { type: 'RESERVE_CARD', actingPlayer: 0, payload: { cardId: fromBoard.id } });
    aiBridge.onActionResult(room, { type: 'RESERVE_FROM_DECK', actingPlayer: 0, payload: { tier: 2 } });

    const { socket } = registerWorker();
    aiBridge.maybeAct(room);
    await sleep(25);
    const request = socket.lastOf('ai_move_request');

    assertEqual(request.knownReserved, [fromBoard.id], 'knownReserved');
    assertEqual(request.state.players[0].reserved, [
      { id: fromBoard.id, tier: 2, hidden: true, known: true },
      { id: -1, tier: 2, hidden: true, known: false },
    ], 'opponent reserve is masked');
    assertEqual(request.state.players[1].reserved[0].cost, room.gameState.players[1].reserved[0].cost,
      'the bot sees its own reserve in full');
  });

  await test('asks for a TILE decision while a noble choice is pending', async () => {
    freshBridge();
    const { room, applied } = makeWorld();
    const byReward = reward => ALL_CARDS.filter(card => card.reward === reward && card.tier === 1).slice(0, 4);
    room.gameState.players[1].cards = [...byReward(0), ...byReward(1), ...byReward(2)];
    const discount = [0, 0, 0, 0, 0];
    for (const card of room.gameState.players[1].cards) discount[card.reward]++;
    const qualifying = ALL_BONUS_TILES.filter(tile =>
      tile.requirement.every((need, color) => discount[color] >= need));
    assert(qualifying.length >= 2, 'fixture offers a real noble choice');
    room.gameState.bonusTiles = qualifying.slice(0, 2);
    room.gameState.turnAction = { type: 'BUY' };
    room.gameState._pendingTileChoice = room.gameState.bonusTiles.map(tile => tile.id);
    const pending = [...room.gameState._pendingTileChoice];

    const { socket } = registerWorker();
    aiBridge.maybeAct(room);
    await sleep(25);
    const request = socket.lastOf('ai_move_request');
    assertEqual(request.kind, 'TILE');
    assertEqual(request.pendingTileChoice, pending);

    // The worker picks the second tile — a plain CHOOSE_TILE is applied as is.
    socket.trigger('ai_move_response', {
      requestId: request.requestId, action: { type: 'CHOOSE_TILE', tileId: pending[1] },
    }, () => {});
    assertEqual(applied, [{ playerIndex: 1, action: { type: 'CHOOSE_TILE', tileId: pending[1] } }]);
    assertEqual(aiBridge._inFlightCount(), 0, 'released');

    // And a move action is no answer to a TILE request: the tile fallback
    // (first pending tile) takes over.
    const second = makeWorld();
    second.room.gameState.players[1].cards = [...byReward(0), ...byReward(1), ...byReward(2)];
    second.room.gameState.bonusTiles = qualifying.slice(0, 2);
    second.room.gameState.turnAction = { type: 'BUY' };
    second.room.gameState._pendingTileChoice = [...pending];
    const worker2 = registerWorker('unit-worker', 'socket-tile-2');
    aiBridge.maybeAct(second.room);
    await sleep(25);
    const request2 = worker2.socket.lastOf('ai_move_request');
    worker2.socket.trigger('ai_move_response', {
      requestId: request2.requestId, action: { type: 'TAKE_GEMS', colors: [0] },
    }, () => {});
    assertEqual(second.applied, [{ playerIndex: 1, action: { type: 'CHOOSE_TILE', tileId: pending[0] } }],
      'the fallback claims the first pending tile');
  });

  suite('aiBridge — fallback paths');

  await test('falls back to the greedy policy when no worker is connected', async () => {
    freshBridge();
    const { room, applied } = makeWorld();
    aiBridge.maybeAct(room);
    await sleep(25);
    assert(applied.length >= 1, 'the bot still moved');
    assertEqual(applied[0].playerIndex, 1);
    assertEqual(room.gameState.currentPlayerIndex, 0, 'the turn moved on');
  });

  await test('falls back when the worker misses the deadline', async () => {
    freshBridge({ delayMs: 1, deadlineMs: 20 });
    const { room, applied } = makeWorld();
    const { socket } = registerWorker();

    aiBridge.maybeAct(room);
    await sleep(80);

    assertEqual(socket.countOf('ai_move_request'), 1, 'the worker was asked');
    assertEqual(socket.countOf('ai_move_cancel'), 1, 'and told to stop');
    assert(applied.length >= 1, 'the fallback moved the bot');
    assertEqual(room.gameState.currentPlayerIndex, 0, 'the turn moved on');

    // A late answer for the cancelled request is ignored.
    const request = socket.lastOf('ai_move_request');
    let ack = null;
    socket.trigger('ai_move_response', {
      requestId: request.requestId, action: { type: 'TAKE_GEMS', colors: [0, 1, 2] },
    }, response => { ack = response; });
    assertEqual(ack, { error: 'Unknown or expired request' }, 'a late answer is refused');
    assertEqual(applied.length, 1, 'nothing extra was applied');
  });

  await test('ignores an unknown request id and a superseded position', async () => {
    freshBridge();
    const { room, applied } = makeWorld();
    const { socket } = registerWorker();
    aiBridge.maybeAct(room);
    await sleep(25);
    const request = socket.lastOf('ai_move_request');

    let ack = null;
    socket.trigger('ai_move_response', {
      requestId: 'not-a-request', action: { type: 'TAKE_GEMS', colors: [0] },
    }, response => { ack = response; });
    assertEqual(ack, { error: 'Unknown or expired request' }, 'unknown id');
    assertEqual(applied.length, 0, 'nothing applied');

    // The seat moved on (a timeout eliminated it, say) before the answer.
    room.gameState.turnNumber += 1;
    room.gameState.currentPlayerIndex = 0;
    socket.trigger('ai_move_response', {
      requestId: request.requestId, action: { type: 'TAKE_GEMS', colors: [0, 1, 2] },
    }, response => { ack = response; });
    assertEqual(ack, { error: 'The position changed before the answer arrived' }, 'stale position');
    assertEqual(applied.length, 0, 'still nothing applied');
    assertEqual(aiBridge._inFlightCount(), 0, 'released');
  });

  await test('an illegal worker answer is replaced by the fallback', async () => {
    freshBridge();
    const { room, applied } = makeWorld();
    const { socket } = registerWorker();
    aiBridge.maybeAct(room);
    await sleep(25);
    const request = socket.lastOf('ai_move_request');

    // Buying a card the bot cannot afford: shape is fine, rules say no.
    const expensive = room.gameState.board[2][0];
    socket.trigger('ai_move_response', {
      requestId: request.requestId, action: { type: 'BUY_CARD', cardId: expensive.id, source: 'board' },
    }, () => {});

    assertEqual(applied.length, 1, 'one fallback action applied');
    assert(applied[0].action.type !== 'BUY_CARD', `unexpected ${applied[0].action.type}`);
    assertEqual(room.gameState.currentPlayerIndex, 0, 'the turn moved on');
  });

  await test('a malformed worker answer is replaced by the fallback', async () => {
    freshBridge();
    const { room, applied } = makeWorld();
    const { socket } = registerWorker();
    aiBridge.maybeAct(room);
    await sleep(25);
    const request = socket.lastOf('ai_move_request');
    socket.trigger('ai_move_response', {
      requestId: request.requestId, action: { type: 'TAKE_GEMS', colors: ['red'] },
    }, () => {});
    assertEqual(applied.length, 1, 'the fallback moved the bot');
    assertEqual(room.gameState.currentPlayerIndex, 0);
  });

  await test('RESIGN and NONE both resign the bot seat', async () => {
    for (const type of ['RESIGN', 'NONE']) {
      freshBridge();
      const { room, resigned } = makeWorld();
      const { socket } = registerWorker();
      aiBridge.maybeAct(room);
      await sleep(25);
      const request = socket.lastOf('ai_move_request');
      socket.trigger('ai_move_response', { requestId: request.requestId, action: { type } }, () => {});
      assertEqual(resigned, [1], `${type} resigns the seat`);
    }
  });

  await test('a worker that disconnects mid-request hands over to the fallback', async () => {
    freshBridge({ delayMs: 1, deadlineMs: 5000 });
    const { room, applied } = makeWorld();
    const { socket } = registerWorker();
    aiBridge.maybeAct(room);
    await sleep(25);
    assertEqual(socket.countOf('ai_move_request'), 1);

    socket.trigger('disconnect');
    assertEqual(aiBridge.isAvailable(), false, 'no worker any more');
    assertEqual(applied.length, 1, 'the pending turn was played by the fallback');
    assertEqual(room.gameState.currentPlayerIndex, 0, 'the turn moved on');
  });

  suite('aiBridge — guards');

  await test('never acts for a human seat, a finished game or twice at once', async () => {
    freshBridge();
    const { room, applied } = makeWorld();
    const { socket } = registerWorker();

    room.gameState.currentPlayerIndex = 0; // human
    aiBridge.maybeAct(room);
    await sleep(20);
    assertEqual(socket.countOf('ai_move_request'), 0, 'human seats are ignored');

    room.gameState.currentPlayerIndex = 1;
    room.gameState.phase = 'GAME_OVER';
    aiBridge.maybeAct(room);
    await sleep(20);
    assertEqual(socket.countOf('ai_move_request'), 0, 'finished games are ignored');

    room.gameState.phase = 'PLAYING';
    aiBridge.maybeAct(room);
    aiBridge.maybeAct(room);
    aiBridge.maybeAct(room);
    assertEqual(aiBridge._inFlightCount(), 1, 're-entrancy guard: one request per room');
    await sleep(25);
    assertEqual(socket.countOf('ai_move_request'), 1, 'exactly one request');
    assertEqual(applied.length, 0, 'nothing applied yet');
  });

  await test('does nothing at all while AI_WORKER_SECRET is unset', async () => {
    freshBridge({ enabled: false });
    const { room, applied } = makeWorld();
    aiBridge.maybeAct(room);
    await sleep(30);
    assertEqual(applied.length, 0, 'no bot move');
    assertEqual(aiBridge._inFlightCount(), 0, 'no request');
  });

  await test('clearRoom cancels the pending request and forgets the room', async () => {
    freshBridge({ delayMs: 1, deadlineMs: 5000 });
    const { room, applied } = makeWorld();
    const { socket } = registerWorker();
    aiBridge.maybeAct(room);
    await sleep(25);
    aiBridge.clearRoom(room.id);
    assertEqual(socket.countOf('ai_move_cancel'), 1, 'the worker was told to stop');
    assertEqual(aiBridge._inFlightCount(), 0);
    await sleep(20);
    assertEqual(applied.length, 0, 'no move was applied after the room went away');
  });

  suite('aiBridge — translation table');

  await test('maps every worker action onto the live protocol', async () => {
    assertEqual(aiBridge.translateWorkerAction({ type: 'TAKE_GEMS', colors: [0, 1, 2] }),
      { actions: [{ type: 'TAKE_GEMS_CONFIRMED', colors: [0, 1, 2] }] });
    assertEqual(aiBridge.translateWorkerAction({ type: 'TAKE_GEMS', colors: [3, 3] }),
      { actions: [{ type: 'TAKE_GEMS_CONFIRMED', colors: [3, 3] }] });
    assertEqual(aiBridge.translateWorkerAction({ type: 'RESERVE_CARD', cardId: 37 }),
      { actions: [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_CARD', cardId: 37 }] });
    assertEqual(aiBridge.translateWorkerAction({ type: 'RESERVE_FROM_DECK', tier: 2 }),
      { actions: [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_FROM_DECK', tier: 2 }] });
    assertEqual(aiBridge.translateWorkerAction({ type: 'BUY_CARD', cardId: 12, source: 'reserved' }),
      { actions: [{ type: 'BUY_CARD', cardId: 12, source: 'reserved' }] });
    assertEqual(aiBridge.translateWorkerAction({ type: 'CHOOSE_TILE', tileId: 4 }, 'TILE'),
      { actions: [{ type: 'CHOOSE_TILE', tileId: 4 }] });
    assertEqual(aiBridge.translateWorkerAction({ type: 'RESIGN' }), { resign: true });
    assertEqual(aiBridge.translateWorkerAction({ type: 'NONE' }), { resign: true });
  });

  await test('rejects malformed or out-of-context actions', async () => {
    const bad = [
      null, undefined, {}, { type: 'WAT' },
      { type: 'TAKE_GEMS' },
      { type: 'TAKE_GEMS', colors: [] },
      { type: 'TAKE_GEMS', colors: [0, 1, 2, 3] },
      { type: 'TAKE_GEMS', colors: [5] },
      { type: 'RESERVE_CARD' },
      { type: 'RESERVE_FROM_DECK', tier: 4 },
      { type: 'BUY_CARD', cardId: 1, source: 'deck' },
      { type: 'CHOOSE_TILE', tileId: 'x' },
    ];
    for (const action of bad) {
      assertEqual(aiBridge.translateWorkerAction(action), null, `rejects ${JSON.stringify(action)}`);
    }
    // A move action is meaningless while a noble choice is pending.
    assertEqual(aiBridge.translateWorkerAction({ type: 'TAKE_GEMS', colors: [0] }, 'TILE'), null);
    assertEqual(aiBridge.translateWorkerAction({ type: 'BUY_CARD', cardId: 1, source: 'board' }, 'TILE'), null);
  });

  // Leave the process exactly as it was found: the in-process e2e server in
  // replay.e2e.js must not accidentally come up with AI enabled.
  delete process.env.AI_WORKER_SECRET;
  aiBridge.reset();
}

module.exports = { run };
