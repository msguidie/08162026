// Unit tests for the AI bridge (docs/AI_BRIDGE.md §1/§3) with a fake socket
// and a fake room. `applyGameAction` is stubbed with the real rules engine so
// the turn really advances, but nothing is broadcast.

const { suite, test, assert, assertEqual } = require('./harness');
const { createInitialGameState, processAction, processResign, ALL_CARDS, ALL_BONUS_TILES } = require('../gameLogic');
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

// A 4-player room whose seats 1 and 2 are bots; seat 1 is to move.
// Two human seats keep the game alive while seats resign one after another.
function makeBotWorld() {
  const gameState = createInitialGameState(
    [
      { username: 'human', avatarSeed: 1 },
      { username: 'Bot Alpha', avatarSeed: 2 },
      { username: 'Bot Beta', avatarSeed: 3 },
      { username: 'human-2', avatarSeed: 4 },
    ],
    { gameMode: 'INDIVIDUAL', unlimitedTime: true },
  );
  gameState.currentPlayerIndex = 1;
  gameState.roundStartPlayer = 0;
  const room = {
    id: 'room-unit-multi',
    gameState,
    playerSockets: [
      { socketId: 'human-socket', username: 'human', playerIndex: 0 },
      { socketId: 'ai:1', username: 'Bot Alpha', playerIndex: 1, isAI: true },
      { socketId: 'ai:2', username: 'Bot Beta', playerIndex: 2, isAI: true },
      { socketId: 'human-socket-2', username: 'human-2', playerIndex: 3 },
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
      processResign(target.gameState, playerIndex);
      return { ok: true };
    },
  });
  return { room, applied, resigned };
}

// Records every availability flip the bridge reports to the server.
function trackAvailability() {
  const flips = [];
  aiBridge.init({ onAvailabilityChange: available => flips.push(available) });
  return flips;
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

  await test('reports every availability flip so the lobby can be re-broadcast', async () => {
    freshBridge();
    const flips = trackAvailability();

    const first = registerWorker('gpu-1', 'socket-a1');
    assertEqual(flips, [true], 'the first worker makes AI available');

    const second = registerWorker('gpu-2', 'socket-a2');
    assertEqual(flips, [true], 'a replacement worker is not a flip');

    first.socket.trigger('disconnect');
    assertEqual(flips, [true], 'the replaced socket going away is not a flip');

    second.socket.trigger('disconnect');
    assertEqual(flips, [true, false], 'the last worker leaving is');
    assertEqual(aiBridge.isAvailable(), false);

    registerWorker('gpu-3', 'socket-a3');
    assertEqual(flips, [true, false, true], 'and a fresh worker flips it back');
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

  await test('a buy that opens a noble choice asks the worker for the tile instead of falling back', async () => {
    freshBridge();
    const { room, applied, resigned } = makeWorld();
    // Buying `target` (reward 4) takes the bot's discount to [3, 3, 3, 0, 3],
    // which meets tiles 5 and 7 at once — so the buy legitimately leaves the
    // turn where it is with `_pendingTileChoice` set.
    const target = ALL_CARDS.find(card => card.id === 4);
    const pick = (reward, count) => ALL_CARDS
      .filter(card => card.tier === 1 && card.reward === reward && card.id !== target.id)
      .slice(0, count);
    room.gameState.players[1].cards = [...pick(0, 3), ...pick(1, 3), ...pick(2, 3), ...pick(4, 2)];
    room.gameState.players[1].gems = [0, 0, 0, 1, 0, 0];
    room.gameState.bonusTiles = ALL_BONUS_TILES.filter(tile => tile.id === 5 || tile.id === 7);
    room.gameState.board[0] = [target, ...room.gameState.board[0].slice(1)];

    const { socket } = registerWorker();
    aiBridge.maybeAct(room);
    await sleep(25);
    const buyRequest = socket.lastOf('ai_move_request');
    assertEqual(buyRequest.kind, 'MOVE', 'the first request is an ordinary move');
    socket.trigger('ai_move_response', {
      requestId: buyRequest.requestId,
      action: { type: 'BUY_CARD', cardId: target.id, source: 'board' },
    }, () => {});

    assertEqual(applied, [{ playerIndex: 1, action: { type: 'BUY_CARD', cardId: target.id, source: 'board' } }],
      'the buy is applied and no fallback CHOOSE_TILE is bolted onto it');
    assertEqual(resigned, [], 'the seat is not resigned');
    assertEqual(room.gameState._pendingTileChoice, [5, 7], 'the noble choice is still the bot\'s to make');

    await sleep(25);
    assertEqual(socket.countOf('ai_move_request'), 2, 'exactly one follow-up request');
    const tileRequest = socket.lastOf('ai_move_request');
    assertEqual(tileRequest.kind, 'TILE', 'and it asks for the tile');
    assertEqual(tileRequest.playerIndex, 1);
    assertEqual(tileRequest.pendingTileChoice, [5, 7]);

    socket.trigger('ai_move_response', {
      requestId: tileRequest.requestId, action: { type: 'CHOOSE_TILE', tileId: 7 },
    }, () => {});
    assertEqual(applied.length, 2, 'the worker picked the noble itself');
    assertEqual(applied[1].action, { type: 'CHOOSE_TILE', tileId: 7 });
    assertEqual(room.gameState.currentPlayerIndex, 0, 'and the turn moved on');
    assertEqual(aiBridge._inFlightCount(), 0, 'released');
  });

  await test('asks for a MOVE while an orphaned noble choice is pending', async () => {
    // docs/KNOWN_ISSUES.md §1: a gem take or a reserve can leave
    // `_pendingTileChoice` set with `turnAction` back to null. CHOOSE_TILE is
    // refused by the engine in that state, so the bridge must ask for a MOVE
    // — and the seat must never be resigned over it.
    freshBridge();
    const { room, applied, resigned } = makeBotWorld();
    // Hand-built orphan: pending choice, turnAction null, turn not advanced.
    room.gameState.bonusTiles = ALL_BONUS_TILES.slice(0, 2);
    room.gameState._pendingTileChoice = room.gameState.bonusTiles.map(tile => tile.id);
    room.gameState.turnAction = null;

    const { socket } = registerWorker('unit-worker', 'socket-orphan');
    aiBridge.maybeAct(room);
    await sleep(25);

    const request = socket.lastOf('ai_move_request');
    assertEqual(request.kind, 'MOVE', 'CHOOSE_TILE is impossible here — ask for a move');
    assertEqual(request.pendingTileChoice, room.gameState._pendingTileChoice,
      'the orphaned choice is still reported to the worker');

    socket.trigger('ai_move_response', {
      requestId: request.requestId, action: { type: 'TAKE_GEMS', colors: [0, 1, 2] },
    }, () => {});

    assertEqual(applied, [{ playerIndex: 1, action: { type: 'TAKE_GEMS_CONFIRMED', colors: [0, 1, 2] } }],
      'the ordinary move went through');
    assertEqual(resigned, [], 'the seat was not resigned over the pending choice');
    assertEqual(room.gameState.currentPlayerIndex, 2, 'the turn moved on');

    // …and the next bot seat is driven normally afterwards.
    await sleep(25);
    assertEqual(socket.lastOf('ai_move_request').playerIndex, 2, 'the next bot seat was asked');
  });

  await test('a move that re-opens an orphaned noble choice is not a stall', async () => {
    // The real shape of the gap: the seat qualifies for two nobles, so every
    // completed gem take re-arms `_pendingTileChoice` and keeps the turn.
    freshBridge();
    const { room, applied, resigned } = makeBotWorld();
    const byReward = reward => ALL_CARDS.filter(card => card.reward === reward && card.tier === 1).slice(0, 4);
    room.gameState.players[1].cards = [...byReward(0), ...byReward(1), ...byReward(2)];
    const discount = [0, 0, 0, 0, 0];
    for (const card of room.gameState.players[1].cards) discount[card.reward]++;
    const qualifying = ALL_BONUS_TILES.filter(tile =>
      tile.requirement.every((need, color) => discount[color] >= need));
    assert(qualifying.length >= 2, 'fixture qualifies for two nobles');
    room.gameState.bonusTiles = qualifying.slice(0, 2);

    const { socket } = registerWorker('unit-worker', 'socket-orphan-2');
    aiBridge.maybeAct(room);
    await sleep(25);
    const first = socket.lastOf('ai_move_request');
    assertEqual(first.kind, 'MOVE');
    socket.trigger('ai_move_response', {
      requestId: first.requestId, action: { type: 'TAKE_GEMS', colors: [0, 1, 2] },
    }, () => {});

    assertEqual(applied.length, 1, 'the take was applied');
    assertEqual(room.gameState.turnAction, null, 'and left the turn action empty');
    assertEqual(room.gameState._pendingTileChoice, qualifying.slice(0, 2).map(tile => tile.id),
      'while arming a choice the engine will not accept');
    assertEqual(room.gameState.currentPlayerIndex, 1, 'the seat keeps the turn');
    assertEqual(resigned, [], 'and is not resigned for it');

    await sleep(25);
    assertEqual(socket.countOf('ai_move_request'), 2, 'the seat is asked to play on');
    const second = socket.lastOf('ai_move_request');
    assertEqual(second.kind, 'MOVE', 'still a MOVE, never an impossible TILE');
    assertEqual(second.playerIndex, 1);
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

  await test('cancels an in-flight request when the position moves on', async () => {
    freshBridge({ delayMs: 1, deadlineMs: 5000 });
    const { room, applied, resigned } = makeBotWorld();
    const { socket } = registerWorker('unit-worker', 'socket-supersede');

    aiBridge.maybeAct(room);
    await sleep(25);
    const first = socket.lastOf('ai_move_request');
    assertEqual(first.playerIndex, 1, 'the bot seat to move was asked');

    // A seat that is NOT to move resigning leaves the request alone.
    processResign(room.gameState, 0);
    aiBridge.maybeAct(room);
    await sleep(20);
    assertEqual(socket.countOf('ai_move_cancel'), 0, 'the position did not change');
    assertEqual(socket.countOf('ai_move_request'), 1, 'and no second request went out');

    // …but the bot seat itself being eliminated (timeout / resign) does: the
    // turn is now seat 2's and the pending request can never be answered.
    processResign(room.gameState, 1);
    assertEqual(room.gameState.currentPlayerIndex, 2, 'the turn moved to the other bot');
    aiBridge.maybeAct(room);

    assertEqual(socket.countOf('ai_move_cancel'), 1, 'the worker was told to stop');
    assertEqual(socket.lastOf('ai_move_cancel'), { requestId: first.requestId });

    // A late answer for the superseded request is refused, not applied.
    let ack = null;
    socket.trigger('ai_move_response', {
      requestId: first.requestId, action: { type: 'TAKE_GEMS', colors: [0, 1, 2] },
    }, response => { ack = response; });
    assertEqual(ack, { error: 'Unknown or expired request' }, 'the stale answer is ignored');
    assertEqual(applied, [], 'nothing was applied for the eliminated seat');

    // …and the seat that is really to move gets its own request.
    await sleep(25);
    assertEqual(socket.countOf('ai_move_request'), 2, 'a new request went out');
    const second = socket.lastOf('ai_move_request');
    assertEqual(second.playerIndex, 2, 'for the current seat');
    assert(second.requestId !== first.requestId, 'with a new request id');

    socket.trigger('ai_move_response', {
      requestId: second.requestId, action: { type: 'TAKE_GEMS', colors: [0, 1, 2] },
    }, () => {});
    assertEqual(applied, [{ playerIndex: 2, action: { type: 'TAKE_GEMS_CONFIRMED', colors: [0, 1, 2] } }],
      'the new seat played');
    assertEqual(resigned, [], 'no bot seat was resigned by the bridge');
    assertEqual(aiBridge._inFlightCount(), 0, 'released');
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
