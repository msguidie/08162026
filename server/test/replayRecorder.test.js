// Unit tests for server/replayRecorder.js — recording semantics (contract §2)
// and a recorder → engine round trip (a recorded game must reconstruct to the
// exact state the live rules produced).

const { suite, test, assert, assertEqual } = require('./harness');
const recorder = require('../replayRecorder');
const store = require('../replayStore');
const { reconstruct } = require('../replayEngine');
const {
  createInitialGameState,
  clientView,
  processAction,
  processResign,
  calculateRatingChanges,
} = require('../gameLogic');

let roomCounter = 0;

function makeRoom(options = {}) {
  const n = options.n || 2;
  const teams = options.teams || null;
  const playerInfos = Array.from({ length: n }, (_, i) => ({
    username: `p${i}`,
    socketId: `socket-${i}`,
    avatarSeed: 100 + i,
    ...(teams ? { teamId: teams[i] } : {}),
  }));
  const gameState = createInitialGameState(playerInfos, {
    gameMode: options.gameMode || 'INDIVIDUAL',
    teamLayout: options.teamLayout,
    unlimitedTime: options.unlimitedTime !== false,
    firstPlayerIndex: options.firstPlayerIndex,
  });
  return {
    id: `game-${Date.now()}-t${roomCounter++}`,
    playerSockets: playerInfos.map((p, i) => ({
      socketId: p.socketId,
      username: p.username,
      playerIndex: i,
      ...(options.aiSeats && options.aiSeats.includes(i) ? { isAI: true } : {}),
    })),
    gameState,
    lastPlayerAction: Date.now(),
    created: Date.now(),
    resultsApplied: false,
    ratingChanges: null,
  };
}

// Minimal room shape for result-block tests (no real rules needed).
function makeSyntheticRoom(gameState, ratingChanges) {
  return {
    id: `game-${Date.now()}-s${roomCounter++}`,
    playerSockets: gameState.players.map((p, i) => ({ socketId: `s${i}`, username: p.username, playerIndex: i })),
    gameState,
    created: Date.now(),
    ratingChanges,
  };
}

function syntheticState(players, extra = {}) {
  return {
    phase: 'GAME_OVER',
    board: [[], [], []],
    decks: [[], [], []],
    bonusTiles: [],
    players,
    currentPlayerIndex: 0,
    numPlayers: players.length,
    resignedPlayers: [],
    gameMode: 'INDIVIDUAL',
    teamLayout: null,
    gameResult: null,
    timeControl: null,
    ...extra,
  };
}

function player(username, score, cardCount, teamId) {
  return {
    username,
    score,
    cards: Array.from({ length: cardCount }, (_, i) => ({ id: i })),
    reserved: [],
    bonusTiles: [],
    gems: [0, 0, 0, 0, 0, 0],
    avatarSeed: 1,
    ...(teamId === undefined ? {} : { teamId }),
  };
}

// ── scripted legal play against the real rules ──

function clone(state) {
  return structuredClone(state);
}

function legal(state, index, action) {
  return !processAction(clone(state), index, action).error;
}

function gemCandidates() {
  const out = [];
  for (let a = 0; a < 5; a++) {
    out.push([a, a]);
    out.push([a]);
    for (let b = a + 1; b < 5; b++) {
      for (let c = b + 1; c < 5; c++) out.push([a, b, c]);
      out.push([a, b]);
    }
  }
  return out;
}

// Mirrors index.js: processAction → replayRecorder.onActionResult →
// (broadcast, which moves _tileClaimed off the state).
function applyAndRecord(room, index, action) {
  const result = processAction(room.gameState, index, action);
  if (result.error) return null;
  recorder.onActionResult(room, result.result);
  if (room.gameState._tileClaimed) delete room.gameState._tileClaimed;
  return result;
}

function playScriptedGame(room, maxTurns = 400) {
  const state = room.gameState;
  let guard = 0;
  while (state.phase === 'PLAYING' && guard++ < maxTurns) {
    const index = state.currentPlayerIndex;

    if (state._pendingTileChoice && state._pendingTileChoice.length > 0) {
      applyAndRecord(room, index, { type: 'CHOOSE_TILE', tileId: state._pendingTileChoice[0] });
      continue;
    }

    // 1. buy anything affordable (keeps the game moving toward an ending)
    const buyable = [
      ...state.players[index].reserved.map(card => ({ cardId: card.id, source: 'reserved' })),
      ...state.board.flat().map(card => ({ cardId: card.id, source: 'board' })),
    ].filter(candidate => legal(state, index, { type: 'BUY_CARD', ...candidate }));
    if (buyable.length > 0) {
      const pick = buyable[guard % buyable.length];
      applyAndRecord(room, index, { type: 'BUY_CARD', ...pick });
      continue;
    }

    // 2. take gems — alternate the mobile and desktop client flows
    const takes = gemCandidates().filter(colors => legal(state, index, { type: 'TAKE_GEMS_CONFIRMED', colors }));
    if (takes.length > 0) {
      const colors = takes[guard % takes.length];
      if (guard % 2 === 0) {
        applyAndRecord(room, index, { type: 'TAKE_GEMS_CONFIRMED', colors });
      } else {
        for (const color of colors) applyAndRecord(room, index, { type: 'SELECT_GEM', color });
      }
      continue;
    }

    // 3. reserve
    if (legal(state, index, { type: 'ENTER_RESERVE' })) {
      applyAndRecord(room, index, { type: 'ENTER_RESERVE' });
      const boardCard = state.board.flat()[0];
      if (boardCard && legal(state, index, { type: 'RESERVE_CARD', cardId: boardCard.id })) {
        applyAndRecord(room, index, { type: 'RESERVE_CARD', cardId: boardCard.id });
        continue;
      }
      const tier = [1, 2, 3].find(t => legal(state, index, { type: 'RESERVE_FROM_DECK', tier: t }));
      if (tier) {
        applyAndRecord(room, index, { type: 'RESERVE_FROM_DECK', tier });
        continue;
      }
    }

    // 4. no legal action left — resign, exactly like a stuck player would
    processResign(state, index);
    recorder.onResign(room, index);
    if (state.numPlayers - state.resignedPlayers.length < 2) state.phase = 'GAME_OVER';
  }
  return guard;
}

function finishRoom(room) {
  room.ratingChanges = calculateRatingChanges(room.gameState.players, room.gameState);
  return recorder.finish(room);
}

function assertRoundTrip(room, json) {
  const { frames } = reconstruct(json);
  assertEqual(frames.length, json.actions.length + 1, 'frames === actions + 1');
  const live = clientView(room.gameState);
  const replayed = frames[frames.length - 1].state;
  assertEqual(replayed.players.map(p => p.score), live.players.map(p => p.score), 'scores');
  assertEqual(replayed.players.map(p => p.cards.length), live.players.map(p => p.cards.length), 'card counts');
  assertEqual(replayed.phase, live.phase, 'phase');
  assertEqual(replayed, live, 'full final state');
}

async function run() {
  suite('replayRecorder — begin');

  await test('captures setup, seats and clock from the live room', () => {
    const room = makeRoom({ n: 3, firstPlayerIndex: 2, aiSeats: [1] });
    const recording = recorder.begin(room);
    assertEqual(recording.v, 1);
    assertEqual(recording.id, room.id);
    assertEqual(recording.t, room.created);
    assertEqual(recording.mode, 'INDIVIDUAL');
    assertEqual(recording.layout, null);
    assertEqual(recording.n, 3);
    assertEqual(recording.clock, false, 'unlimitedTime rooms have no clock');
    assertEqual(recording.first, 2);
    assertEqual(recording.setup.board, room.gameState.board.map(row => row.map(c => c.id)));
    assertEqual(recording.setup.decks, room.gameState.decks.map(d => d.map(c => c.id)));
    assertEqual(recording.setup.tiles, room.gameState.bonusTiles.map(t => t.id));
    assertEqual(recording.players, [
      { u: 'p0', a: 100, ai: false },
      { u: 'p1', a: 101, ai: true },
      { u: 'p2', a: 102, ai: false },
    ], 'isAI is read defensively from playerSockets');
    assertEqual(recording.actions, []);
    recorder.discard(room);
  });

  await test('records the clock flag for timed three-player games', () => {
    const room = makeRoom({ n: 3, unlimitedTime: false });
    const recording = recorder.begin(room);
    assertEqual(recording.clock, true);
    assertEqual(recording.first, room.gameState.currentPlayerIndex);
    recorder.discard(room);
  });

  await test('team games carry teamId and layout', () => {
    const room = makeRoom({ n: 4, gameMode: 'TEAM', teamLayout: 'OPPOSITE', teams: [0, 1, 0, 1] });
    const recording = recorder.begin(room);
    assertEqual(recording.mode, 'TEAM');
    assertEqual(recording.layout, 'OPPOSITE');
    assertEqual(recording.players.map(p => p.team), [0, 1, 0, 1]);
    recorder.discard(room);
  });

  suite('replayRecorder — action mapping (contract §2)');

  await test('maps completed actions and skips the rest', () => {
    const room = makeRoom({ n: 2 });
    recorder.begin(room);
    const record = result => recorder.onActionResult(room, result);

    assertEqual(record({ type: 'SELECT_GEM', actingPlayer: 0, payload: { selected: [1], completed: false } }), null);
    assertEqual(record({ type: 'SELECT_GEM', actingPlayer: 0, payload: { selected: [1, 2, 3] } }), [0, 'G', [1, 2, 3]]);
    assertEqual(record({ type: 'TAKE_GEMS_CONFIRMED', actingPlayer: 1, payload: { selected: [0, 0] } }), [1, 'G', [0, 0]]);
    assertEqual(record({ type: 'ENTER_RESERVE', actingPlayer: 0, payload: { goldTaken: true } }), null);
    assertEqual(record({ type: 'RESERVE_CARD', actingPlayer: 0, payload: { cardId: 37, tier: 1, fromDeck: false } }), [0, 'R', 37]);
    assertEqual(record({ type: 'RESERVE_FROM_DECK', actingPlayer: 1, payload: { tier: 2, fromDeck: true } }), [1, 'RD', 2]);
    assertEqual(record({ type: 'BUY_CARD', actingPlayer: 0, payload: { cardId: 12, source: 'board' } }), [0, 'B', 12, 'b']);
    assertEqual(record({ type: 'BUY_CARD', actingPlayer: 1, payload: { cardId: 13, source: 'reserved' } }), [1, 'B', 13, 'r']);
    assertEqual(record({ type: 'CHOOSE_TILE', actingPlayer: 0, payload: { tileId: 4, playerIndex: 0 } }), [0, 'N', 4]);
    assertEqual(record({ type: 'CANCEL_GEMS', actingPlayer: 0, payload: {} }), null);
    assertEqual(record({ type: 'NONSENSE', actingPlayer: 0, payload: {} }), null);
    assertEqual(recorder.onResign(room, 1), [1, 'X']);
    assertEqual(recorder.onTimeout(room, 0), [0, 'T']);

    recorder.discard(room);
  });

  await test('ignores rooms that were never started or already discarded', () => {
    const room = makeRoom({ n: 2 });
    assertEqual(recorder.onActionResult(room, { type: 'TAKE_GEMS_CONFIRMED', actingPlayer: 0, payload: { selected: [0] } }), null);
    assertEqual(recorder.onResign(room, 0), null);
    assertEqual(recorder.onTimeout(room, 0), null);
    assertEqual(recorder.finish(room), null);
    assertEqual(recorder.discard(room), false);
  });

  await test('never throws on malformed input', () => {
    assertEqual(recorder.begin(null), null);
    assertEqual(recorder.begin({ id: 'x' }), null);
    assertEqual(recorder.onActionResult(null, null), null);
    assertEqual(recorder.onActionResult({ id: 'x' }, undefined), null);
    assertEqual(recorder.onResign(null, 0), null);
    assertEqual(recorder.onTimeout({ id: 'x' }, 'nope'), null);
    assertEqual(recorder.finish(null), null);
    assertEqual(recorder.discard(null), false);
  });

  suite('replayRecorder — result block');

  await test('INDIVIDUAL winners share the top rank among non-resigned seats', () => {
    const state = syntheticState([
      player('a', 10, 5), player('b', 10, 4), player('c', 10, 4), player('d', 0, 0),
    ], { resignedPlayers: [3] });
    const room = makeSyntheticRoom(state, [3, 5, 5, 0]);
    recorder.begin(room);
    const json = recorder.finish(room);
    assertEqual(json.result.winners, [1, 2], 'fewest cards wins the tie');
    assertEqual(json.result.scores, [10, 10, 10, 0]);
    assertEqual(json.result.cards, [5, 4, 4, 0]);
    assertEqual(json.result.resigned, [3]);
    assertEqual(json.result.winningTeamIds, null);
    assertEqual(json.result.reason, 'SCORE');
    assertEqual(json.result.rating, [3, 5, 5, 0]);
  });

  await test('an INDIVIDUAL game with one player left is a forfeit', () => {
    const state = syntheticState([player('a', 4, 3), player('b', 0, 0)], { resignedPlayers: [1] });
    const room = makeSyntheticRoom(state, [5, 0]);
    recorder.begin(room);
    const json = recorder.finish(room);
    assertEqual(json.result.reason, 'FORFEIT');
    assertEqual(json.result.winners, [0]);
  });

  await test('team modes carry gameResult and no individual winners', () => {
    const state = syntheticState([
      player('a', 18, 9, 0), player('b', 6, 4, 1), player('c', 14, 8, 1),
    ], {
      gameMode: 'ONE_V_TWO',
      resignedPlayers: [],
      gameResult: { reason: 'SCORE', winningTeamIds: [1] },
    });
    const room = makeSyntheticRoom(state, [0, 5, 5]);
    recorder.begin(room);
    const json = recorder.finish(room);
    assertEqual(json.result.winners, null);
    assertEqual(json.result.winningTeamIds, [1]);
    assertEqual(json.result.reason, 'SCORE');
    assertEqual(json.result.rating, [0, 5, 5]);
  });

  await test('a missing ratingChanges falls back to zeroes instead of throwing', () => {
    const state = syntheticState([player('a', 1, 1), player('b', 2, 1)]);
    const room = makeSyntheticRoom(state, null);
    recorder.begin(room);
    const json = recorder.finish(room);
    assertEqual(json.result.rating, [0, 0]);
  });

  suite('replayRecorder — lifecycle');

  await test('finish stores the replay and clears the recording', async () => {
    const state = syntheticState([player('a', 3, 2), player('b', 1, 1)]);
    const room = makeSyntheticRoom(state, [5, 0]);
    recorder.begin(room);
    const before = recorder.activeCount();
    const json = recorder.finish(room);
    assert(json !== null, 'finish returns the stored JSON');
    assertEqual(recorder.activeCount(), before - 1);
    assertEqual(recorder.finish(room), null, 'finish is idempotent');
    assertEqual(await store.getReplay(room.id), json, 'replay reached the store');
    assertEqual(json.e > 0, true, 'end timestamp recorded');
  });

  await test('discard drops the recording so nothing is stored', async () => {
    const room = makeRoom({ n: 2 });
    recorder.begin(room);
    recorder.onResign(room, 0);
    assertEqual(recorder.discard(room), true);
    assertEqual(recorder.finish(room), null);
    assertEqual(await store.getReplay(room.id), null);
  });

  suite('replayRecorder — round trip through replayEngine');

  await test('a scripted 2-player game reconstructs to the identical final state', () => {
    const room = makeRoom({ n: 2 });
    recorder.begin(room);
    playScriptedGame(room);
    assertEqual(room.gameState.phase, 'GAME_OVER');
    const json = finishRoom(room);
    assert(json.actions.length > 10, `expected a real game, got ${json.actions.length} actions`);
    assertRoundTrip(room, json);
  });

  await test('a scripted 1v2 game reconstructs to the identical final state', () => {
    const room = makeRoom({ n: 3, gameMode: 'ONE_V_TWO', teams: [0, 1, 1] });
    recorder.begin(room);
    playScriptedGame(room);
    const json = finishRoom(room);
    assertEqual(json.mode, 'ONE_V_TWO');
    assertEqual(json.first, 0, 'the solo seat always opens');
    assertRoundTrip(room, json);
  });

  await test('a scripted 2v2 game reconstructs to the identical final state', () => {
    const room = makeRoom({ n: 4, gameMode: 'TEAM', teamLayout: 'OPPOSITE', teams: [0, 1, 0, 1] });
    recorder.begin(room);
    playScriptedGame(room);
    const json = finishRoom(room);
    assertEqual(json.layout, 'OPPOSITE');
    assertRoundTrip(room, json);
  });

  await test('onTimeout is replayed exactly like the live elimination', () => {
    const room = makeRoom({ n: 3 });
    recorder.begin(room);
    const state = room.gameState;

    // A few normal turns, then eliminate the active seat like
    // index.js → eliminateTimedOutPlayer does.
    for (let i = 0; i < 3; i++) {
      const index = state.currentPlayerIndex;
      const colors = gemCandidates().find(c => legal(state, index, { type: 'TAKE_GEMS_CONFIRMED', colors: c }));
      applyAndRecord(room, index, { type: 'TAKE_GEMS_CONFIRMED', colors });
    }
    const timedOut = state.currentPlayerIndex;
    processResign(state, timedOut);
    recorder.onTimeout(room, timedOut);
    if (state.numPlayers - state.resignedPlayers.length < 2) state.phase = 'GAME_OVER';
    assertEqual(state.phase, 'PLAYING', 'two seats remain');

    playScriptedGame(room);
    const json = finishRoom(room);
    assertEqual(json.actions[3], [timedOut, 'T']);
    assert(json.result.resigned.includes(timedOut), 'timed out seat is listed as resigned');
    assertRoundTrip(room, json);
  });
}

module.exports = { run };
