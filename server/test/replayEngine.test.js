// Unit tests for server/replayEngine.js — reconstruction of hand-built replays.

const { suite, test, assert, assertEqual, assertThrows } = require('./harness');
const fixtures = require('./fixtures');
const {
  reconstruct,
  buildInitialState,
  compactActionToProtocol,
  ReplayCorruptError,
} = require('../replayEngine');

function ids(cards) {
  return cards.map(card => card.id);
}

function frameWith(frames, predicate) {
  return frames.filter(predicate);
}

async function run() {
  suite('replayEngine — compactActionToProtocol');

  await test('maps every action code to the live protocol', () => {
    assertEqual(compactActionToProtocol([0, 'G', [1, 2]]).steps,
      [{ type: 'TAKE_GEMS_CONFIRMED', colors: [1, 2] }]);
    assertEqual(compactActionToProtocol([1, 'R', 37]).steps,
      [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_CARD', cardId: 37 }]);
    assertEqual(compactActionToProtocol([2, 'RD', 2]).steps,
      [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_FROM_DECK', tier: 2 }]);
    assertEqual(compactActionToProtocol([0, 'B', 12, 'b']).steps,
      [{ type: 'BUY_CARD', cardId: 12, source: 'board' }]);
    assertEqual(compactActionToProtocol([0, 'B', 12, 'r']).steps,
      [{ type: 'BUY_CARD', cardId: 12, source: 'reserved' }]);
    assertEqual(compactActionToProtocol([0, 'N', 4]).steps,
      [{ type: 'CHOOSE_TILE', tileId: 4 }]);
    assertEqual(compactActionToProtocol([1, 'X']).kind, 'RESIGN');
    assertEqual(compactActionToProtocol([2, 'T']).kind, 'TIMEOUT');
    assertEqual(compactActionToProtocol([2, 'Z']), null);
    assertEqual(compactActionToProtocol('nope'), null);
  });

  suite('replayEngine — buildInitialState');

  await test('rebuilds the exact setup and strips the clock', () => {
    const state = buildInitialState(fixtures.nobleGame);
    assertEqual(state.board.map(ids), fixtures.nobleGame.setup.board, 'board');
    assertEqual(state.decks.map(ids), fixtures.nobleGame.setup.decks, 'decks');
    assertEqual(state.deckCounts, [36, 3, 3], 'deckCounts');
    assertEqual(state.bonusTiles.map(tile => tile.id), [0, 4], 'tiles');
    assert(state.timeControl === null, 'timeControl must be null');
    assertEqual(state.currentPlayerIndex, 0);
    assertEqual(state.roundStartPlayer, 0);
    assertEqual(state.players.map(p => p.username), ['alice', 'bob']);
    assertEqual(state.gems, [4, 4, 4, 4, 4, 5], 'two-player supply');
  });

  await test('honours firstPlayerIndex outside 1v2 and forces seat 0 for 1v2', () => {
    assertEqual(buildInitialState(fixtures.timedOut).currentPlayerIndex, 1);
    assertEqual(buildInitialState(fixtures.teamGame).currentPlayerIndex, 2);
    const solo = buildInitialState(fixtures.oneVsTwo);
    assertEqual(solo.currentPlayerIndex, 0);
    assertEqual(solo.roundStartPlayer, 0);
    assertEqual(solo.gameMode, 'ONE_V_TWO');
    assertEqual(solo.players.map(p => p.teamId), [0, 1, 1]);
  });

  await test('does not mutate the shared card table between reconstructions', () => {
    const a = buildInitialState(fixtures.reserves);
    const b = buildInitialState(fixtures.reserves);
    assert(a.board[0][0] !== b.board[0][0], 'card objects must be per-replay copies');
    assertEqual(ids(a.board[0]), ids(b.board[0]));
  });

  suite('replayEngine — reconstruct (noble choice game)');

  const noble = reconstruct(fixtures.nobleGame);

  await test('produces actions.length + 1 frames', () => {
    assertEqual(noble.frames.length, fixtures.nobleGame.actions.length + 1);
    assertEqual(noble.frames.length, 62);
  });

  await test('frame 0 is the pre-game snapshot', () => {
    const frame = noble.frames[0];
    assertEqual(frame.i, 0);
    assertEqual(frame.turn, 0);
    assertEqual(frame.actor, null);
    assertEqual(frame.action, null);
    assertEqual(frame.result, null);
    assertEqual(frame.pendingTileChoice, null);
    assert(frame.state.decks === undefined, 'decks must be stripped from frame state');
    assertEqual(frame.state.deckCounts, [36, 3, 3]);
    assertEqual(frame.state.timeControl, null);
    assertEqual(frame.state.phase, 'PLAYING');
  });

  await test('meta follows contract §4', () => {
    assertEqual(noble.meta.mode, 'INDIVIDUAL');
    assertEqual(noble.meta.layout, null);
    assertEqual(noble.meta.n, 2);
    assertEqual(noble.meta.clock, false);
    assertEqual(noble.meta.first, 0);
    assertEqual(noble.meta.t, fixtures.nobleGame.t);
    assertEqual(noble.meta.e, fixtures.nobleGame.e);
    assertEqual(noble.meta.result, fixtures.nobleGame.result);
    assertEqual(noble.meta.players, [
      { username: 'alice', avatarSeed: 1, isAI: false },
      { username: 'bob', avatarSeed: 2, isAI: false },
    ]);
  });

  await test('turn numbers never go backwards', () => {
    for (let i = 1; i < noble.frames.length; i++) {
      assert(noble.frames[i].state.turnNumber >= noble.frames[i - 1].state.turnNumber,
        `turnNumber decreased at frame ${i}`);
      assertEqual(noble.frames[i].i, i, 'frame index');
      assertEqual(noble.frames[i].turn, noble.frames[i].state.turnNumber, 'frame.turn');
    }
  });

  await test('gem takes report the exact selection', () => {
    const frame = noble.frames[1];
    assertEqual(frame.actor, 0);
    assertEqual(frame.action, [0, 'G', [0, 0]]);
    assertEqual(frame.result.type, 'TAKE_GEMS_CONFIRMED');
    assertEqual(frame.result.actingPlayer, 0);
    assertEqual(frame.result.payload, { selected: [0, 0] });
    assertEqual(frame.state.players[0].gems, [2, 0, 0, 0, 0, 0]);
    assertEqual(frame.state.gems, [2, 4, 4, 4, 4, 5]);
  });

  await test('a simultaneous two-noble qualification becomes a pending choice', () => {
    const pending = frameWith(noble.frames, f => f.pendingTileChoice !== null);
    assertEqual(pending.length, 1, 'exactly one frame offers a noble choice');
    assertEqual(pending[0].pendingTileChoice, [0, 4]);
    assertEqual(pending[0].action, [0, 'B', 20, 'r']);
    assertEqual(pending[0].state.turnAction, { type: 'BUY' }, 'turn stays open for the choice');
    const chooseIndex = noble.frames.indexOf(pending[0]) + 1;
    const choice = noble.frames[chooseIndex];
    assertEqual(choice.action, [0, 'N', 0]);
    assertEqual(choice.result.type, 'CHOOSE_TILE');
    assertEqual(choice.result.payload, { tileId: 0, playerIndex: 0 });
    assertEqual(choice.pendingTileChoice, null);
    assertEqual(choice.state.players[0].bonusTiles.map(t => t.id), [0]);
  });

  await test('the leftover noble auto-claims on the next completed turn', () => {
    const claimed = frameWith(noble.frames, f => f.result && f.result.tileClaimed);
    assertEqual(claimed.length, 1, 'exactly one auto-claim');
    assertEqual(claimed[0].result.tileClaimed, { tileId: 4, playerIndex: 0 });
    assertEqual(claimed[0].state.players[0].bonusTiles.map(t => t.id), [0, 4]);
    assertEqual(claimed[0].state.bonusTiles, []);
    assert(claimed[0].state._tileClaimed === undefined, '_tileClaimed must be moved onto the result');
  });

  await test('reserved cards stay fully visible in replay frames', () => {
    const withReserve = frameWith(noble.frames, f => f.state.players.some(p => p.reserved.length > 0));
    assert(withReserve.length > 0, 'fixture should contain reserved cards');
    for (const frame of withReserve) {
      for (const player of frame.state.players) {
        for (const card of player.reserved) {
          assert(card.hidden === undefined, 'replay frames must not hide reserved cards');
          assert(card.id >= 0, 'reserved card must carry its real id');
        }
      }
    }
  });

  await test('buying out of the reserve is replayed as source "reserved"', () => {
    const frame = noble.frames.find(f => f.action && f.action[1] === 'B' && f.action[3] === 'r');
    assertEqual(frame.result.type, 'BUY_CARD');
    assertEqual(frame.result.payload.source, 'reserved');
    assert(Array.isArray(frame.result.payload.gemsReturned), 'gemsReturned drives the viewer animation');
  });

  await test('the final frame matches the stored result block', () => {
    const last = noble.frames[noble.frames.length - 1];
    assertEqual(last.state.phase, 'GAME_OVER');
    assertEqual(last.state.players.map(p => p.score), fixtures.nobleGame.result.scores);
    assertEqual(last.state.players.map(p => p.cards.length), fixtures.nobleGame.result.cards);
    assertEqual(last.state.resignedPlayers, [1]);
    assertEqual(last.result, {
      type: 'RESIGN',
      actingPlayer: 1,
      payload: { resignedPlayerIndex: 1 },
    });
  });

  suite('replayEngine — forced short gem takes');

  await test('replays takes that are cut short by the ten-token cap', () => {
    const { frames } = reconstruct(fixtures.forcedShortTakes);
    assertEqual(frames.length, fixtures.forcedShortTakes.actions.length + 1);
    assertEqual(frames[5].state.players[0].gems, [2, 1, 2, 2, 2, 0], 'nine tokens after three takes');
    assertEqual(frames[7].result.payload, { selected: [4] }, 'single-gem take completes the turn');
    assertEqual(frames[8].result.payload, { selected: [1] });
    assertEqual(frames[7].state.players[0].gems.reduce((a, b) => a + b, 0), 10);
    assertEqual(frames[8].state.players[1].gems.reduce((a, b) => a + b, 0), 10);
    assertEqual(frames[8].state.gems, [0, 0, 0, 0, 0, 5], 'colour supply is exhausted');
    const last = frames[frames.length - 1];
    assertEqual(last.state.phase, 'GAME_OVER');
    assertEqual(last.state.resignedPlayers, [0]);
  });

  suite('replayEngine — reserves');

  const reserved = reconstruct(fixtures.reserves);

  await test('deck reserve recomputes the hidden card and the gold', () => {
    const frame = reserved.frames[1];
    assertEqual(frame.result.type, 'RESERVE_FROM_DECK');
    assertEqual(frame.result.payload, { tier: 1, fromDeck: true, cardId: 25, goldTaken: true });
    assertEqual(ids(frame.state.players[0].reserved), [25]);
    assertEqual(frame.state.players[0].gems, [0, 0, 0, 0, 0, 1]);
    assertEqual(frame.state.deckCounts, [2, 2, 2]);
  });

  await test('face-up reserve refills the board from the deck', () => {
    const frame = reserved.frames[2];
    assertEqual(frame.result.type, 'RESERVE_CARD');
    assertEqual(frame.result.payload, { cardId: 0, tier: 1, fromDeck: false, goldTaken: true });
    assertEqual(ids(frame.state.board[0]), [5, 10, 15, 26], 'board refilled from the deck top');
    assertEqual(frame.state.deckCounts, [1, 2, 2]);
    assertEqual(ids(frame.state.players[1].reserved), [0]);
  });

  await test('a reserved card can be bought with gold', () => {
    const frame = reserved.frames[5];
    assertEqual(frame.result.type, 'BUY_CARD');
    assertEqual(frame.result.payload, {
      cardId: 25,
      source: 'reserved',
      reward: 0,
      points: 0,
      gemsReturned: [0, 0, 0, 0, 2, 1],
    });
    assertEqual(ids(frame.state.players[0].cards), [25]);
    assertEqual(frame.state.players[0].reserved, []);
    assertEqual(frame.state.players[0].gems, [0, 0, 0, 0, 0, 0]);
  });

  suite('replayEngine — resign / timeout / team endings');

  await test('a timeout is replayed through processResign', () => {
    const { frames } = reconstruct(fixtures.timedOut);
    assertEqual(frames.length, 5);
    const timeout = frames[2];
    assertEqual(timeout.result, {
      type: 'TIMEOUT',
      actingPlayer: 2,
      payload: { timedOutPlayerIndex: 2 },
    });
    assertEqual(timeout.state.resignedPlayers, [2]);
    assertEqual(timeout.state.currentPlayerIndex, 0, 'turn passes to the next active seat');
    assertEqual(timeout.state.phase, 'PLAYING');
    const last = frames[frames.length - 1];
    assertEqual(last.state.phase, 'GAME_OVER', 'one active player left ends the game');
    assertEqual(last.state.resignedPlayers, [2, 1]);
    assertEqual(last.result.type, 'RESIGN');
  });

  await test('1v2 forfeit resolves to the other team', () => {
    const { meta, frames } = reconstruct(fixtures.oneVsTwo);
    assertEqual(meta.players[0], { username: 'solo', avatarSeed: 1, teamId: 0, isAI: false });
    assertEqual(frames.map(f => f.state.currentPlayerIndex).slice(0, 4), [0, 1, 2, 0]);
    const last = frames[frames.length - 1];
    assertEqual(last.state.phase, 'GAME_OVER');
    assertEqual(last.state.gameResult, {
      reason: 'FORFEIT',
      forfeitingTeamId: 1,
      winningTeamIds: [0],
    });
  });

  await test('2v2 OPPOSITE seating and forfeit', () => {
    const { frames } = reconstruct(fixtures.teamGame);
    assertEqual(frames[0].state.teamLayout, 'OPPOSITE');
    assertEqual(frames[0].state.players.map(p => p.teamId), [0, 1, 0, 1]);
    assertEqual(frames.map(f => f.state.currentPlayerIndex).slice(0, 5), [2, 3, 0, 1, 2]);
    const last = frames[frames.length - 1];
    assertEqual(last.state.phase, 'GAME_OVER');
    assertEqual(last.state.gameResult.winningTeamIds, [1]);
    assertEqual(last.state.gameResult.reason, 'FORFEIT');
  });

  suite('replayEngine — corrupt replays');

  await test('an illegal action throws ReplayCorruptError with its index', async () => {
    const broken = fixtures.clone(fixtures.reserves);
    broken.actions[4] = [0, 'B', 71, 'b']; // tier-3 card nobody can afford
    const err = await assertThrows(() => reconstruct(broken));
    assert(err instanceof ReplayCorruptError, 'expected ReplayCorruptError');
    assertEqual(err.actionIndex, 4);
    assert(/BUY_CARD rejected/.test(err.message), err.message);
  });

  await test('an out-of-turn action is rejected', async () => {
    const broken = fixtures.clone(fixtures.forcedShortTakes);
    broken.actions[1] = [0, 'G', [3, 4]];
    const err = await assertThrows(() => reconstruct(broken));
    assertEqual(err.actionIndex, 1);
    assert(/Not your turn/.test(err.message), err.message);
  });

  await test('an unknown action code is rejected', async () => {
    const broken = fixtures.clone(fixtures.forcedShortTakes);
    broken.actions[2] = [0, 'Q', 1];
    const err = await assertThrows(() => reconstruct(broken));
    assertEqual(err.actionIndex, 2);
    assert(/Malformed action/.test(err.message), err.message);
  });

  await test('an unknown seat is rejected', async () => {
    const broken = fixtures.clone(fixtures.forcedShortTakes);
    broken.actions[0] = [9, 'G', [0, 1, 2]];
    const err = await assertThrows(() => reconstruct(broken));
    assertEqual(err.actionIndex, 0);
    assert(/unknown player/.test(err.message), err.message);
  });

  await test('a malformed header is rejected before any action', async () => {
    const noSetup = fixtures.clone(fixtures.reserves);
    delete noSetup.setup;
    const err = await assertThrows(() => reconstruct(noSetup));
    assertEqual(err.actionIndex, -1);

    const badMode = fixtures.clone(fixtures.reserves);
    badMode.mode = 'SOLITAIRE';
    assertEqual((await assertThrows(() => reconstruct(badMode))).actionIndex, -1);

    const badFirst = fixtures.clone(fixtures.reserves);
    badFirst.first = 5;
    assertEqual((await assertThrows(() => reconstruct(badFirst))).actionIndex, -1);

    const badCard = fixtures.clone(fixtures.reserves);
    badCard.setup.board[0][0] = 999;
    assertEqual((await assertThrows(() => reconstruct(badCard))).actionIndex, -1);
  });

  await test('resigning after the game is over is rejected', async () => {
    const broken = fixtures.clone(fixtures.forcedShortTakes);
    broken.actions.push([1, 'X']);
    const err = await assertThrows(() => reconstruct(broken));
    assertEqual(err.actionIndex, broken.actions.length - 1);
  });
}

module.exports = { run };
