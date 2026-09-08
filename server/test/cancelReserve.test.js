// CANCEL_RESERVE: backing out of reserve mode before a card is picked.

const { suite, test, assert, assertEqual } = require('./harness');
const { createInitialGameState, processAction, totalGems } = require('../gameLogic');

function newGame() {
  return createInitialGameState(
    [{ username: 'a', avatarSeed: 1 }, { username: 'b', avatarSeed: 2 }],
    { firstPlayerIndex: 0, unlimitedTime: true },
  );
}

function run() {
  suite('CANCEL_RESERVE', () => {
    test('returns the gold and clears the turn action', () => {
      const state = newGame();
      const goldBefore = state.gems[5];

      const entered = processAction(state, 0, { type: 'ENTER_RESERVE' });
      assert(entered.ok, 'ENTER_RESERVE should succeed');
      assertEqual(entered.result.payload.goldTaken, true, 'gold taken');
      assertEqual(state.gems[5], goldBefore - 1, 'supply gold down by one');
      assertEqual(state.players[0].gems[5], 1, 'player holds the gold');

      const cancelled = processAction(state, 0, { type: 'CANCEL_RESERVE' });
      assert(cancelled.ok, 'CANCEL_RESERVE should succeed');
      assertEqual(cancelled.completed, false, 'the turn is not consumed');
      assertEqual(cancelled.result.payload.goldReturned, true, 'gold reported returned');
      assertEqual(state.gems[5], goldBefore, 'supply gold restored');
      assertEqual(state.players[0].gems[5], 0, 'player gave the gold back');
      assertEqual(state.turnAction, null, 'turn action cleared');
      assertEqual(state.currentPlayerIndex, 0, 'still the same player to act');
      assertEqual(state.turnNumber, 0, 'turn number untouched');
    });

    test('the player can act normally afterwards', () => {
      const state = newGame();
      processAction(state, 0, { type: 'ENTER_RESERVE' });
      processAction(state, 0, { type: 'CANCEL_RESERVE' });

      const took = processAction(state, 0, { type: 'TAKE_GEMS_CONFIRMED', colors: [0, 1, 2] });
      assert(took.ok, 'taking gems after cancelling should succeed');
      assertEqual(totalGems(state.players[0]), 3, 'exactly the three gems, no leftover gold');
      assertEqual(state.currentPlayerIndex, 1, 'turn advanced');
    });

    test('is a no-op error outside reserve mode', () => {
      const state = newGame();
      assertEqual(processAction(state, 0, { type: 'CANCEL_RESERVE' }).error, 'Not in reserve mode',
        'rejected with nothing to cancel');

      processAction(state, 0, { type: 'SELECT_GEM', color: 0 });
      assertEqual(processAction(state, 0, { type: 'CANCEL_RESERVE' }).error, 'Not in reserve mode',
        'rejected during a gem selection');
    });

    test('no gold is returned when none was available', () => {
      const state = newGame();
      state.gems[5] = 0;
      const entered = processAction(state, 0, { type: 'ENTER_RESERVE' });
      assertEqual(entered.result.payload.goldTaken, false, 'no gold to take');

      const cancelled = processAction(state, 0, { type: 'CANCEL_RESERVE' });
      assertEqual(cancelled.result.payload.goldReturned, false, 'nothing to give back');
      assertEqual(state.gems[5], 0, 'supply unchanged');
      assertEqual(state.players[0].gems[5], 0, 'player unchanged');
      assertEqual(state.turnAction, null, 'turn action cleared');
    });

    test('a cancelled reserve is not recorded in the replay', () => {
      const { compactFromActionResult } = require('../replayRecorder');
      assertEqual(compactFromActionResult({
        type: 'CANCEL_RESERVE', actingPlayer: 0, payload: { goldReturned: true },
      }), null, 'CANCEL_RESERVE produces no replay entry');
    });
  });
}

module.exports = { run };
