// Bonus tiles: individual games reveal n+1 tiles worth 3 / 2 / 1.5 points at
// 2 / 3 / 4 players; the team modes (1v2, 2v2) reveal three tiles worth 3.
// The value is stamped on every tile when the game is dealt, so scoring, the
// client and replays all read `tile.points`.

const { suite, test, assert, assertEqual } = require('./harness');
const {
  ALL_CARDS,
  ALL_BONUS_TILES,
  tilePointsFor,
  revealedTilesFor,
  createInitialGameState,
  processAction,
} = require('../gameLogic');
const { reconstruct } = require('../replayEngine');
const fixtures = require('./fixtures');

function makeGame(n, options = {}) {
  const players = [];
  for (let i = 0; i < n; i++) {
    players.push({
      username: `p${i}`,
      avatarSeed: i + 1,
      // 1v2: the solo is seat 0 (team 0), the duo seats 1-2 (team 1)
      ...(options.gameMode === 'ONE_V_TWO' ? { teamId: i === 0 ? 0 : 1 }
        : options.gameMode === 'TEAM' ? { teamId: i % 2 } : {}),
    });
  }
  return createInitialGameState(players, { firstPlayerIndex: 0, unlimitedTime: true, ...options });
}

// Zero-point tier-1 cards, so the score under test comes from the tile alone.
function cardsGranting(requirement) {
  const out = [];
  for (let color = 0; color < 5; color++) {
    const pool = ALL_CARDS.filter(c => c.tier === 1 && c.reward === color && c.points === 0);
    for (let k = 0; k < requirement[color]; k++) out.push(pool[k]);
  }
  return out;
}

// Give seat 0 exactly one tile's requirement, then end its turn: `advanceTurn`
// auto-claims a single qualifying tile.  No second tile can qualify — the 4/4
// tiles are distinct colour pairs and the 3/3/3 tiles distinct colour triples.
function claimOneTile(state) {
  const tile = state.bonusTiles[0];
  state.players[0].cards = cardsGranting(tile.requirement);
  const took = processAction(state, 0, { type: 'TAKE_GEMS_CONFIRMED', colors: [0, 1, 2] });
  assert(took.ok, `taking gems should succeed: ${took.error}`);
  assertEqual(state.players[0].bonusTiles.map(t => t.id), [tile.id], 'the tile was claimed');
  return tile;
}

async function run() {
  suite('bonus tile points — by player count');

  await test('individual: 3 / 2 / 1.5 points at 2 / 3 / 4 players; 1v2 and 2v2: 3', () => {
    assertEqual(tilePointsFor(2), 3, 'two players');
    assertEqual(tilePointsFor(3), 2, 'three players');
    assertEqual(tilePointsFor(4), 1.5, 'four players');
    assertEqual(tilePointsFor(3, 'ONE_V_TWO'), 3, '1v2');
    assertEqual(tilePointsFor(4, 'TEAM'), 3, '2v2');
    assertEqual(revealedTilesFor(2), 3, 'two players reveal 3');
    assertEqual(revealedTilesFor(3), 4, 'three players reveal 4');
    assertEqual(revealedTilesFor(4), 5, 'four players reveal 5');
    assertEqual(revealedTilesFor(3, 'ONE_V_TWO'), 3, '1v2 reveals 3');
    assertEqual(revealedTilesFor(4, 'TEAM'), 3, '2v2 reveals 3');
  });

  await test('every dealt tile carries the value of its own game', () => {
    const cases = [
      [makeGame(2), 3, 3, '2p individual'],
      [makeGame(3), 2, 4, '3p individual'],
      [makeGame(4), 1.5, 5, '4p individual'],
      [makeGame(3, { gameMode: 'ONE_V_TWO' }), 3, 3, '1v2'],
      [makeGame(4, { gameMode: 'TEAM', teamLayout: 'ADJACENT' }), 3, 3, '2v2 adjacent'],
      [makeGame(4, { gameMode: 'TEAM', teamLayout: 'OPPOSITE' }), 3, 3, '2v2 opposite'],
    ];
    for (const [state, points, count, label] of cases) {
      assertEqual(state.config.tilePoints, points, `${label}: config.tilePoints`);
      assertEqual(state.config.revealedTiles, count, `${label}: config.revealedTiles`);
      assertEqual(state.bonusTiles.length, count, `${label}: tiles revealed`);
      for (const tile of state.bonusTiles) {
        assertEqual(tile.points, points, `${label}: tile ${tile.id} value`);
      }
    }
  });

  await test('claiming a tile adds that value to the score', () => {
    const two = makeGame(2);
    claimOneTile(two);
    assertEqual(two.players[0].score, 3, 'two players: +3');

    const three = makeGame(3);
    claimOneTile(three);
    assertEqual(three.players[0].score, 2, 'three players: +2');

    const four = makeGame(4);
    claimOneTile(four);
    assertEqual(four.players[0].score, 1.5, 'four players: +1.5');

    const ovt = makeGame(3, { gameMode: 'ONE_V_TWO' });
    claimOneTile(ovt);
    assertEqual(ovt.players[0].score, 3, '1v2: +3');

    const team = makeGame(4, { gameMode: 'TEAM', teamLayout: 'ADJACENT' });
    claimOneTile(team);
    assertEqual(team.players[0].score, 3, '2v2: +3');
  });

  await test('two half tiles add up to a whole score at four players', () => {
    const state = makeGame(4);
    const first = claimOneTile(state);
    assertEqual(state.players[0].score, 1.5, 'one tile: half a point short of 2');

    // Reveal exactly one further tile so the second claim is automatic too,
    // and swap the tableau for that tile's requirement.
    const second = ALL_BONUS_TILES.find(t => t.id !== first.id);
    state.bonusTiles = [{ ...second, points: state.config.tilePoints }];
    state.players[0].cards = cardsGranting(second.requirement);
    for (const seat of [1, 2, 3]) {
      const turn = processAction(state, seat, { type: 'TAKE_GEMS_CONFIRMED', colors: [0, 1, 2] });
      assert(turn.ok, `seat ${seat} takes gems: ${turn.error}`);
    }
    const again = processAction(state, 0, { type: 'TAKE_GEMS_CONFIRMED', colors: [0, 1, 3] });
    assert(again.ok, `seat 0 takes gems again: ${again.error}`);

    assertEqual(state.players[0].bonusTiles.map(t => t.id), [first.id, second.id], 'both tiles');
    assertEqual(state.players[0].score, 3, 'two halves add up to a whole 3');
  });

  suite('bonus tile points — replays');

  await test('a replay scores nobles the way the recorded game did', () => {
    // v1 files carry no `setup.tp`: every game they recorded paid a flat 3.
    const legacy = JSON.parse(JSON.stringify(fixtures.teamGame));
    legacy.setup.tiles = [0, 5];
    assert(legacy.setup.tp === undefined, 'the fixture is a pre-rule replay');
    for (const tile of reconstruct(legacy).frames[0].state.bonusTiles) {
      assertEqual(tile.points, 3, 'a pre-rule 4p replay keeps its 3-point nobles');
    }

    // A game recorded now carries what its own nobles paid.
    const current = JSON.parse(JSON.stringify(legacy));
    current.setup.tp = 1.5;
    const rebuilt = reconstruct(current);
    const tiles = rebuilt.frames[0].state.bonusTiles;
    assertEqual(tiles.map(t => t.id), [0, 5], 'the replayed tiles');
    for (const tile of tiles) assertEqual(tile.points, 1.5, 'replayed 4p tile');
    assertEqual(rebuilt.frames[0].state.config.tilePoints, 1.5, 'and the config agrees');
  });

  await test('the recorder stores the value its game was dealt', () => {
    const recorder = require('../replayRecorder');
    const cases = [
      [makeGame(2), 3, '2p'], [makeGame(3), 2, '3p'], [makeGame(4), 1.5, '4p'],
      [makeGame(3, { gameMode: 'ONE_V_TWO' }), 3, '1v2'],
      [makeGame(4, { gameMode: 'TEAM', teamLayout: 'OPPOSITE' }), 3, '2v2'],
    ];
    for (const [state, expected, label] of cases) {
      const room = { id: `room-${label}`, gameState: state, created: 1725280000000,
                     playerSockets: state.players.map(() => ({})) };
      const recording = recorder.begin(room);
      assertEqual(recording.setup.tp, expected, `${label} recorded tile value`);
      assertEqual(recording.setup.tiles.length, state.config.revealedTiles, `${label} recorded tiles`);
      recorder.discard(room);
    }
  });
}

module.exports = { run };
