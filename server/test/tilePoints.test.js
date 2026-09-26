// Bonus tiles are worth 3 points at two players, 2 at three and 1.5 at four
// (gameLogic.tilePointsFor).  The value is stamped on every tile when the game
// is dealt, so scoring, the client and replays all read `tile.points`.

const { suite, test, assert, assertEqual } = require('./harness');
const {
  ALL_CARDS,
  ALL_BONUS_TILES,
  tilePointsFor,
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
      ...(options.gameMode && options.gameMode !== 'INDIVIDUAL' ? { teamId: i % 2 } : {}),
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

  await test('tilePointsFor: 3 at two players, 2 at three, 1.5 at four', () => {
    assertEqual(tilePointsFor(2), 3, 'two players');
    assertEqual(tilePointsFor(3), 2, 'three players');
    assertEqual(tilePointsFor(4), 1.5, 'four players');
  });

  await test('every dealt tile carries the value of its own game', () => {
    const cases = [
      [makeGame(2), 3, '2p individual'],
      [makeGame(3), 2, '3p individual'],
      [makeGame(4), 1.5, '4p individual'],
      [makeGame(3, { gameMode: 'ONE_V_TWO' }), 2, '1v2'],
      [makeGame(4, { gameMode: 'TEAM', teamLayout: 'ADJACENT' }), 1.5, '2v2'],
    ];
    for (const [state, expected, label] of cases) {
      assertEqual(state.config.tilePoints, expected, `${label}: config.tilePoints`);
      assertEqual(state.bonusTiles.length, state.numPlayers + 1, `${label}: n+1 tiles revealed`);
      for (const tile of state.bonusTiles) {
        assertEqual(tile.points, expected, `${label}: tile ${tile.id} value`);
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

  await test('a replay rebuilds tiles at its own game size', () => {
    const two = reconstruct(fixtures.nobleGame);
    const twoTiles = two.frames[0].state.bonusTiles;
    assert(twoTiles.length > 0, 'the 2p fixture reveals tiles');
    for (const tile of twoTiles) assertEqual(tile.points, 3, 'replayed 2p tile');

    const four = JSON.parse(JSON.stringify(fixtures.teamGame));
    four.setup.tiles = [0, 5];
    const rebuilt = reconstruct(four);
    const fourTiles = rebuilt.frames[0].state.bonusTiles;
    assertEqual(fourTiles.map(t => t.id), [0, 5], 'the replayed tiles');
    for (const tile of fourTiles) assertEqual(tile.points, 1.5, 'replayed 4p tile');
  });
}

module.exports = { run };
