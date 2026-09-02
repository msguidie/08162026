// Unit tests for the deterministic fallback policy (docs/AI_BRIDGE.md §2).
// Every state here is a real `createInitialGameState` result with the board,
// hand and reserve overwritten, so the black-box validation runs against the
// real rules engine.

const { suite, test, assert, assertEqual } = require('./harness');
const {
  createInitialGameState, ALL_CARDS, ALL_BONUS_TILES, processAction,
} = require('../gameLogic');
const aiFallback = require('../aiFallback');

function baseState() {
  const state = createInitialGameState(
    [{ username: 'bot', avatarSeed: 1 }, { username: 'human', avatarSeed: 2 }],
    { gameMode: 'INDIVIDUAL', unlimitedTime: true },
  );
  state.currentPlayerIndex = 0;
  state.roundStartPlayer = 0;
  state.turnNumber = 0;
  state.board = [[], [], []];
  state.decks = [[], [], []];
  state.deckCounts = [0, 0, 0];
  state.players[0].gems = [0, 0, 0, 0, 0, 0];
  state.players[0].cards = [];
  state.players[0].reserved = [];
  state.bonusTiles = [];
  return state;
}

const card = id => ALL_CARDS.find(c => c.id === id);
const cardsOfTier = tier => ALL_CARDS.filter(c => c.tier === tier);

async function run() {
  suite('aiFallback — greedy policy');

  await test('buys the affordable card with the most points', () => {
    const state = baseState();
    // A 5-point tier-3 card and a 0-point tier-1 card, both affordable.
    const rich = cardsOfTier(3).find(c => c.points === 5);
    const cheap = cardsOfTier(1).find(c => c.points === 0 && c.cost.reduce((a, b) => a + b, 0) === 4);
    state.board[2] = [rich];
    state.board[0] = [cheap];
    state.players[0].gems = [7, 7, 7, 7, 7, 0];

    const actions = aiFallback.chooseFallbackActions(state, 0);
    assertEqual(actions, [{ type: 'BUY_CARD', cardId: rich.id, source: 'board' }], 'greedy buy');
    assert(aiFallback.isLegalSequence(state, 0, actions), 'the chosen buy is legal');
  });

  await test('prefers a reserved card over an equally valuable board card', () => {
    const state = baseState();
    const [first, second] = cardsOfTier(3).filter(c => c.points === 4).slice(0, 2);
    state.board[2] = [first];
    state.players[0].reserved = [second];
    state.players[0].gems = [7, 7, 7, 7, 7, 0];

    const actions = aiFallback.chooseFallbackActions(state, 0);
    assertEqual(actions, [{ type: 'BUY_CARD', cardId: second.id, source: 'reserved' }], 'reserved first');
  });

  await test('takes a legal number of gems right below the ten-token cap', () => {
    const state = baseState();
    // 9 tokens in hand: exactly one more gem may be taken.
    state.players[0].gems = [3, 3, 3, 0, 0, 0];
    state.board[2] = [cardsOfTier(3).find(c => c.points === 5)];   // unaffordable
    const actions = aiFallback.chooseFallbackActions(state, 0);
    assertEqual(actions.length, 1, 'a single protocol action');
    assertEqual(actions[0].type, 'TAKE_GEMS_CONFIRMED');
    assertEqual(actions[0].colors.length, 1, 'only one gem fits');
    assert(aiFallback.isLegalSequence(state, 0, actions), 'the take is legal');
  });

  await test('takes two gems with eight tokens in hand and three with none', () => {
    const eight = baseState();
    eight.players[0].gems = [3, 3, 2, 0, 0, 0];
    eight.board[2] = [cardsOfTier(3).find(c => c.points === 5)];
    const twoActions = aiFallback.chooseFallbackActions(eight, 0);
    assertEqual(twoActions[0].type, 'TAKE_GEMS_CONFIRMED');
    assertEqual(twoActions[0].colors.length, 2, 'two gems fit');
    assert(aiFallback.isLegalSequence(eight, 0, twoActions), 'legal');

    const empty = baseState();
    empty.board[2] = [cardsOfTier(3).find(c => c.points === 5)];
    const threeActions = aiFallback.chooseFallbackActions(empty, 0);
    assertEqual(threeActions[0].type, 'TAKE_GEMS_CONFIRMED');
    assertEqual(threeActions[0].colors.length, 3, 'three distinct gems');
    assertEqual(new Set(threeActions[0].colors).size, 3, 'distinct');
    assert(aiFallback.isLegalSequence(empty, 0, threeActions), 'legal');
  });

  await test('aims the gem take at the cheapest attractive board card', () => {
    const state = baseState();
    // cost[4] === 3 only: the greedy take must include colour 4.
    const target = cardsOfTier(1).find(c => c.cost[4] === 3 && c.cost.reduce((a, b) => a + b, 0) === 3);
    state.board[0] = [target];
    const actions = aiFallback.chooseFallbackActions(state, 0);
    assertEqual(actions[0].type, 'TAKE_GEMS_CONFIRMED');
    assert(actions[0].colors.includes(4), `expected colour 4 in ${JSON.stringify(actions[0].colors)}`);
  });

  await test('reserves the best board card when no gem may be taken', () => {
    const state = baseState();
    state.gems = [0, 0, 0, 0, 0, 5];                 // no colour left in the supply
    const rich = cardsOfTier(3).find(c => c.points === 5);
    const poor = cardsOfTier(1).find(c => c.points === 0);
    state.board = [[poor], [], [rich]];
    const actions = aiFallback.chooseFallbackActions(state, 0);
    assertEqual(actions, [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_CARD', cardId: rich.id }], 'reserve best');
    assert(aiFallback.isLegalSequence(state, 0, actions), 'legal');
  });

  await test('finishes a half-open reserve turn without entering reserve twice', () => {
    const state = baseState();
    const target = cardsOfTier(2)[0];
    state.board[1] = [target];
    processAction(state, 0, { type: 'ENTER_RESERVE' });
    assertEqual(state.turnAction.type, 'RESERVE', 'mid-reserve fixture');
    const actions = aiFallback.chooseFallbackActions(state, 0);
    assertEqual(actions, [{ type: 'RESERVE_CARD', cardId: target.id }], 'completes the reserve');
  });

  await test('cancels a stray gem selection before acting', () => {
    const state = baseState();
    const rich = cardsOfTier(3).find(c => c.points === 5);
    state.board[2] = [rich];
    state.players[0].gems = [7, 7, 7, 7, 7, 0];
    processAction(state, 0, { type: 'SELECT_GEM', color: 0 });
    assertEqual(state.turnAction.type, 'TAKE_GEMS', 'stray selection fixture');
    const actions = aiFallback.chooseFallbackActions(state, 0);
    assertEqual(actions[0], { type: 'CANCEL_GEMS' }, 'clears the selection first');
    assertEqual(actions[1], { type: 'BUY_CARD', cardId: rich.id, source: 'board' });
    assert(aiFallback.isLegalSequence(state, 0, actions), 'legal');
  });

  await test('returns NONE when the seat is completely stuck', () => {
    const state = baseState();
    // Ten tokens (nothing may be taken), a full reserve (nothing may be
    // reserved) and only unaffordable cards on the board.
    state.players[0].gems = [2, 2, 2, 2, 2, 0];
    state.players[0].reserved = cardsOfTier(3).slice(0, 3);
    state.board[2] = cardsOfTier(3).slice(3, 7);
    state.deckCounts = [0, 0, 0];

    const choice = aiFallback.chooseFallbackActions(state, 0);
    assertEqual(choice, { type: 'NONE' }, 'no legal move');
    assert(aiFallback.isNone(choice), 'isNone');
  });

  await test('is a no-op for a seat that is not to move or a finished game', () => {
    const state = baseState();
    state.board[0] = [cardsOfTier(1)[0]];
    assertEqual(aiFallback.chooseFallbackActions(state, 1), { type: 'NONE' }, 'not this seat');
    const over = baseState();
    over.phase = 'GAME_OVER';
    assertEqual(aiFallback.chooseFallbackActions(over, 0), { type: 'NONE' }, 'game over');
  });

  suite('aiFallback — noble choice');

  await test('claims the first pending tile', () => {
    const state = baseState();
    // Four cards each of rewards 0, 1 and 2 qualify for exactly the two
    // "4 + 4" tiles over those colours — a genuine noble CHOICE.
    const byReward = reward => ALL_CARDS.filter(c => c.reward === reward && c.tier === 1).slice(0, 4);
    state.players[0].cards = [...byReward(0), ...byReward(1), ...byReward(2)];
    const discount = [0, 0, 0, 0, 0];
    for (const owned of state.players[0].cards) discount[owned.reward]++;
    const qualifying = ALL_BONUS_TILES.filter(tile =>
      tile.requirement.every((need, color) => discount[color] >= need));
    assert(qualifying.length >= 2, 'fixture has at least two qualifying tiles');
    state.bonusTiles = qualifying.slice(0, 2);
    state.turnAction = { type: 'BUY' };
    state._pendingTileChoice = state.bonusTiles.map(tile => tile.id);

    const actions = aiFallback.chooseFallbackActions(state, 0);
    assertEqual(actions, [{ type: 'CHOOSE_TILE', tileId: state._pendingTileChoice[0] }], 'first pending tile');
    assertEqual(aiFallback.chooseTileActions(state, 0), actions, 'same through chooseTileActions');
    assert(aiFallback.isLegalSequence(state, 0, actions), 'legal');
  });

  await test('simulate never mutates the state it is given', () => {
    const state = baseState();
    state.board[0] = [cardsOfTier(1)[0]];
    const before = JSON.stringify(state);
    aiFallback.simulate(state, 0, [{ type: 'TAKE_GEMS_CONFIRMED', colors: [0, 1, 2] }]);
    aiFallback.chooseFallbackActions(state, 0);
    assertEqual(JSON.stringify(state), before, 'state untouched');
  });
}

module.exports = { run };
