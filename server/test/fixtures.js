// Hand-built replay fixtures (docs/REPLAY_FORMAT.md §1).
//
// `nobleGame` was produced with a scripted greedy run over gameLogic.js and is
// checked in verbatim: player 0 buys four cards each of reward 1, 4 and 0, so
// tiles 0 (4x c0 + 4x c1) and 4 (4x c4 + 4x c0) qualify on the same purchase —
// a noble CHOICE — after which the leftover tile auto-claims on the next turn.
// Player 1 only buys reward 2/3 cards and can therefore never qualify.

const NOBLE_DECK_1 = [
  39, 36, 35, 34, 31, 30, 14, 11, 10, 9,
  6, 5, 33, 32, 13, 12, 8, 20, 7, 15,
  38, 0, 37, 25, 23, 24, 22, 19, 18, 4,
  17, 29, 3, 21, 2, 16,
];

const nobleGame = {
  v: 1,
  id: 'game-1725280000000-fixt',
  t: 1725280000000,
  e: 1725281800000,
  mode: 'INDIVIDUAL',
  layout: null,
  n: 2,
  clock: false,
  players: [
    { u: 'alice', a: 1, ai: false },
    { u: 'bob', a: 2, ai: false },
  ],
  first: 0,
  setup: {
    board: [[26, 27, 1, 28], [40, 41, 42, 43], [70, 71, 72, 73]],
    decks: [NOBLE_DECK_1, [44, 45, 46], [74, 75, 76]],
    tiles: [0, 4],
  },
  actions: [
    [0, 'G', [0, 0]], [1, 'G', [1, 1]], [0, 'G', [0, 1, 2]], [1, 'G', [0, 1, 2]],
    [0, 'B', 26, 'b'], [1, 'B', 27, 'b'], [0, 'G', [0, 1, 4]], [1, 'G', [0, 1, 2]],
    [0, 'B', 1, 'b'], [1, 'B', 28, 'b'], [0, 'G', [4, 4]], [1, 'B', 2, 'b'],
    [0, 'B', 16, 'b'], [1, 'G', [2, 2]], [0, 'G', [3, 3]], [1, 'B', 17, 'b'],
    [0, 'G', [0, 1, 4]], [1, 'G', [0, 1, 4]], [0, 'B', 21, 'b'], [1, 'B', 3, 'b'],
    [0, 'G', [3, 3]], [1, 'G', [0, 1, 3]], [0, 'G', [0, 1, 3]], [1, 'B', 18, 'b'],
    [0, 'B', 29, 'b'], [1, 'G', [4, 4]], [0, 'G', [1, 2, 3]], [1, 'B', 22, 'b'],
    [0, 'B', 4, 'b'], [1, 'G', [0, 2, 3]], [0, 'G', [4, 4]], [1, 'G', [0, 2, 3]],
    [0, 'B', 25, 'b'], [1, 'B', 37, 'b'], [0, 'G', [0, 2, 3]], [1, 'R', 23],
    [0, 'B', 19, 'b'], [1, 'G', [4, 4]], [0, 'B', 0, 'b'], [1, 'B', 7, 'b'],
    [0, 'G', [2, 3, 4]], [1, 'G', [3, 4]], [0, 'B', 15, 'b'], [1, 'B', 8, 'b'],
    [0, 'R', 24], [1, 'G', [0, 3, 4]], [0, 'B', 24, 'r'], [1, 'G', [4]],
    [0, 'G', [0, 2, 3]], [1, 'B', 13, 'b'], [0, 'R', 20], [1, 'G', [0, 4]],
    [0, 'R', 38], [1, 'R', 32], [0, 'B', 38, 'r'], [1, 'B', 32, 'r'],
    [0, 'B', 20, 'r'], [0, 'N', 0], [1, 'G', [0, 2, 4]], [0, 'G', [3, 4]],
    [1, 'X'],
  ],
  result: {
    scores: [7, 0],
    cards: [13, 0],
    resigned: [1],
    winners: [0],
    winningTeamIds: null,
    reason: 'FORFEIT',
    rating: [5, 0],
  },
};

// Two players fill up to the ten-token cap, so the last two takes are forced
// short (a single gem completes the turn).
const forcedShortTakes = {
  v: 1,
  id: 'game-1725280100000-shrt',
  t: 1725280100000,
  e: 1725280200000,
  mode: 'INDIVIDUAL',
  layout: null,
  n: 2,
  clock: false,
  players: [
    { u: 'alice', a: 1, ai: false },
    { u: 'bob', a: 2, ai: false },
  ],
  first: 0,
  setup: {
    board: [[0, 1, 2, 3], [40, 41, 42, 43], [70, 71, 72, 73]],
    decks: [[4, 5, 6], [44, 45], [74, 75]],
    tiles: [],
  },
  actions: [
    [0, 'G', [0, 1, 2]], [1, 'G', [0, 1, 2]],
    [0, 'G', [3, 4, 0]], [1, 'G', [3, 4, 1]],
    [0, 'G', [2, 3, 4]], [1, 'G', [0, 2, 3]],
    [0, 'G', [4]], [1, 'G', [1]],
    [0, 'X'],
  ],
  result: {
    scores: [0, 0],
    cards: [0, 0],
    resigned: [0],
    winners: [1],
    winningTeamIds: null,
    reason: 'FORFEIT',
    rating: [0, 5],
  },
};

// Deck reserve (RD, hidden card), face-up reserve (R, board refill) and a
// purchase out of the reserve.
const reserves = {
  v: 1,
  id: 'game-1725280200000-resv',
  t: 1725280200000,
  e: 1725280300000,
  mode: 'INDIVIDUAL',
  layout: null,
  n: 2,
  clock: false,
  players: [
    { u: 'alice', a: 1, ai: false },
    { u: 'bob', a: 2, ai: false },
  ],
  first: 0,
  setup: {
    board: [[0, 5, 10, 15], [40, 41, 42, 43], [70, 71, 72, 73]],
    decks: [[20, 26, 25], [44, 45], [74, 75]], // pop() → 25, then 26, then 20
    tiles: [],
  },
  actions: [
    [0, 'RD', 1],
    [1, 'R', 0],
    [0, 'G', [4, 4]],
    [1, 'G', [0, 1, 2]],
    [0, 'B', 25, 'r'],
    [1, 'X'],
  ],
  result: {
    scores: [0, 0],
    cards: [1, 0],
    resigned: [1],
    winners: [0],
    winningTeamIds: null,
    reason: 'FORFEIT',
    rating: [5, 0],
  },
};

// Three-player individual game where seat 2 is eliminated by the clock.
const timedOut = {
  v: 1,
  id: 'game-1725280300000-time',
  t: 1725280300000,
  e: 1725280400000,
  mode: 'INDIVIDUAL',
  layout: null,
  n: 3,
  clock: true,
  players: [
    { u: 'alice', a: 1, ai: false },
    { u: 'bob', a: 2, ai: false },
    { u: 'carol', a: 3, ai: false },
  ],
  first: 1,
  setup: {
    board: [[0, 1, 2, 3], [40, 41, 42, 43], [70, 71, 72, 73]],
    decks: [[4, 5, 6], [44, 45], [74, 75]],
    tiles: [],
  },
  actions: [
    [1, 'G', [0, 1, 2]],
    [2, 'T'],
    [0, 'G', [0, 1, 2]],
    [1, 'X'],
  ],
  result: {
    scores: [0, 0, 0],
    cards: [0, 0, 0],
    resigned: [1, 2],
    winners: [0],
    winningTeamIds: null,
    reason: 'FORFEIT',
    rating: [5, 0, 0],
  },
};

// 1v2: the solo seat is always index 0 and always opens.
const oneVsTwo = {
  v: 1,
  id: 'game-1725280400000-ovt1',
  t: 1725280400000,
  e: 1725280500000,
  mode: 'ONE_V_TWO',
  layout: null,
  n: 3,
  clock: false,
  players: [
    { u: 'solo', a: 1, team: 0, ai: false },
    { u: 'duoA', a: 2, team: 1, ai: false },
    { u: 'duoB', a: 3, team: 1, ai: false },
  ],
  first: 0,
  setup: {
    board: [[0, 1, 2, 3], [40, 41, 42, 43], [70, 71, 72, 73]],
    decks: [[4, 5, 6], [44, 45], [74, 75]],
    tiles: [],
  },
  actions: [
    [0, 'G', [0, 1, 2]],
    [1, 'G', [0, 1, 2]],
    [2, 'G', [0, 1, 2]],
    [1, 'X'],
  ],
  result: {
    scores: [0, 0, 0],
    cards: [0, 0, 0],
    resigned: [1],
    winners: null,
    winningTeamIds: [0],
    reason: 'FORFEIT',
    rating: [5, 0, 0],
  },
};

// 2v2 with the OPPOSITE seat layout: seats alternate teams.
const teamGame = {
  v: 1,
  id: 'game-1725280500000-team',
  t: 1725280500000,
  e: 1725280600000,
  mode: 'TEAM',
  layout: 'OPPOSITE',
  n: 4,
  clock: false,
  players: [
    { u: 'a0', a: 1, team: 0, ai: false },
    { u: 'b0', a: 2, team: 1, ai: false },
    { u: 'a1', a: 3, team: 0, ai: false },
    { u: 'b1', a: 4, team: 1, ai: false },
  ],
  first: 2,
  setup: {
    board: [[0, 1, 2, 3], [40, 41, 42, 43], [70, 71, 72, 73]],
    decks: [[4, 5, 6], [44, 45], [74, 75]],
    tiles: [],
  },
  actions: [
    [2, 'G', [0, 1, 2]],
    [3, 'G', [0, 1, 2]],
    [0, 'G', [0, 1, 2]],
    [1, 'G', [0, 1, 2]],
    [2, 'X'],
  ],
  result: {
    scores: [0, 0, 0, 0],
    cards: [0, 0, 0, 0],
    resigned: [2],
    winners: null,
    winningTeamIds: [1],
    reason: 'FORFEIT',
    rating: [0, 5, 0, 5],
  },
};

function clone(replay) {
  return JSON.parse(JSON.stringify(replay));
}

module.exports = {
  nobleGame,
  forcedShortTakes,
  reserves,
  timedOut,
  oneVsTwo,
  teamGame,
  clone,
};
