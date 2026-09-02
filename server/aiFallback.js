// ═══════════════════════════════════════════════════════════
// Deterministic greedy fallback policy (docs/AI_BRIDGE.md §2).
//
// Used when no worker is connected, the worker times out, or the worker
// returns an action the rules engine rejects.
//
// Every candidate is validated BLACK-BOX: the protocol action sequence is
// replayed on a `structuredClone` of the game state through the very same
// `processAction` the socket handlers use. No engine internals are touched,
// so this file cannot drift away from the rules.
// ═══════════════════════════════════════════════════════════

const { processAction } = require('./gameLogic');

const NONE = { type: 'NONE' };

// ── black-box validation ───────────────────────────────────

/** Replays `actions` on a clone. Returns the resulting state, or null if any step is illegal. */
function simulate(gameState, playerIndex, actions) {
  try {
    if (!gameState || !Array.isArray(actions) || actions.length === 0) return null;
    const clone = structuredClone(gameState);
    for (const action of actions) {
      const result = processAction(clone, playerIndex, action);
      if (!result || result.error) return null;
    }
    return clone;
  } catch (err) {
    return null;
  }
}

function isLegalSequence(gameState, playerIndex, actions) {
  return simulate(gameState, playerIndex, actions) !== null;
}

// ── small pure helpers (no engine internals) ───────────────

function discountOf(player) {
  const discount = [0, 0, 0, 0, 0];
  for (const card of player.cards || []) {
    if (Number.isInteger(card?.reward)) discount[card.reward]++;
  }
  return discount;
}

function totalCost(card) {
  return (card?.cost || []).reduce((sum, value) => sum + value, 0);
}

/** Colour-by-colour shortfall of `card` for `player`, ignoring gold (a ranking heuristic only). */
function needFor(player, card) {
  const discount = discountOf(player);
  const need = [0, 0, 0, 0, 0];
  for (let i = 0; i < 5; i++) {
    need[i] = Math.max(0, (card.cost?.[i] ?? 0) - discount[i] - (player.gems?.[i] ?? 0));
  }
  return need;
}

function deficit(player, card) {
  return needFor(player, card).reduce((sum, value) => sum + value, 0);
}

function boardCards(state) {
  return [0, 1, 2].flatMap(tier => state.board?.[tier] || []).filter(card => Number.isInteger(card?.id));
}

// ── candidate builders ─────────────────────────────────────

// 1. buy the affordable card with the most points
//    (ties: reserved before board, then cheapest, then lowest id)
function buyCandidates(state, player) {
  const reserved = (player.reserved || [])
    .filter(card => Number.isInteger(card?.id) && card.id >= 0)
    .map(card => ({ card, source: 'reserved' }));
  const board = boardCards(state).map(card => ({ card, source: 'board' }));
  return [...reserved, ...board].sort((a, b) =>
    (b.card.points ?? 0) - (a.card.points ?? 0)
    || (a.source === b.source ? 0 : a.source === 'reserved' ? -1 : 1)
    || totalCost(a.card) - totalCost(b.card)
    || a.card.id - b.card.id,
  ).map(entry => [{ type: 'BUY_CARD', cardId: entry.card.id, source: entry.source }]);
}

/** The cheapest still-attractive board card — the take-gems heuristic aims at it. */
function targetCard(state, player) {
  const cards = boardCards(state);
  if (cards.length === 0) return null;
  return cards.slice().sort((a, b) =>
    deficit(player, a) - deficit(player, b)
    || (b.points ?? 0) - (a.points ?? 0)
    || totalCost(a) - totalCost(b)
    || a.id - b.id,
  )[0];
}

function gemCombos() {
  const combos = [];
  for (let a = 0; a < 5; a++) {
    for (let b = a + 1; b < 5; b++) {
      for (let c = b + 1; c < 5; c++) combos.push([a, b, c]);
    }
  }
  for (let a = 0; a < 5; a++) {
    for (let b = a + 1; b < 5; b++) combos.push([a, b]);
  }
  for (let a = 0; a < 5; a++) combos.push([a, a]);
  for (let a = 0; a < 5; a++) combos.push([a]);
  return combos;
}

// 2. take up to 3 colours that most reduce the deficit of the target card.
//    Shorter takes stay in the list so a player near the 10-token cap still
//    finds a legal move (the engine, not this file, decides what fits).
function gemCandidateSequences(state, player) {
  const target = targetCard(state, player);
  const need = target ? needFor(player, target) : [0, 0, 0, 0, 0];
  const scored = gemCombos().map(colors => {
    const taken = [0, 0, 0, 0, 0];
    for (const color of colors) taken[color]++;
    let gain = 0;
    for (let i = 0; i < 5; i++) gain += Math.min(need[i], taken[i]);
    return { colors, gain };
  });
  scored.sort((a, b) =>
    b.gain - a.gain
    || b.colors.length - a.colors.length
    || a.colors.join(',').localeCompare(b.colors.join(',')),
  );
  return scored.map(entry => [{ type: 'TAKE_GEMS_CONFIRMED', colors: [...entry.colors] }]);
}

// 3. reserve the best board card (most points, then cheapest, then lowest id).
function reserveCandidates(state, player, { enteredAlready = false } = {}) {
  const prefix = enteredAlready ? [] : [{ type: 'ENTER_RESERVE' }];
  const cards = boardCards(state).sort((a, b) =>
    (b.points ?? 0) - (a.points ?? 0)
    || totalCost(a) - totalCost(b)
    || a.id - b.id,
  );
  const sequences = cards.map(card => [...prefix, { type: 'RESERVE_CARD', cardId: card.id }]);
  // Last resort so a bot with an empty board still has a move: blind deck reserve.
  for (const tier of [1, 2, 3]) {
    if ((state.deckCounts?.[tier - 1] ?? 0) > 0) {
      sequences.push([...prefix, { type: 'RESERVE_FROM_DECK', tier }]);
    }
  }
  return sequences;
}

// ── public API ─────────────────────────────────────────────

/**
 * Greedy move for `playerIndex`.
 * @returns {Array<object>} a protocol action list, or `{ type: 'NONE' }` when nothing is legal.
 */
function chooseFallbackActions(gameState, playerIndex) {
  try {
    if (!gameState || gameState.phase !== 'PLAYING') return NONE;
    if (gameState.currentPlayerIndex !== playerIndex) return NONE;
    const player = gameState.players?.[playerIndex];
    if (!player) return NONE;

    // A tile choice is pending: only CHOOSE_TILE is legal.
    if (Array.isArray(gameState._pendingTileChoice) && gameState._pendingTileChoice.length > 0) {
      return chooseTileActions(gameState, playerIndex);
    }

    const turnAction = gameState.turnAction;
    let candidates;
    if (turnAction?.type === 'RESERVE') {
      // Mid-reserve (gold already taken): the turn can only be finished by picking a card.
      candidates = reserveCandidates(gameState, player, { enteredAlready: true });
    } else {
      // A half-finished gem selection blocks everything else — clear it first.
      const prefix = turnAction?.type === 'TAKE_GEMS' ? [{ type: 'CANCEL_GEMS' }] : [];
      candidates = [
        ...buyCandidates(gameState, player),
        ...gemCandidateSequences(gameState, player),
        ...reserveCandidates(gameState, player),
      ].map(actions => [...prefix, ...actions]);
    }

    for (const actions of candidates) {
      if (isLegalSequence(gameState, playerIndex, actions)) return actions;
    }
    return NONE;
  } catch (err) {
    return NONE;
  }
}

/** Tile fallback: the first pending tile (later ones are tried only if it is somehow illegal). */
function chooseTileActions(gameState, playerIndex) {
  try {
    const pending = gameState?._pendingTileChoice;
    if (!Array.isArray(pending) || pending.length === 0) return NONE;
    for (const tileId of pending) {
      const actions = [{ type: 'CHOOSE_TILE', tileId }];
      if (isLegalSequence(gameState, playerIndex, actions)) return actions;
    }
    return NONE;
  } catch (err) {
    return NONE;
  }
}

function isNone(choice) {
  return !Array.isArray(choice);
}

module.exports = {
  chooseFallbackActions,
  chooseTileActions,
  isLegalSequence,
  simulate,
  isNone,
  NONE,
};
