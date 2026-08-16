// ═══════════════════════════════════════════════════════════
// Server-side game logic — single source of truth
// ═══════════════════════════════════════════════════════════

// ── Card & tile data (mirrored from src/data/cards.ts) ──

const ALL_CARDS = [];
const ALL_BONUS_TILES = [];

let nextId = 0;
function addCycle(tier, points, template) {
  for (let i = 0; i < 5; i++) {
    const cost = [0, 0, 0, 0, 0];
    for (let j = 0; j < 5; j++) {
      cost[j] = template[(j - i + 5) % 5];
    }
    ALL_CARDS.push({ id: nextId++, tier, reward: i, points, cost });
  }
}

// TIER 1: 40 Cards
addCycle(1, 0, [1, 1, 0, 1, 1]);
addCycle(1, 0, [1, 2, 1, 0, 1]);
addCycle(1, 0, [0, 2, 1, 0, 2]);
addCycle(1, 0, [3, 0, 0, 1, 0]);
addCycle(1, 0, [0, 0, 2, 2, 0]);
addCycle(1, 0, [0, 0, 0, 0, 3]);
addCycle(1, 0, [0, 2, 1, 1, 1]);
addCycle(1, 1, [0, 4, 0, 0, 0]);

// TIER 2: 30 Cards
addCycle(2, 3, [6, 0, 0, 0, 0]);
addCycle(2, 1, [0, 2, 0, 3, 3]);
addCycle(2, 2, [0, 0, 0, 0, 5]);
addCycle(2, 2, [0, 4, 0, 1, 3]);
addCycle(2, 2, [0, 1, 4, 2, 0]);
addCycle(2, 3, [0, 0, 0, 0, 6]);

// TIER 3: 20 Cards
addCycle(3, 3, [0, 3, 3, 3, 5]);
addCycle(3, 4, [0, 0, 0, 7, 0]);
addCycle(3, 4, [0, 0, 0, 3, 6]);
addCycle(3, 5, [0, 3, 0, 0, 7]);

let tileId = 0;
for (let i = 0; i < 5; i++) {
  const req = [0, 0, 0, 0, 0];
  req[i] = 4; req[(i + 1) % 5] = 4;
  ALL_BONUS_TILES.push({ id: tileId++, points: 3, requirement: req });
}
for (let i = 0; i < 5; i++) {
  const req = [0, 0, 0, 0, 0];
  req[i] = 3; req[(i + 2) % 5] = 3; req[(i + 4) % 5] = 3;
  ALL_BONUS_TILES.push({ id: tileId++, points: 3, requirement: req });
}

// ── Helpers ──

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function getDiscount(player) {
  const d = [0, 0, 0, 0, 0];
  for (const card of player.cards) d[card.reward]++;
  return d;
}

function totalGems(player) {
  return player.gems.reduce((a, b) => a + b, 0);
}

function canAfford(player, card) {
  const discount = getDiscount(player);
  let goldNeeded = 0;
  for (let i = 0; i < 5; i++) {
    const need = Math.max(0, card.cost[i] - discount[i]);
    if (player.gems[i] < need) goldNeeded += need - player.gems[i];
  }
  return goldNeeded <= player.gems[5];
}

function payForCard(player, card) {
  const discount = getDiscount(player);
  const paid = [0, 0, 0, 0, 0, 0];
  let goldUsed = 0;
  for (let i = 0; i < 5; i++) {
    const need = Math.max(0, card.cost[i] - discount[i]);
    const fromColor = Math.min(player.gems[i], need);
    paid[i] = fromColor;
    player.gems[i] -= fromColor;
    goldUsed += need - fromColor;
  }
  paid[5] = goldUsed;
  player.gems[5] -= goldUsed;
  return paid;
}

function qualifiesForTile(player, tile) {
  const rewards = getDiscount(player);
  for (let i = 0; i < 5; i++) {
    if (rewards[i] < tile.requirement[i]) return false;
  }
  return true;
}

function getQualifiedTiles(player, tiles) {
  return tiles.filter(t => qualifiesForTile(player, t));
}

function canSelectGem(color, selected, supply, playerGemCount, config) {
  if (supply[color] <= 0) return false;
  if (playerGemCount + selected.length >= config.maxTokensInHand) return false;
  if (selected.length === 0) return true;
  if (selected.length === 1) {
    if (selected[0] === color) return supply[color] >= config.take2MinStack - 1;
    return true;
  }
  if (selected.length === 2) {
    if (selected[0] === selected[1]) return false;
    if (selected.includes(color)) return false;
    return true;
  }
  return false;
}

function isGemTakeComplete(selected, supply, playerGemCount, config) {
  const maxCanHold = config.maxTokensInHand - playerGemCount;
  if (selected.length >= maxCanHold) return true;
  if (selected.length === 2 && selected[0] === selected[1]) return true;
  if (selected.length === 3) return true;
  if (selected.length === 2 && selected[0] !== selected[1]) {
    const usedColors = new Set(selected);
    const hasAvailable = [0, 1, 2, 3, 4].some(c => !usedColors.has(c) && supply[c] > 0);
    if (!hasAvailable) return true;
  }
  if (selected.length === 1) {
    const color = selected[0];
    const canTakeSame = supply[color] >= config.take2MinStack - 1;
    const hasOtherColor = [0, 1, 2, 3, 4].some(c => c !== color && supply[c] > 0);
    if (!canTakeSame && !hasOtherColor) return true;
  }
  return false;
}

function calculateRatingChanges(players, state) {
  if ((state?.gameMode === 'TEAM' || state?.gameMode === 'ONE_V_TWO') && state.gameResult?.winningTeamIds) {
    const winners = state.gameResult.winningTeamIds;
    if (winners.length !== 1) return new Array(players.length).fill(3);
    return players.map(p => p.teamId === winners[0] ? 5 : 0);
  }

  const ranked = players.map((p, i) => ({ index: i, score: p.score, cardCount: p.cards.length }));
  ranked.sort((a, b) => b.score !== a.score ? b.score - a.score : a.cardCount - b.cardCount);
  const changes = new Array(players.length).fill(0);
  let rank = 0;
  for (let i = 0; i < ranked.length; i++) {
    if (i > 0 && (ranked[i].score !== ranked[i - 1].score || ranked[i].cardCount !== ranked[i - 1].cardCount)) rank = i;
    changes[ranked[i].index] = rank === 0 ? 5 : rank === 1 ? 3 : rank === 2 ? 1 : 0;
  }
  return changes;
}

// ── Create initial game state ──

function createInitialGameState(players, options = {}) {
  const n = players.length;
  const gameMode = options.gameMode === 'TEAM' ? 'TEAM'
    : options.gameMode === 'ONE_V_TWO' ? 'ONE_V_TWO'
      : 'INDIVIDUAL';
  const teamLayout = gameMode === 'TEAM' && options.teamLayout === 'OPPOSITE' ? 'OPPOSITE'
    : gameMode === 'TEAM' ? 'ADJACENT' : null;
  const isTeamGame = gameMode !== 'INDIVIDUAL';
  const config = {
    tokensPerColor: n <= 2 ? 4 : n === 3 ? 5 : 7,
    wildTokens: 5,
    revealedTiles: n + 1,
    cardsPerRow: 4,
    maxTokensInHand: 10,
    maxReserved: 3,
    winThreshold: 15,
    take2MinStack: 4,
  };

  const tier1 = shuffle(ALL_CARDS.filter(c => c.tier === 1));
  const tier2 = shuffle(ALL_CARDS.filter(c => c.tier === 2));
  const tier3 = shuffle(ALL_CARDS.filter(c => c.tier === 3));

  const board = [tier1.splice(0, 4), tier2.splice(0, 4), tier3.splice(0, 4)];
  const decks = [tier1, tier2, tier3];
  const bonusTiles = shuffle(ALL_BONUS_TILES).slice(0, config.revealedTiles);
  const gems = [config.tokensPerColor, config.tokensPerColor, config.tokensPerColor, config.tokensPerColor, config.tokensPerColor, config.wildTokens];

  const gamePlayers = players.map(p => ({
    username: p.username,
    gems: [0, 0, 0, 0, 0, 0],
    cards: [],
    reserved: [],
    bonusTiles: [],
    score: 0,
    avatarSeed: p.avatarSeed,
    ...(isTeamGame ? { teamId: p.teamId } : {}),
  }));

  // The solo player occupies seat zero and always opens a 1v2 game.
  // All other modes start with a randomly selected player.
  const actualFirstPlayer = gameMode === 'ONE_V_TWO' ? 0 : Math.floor(Math.random() * n);
  const now = Date.now();
  const teams = isTeamGame ? [0, 1].map(id => ({
    id,
    playerIndices: gamePlayers.map((p, index) => p.teamId === id ? index : -1).filter(index => index >= 0),
  })) : [];

  return {
    phase: 'PLAYING',
    board,
    decks, // server-only, not sent to clients
    deckCounts: [decks[0].length, decks[1].length, decks[2].length],
    gems,
    bonusTiles,
    players: gamePlayers,
    currentPlayerIndex: actualFirstPlayer,
    roundStartPlayer: actualFirstPlayer, // player who starts each round — used for final round detection
    turnAction: null,
    finalRoundTriggeredBy: null,
    turnNumber: 0,
    numPlayers: n,
    config,
    resignedPlayers: [],
    gameMode,
    teamLayout,
    teams,
    gameResult: null,
    timeControl: n === 3 || n === 4 ? {
      mainTimeMs: 3 * 60 * 1000,
      countdownMs: 8 * 1000,
      playerTimeRemainingMs: new Array(n).fill(3 * 60 * 1000),
      activeSince: now,
      countdownDeadline: null,
      serverNow: now,
    } : null,
  };
}

// Strip server-only data (decks with full card info) before sending to clients
function clientView(state) {
  const { decks, ...rest } = state;
  return {
    ...rest,
    deckCounts: [decks[0].length, decks[1].length, decks[2].length],
    timeControl: rest.timeControl ? { ...rest.timeControl, serverNow: Date.now() } : null,
  };
}

// Client view of reserved cards: each player only sees their own reserved cards in detail
// Other players' reserved cards are hidden (count only shown in the panel)
function clientViewForPlayer(state, playerIndex) {
  const view = clientView(state);
  // Show full reserved cards for requesting player, hide others'
  view.players = view.players.map((p, i) => {
    if (i === playerIndex) return p;
    return { ...p, reserved: p.reserved.map(() => ({ id: -1, tier: 0, reward: 0, points: 0, cost: [0, 0, 0, 0, 0], hidden: true })) };
  });
  return view;
}

// ── Get next active player (skip resigned) ──

function getNextActivePlayer(state, currentIdx) {
  const resigned = state.resignedPlayers || [];
  let next = (currentIdx + 1) % state.numPlayers;
  let attempts = 0;
  while (resigned.includes(next) && attempts < state.numPlayers) {
    next = (next + 1) % state.numPlayers;
    attempts++;
  }
  return next;
}

function updateTimeControl(state, now = Date.now()) {
  const clock = state.timeControl;
  if (!clock || state.phase !== 'PLAYING') return { enabled: false, countdownStarted: false, expired: false };

  clock.serverNow = now;
  if (clock.countdownDeadline !== null) {
    return {
      enabled: true,
      countdownStarted: false,
      expired: now >= clock.countdownDeadline,
    };
  }

  const playerIndex = state.currentPlayerIndex;
  const remaining = Math.max(0, clock.playerTimeRemainingMs[playerIndex] ?? 0);
  const mainDeadline = clock.activeSince + remaining;
  if (now < mainDeadline) return { enabled: true, countdownStarted: false, expired: false };

  clock.playerTimeRemainingMs[playerIndex] = 0;
  clock.activeSince = mainDeadline;
  clock.countdownDeadline = mainDeadline + clock.countdownMs;
  return {
    enabled: true,
    countdownStarted: true,
    expired: now >= clock.countdownDeadline,
  };
}

function consumeTurnTime(state, playerIndex, now = Date.now()) {
  const clock = state.timeControl;
  if (!clock) return;
  if (clock.countdownDeadline === null) {
    const elapsed = Math.max(0, now - clock.activeSince);
    clock.playerTimeRemainingMs[playerIndex] = Math.max(
      0,
      (clock.playerTimeRemainingMs[playerIndex] ?? 0) - elapsed,
    );
  }
  clock.activeSince = now;
  clock.serverNow = now;
}

function startTurnTimeControl(state, now = Date.now()) {
  const clock = state.timeControl;
  if (!clock || state.phase !== 'PLAYING') return;
  clock.activeSince = now;
  clock.countdownDeadline = (clock.playerTimeRemainingMs[state.currentPlayerIndex] ?? 0) <= 0
    ? now + clock.countdownMs
    : null;
  clock.serverNow = now;
}

function withForcedFlag(response) {
  if (response?.result) {
    response.result.payload = { ...response.result.payload, forced: true };
  }
  return response;
}

function processAutoAction(state, playerIndex) {
  if (state.phase !== 'PLAYING' || state.currentPlayerIndex !== playerIndex) {
    return { error: 'Player turn is no longer active' };
  }

  if (state.turnAction?.type === 'RESERVE') {
    const faceUpCards = state.board.flat();
    if (faceUpCards.length > 0) {
      const card = faceUpCards[Math.floor(Math.random() * faceUpCards.length)];
      return withForcedFlag(processAction(state, playerIndex, { type: 'RESERVE_CARD', cardId: card.id }));
    }

    // A fully exhausted board is exceptional. Return the already-granted gold
    // and finish the turn instead of leaving the game permanently blocked.
    if (state.turnAction.goldTaken && state.players[playerIndex].gems[5] > 0) {
      state.players[playerIndex].gems[5]--;
      state.gems[5]++;
    }
    state.turnAction = null;
    finishTurn(state);
    return { ok: true, result: { type: 'AUTO_PASS', actingPlayer: playerIndex, payload: { forced: true } } };
  }

  if (state.turnAction?.type === 'BUY' && state._pendingTileChoice?.length > 0) {
    const choices = state._pendingTileChoice;
    const tileId = choices[Math.floor(Math.random() * choices.length)];
    return withForcedFlag(processAction(state, playerIndex, { type: 'CHOOSE_TILE', tileId }));
  }

  const startingTurnNumber = state.turnNumber;
  let lastResponse = null;
  for (let color = 0; color < 5; color++) {
    const selected = state.turnAction?.type === 'TAKE_GEMS' ? state.turnAction.selected : [];
    if (selected.includes(color)) continue;
    const response = processAction(state, playerIndex, { type: 'SELECT_GEM', color });
    if (response.ok) lastResponse = response;
    if (state.phase !== 'PLAYING' || state.turnNumber !== startingTurnNumber) {
      return withForcedFlag(lastResponse);
    }
  }

  if (state.turnAction?.type === 'TAKE_GEMS' && state.turnAction.selected.length > 0) {
    const selected = [...state.turnAction.selected];
    const player = state.players[playerIndex];
    for (const color of selected) {
      player.gems[color]++;
      state.gems[color]--;
    }
    state.turnAction = null;
    advanceTurn(state);
    return {
      ok: true,
      result: { type: 'SELECT_GEM', actingPlayer: playerIndex, payload: { selected, forced: true } },
    };
  }

  state.turnAction = null;
  finishTurn(state);
  return { ok: true, result: { type: 'AUTO_PASS', actingPlayer: playerIndex, payload: { forced: true } } };
}

// ── Process actions ──

function processAction(state, playerIndex, action) {
  if (state.phase !== 'PLAYING') return { error: 'Game is over' };
  if (state.currentPlayerIndex !== playerIndex) return { error: 'Not your turn' };
  if (!action || typeof action.type !== 'string') return { error: 'Invalid action' };

  const player = state.players[playerIndex];
  const result = { type: action.type, actingPlayer: playerIndex, payload: {} };

  switch (action.type) {
    case 'SELECT_GEM': {
      const { color } = action;
      if (!Number.isInteger(color) || color < 0 || color > 4) return { error: 'Invalid gem color' };

      // Once reserve mode has granted gold, only selecting a reserve card may
      // complete the turn. This server-side guard also survives reconnects.
      if (state.turnAction && state.turnAction.type !== 'TAKE_GEMS') {
        return { error: 'Finish your current action first' };
      }

      // Initialize turn action if needed
      if (!state.turnAction) {
        state.turnAction = { type: 'TAKE_GEMS', selected: [] };
      }

      const selected = state.turnAction.selected;
      const supply = state.gems.slice(0, 5);
      const adjustedSupply = [...supply];
      for (const s of selected) adjustedSupply[s]--;

      if (!canSelectGem(color, selected, adjustedSupply, totalGems(player), state.config)) {
        return { error: 'Cannot select this gem' };
      }

      selected.push(color);

      // Check adjusted supply after this selection
      const newAdjustedSupply = [...supply];
      for (const s of selected) newAdjustedSupply[s]--;

      if (isGemTakeComplete(selected, newAdjustedSupply, totalGems(player), state.config)) {
        // Apply the gem take
        for (const s of selected) { player.gems[s]++; state.gems[s]--; }
        result.payload = { selected: [...selected] };
        state.turnAction = null;
        advanceTurn(state);
        return { ok: true, result, completed: true };
      }

      result.payload = { selected: [...selected], completed: false };
      return { ok: true, result, completed: false };
    }

    case 'ENTER_RESERVE': {
      if (player.reserved.length >= state.config.maxReserved) return { error: 'Reserve full' };
      if (state.turnAction) return { error: 'Finish or cancel your current action first' };

      // Take gold immediately
      let goldTaken = false;
      if (state.gems[5] > 0 && totalGems(player) < state.config.maxTokensInHand) {
        player.gems[5]++;
        state.gems[5]--;
        goldTaken = true;
      }
      state.turnAction = { type: 'RESERVE', goldTaken, cardPicked: false };
      result.payload = { goldTaken };
      return { ok: true, result, completed: false };
    }

    case 'RESERVE_CARD': {
      if (!state.turnAction || state.turnAction.type !== 'RESERVE') return { error: 'Not in reserve mode' };
      if (player.reserved.length >= state.config.maxReserved) return { error: 'Reserve full' };

      const { cardId } = action;
      const tierIdx = [0, 1, 2].find(t => state.board[t].some(c => c.id === cardId));
      if (tierIdx === undefined) return { error: 'Card not on board' };

      const cardIdx = state.board[tierIdx].findIndex(c => c.id === cardId);
      const card = state.board[tierIdx].splice(cardIdx, 1)[0];
      player.reserved.push(card);
      if (state.decks[tierIdx].length > 0) state.board[tierIdx].push(state.decks[tierIdx].pop());
      state.deckCounts = [state.decks[0].length, state.decks[1].length, state.decks[2].length];

      result.payload = { cardId: card.id, tier: card.tier, fromDeck: false };
      state.turnAction = null;
      advanceTurn(state);
      return { ok: true, result, completed: true };
    }

    case 'RESERVE_FROM_DECK': {
      if (!state.turnAction || state.turnAction.type !== 'RESERVE') return { error: 'Not in reserve mode' };
      if (player.reserved.length >= state.config.maxReserved) return { error: 'Reserve full' };

      const { tier } = action;
      const ti = tier - 1;
      if (ti < 0 || ti > 2 || state.decks[ti].length === 0) return { error: 'Invalid or empty deck' };

      const card = state.decks[ti].pop();
      player.reserved.push(card);
      state.deckCounts = [state.decks[0].length, state.decks[1].length, state.decks[2].length];

      result.payload = { tier, fromDeck: true };
      state.turnAction = null;
      advanceTurn(state);
      return { ok: true, result, completed: true };
    }

    case 'BUY_CARD': {
      if (state.turnAction) return { error: 'Finish or cancel your current action first' };
      const { cardId, source } = action; // source: 'board' | 'reserved'
      if (source !== 'board' && source !== 'reserved') return { error: 'Invalid card source' };
      let card;

      if (source === 'board') {
        for (let t = 0; t < 3; t++) {
          const idx = state.board[t].findIndex(c => c.id === cardId);
          if (idx !== -1) {
            card = state.board[t][idx];
            break;
          }
        }
      } else {
        card = player.reserved.find(c => c.id === cardId);
      }

      if (!card) return { error: 'Card not found' };
      if (!canAfford(player, card)) return { error: 'Cannot afford' };

      const gemsReturned = payForCard(player, card);

      if (source === 'board') {
        for (let t = 0; t < 3; t++) {
          const idx = state.board[t].findIndex(c => c.id === cardId);
          if (idx !== -1) {
            state.board[t].splice(idx, 1);
            if (state.decks[t].length > 0) state.board[t].push(state.decks[t].pop());
            state.deckCounts = [state.decks[0].length, state.decks[1].length, state.decks[2].length];
            break;
          }
        }
      } else {
        player.reserved = player.reserved.filter(c => c.id !== cardId);
      }

      player.cards.push(card);
      player.score += card.points;
      for (let i = 0; i < 6; i++) state.gems[i] += gemsReturned[i];

      state.turnAction = { type: 'BUY' };
      result.payload = { cardId, source, reward: card.reward, points: card.points, gemsReturned };
      advanceTurn(state);
      return { ok: true, result, completed: true };
    }

    case 'CHOOSE_TILE': {
      const { tileId } = action;
      if (state.turnAction?.type !== 'BUY' || !state._pendingTileChoice?.includes(tileId)) {
        return { error: 'No matching noble choice is pending' };
      }
      const tileIdx = state.bonusTiles.findIndex(t => t.id === tileId);
      if (tileIdx === -1) return { error: 'Tile not found' };
      const tile = state.bonusTiles[tileIdx];
      if (!qualifiesForTile(player, tile)) return { error: 'Not qualified' };

      state.bonusTiles.splice(tileIdx, 1);
      player.bonusTiles.push(tile);
      player.score += tile.points;

      result.payload = { tileId, playerIndex };
      // After choosing tile, finish the turn
      finishTurn(state);
      return { ok: true, result, completed: true };
    }

    case 'CANCEL_GEMS': {
      // Allow canceling gem selection (reset turn action)
      if (state.turnAction?.type === 'TAKE_GEMS') {
        state.turnAction = null;
        return { ok: true, result: { type: 'CANCEL_GEMS', actingPlayer: playerIndex, payload: {} }, completed: false };
      }
      return { error: 'Nothing to cancel' };
    }

    default:
      return { error: 'Unknown action' };
  }
}

function advanceTurn(state) {
  const currentPlayer = state.players[state.currentPlayerIndex];
  const qualified = getQualifiedTiles(currentPlayer, state.bonusTiles);

  if (qualified.length === 1) {
    // Auto-claim single tile
    const tile = qualified[0];
    state.bonusTiles = state.bonusTiles.filter(t => t.id !== tile.id);
    currentPlayer.bonusTiles.push(tile);
    currentPlayer.score += tile.points;
    state._tileClaimed = { tileId: tile.id, playerIndex: state.currentPlayerIndex };
    finishTurn(state);
  } else if (qualified.length > 1) {
    // Player must choose — don't advance yet
    state._pendingTileChoice = qualified.map(t => t.id);
  } else {
    finishTurn(state);
  }
}

function getTeamStats(state) {
  if (state.gameMode === 'INDIVIDUAL') return [];
  return [0, 1].map(teamId => {
    const members = state.players.filter(p => p.teamId === teamId);
    const scores = members.map(p => p.score).sort((a, b) => b - a);
    return {
      teamId,
      total: scores.reduce((sum, score) => sum + score, 0),
      secondScore: scores.length === 2 ? scores[1] : -Infinity,
      cardCount: members.reduce((sum, player) => sum + player.cards.length, 0),
    };
  });
}

function getQualifyingTeamIds(state) {
  const stats = getTeamStats(state);
  if (stats.length !== 2) return [];
  if (state.gameMode === 'ONE_V_TWO') {
    return stats
      .filter(team => team.teamId === 0 ? team.total >= 15 : team.total >= 32)
      .map(team => team.teamId);
  }
  return stats
    .filter((team, index) => team.total > 30 && team.secondScore >= stats[1 - index].secondScore)
    .map(team => team.teamId);
}

function resolveTeamWinners(state, qualifyingTeamIds) {
  if (qualifyingTeamIds.length <= 1) return qualifyingTeamIds;
  const candidates = getTeamStats(state).filter(team => qualifyingTeamIds.includes(team.teamId));
  candidates.sort((a, b) => b.total - a.total || a.cardCount - b.cardCount);
  const best = candidates[0];
  return candidates
    .filter(team => team.total === best.total && team.cardCount === best.cardCount)
    .map(team => team.teamId);
}

function resolveOneVsTwoWinners(state) {
  const stats = getTeamStats(state);
  const solo = stats.find(team => team.teamId === 0);
  const duo = stats.find(team => team.teamId === 1);
  if (!solo || !duo) return [];

  const soloQualified = solo.total >= 15;
  const duoQualified = duo.total >= 32;
  if (soloQualified && !duoQualified) return [0];
  if (duoQualified && !soloQualified) return [1];
  if (!soloQualified && !duoQualified) return [];

  const soloExcess = solo.total - 15;
  const duoExcess = duo.total - 32;
  if (soloExcess === duoExcess) return [0, 1];
  return [soloExcess > duoExcess ? 0 : 1];
}

function finishTurn(state) {
  const currentPlayer = state.players[state.currentPlayerIndex];

  if (state.finalRoundTriggeredBy === null) {
    if (state.gameMode === 'TEAM') {
      if (getQualifyingTeamIds(state).length > 0) state.finalRoundTriggeredBy = state.currentPlayerIndex;
    } else if (state.gameMode === 'ONE_V_TWO') {
      if (getQualifyingTeamIds(state).length > 0) state.finalRoundTriggeredBy = state.currentPlayerIndex;
    } else if (currentPlayer.score >= state.config.winThreshold) {
      state.finalRoundTriggeredBy = state.currentPlayerIndex;
    }
  }

  const nextPlayer = getNextActivePlayer(state, state.currentPlayerIndex);

  // Final round: game ends when play returns to the round leader
  // (the player who started the game — they never get another turn).
  // If the triggering player IS the round leader, all other players finish their turn first.
  // If the triggering player is the last in the round, game ends immediately after their turn.
  if (state.finalRoundTriggeredBy !== null) {
    // Use roundStartPlayer (or fallback to first active if resigned)
    let roundLeader = state.roundStartPlayer;
    const resigned = state.resignedPlayers || [];
    if (roundLeader === undefined || resigned.includes(roundLeader)) {
      roundLeader = getFirstActivePlayer(state);
    }
    if (nextPlayer === roundLeader) {
      if (state.gameMode === 'TEAM') {
        const qualifyingTeamIds = getQualifyingTeamIds(state);
        if (qualifyingTeamIds.length > 0) {
          state.phase = 'GAME_OVER';
          state.gameResult = {
            reason: 'SCORE',
            winningTeamIds: resolveTeamWinners(state, qualifyingTeamIds),
          };
          state.turnAction = null;
          return;
        }
        // Team conditions can become false while the remaining players act.
        // Clear the final round and continue normal play from the round leader.
        state.finalRoundTriggeredBy = null;
      } else if (state.gameMode === 'ONE_V_TWO') {
        // Unlike 2v2, reaching either threshold makes the 1v2 final round irrevocable.
        state.phase = 'GAME_OVER';
        state.gameResult = {
          reason: 'SCORE',
          winningTeamIds: resolveOneVsTwoWinners(state),
        };
        state.turnAction = null;
        return;
      } else {
        state.phase = 'GAME_OVER';
        state.turnAction = null;
        return;
      }
    }
  }

  state.currentPlayerIndex = nextPlayer;
  state.turnAction = null;
  state._pendingTileChoice = null;
  state.turnNumber++;
}

function getFirstActivePlayer(state) {
  const resigned = state.resignedPlayers || [];
  for (let i = 0; i < state.numPlayers; i++) {
    if (!resigned.includes(i)) return i;
  }
  return 0;
}

function processResign(state, playerIndex) {
  if (!state.resignedPlayers) state.resignedPlayers = [];
  if (state.resignedPlayers.includes(playerIndex)) return;
  state.resignedPlayers.push(playerIndex);

  const player = state.players[playerIndex];
  if (state.gameMode === 'TEAM' || state.gameMode === 'ONE_V_TWO') {
    const forfeitingTeamId = player.teamId;
    state.phase = 'GAME_OVER';
    state.turnAction = null;
    state._pendingTileChoice = null;
    state.finalRoundTriggeredBy = null;
    state.gameResult = {
      reason: 'FORFEIT',
      forfeitingTeamId,
      winningTeamIds: [forfeitingTeamId === 0 ? 1 : 0],
    };
    return;
  }

  for (let i = 0; i < 6; i++) { state.gems[i] += player.gems[i]; player.gems[i] = 0; }
  player.cards = []; player.reserved = []; player.bonusTiles = []; player.score = 0;

  // If the round leader resigned, advance the round leader to next active
  if (state.roundStartPlayer === playerIndex) {
    state.roundStartPlayer = getNextActivePlayer(state, playerIndex);
  }

  const activeCount = state.numPlayers - state.resignedPlayers.length;
  if (activeCount < 2) {
    state.phase = 'GAME_OVER';
    state.turnAction = null;
  } else if (state.currentPlayerIndex === playerIndex) {
    state.currentPlayerIndex = getNextActivePlayer(state, playerIndex);
    state.turnAction = null;
    state._pendingTileChoice = null;
  }
}

module.exports = {
  createInitialGameState,
  clientView,
  clientViewForPlayer,
  processAction,
  processResign,
  calculateRatingChanges,
  getTeamStats,
  getQualifyingTeamIds,
  resolveTeamWinners,
  resolveOneVsTwoWinners,
  finishTurn,
  updateTimeControl,
  consumeTurnTime,
  startTurnTimeControl,
  processAutoAction,
  totalGems,
};
