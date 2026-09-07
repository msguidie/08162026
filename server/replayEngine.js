// ═══════════════════════════════════════════════════════════
// Replay reconstruction — replays a stored replay JSON through the
// SAME gameLogic.js functions the live server uses, so replay states
// can never drift from the rules.
// Contract: docs/REPLAY_FORMAT.md §3
// ═══════════════════════════════════════════════════════════

const {
  ALL_CARDS,
  ALL_BONUS_TILES,
  createInitialGameState,
  clientView,
  processAction,
  processResign,
} = require('./gameLogic');

const GEM_TAKE = 'G';
const RESERVE_BOARD = 'R';
const RESERVE_DECK = 'RD';
const BUY = 'B';
const CHOOSE_NOBLE = 'N';
const RESIGN = 'X';
const TIMEOUT = 'T';

class ReplayCorruptError extends Error {
  constructor(actionIndex, message) {
    super(message);
    this.name = 'ReplayCorruptError';
    this.actionIndex = actionIndex;
  }
}

const cardsById = new Map(ALL_CARDS.map(card => [card.id, card]));
const tilesById = new Map(ALL_BONUS_TILES.map(tile => [tile.id, tile]));

// Cards/tiles are never mutated by gameLogic, but replays get their own
// copies so a reconstruction can never touch the shared card table.
function cloneCard(id, actionIndex) {
  const card = cardsById.get(id);
  if (!card) throw new ReplayCorruptError(actionIndex, `Unknown card id ${id}`);
  return { id: card.id, tier: card.tier, reward: card.reward, points: card.points, cost: [...card.cost] };
}

function cloneTile(id, actionIndex) {
  const tile = tilesById.get(id);
  if (!tile) throw new ReplayCorruptError(actionIndex, `Unknown tile id ${id}`);
  return { id: tile.id, points: tile.points, requirement: [...tile.requirement] };
}

function deepClone(value) {
  if (typeof structuredClone === 'function') return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

// Map one stored compact action to the protocol actions the live server
// receives from the client. `steps` are fed to processAction in order.
function compactActionToProtocol(action) {
  if (!Array.isArray(action) || action.length < 2) return null;
  const actor = action[0];
  const code = action[1];
  if (!Number.isInteger(actor)) return null;

  switch (code) {
    case GEM_TAKE:
      if (!Array.isArray(action[2])) return null;
      return { actor, code, kind: 'ACTION', steps: [{ type: 'TAKE_GEMS_CONFIRMED', colors: [...action[2]] }] };
    case RESERVE_BOARD:
      if (!Number.isInteger(action[2])) return null;
      return {
        actor,
        code,
        kind: 'ACTION',
        steps: [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_CARD', cardId: action[2] }],
      };
    case RESERVE_DECK:
      if (!Number.isInteger(action[2])) return null;
      return {
        actor,
        code,
        kind: 'ACTION',
        steps: [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_FROM_DECK', tier: action[2] }],
      };
    case BUY: {
      if (!Number.isInteger(action[2])) return null;
      if (action[3] !== 'b' && action[3] !== 'r') return null;
      const source = action[3] === 'b' ? 'board' : 'reserved';
      return { actor, code, kind: 'ACTION', steps: [{ type: 'BUY_CARD', cardId: action[2], source }] };
    }
    case CHOOSE_NOBLE:
      if (!Number.isInteger(action[2])) return null;
      return { actor, code, kind: 'ACTION', steps: [{ type: 'CHOOSE_TILE', tileId: action[2] }] };
    case RESIGN:
      return { actor, code, kind: 'RESIGN', steps: [] };
    case TIMEOUT:
      return { actor, code, kind: 'TIMEOUT', steps: [] };
    default:
      return null;
  }
}

function validateReplay(json) {
  if (!isPlainObject(json)) throw new ReplayCorruptError(-1, 'Replay is not an object');
  if (!Array.isArray(json.players) || json.players.length === 0) {
    throw new ReplayCorruptError(-1, 'Replay has no players');
  }
  const n = Number.isInteger(json.n) ? json.n : json.players.length;
  if (n !== json.players.length) throw new ReplayCorruptError(-1, 'Replay player count mismatch');
  if (json.mode !== 'INDIVIDUAL' && json.mode !== 'TEAM' && json.mode !== 'ONE_V_TWO') {
    throw new ReplayCorruptError(-1, `Unknown game mode ${json.mode}`);
  }
  if (!Number.isInteger(json.first) || json.first < 0 || json.first >= n) {
    throw new ReplayCorruptError(-1, 'Replay has an invalid first player');
  }
  const setup = json.setup;
  if (!isPlainObject(setup) || !Array.isArray(setup.board) || setup.board.length !== 3
    || !Array.isArray(setup.decks) || setup.decks.length !== 3 || !Array.isArray(setup.tiles)) {
    throw new ReplayCorruptError(-1, 'Replay setup is malformed');
  }
  if (setup.board.some(row => !Array.isArray(row)) || setup.decks.some(deck => !Array.isArray(deck))) {
    throw new ReplayCorruptError(-1, 'Replay setup is malformed');
  }
  if (!Array.isArray(json.actions)) throw new ReplayCorruptError(-1, 'Replay has no action list');
  return n;
}

// Contract §3 steps 1–2.
function buildInitialState(json) {
  const n = validateReplay(json);
  const players = json.players.map(p => ({
    username: p.u,
    avatarSeed: p.a,
    teamId: p.team,
  }));

  const state = createInitialGameState(players, {
    gameMode: json.mode,
    teamLayout: json.layout || null,
    unlimitedTime: true,
    // ONE_V_TWO forces seat 0 inside createInitialGameState; the explicit
    // assignment below covers every mode either way.
    firstPlayerIndex: json.first,
  });

  state.board = json.setup.board.map(row => row.map(id => cloneCard(id, -1)));
  state.decks = json.setup.decks.map(deck => deck.map(id => cloneCard(id, -1)));
  state.deckCounts = [state.decks[0].length, state.decks[1].length, state.decks[2].length];
  state.bonusTiles = json.setup.tiles.map(id => cloneTile(id, -1));
  state.currentPlayerIndex = json.first;
  state.roundStartPlayer = json.first;
  state.timeControl = null;
  return state;
}

function buildMeta(json) {
  return {
    t: json.t ?? null,
    e: json.e ?? null,
    mode: json.mode,
    layout: json.layout ?? null,
    n: json.players.length,
    clock: json.clock === true,
    first: json.first,
    result: json.result ? deepClone(json.result) : null,
    players: json.players.map(p => ({
      username: p.u,
      avatarSeed: p.a ?? null,
      ...(p.team === 0 || p.team === 1 ? { teamId: p.team } : {}),
      isAI: p.ai === true,
    })),
  };
}

function makeFrame(index, state, actor, action, result, pendingTileChoice) {
  return {
    i: index,
    turn: state.turnNumber,
    actor,
    action,
    result,
    state: deepClone(clientView(state)),
    pendingTileChoice: pendingTileChoice ? [...pendingTileChoice] : null,
  };
}

// The live server moves `_tileClaimed` onto the emitted action result
// (index.js → broadcastProcessedAction); replays must do the same.
function takeTileClaimed(state, result) {
  if (state._tileClaimed) {
    result.tileClaimed = state._tileClaimed;
    delete state._tileClaimed;
  }
  return result;
}

function reconstruct(replayJson) {
  const state = buildInitialState(replayJson);
  const meta = buildMeta(replayJson);
  const frames = [makeFrame(0, state, null, null, null, null)];

  replayJson.actions.forEach((compact, actionIndex) => {
    const plan = compactActionToProtocol(compact);
    if (!plan) throw new ReplayCorruptError(actionIndex, `Malformed action entry ${JSON.stringify(compact)}`);
    if (plan.actor < 0 || plan.actor >= state.numPlayers) {
      throw new ReplayCorruptError(actionIndex, `Action refers to unknown player ${plan.actor}`);
    }

    let result;

    if (plan.kind === 'RESIGN' || plan.kind === 'TIMEOUT') {
      if (state.phase !== 'PLAYING') {
        throw new ReplayCorruptError(actionIndex, 'Cannot resign after the game is over');
      }
      processResign(state, plan.actor);
      // Mirror index.js: a game with fewer than two active players is over.
      const activeCount = state.numPlayers - (state.resignedPlayers?.length || 0);
      if (activeCount < 2) state.phase = 'GAME_OVER';
      result = plan.kind === 'RESIGN'
        ? { type: 'RESIGN', actingPlayer: plan.actor, payload: { resignedPlayerIndex: plan.actor } }
        : { type: 'TIMEOUT', actingPlayer: plan.actor, payload: { timedOutPlayerIndex: plan.actor } };
    } else {
      // Extra facts the live client learns from the ENTER_RESERVE result or
      // from its own hand; recomputed here so a viewer can animate the frame.
      const extras = {};
      for (const step of plan.steps) {
        if (step.type === 'RESERVE_FROM_DECK') {
          const deck = state.decks[step.tier - 1];
          if (Array.isArray(deck) && deck.length > 0) extras.cardId = deck[deck.length - 1].id;
        }
        const stepResult = processAction(state, plan.actor, step);
        if (stepResult.error) {
          throw new ReplayCorruptError(actionIndex, `${step.type} rejected: ${stepResult.error}`);
        }
        if (step.type === 'ENTER_RESERVE') {
          extras.goldTaken = stepResult.result.payload.goldTaken;
          continue;
        }
        result = stepResult.result;
      }
      if (!result) throw new ReplayCorruptError(actionIndex, 'Action produced no result');
      result.payload = { ...result.payload, ...extras };
    }

    takeTileClaimed(state, result);
    const pendingTileChoice = state._pendingTileChoice || null;
    frames.push(makeFrame(actionIndex + 1, state, plan.actor, deepClone(compact), result, pendingTileChoice));
  });

  return { meta, frames };
}

module.exports = {
  reconstruct,
  buildInitialState,
  compactActionToProtocol,
  ReplayCorruptError,
};
