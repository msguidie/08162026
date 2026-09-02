// ═══════════════════════════════════════════════════════════
// Replay recorder — captures the minimal per-game log described in
// docs/REPLAY_FORMAT.md §1/§2.
//
// Every exported function is internally guarded: a recorder failure must
// never propagate into a socket handler or change gameplay in any way.
// ═══════════════════════════════════════════════════════════

const replayStore = require('./replayStore');

// roomId → recording. Bounded so a room that is dropped without a
// finish()/discard() call can never grow the process unboundedly.
const recordings = new Map();
const MAX_ACTIVE_RECORDINGS = 200;

function log(message, err) {
  console.error(`[replay] ${message}${err ? `: ${err.message || err}` : ''}`);
}

function guard(name, fn, fallback = null) {
  try {
    return fn();
  } catch (err) {
    log(`${name} failed`, err);
    return fallback;
  }
}

function trimActiveRecordings() {
  while (recordings.size > MAX_ACTIVE_RECORDINGS) {
    const oldest = recordings.keys().next();
    if (oldest.done) return;
    recordings.delete(oldest.value);
  }
}

function isRecordable(room) {
  return !!(room && room.id && room.gameState && Array.isArray(room.playerSockets));
}

// ── begin ──────────────────────────────────────────────────
// Called right after the room (and its initial game state) is created.
function begin(room) {
  return guard('begin', () => {
    if (!isRecordable(room)) return null;
    const state = room.gameState;
    const players = state.players.map((player, index) => {
      const socketEntry = room.playerSockets[index] || {};
      const entry = {
        u: player.username ?? socketEntry.username ?? `player-${index}`,
        a: player.avatarSeed ?? null,
      };
      // teamId only exists in TEAM / ONE_V_TWO games.
      if (player.teamId === 0 || player.teamId === 1) entry.team = player.teamId;
      // `isAI` may be added to playerSockets entries later; read defensively.
      entry.ai = socketEntry.isAI === true;
      return entry;
    });

    const recording = {
      v: 1,
      id: room.id,
      t: room.created || Date.now(),
      e: null,
      mode: state.gameMode,
      layout: state.teamLayout ?? null,
      n: state.numPlayers,
      clock: !!state.timeControl,
      players,
      first: state.currentPlayerIndex,
      setup: {
        board: state.board.map(row => row.map(card => card.id)),
        decks: state.decks.map(deck => deck.map(card => card.id)),
        tiles: state.bonusTiles.map(tile => tile.id),
      },
      actions: [],
    };

    recordings.set(room.id, recording);
    trimActiveRecordings();
    return recording;
  });
}

// ── per-action ─────────────────────────────────────────────
// Maps a gameLogic ActionResult onto the compact action codes (§1).
// Returns the appended entry, or null when the action is not recordable
// (ENTER_RESERVE, CANCEL_GEMS, an unfinished SELECT_GEM, …).
function compactFromActionResult(actionResult) {
  if (!actionResult || typeof actionResult.type !== 'string') return null;
  const actor = actionResult.actingPlayer;
  if (!Number.isInteger(actor)) return null;
  const payload = actionResult.payload || {};

  switch (actionResult.type) {
    case 'SELECT_GEM':
      // gameLogic omits `completed` from the payload once the take is applied.
      if (payload.completed === false) return null;
      if (!Array.isArray(payload.selected)) return null;
      return [actor, 'G', [...payload.selected]];
    case 'TAKE_GEMS_CONFIRMED':
      if (!Array.isArray(payload.selected)) return null;
      return [actor, 'G', [...payload.selected]];
    case 'RESERVE_CARD':
      if (!Number.isInteger(payload.cardId)) return null;
      return [actor, 'R', payload.cardId];
    case 'RESERVE_FROM_DECK':
      if (!Number.isInteger(payload.tier)) return null;
      return [actor, 'RD', payload.tier];
    case 'BUY_CARD':
      if (!Number.isInteger(payload.cardId)) return null;
      return [actor, 'B', payload.cardId, payload.source === 'board' ? 'b' : 'r'];
    case 'CHOOSE_TILE':
      if (!Number.isInteger(payload.tileId)) return null;
      return [actor, 'N', payload.tileId];
    default:
      return null;
  }
}

function onActionResult(room, actionResult) {
  return guard('onActionResult', () => {
    const recording = room && recordings.get(room.id);
    if (!recording) return null;
    const entry = compactFromActionResult(actionResult);
    if (!entry) return null;
    recording.actions.push(entry);
    return entry;
  });
}

function onResign(room, playerIndex) {
  return guard('onResign', () => {
    const recording = room && recordings.get(room.id);
    if (!recording || !Number.isInteger(playerIndex)) return null;
    const entry = [playerIndex, 'X'];
    recording.actions.push(entry);
    return entry;
  });
}

function onTimeout(room, playerIndex) {
  return guard('onTimeout', () => {
    const recording = room && recordings.get(room.id);
    if (!recording || !Number.isInteger(playerIndex)) return null;
    const entry = [playerIndex, 'T'];
    recording.actions.push(entry);
    return entry;
  });
}

// ── result block ───────────────────────────────────────────
// INDIVIDUAL: winners are every non-resigned player sharing the top rank
// (score desc, then fewest cards) — the same ordering gameLogic uses for
// rating changes. Team modes carry gameState.gameResult instead.
function individualWinners(state) {
  const resigned = state.resignedPlayers || [];
  const candidates = state.players
    .map((player, index) => ({ index, score: player.score, cards: player.cards.length }))
    .filter(entry => !resigned.includes(entry.index));
  if (candidates.length === 0) return [];
  candidates.sort((a, b) => (b.score !== a.score ? b.score - a.score : a.cards - b.cards));
  const best = candidates[0];
  return candidates
    .filter(entry => entry.score === best.score && entry.cards === best.cards)
    .map(entry => entry.index);
}

// index.js only passes `excludeResigned` to applyRoomRatings when an INDIVIDUAL
// game ends inside resignPlayer()/eliminateTimedOutPlayer() — i.e. the very last
// thing that happened was that resignation or timeout. A game that ends on a
// normal move credits every seat, resigned ones included, so the stored deltas
// must follow the same rule.
function endedOnResignation(recording) {
  const actions = recording && Array.isArray(recording.actions) ? recording.actions : [];
  const last = actions[actions.length - 1];
  return Array.isArray(last) && (last[1] === 'X' || last[1] === 'T');
}

function buildResult(room, recording) {
  const state = room.gameState;
  const scores = state.players.map(player => player.score);
  const cards = state.players.map(player => player.cards.length);
  const resigned = [...(state.resignedPlayers || [])].sort((a, b) => a - b);
  const ratingChanges = Array.isArray(room.ratingChanges)
    ? [...room.ratingChanges]
    : new Array(state.numPlayers).fill(0);

  if (state.gameMode === 'TEAM' || state.gameMode === 'ONE_V_TWO') {
    return {
      scores,
      cards,
      resigned,
      winners: null,
      winningTeamIds: state.gameResult?.winningTeamIds ?? null,
      reason: state.gameResult?.reason ?? null,
      rating: ratingChanges,
    };
  }

  // An INDIVIDUAL game that ran out of active players ended by forfeit.
  const forfeited = resigned.length >= state.numPlayers - 1 && resigned.length > 0;
  // Mirror applyRoomRatings(room, excludeResigned): only the resign/timeout
  // endings skip resigned seats. When the game ended on a normal move the server
  // credited every seat, so store room.ratingChanges verbatim.
  const rating = endedOnResignation(recording)
    ? ratingChanges.map((delta, seat) => (resigned.includes(seat) ? 0 : delta))
    : ratingChanges;
  return {
    scores,
    cards,
    resigned,
    winners: individualWinners(state),
    winningTeamIds: null,
    reason: forfeited ? 'FORFEIT' : 'SCORE',
    rating,
  };
}

// ── finish / discard ───────────────────────────────────────
// Call once the room reached GAME_OVER *and* broadcastProcessedAction has
// run, so room.ratingChanges is populated.
function finish(room) {
  return guard('finish', () => {
    const recording = room && recordings.get(room.id);
    if (!recording) return null;
    recordings.delete(room.id);
    if (!isRecordable(room)) return null;

    recording.e = Date.now();
    recording.result = buildResult(room, recording);
    replayStore.add(recording);
    return recording;
  });
}

function discard(room) {
  return guard('discard', () => {
    if (!room || !room.id) return false;
    return recordings.delete(room.id);
  });
}

function activeCount() {
  return recordings.size;
}

module.exports = {
  begin,
  onActionResult,
  onResign,
  onTimeout,
  finish,
  discard,
  activeCount,
  compactFromActionResult,
};
