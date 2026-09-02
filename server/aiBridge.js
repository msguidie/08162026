// ═══════════════════════════════════════════════════════════
// AI bridge — Render server ⇄ local GPU worker (docs/AI_BRIDGE.md §1).
//
// The worker is a socket.io CLIENT that connects out to this server, so a
// free-tier deploy needs no inbound port. One worker is active at a time.
//
// This module drives bot seats: after every broadcast the server asks
// `maybeAct(room)`; if the seat to move is a bot, the bridge waits a short
// UX beat, asks the worker for a move, and applies the answer through the
// SAME `applyGameAction` path a human socket uses. Missing worker, timeout
// or an illegal answer all fall through to `aiFallback`.
//
// Nothing in here may throw into the game flow — every entry point is wrapped.
// ═══════════════════════════════════════════════════════════

const { clientViewForPlayer } = require('./gameLogic');
const aiFallback = require('./aiFallback');

const DEFAULT_DELAY_MS = 600;      // UX pacing before asking for a move
const DEFAULT_DEADLINE_MS = 15000; // contract §1 budget

function envInt(name, fallback) {
  const raw = Number.parseInt(process.env[name] ?? '', 10);
  return Number.isFinite(raw) && raw >= 0 ? raw : fallback;
}

const config = {
  delayMs: envInt('AI_MOVE_DELAY_MS', DEFAULT_DELAY_MS),
  deadlineMs: envInt('AI_MOVE_DEADLINE_MS', DEFAULT_DEADLINE_MS),
};

// Injected by server/index.js so this module never requires it back.
const deps = {
  getRoom: () => null,
  applyGameAction: () => ({ error: 'AI bridge is not wired up' }),
  resignPlayer: () => ({ error: 'AI bridge is not wired up' }),
};

let worker = null;                     // { socket, name, version, modes }
let requestCounter = 0;
const inFlight = new Map();            // roomId  → entry
const requestIndex = new Map();        // requestId → entry
const knownReserved = new Map();       // roomId  → Set(cardId reserved from the board)

// ── plumbing ───────────────────────────────────────────────

function log(message) {
  console.log(`[ai] ${message}`);
}

/** Opt-in trace of applied worker moves: AI_DEBUG=1. Silent by default. */
function debug(message) {
  if (process.env.AI_DEBUG === '1') console.log(`[ai] ${message}`);
}

function warn(message) {
  console.error(`[ai] ${message}`);
}

function safe(fn, label = 'hook') {
  try {
    return fn();
  } catch (err) {
    warn(`${label} failed: ${err?.message || err}`);
    return undefined;
  }
}

function later(fn, ms) {
  const timer = setTimeout(() => safe(fn, 'timer'), ms);
  if (typeof timer.unref === 'function') timer.unref();
  return timer;
}

function init(next = {}) {
  Object.assign(deps, next);
}

/** Test seam: override the UX delay / deadline. */
function configure(next = {}) {
  if (Number.isFinite(next.delayMs)) config.delayMs = next.delayMs;
  if (Number.isFinite(next.deadlineMs)) config.deadlineMs = next.deadlineMs;
  return { ...config };
}

function isEnabled() {
  return !!process.env.AI_WORKER_SECRET;
}

function isAvailable() {
  return isEnabled() && !!worker;
}

function status() {
  if (!isAvailable()) return { enabled: isEnabled(), available: false };
  return {
    enabled: true,
    available: true,
    name: worker.name,
    modes: worker.modes,
  };
}

/** Test seam: drop all state (worker, timers, per-room knowledge). */
function reset() {
  for (const entry of inFlight.values()) clearTimer(entry);
  inFlight.clear();
  requestIndex.clear();
  knownReserved.clear();
  worker = null;
  requestCounter = 0;
  config.delayMs = envInt('AI_MOVE_DELAY_MS', DEFAULT_DELAY_MS);
  config.deadlineMs = envInt('AI_MOVE_DEADLINE_MS', DEFAULT_DEADLINE_MS);
}

// ── worker socket handlers (attached to every connection) ──

function attach(socket) {
  safe(() => {
    socket.on('ai_worker_register', (data = {}, ack) => safe(() => onRegister(socket, data, ack), 'ai_worker_register'));
    socket.on('ai_move_response', (data = {}, ack) => safe(() => onMoveResponse(socket, data, ack), 'ai_move_response'));
    socket.on('disconnect', () => safe(() => onSocketDisconnect(socket), 'ai worker disconnect'));
  }, 'attach');
}

function onRegister(socket, data, ack) {
  if (!isEnabled()) { ack?.({ error: 'AI is not enabled on this server' }); return; }
  if (!data || data.secret !== process.env.AI_WORKER_SECRET) {
    warn(`worker registration rejected (bad secret) from ${socket.id}`);
    ack?.({ error: 'Invalid worker secret' });
    return;
  }
  const previous = worker;
  worker = {
    socket,
    name: typeof data.name === 'string' && data.name ? data.name.slice(0, 64) : 'worker',
    version: typeof data.version === 'string' ? data.version.slice(0, 32) : null,
    modes: Array.isArray(data.modes) ? data.modes.filter(mode => typeof mode === 'string').slice(0, 8) : [],
  };
  if (previous && previous.socket !== socket) log(`worker ${previous.name} replaced by ${worker.name}`);
  log(`worker registered: ${worker.name}${worker.version ? ` v${worker.version}` : ''} [${worker.modes.join(', ')}]`);
  ack?.({ ok: true });
}

function onSocketDisconnect(socket) {
  if (!worker || worker.socket !== socket) return;
  log(`worker ${worker.name} disconnected — bot seats continue on the fallback policy`);
  worker = null;
  // Anything already asked for is answered by the fallback right away.
  for (const entry of [...inFlight.values()]) {
    if (entry.sent && !entry.applying) {
      clearTimer(entry);
      applyFallback(deps.getRoom(entry.roomId), entry, 'worker disconnected');
    }
  }
}

// ── translation table (contract §1) ────────────────────────

function isColorList(colors) {
  return Array.isArray(colors)
    && colors.length >= 1 && colors.length <= 3
    && colors.every(color => Number.isInteger(color) && color >= 0 && color <= 4);
}

/**
 * Worker action → protocol actions understood by `processAction`.
 * @returns {{actions: object[]}|{resign: true}|null} null when the action is malformed.
 */
function translateWorkerAction(action, kind = 'MOVE') {
  if (!action || typeof action.type !== 'string') return null;
  switch (action.type) {
    case 'TAKE_GEMS':
      if (kind === 'TILE' || !isColorList(action.colors)) return null;
      return { actions: [{ type: 'TAKE_GEMS_CONFIRMED', colors: [...action.colors] }] };
    case 'RESERVE_CARD':
      if (kind === 'TILE' || !Number.isInteger(action.cardId)) return null;
      return { actions: [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_CARD', cardId: action.cardId }] };
    case 'RESERVE_FROM_DECK':
      if (kind === 'TILE' || ![1, 2, 3].includes(action.tier)) return null;
      return { actions: [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_FROM_DECK', tier: action.tier }] };
    case 'BUY_CARD':
      if (kind === 'TILE' || !Number.isInteger(action.cardId)) return null;
      if (action.source !== 'board' && action.source !== 'reserved') return null;
      return { actions: [{ type: 'BUY_CARD', cardId: action.cardId, source: action.source }] };
    case 'CHOOSE_TILE':
      if (!Number.isInteger(action.tileId)) return null;
      return { actions: [{ type: 'CHOOSE_TILE', tileId: action.tileId }] };
    case 'RESIGN':
    case 'NONE':
      return { resign: true };
    default:
      return null;
  }
}

// ── known reserved cards (public knowledge) ────────────────

/**
 * Hook called next to the replay hook in `applyGameAction`: a card reserved
 * FROM THE BOARD is public knowledge, one taken from a deck stays secret.
 */
function onActionResult(room, actionResult) {
  safe(() => {
    if (!room?.id || !actionResult) return;
    if (actionResult.type !== 'RESERVE_CARD') return;
    const cardId = actionResult.payload?.cardId;
    if (!Number.isInteger(cardId)) return;
    if (!knownReserved.has(room.id)) knownReserved.set(room.id, new Set());
    knownReserved.get(room.id).add(cardId);
  }, 'onActionResult');
}

function knownReservedIds(roomId) {
  return knownReserved.get(roomId) || new Set();
}

// ── observation payload (contract §1) ──────────────────────

function buildObservation(room, playerIndex) {
  const state = room.gameState;
  const view = clientViewForPlayer(state, playerIndex);
  const known = knownReservedIds(room.id);
  view.players = view.players.map((player, index) => {
    if (index === playerIndex) return player;
    const actual = state.players[index]?.reserved || [];
    return {
      ...player,
      reserved: actual.map(card => ({
        id: known.has(card.id) ? card.id : -1,
        tier: card.tier,
        hidden: true,
        known: known.has(card.id),
      })),
    };
  });
  return view;
}

function buildRequest(room, entry) {
  return {
    requestId: entry.requestId,
    roomId: room.id,
    playerIndex: entry.playerIndex,
    kind: entry.kind,
    deadlineMs: entry.deadlineAt,
    state: buildObservation(room, entry.playerIndex),
    knownReserved: [...knownReservedIds(room.id)],
    pendingTileChoice: room.gameState._pendingTileChoice || null,
  };
}

// ── turn driver ────────────────────────────────────────────

function clearTimer(entry) {
  if (entry?.timer) { clearTimeout(entry.timer); entry.timer = null; }
}

function release(entry) {
  clearTimer(entry);
  requestIndex.delete(entry.requestId);
  if (inFlight.get(entry.roomId) === entry) inFlight.delete(entry.roomId);
}

function cancelWithWorker(entry) {
  if (entry.sent && worker) {
    safe(() => worker.socket.emit('ai_move_cancel', { requestId: entry.requestId }), 'ai_move_cancel');
  }
}

function isStale(entry) {
  const room = deps.getRoom(entry.roomId);
  if (!room?.gameState) return true;
  const state = room.gameState;
  return state.phase !== 'PLAYING'
    || state.turnNumber !== entry.turnNumber
    || state.currentPlayerIndex !== entry.playerIndex;
}

function isBotSeat(room, playerIndex) {
  return room?.playerSockets?.[playerIndex]?.isAI === true;
}

/**
 * Called after the `game_start` emissions and at the end of every
 * `broadcastProcessedAction`. No-op unless a bot has to move.
 */
function maybeAct(room) {
  safe(() => {
    if (!isEnabled()) return;
    if (!room?.id || !room.gameState) return;
    const state = room.gameState;
    if (state.phase !== 'PLAYING') { clearRoom(room.id); return; }
    const playerIndex = state.currentPlayerIndex;
    if (!isBotSeat(room, playerIndex)) return;
    if (state.resignedPlayers?.includes(playerIndex)) return;
    if (inFlight.has(room.id)) return; // re-entrancy guard: one request per room

    const entry = {
      requestId: `ai-${++requestCounter}`,
      roomId: room.id,
      playerIndex,
      turnNumber: state.turnNumber,
      kind: Array.isArray(state._pendingTileChoice) && state._pendingTileChoice.length > 0 ? 'TILE' : 'MOVE',
      sent: false,
      applying: false,
      timer: null,
      deadlineAt: 0,
    };
    inFlight.set(room.id, entry);
    requestIndex.set(entry.requestId, entry);
    entry.timer = later(() => dispatch(entry), config.delayMs);
  }, 'maybeAct');
}

function dispatch(entry) {
  entry.timer = null;
  if (inFlight.get(entry.roomId) !== entry) return;
  if (isStale(entry)) { release(entry); return; }
  const room = deps.getRoom(entry.roomId);
  if (!worker) { applyFallback(room, entry, 'no worker connected'); return; }

  entry.sent = true;
  entry.deadlineAt = Date.now() + config.deadlineMs;
  entry.timer = later(() => onDeadline(entry), config.deadlineMs);
  safe(() => worker.socket.emit('ai_move_request', buildRequest(room, entry)), 'ai_move_request');
}

function onDeadline(entry) {
  entry.timer = null;
  if (inFlight.get(entry.roomId) !== entry || entry.applying) return;
  cancelWithWorker(entry);
  applyFallback(deps.getRoom(entry.roomId), entry, `worker did not answer within ${config.deadlineMs} ms`);
}

function onMoveResponse(socket, data, ack) {
  if (!isEnabled()) { ack?.({ error: 'AI is not enabled on this server' }); return; }
  if (!worker || worker.socket !== socket) { ack?.({ error: 'Not the active worker' }); return; }

  const entry = requestIndex.get(data?.requestId);
  // Stale-response protection: unknown id, superseded request, or a reply
  // that arrives before the request was even sent.
  if (!entry || !entry.sent || entry.applying || inFlight.get(entry.roomId) !== entry) {
    ack?.({ error: 'Unknown or expired request' });
    return;
  }
  clearTimer(entry);
  if (isStale(entry)) {
    release(entry);
    ack?.({ error: 'The position changed before the answer arrived' });
    return;
  }

  const room = deps.getRoom(entry.roomId);
  const translated = translateWorkerAction(data.action, entry.kind);
  if (!translated) {
    applyFallback(room, entry, `worker returned an unusable action: ${JSON.stringify(data?.action)}`);
    ack?.({ ok: true });
    return;
  }
  applyChoice(room, entry, translated, 'worker');
  ack?.({ ok: true });
}

/** Validates the sequence black-box, then applies it (or falls back). */
function applyChoice(room, entry, translated, source) {
  if (!room?.gameState) { release(entry); return; }
  if (!translated.resign
    && !aiFallback.isLegalSequence(room.gameState, entry.playerIndex, translated.actions)) {
    applyFallback(room, entry, `${source} action rejected by the rules engine: ${JSON.stringify(translated.actions)}`);
    return;
  }
  runSequence(room, entry, translated, source);
}

function applyFallback(room, entry, reason) {
  if (inFlight.get(entry.roomId) !== entry) return;
  warn(`fallback for ${entry.roomId} seat ${entry.playerIndex}: ${reason}`);
  if (!room?.gameState) { release(entry); return; }
  const choice = entry.kind === 'TILE'
    ? aiFallback.chooseTileActions(room.gameState, entry.playerIndex)
    : aiFallback.chooseFallbackActions(room.gameState, entry.playerIndex);
  const translated = aiFallback.isNone(choice) ? { resign: true } : { actions: choice };
  if (translated.resign) log(`seat ${entry.playerIndex} of ${entry.roomId} has no legal move — resigning`);
  runSequence(room, entry, translated, 'fallback');
}

function runSequence(room, entry, translated, source) {
  const state = room.gameState;
  const turnBefore = state.turnNumber;
  const phaseBefore = state.phase;
  // The entry stays in `inFlight` while applying: each applied action
  // broadcasts, and every broadcast calls maybeAct() again.
  entry.applying = true;
  let failure = null;

  safe(() => {
    if (translated.resign) {
      deps.resignPlayer(room, entry.playerIndex);
      return;
    }
    for (const action of translated.actions) {
      const result = deps.applyGameAction(room, entry.playerIndex, action) || {};
      if (result.error) { failure = result.error; break; }
    }
  }, 'apply');

  const after = deps.getRoom(entry.roomId);
  // A buy that qualifies two or more nobles legitimately leaves the turn where
  // it is with `_pendingTileChoice` set: the seat still owes a CHOOSE_TILE, and
  // the maybeAct() below asks the worker for it (kind 'TILE'). That is progress,
  // not a stall — treating it as one would hand every noble choice to the
  // fallback and make TILE requests unreachable.
  const awaitingTileChoice = !!after?.gameState?._pendingTileChoice?.length;
  const stalled = !!after?.gameState
    && after.gameState.phase === 'PLAYING'
    && after.gameState.phase === phaseBefore
    && after.gameState.turnNumber === turnBefore
    && after.gameState.currentPlayerIndex === entry.playerIndex
    && !awaitingTileChoice
    && !translated.resign;

  if (failure) warn(`${source} action failed for ${entry.roomId} seat ${entry.playerIndex}: ${failure}`);

  if (stalled && source !== 'fallback') {
    entry.applying = false;
    applyFallback(after, entry, `${source} sequence left the turn unfinished`);
    return;
  }
  if (stalled) {
    // The fallback itself could not finish the turn — resign so the game
    // can never wedge on a bot seat.
    entry.applying = false;
    warn(`fallback could not finish the turn for ${entry.roomId} seat ${entry.playerIndex} — resigning`);
    safe(() => deps.resignPlayer(after, entry.playerIndex), 'resign');
  }

  if (source === 'worker' && !failure && !stalled) {
    const kind = translated.resign
      ? 'RESIGN'
      : translated.actions.map(a => a.type).join('+');
    debug(`applied ${kind} from worker for ${entry.roomId} seat ${entry.playerIndex}`);
  }

  entry.applying = false;
  release(entry);
  maybeAct(deps.getRoom(entry.roomId));
}

/** Room deleted or finished: drop pending work and tell the worker to stop. */
function clearRoom(roomId) {
  safe(() => {
    const entry = inFlight.get(roomId);
    if (entry) {
      if (entry.applying) return;
      clearTimer(entry);
      cancelWithWorker(entry);
      release(entry);
    }
    knownReserved.delete(roomId);
  }, 'clearRoom');
}

module.exports = {
  init,
  attach,
  maybeAct,
  onActionResult,
  clearRoom,
  status,
  isEnabled,
  isAvailable,
  configure,
  reset,
  translateWorkerAction,
  buildObservation,
  // test seams
  _inFlightCount: () => inFlight.size,
  _worker: () => worker,
};
