#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════
// Mock AI worker (dev + tests) — docs/AI_BRIDGE.md §1.
//
// Stands in for splendor_ai/worker/*: registers over socket.io with
// AI_WORKER_SECRET and answers every `ai_move_request` with a RANDOM LEGAL
// action. Legality is decided black-box, by replaying the candidate through
// the real `server/gameLogic.js` on a state rebuilt from the request payload.
//
// The payload never contains the decks (they are server-only), so the local
// validation state gets placeholder deck cards matching `deckCounts` — enough
// for RESERVE_FROM_DECK and board refills, and it never leaves this process.
//
//   AI_WORKER_SECRET=dev node scripts/dev/mockAiWorker.mjs
//
// Env: SERVER_URL (default http://127.0.0.1:10000), AI_WORKER_SECRET,
//      MOCK_AI_NAME, MOCK_AI_THINK_MS (fake thinking time), MOCK_AI_VERBOSE.
// ═══════════════════════════════════════════════════════════

import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { io } from 'socket.io-client';

const require = createRequire(import.meta.url);
const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const { processAction } = require(path.join(rootDir, 'server/gameLogic.js'));

const SERVER_URL = process.env.SERVER_URL || 'http://127.0.0.1:10000';
const SECRET = process.env.AI_WORKER_SECRET || '';
const NAME = process.env.MOCK_AI_NAME || 'mock-worker';
const THINK_MS = Number.parseInt(process.env.MOCK_AI_THINK_MS || '0', 10) || 0;
const VERBOSE = process.env.MOCK_AI_VERBOSE === '1';

function log(...args) { console.log('[mock-ai]', ...args); }
function debug(...args) { if (VERBOSE) log(...args); }

function shuffled(list) {
  const copy = [...list];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

// ── local validation state ─────────────────────────────────

function placeholderCard(tier, index) {
  return { id: -1000 - index, tier, reward: 0, points: 0, cost: [0, 0, 0, 0, 0] };
}

function validationState(payloadState) {
  const state = structuredClone(payloadState);
  const counts = state.deckCounts || [0, 0, 0];
  state.decks = [0, 1, 2].map(tierIndex =>
    Array.from({ length: counts[tierIndex] || 0 }, (_, i) => placeholderCard(tierIndex + 1, i)));
  return state;
}

// The bridge translates worker actions to the protocol; mirror that table here
// so "legal for the worker" means exactly "legal for the server".
function toProtocol(action) {
  switch (action.type) {
    case 'TAKE_GEMS': return [{ type: 'TAKE_GEMS_CONFIRMED', colors: action.colors }];
    case 'RESERVE_CARD': return [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_CARD', cardId: action.cardId }];
    case 'RESERVE_FROM_DECK': return [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_FROM_DECK', tier: action.tier }];
    case 'BUY_CARD': return [{ type: 'BUY_CARD', cardId: action.cardId, source: action.source }];
    case 'CHOOSE_TILE': return [{ type: 'CHOOSE_TILE', tileId: action.tileId }];
    default: return null;
  }
}

function isLegal(payloadState, playerIndex, action) {
  const actions = toProtocol(action);
  if (!actions) return false;
  try {
    const state = validationState(payloadState);
    for (const step of actions) {
      const result = processAction(state, playerIndex, step);
      if (!result || result.error) return false;
    }
    return true;
  } catch (err) {
    return false;
  }
}

// ── candidate moves ────────────────────────────────────────

function gemCombos() {
  const combos = [];
  for (let a = 0; a < 5; a++) {
    combos.push([a]);
    combos.push([a, a]);
    for (let b = a + 1; b < 5; b++) {
      combos.push([a, b]);
      for (let c = b + 1; c < 5; c++) combos.push([a, b, c]);
    }
  }
  return combos;
}

function moveCandidates(state, playerIndex) {
  const me = state.players[playerIndex];
  const board = [0, 1, 2].flatMap(tier => state.board?.[tier] || []);
  const candidates = [
    ...board.map(card => ({ type: 'BUY_CARD', cardId: card.id, source: 'board' })),
    ...(me.reserved || []).filter(card => card.id >= 0).map(card => ({ type: 'BUY_CARD', cardId: card.id, source: 'reserved' })),
    ...gemCombos().map(colors => ({ type: 'TAKE_GEMS', colors })),
    ...board.map(card => ({ type: 'RESERVE_CARD', cardId: card.id })),
    ...[1, 2, 3].map(tier => ({ type: 'RESERVE_FROM_DECK', tier })),
  ];
  return shuffled(candidates);
}

function chooseAction(request) {
  const { state, playerIndex, kind, pendingTileChoice } = request;
  if (kind === 'TILE') {
    for (const tileId of shuffled(pendingTileChoice || [])) {
      if (isLegal(state, playerIndex, { type: 'CHOOSE_TILE', tileId })) return { type: 'CHOOSE_TILE', tileId };
    }
    return { type: 'NONE' };
  }
  for (const candidate of moveCandidates(state, playerIndex)) {
    if (isLegal(state, playerIndex, candidate)) return candidate;
  }
  return { type: 'NONE' };
}

// ── socket ─────────────────────────────────────────────────

const socket = io(SERVER_URL, { transports: ['websocket'], forceNew: true, reconnection: false });
const cancelled = new Set();

socket.on('connect', () => {
  socket.emit('ai_worker_register', {
    secret: SECRET,
    name: NAME,
    version: '0.1.0-mock',
    modes: ['INDIVIDUAL', 'ONE_V_TWO', 'TEAM'],
  }, (ack = {}) => {
    if (ack.error) {
      log(`registration refused: ${ack.error}`);
      socket.disconnect();
      process.exitCode = 1;
      return;
    }
    log(`registered as ${NAME} at ${SERVER_URL}`);
  });
});

socket.on('ai_move_cancel', ({ requestId } = {}) => {
  cancelled.add(requestId);
  debug(`cancelled ${requestId}`);
});

socket.on('ai_move_request', (request = {}) => {
  const started = Date.now();
  const answer = () => {
    if (cancelled.delete(request.requestId)) return;
    let action;
    try {
      action = chooseAction(request);
    } catch (err) {
      log(`failed to pick a move: ${err?.message || err}`);
      action = { type: 'NONE' };
    }
    debug(`${request.requestId} seat ${request.playerIndex} → ${JSON.stringify(action)}`);
    socket.emit('ai_move_response', {
      requestId: request.requestId,
      action,
      info: { ms: Date.now() - started },
    }, (ack = {}) => {
      if (ack.error) debug(`server refused ${request.requestId}: ${ack.error}`);
    });
  };
  if (THINK_MS > 0) setTimeout(answer, THINK_MS);
  else answer();
});

socket.on('connect_error', err => {
  log(`connection failed: ${err?.message || err}`);
  process.exitCode = 1;
});

socket.on('disconnect', reason => log(`disconnected (${reason})`));

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => { socket.disconnect(); process.exit(0); });
}
