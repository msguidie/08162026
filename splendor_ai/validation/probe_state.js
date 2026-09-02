#!/usr/bin/env node
/**
 * Position probe: build an ARBITRARY position inside the authoritative Node
 * engine, then report what it accepts and what it does.
 *
 * `gen_trajectories.js` can only reach positions that random play reaches;
 * this lets the pytest suite pin down the awkward corners (10-token cap,
 * forced short takes, orphaned noble choices, TEAM final-round revocation,
 * 1v2 excess ties, ...) and still compare against the real gameLogic.js
 * rather than against the Python port's own opinion.
 *
 * stdin  : one JSON request (see REQUEST below)
 * stdout : one JSON response {"resolved": {...}, "results": [...]}
 *          `resolved` is the fully materialised starting position (full decks
 *          included) so the Python side can build the identical state without
 *          having to re-derive gameLogic.js's defaults.
 *
 * REQUEST
 * {
 *   "mode": "INDIVIDUAL"|"TEAM"|"ONE_V_TWO",
 *   "layout": null|"ADJACENT"|"OPPOSITE",
 *   "n": 2..4,
 *   "teams": [teamId per seat]            // optional, defaults per mode
 *   "state": {                            // every field optional
 *     "board":  [[cardId,...] x3],
 *     "decks":  [[cardId,...] x3],        // server order, pop() takes LAST
 *     "tiles":  [tileId,...],
 *     "gems":   [6 ints],
 *     "players":[{ "gems":[6], "cards":[ids], "reserved":[ids],
 *                  "tiles":[ids], "score": int }],
 *     "current": int, "roundStart": int, "turnNumber": int,
 *     "finalRoundTriggeredBy": null|int, "resigned": [int],
 *     "phase": "PLAYING"|"GAME_OVER",
 *     "turnAction": null|{"type":"BUY"|"RESERVE"|"TAKE_GEMS", ...},
 *     "pendingTileChoice": null|[tileId], "gameResult": null|{...}
 *   },
 *   "ops": [ ["probe"], ["apply", <compact code>], ["resign", seat],
 *            ["timeout", seat] ]
 * }
 *
 * Compact codes are the same as gen_trajectories.js:
 *   ["G",[colors]] ["R",cardId] ["RD",tier] ["B",cardId,"b"|"r"] ["N",tileId]
 */

'use strict';

const path = require('path');
const GL = require(path.join(__dirname, '..', '..', 'server', 'gameLogic.js'));
const {
  createInitialGameState, processAction, processResign,
  calculateRatingChanges, getTeamStats, getQualifyingTeamIds,
  ALL_CARDS, ALL_BONUS_TILES,
} = GL;

const CARD = [];
for (const c of ALL_CARDS) CARD[c.id] = c;
const TILE = [];
for (const t of ALL_BONUS_TILES) TILE[t.id] = t;

const SEAT_TEAMS = {
  'ONE_V_TWO': [0, 1, 1],
  'TEAM/ADJACENT': [0, 0, 1, 1],
  'TEAM/OPPOSITE': [0, 1, 0, 1],
};

// ── shared with gen_trajectories.js ───────────────────────────────────────

const GEM_MULTISETS = (() => {
  const out = [];
  for (let a = 0; a < 5; a++) out.push([a]);
  for (let a = 0; a < 5; a++) for (let b = a; b < 5; b++) out.push([a, b]);
  for (let a = 0; a < 5; a++) for (let b = a; b < 5; b++) for (let c = b; c < 5; c++) out.push([a, b, c]);
  return out;
})();

function codeToMessages(code) {
  switch (code[0]) {
    case 'G': return [{ type: 'TAKE_GEMS_CONFIRMED', colors: code[1].slice() }];
    case 'R': return [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_CARD', cardId: code[1] }];
    case 'RD': return [{ type: 'ENTER_RESERVE' }, { type: 'RESERVE_FROM_DECK', tier: code[1] }];
    case 'B': return [{ type: 'BUY_CARD', cardId: code[1], source: code[2] === 'b' ? 'board' : 'reserved' }];
    case 'N': return [{ type: 'CHOOSE_TILE', tileId: code[1] }];
    default: throw new Error(`bad code ${JSON.stringify(code)}`);
  }
}

function candidateCodes(state) {
  const out = [];
  for (const colors of GEM_MULTISETS) out.push(['G', colors]);
  for (let t = 0; t < 3; t++) for (let s = 0; s < 4; s++) if (state.board[t][s]) out.push(['R', state.board[t][s].id]);
  for (let t = 0; t < 3; t++) out.push(['RD', t + 1]);
  for (let t = 0; t < 3; t++) for (let s = 0; s < 4; s++) if (state.board[t][s]) out.push(['B', state.board[t][s].id, 'b']);
  for (const c of state.players[state.currentPlayerIndex].reserved) out.push(['B', c.id, 'r']);
  for (const t of state.bonusTiles) out.push(['N', t.id]);
  return out;
}

function cloneState(s) {
  return {
    ...s,
    board: [s.board[0].slice(), s.board[1].slice(), s.board[2].slice()],
    decks: [s.decks[0].slice(), s.decks[1].slice(), s.decks[2].slice()],
    deckCounts: s.deckCounts.slice(),
    gems: s.gems.slice(),
    bonusTiles: s.bonusTiles.slice(),
    players: s.players.map(p => ({
      ...p, gems: p.gems.slice(), cards: p.cards.slice(),
      reserved: p.reserved.slice(), bonusTiles: p.bonusTiles.slice(),
    })),
    resignedPlayers: (s.resignedPlayers || []).slice(),
    turnAction: s.turnAction
      ? (s.turnAction.selected ? { ...s.turnAction, selected: s.turnAction.selected.slice() } : { ...s.turnAction })
      : s.turnAction,
    gameResult: s.gameResult ? { ...s.gameResult } : s.gameResult,
    _pendingTileChoice: s._pendingTileChoice ? s._pendingTileChoice.slice() : s._pendingTileChoice,
    _tileClaimed: s._tileClaimed ? { ...s._tileClaimed } : s._tileClaimed,
    timeControl: null,
  };
}

function snap(s) {
  return {
    b: [s.board[0].map(c => c.id), s.board[1].map(c => c.id), s.board[2].map(c => c.id)],
    dc: s.deckCounts.slice(),
    dt: [0, 1, 2].map(t => (s.decks[t].length ? s.decks[t][s.decks[t].length - 1].id : -1)),
    g: s.gems.slice(),
    tl: s.bonusTiles.map(t => t.id),
    p: s.players.map(p => [p.gems.slice(), p.cards.map(c => c.id), p.reserved.map(c => c.id), p.bonusTiles.map(t => t.id), p.score]),
    cp: s.currentPlayerIndex,
    rs: s.roundStartPlayer === undefined ? null : s.roundStartPlayer,
    tn: s.turnNumber,
    ph: s.phase,
    fr: s.finalRoundTriggeredBy === undefined ? null : s.finalRoundTriggeredBy,
    rg: (s.resignedPlayers || []).slice(),
    gr: s.gameResult || null,
    pt: s._pendingTileChoice || null,
    ta: s.turnAction ? s.turnAction.type : null,
  };
}

function eventOf(results, tileClaimed, goldTaken) {
  const last = results[results.length - 1];
  const pl = last.payload || {};
  const ev = { type: last.type };
  for (const k of ['selected', 'gemsReturned', 'tier', 'cardId', 'source',
                   'reward', 'points', 'tileId', 'fromDeck']) {
    if (pl[k] !== undefined) ev[k] = Array.isArray(pl[k]) ? pl[k].slice() : pl[k];
  }
  if (goldTaken !== undefined) ev.goldTaken = goldTaken;
  ev.tileClaimed = tileClaimed || null;
  return ev;
}

// ── state construction ────────────────────────────────────────────────────

function buildState(req) {
  const mode = req.mode || 'INDIVIDUAL';
  const layout = mode === 'TEAM' ? (req.layout || 'ADJACENT') : null;
  const n = req.n || 2;
  const teams = req.teams
    || (mode === 'INDIVIDUAL' ? null
      : SEAT_TEAMS[mode === 'ONE_V_TWO' ? 'ONE_V_TWO' : `TEAM/${layout}`]);

  const infos = [];
  for (let i = 0; i < n; i++) {
    infos.push({ username: `p${i}`, avatarSeed: i, ...(teams ? { teamId: teams[i] } : {}) });
  }
  const opts = { gameMode: mode, unlimitedTime: true, firstPlayerIndex: (req.state && req.state.current) || 0 };
  if (mode === 'TEAM') opts.teamLayout = layout;
  const s = createInitialGameState(infos, opts);
  s.timeControl = null;

  const o = req.state || {};
  if (o.board) s.board = o.board.map(row => row.map(id => CARD[id]));
  if (o.decks) s.decks = o.decks.map(row => row.map(id => CARD[id]));
  s.deckCounts = [s.decks[0].length, s.decks[1].length, s.decks[2].length];
  if (o.tiles) s.bonusTiles = o.tiles.map(id => TILE[id]);
  if (o.gems) s.gems = o.gems.slice();
  if (o.players) {
    o.players.forEach((pp, i) => {
      const p = s.players[i];
      if (pp.gems) p.gems = pp.gems.slice();
      if (pp.cards) p.cards = pp.cards.map(id => CARD[id]);
      if (pp.reserved) p.reserved = pp.reserved.map(id => CARD[id]);
      if (pp.tiles) p.bonusTiles = pp.tiles.map(id => TILE[id]);
      p.score = pp.score !== undefined
        ? pp.score
        : p.cards.reduce((a, c) => a + c.points, 0) + p.bonusTiles.reduce((a, t) => a + t.points, 0);
    });
  }
  if (o.current !== undefined) s.currentPlayerIndex = o.current;
  s.roundStartPlayer = o.roundStart !== undefined ? o.roundStart : s.currentPlayerIndex;
  if (o.turnNumber !== undefined) s.turnNumber = o.turnNumber;
  s.finalRoundTriggeredBy = o.finalRoundTriggeredBy === undefined ? null : o.finalRoundTriggeredBy;
  s.resignedPlayers = (o.resigned || []).slice();
  if (o.phase) s.phase = o.phase;
  s.turnAction = o.turnAction === undefined ? null : o.turnAction;
  if (o.pendingTileChoice) s._pendingTileChoice = o.pendingTileChoice.slice();
  if (o.gameResult !== undefined) s.gameResult = o.gameResult;
  return s;
}

// ── main ──────────────────────────────────────────────────────────────────

function resolved(s, req) {
  return {
    mode: s.gameMode, layout: s.teamLayout, n: s.numPlayers,
    teams: s.players.map(p => (p.teamId === undefined ? null : p.teamId)),
    board: [0, 1, 2].map(t => s.board[t].map(c => c.id)),
    decks: [0, 1, 2].map(t => s.decks[t].map(c => c.id)),
    tiles: s.bonusTiles.map(t => t.id),
    gems: s.gems.slice(),
    players: s.players.map(p => ({
      gems: p.gems.slice(), cards: p.cards.map(c => c.id),
      reserved: p.reserved.map(c => c.id), tiles: p.bonusTiles.map(t => t.id),
      score: p.score,
    })),
    current: s.currentPlayerIndex, roundStart: s.roundStartPlayer,
    turnNumber: s.turnNumber, phase: s.phase,
    finalRoundTriggeredBy: s.finalRoundTriggeredBy === undefined ? null : s.finalRoundTriggeredBy,
    resigned: (s.resignedPlayers || []).slice(),
    turnAction: s.turnAction ? s.turnAction.type : null,
    pendingTileChoice: s._pendingTileChoice || null,
    gameResult: s.gameResult || null,
    config: s.config,
  };
}

function run(req) {
  const state = buildState(req);
  const resolvedState = resolved(state, req);
  const results = [];
  for (const op of (req.ops || [['probe']])) {
    if (op[0] === 'probe') {
      const accepted = [];
      for (const code of candidateCodes(state)) {
        const c = cloneState(state);
        const pi = state.currentPlayerIndex;
        let ok = true;
        for (const m of codeToMessages(code)) {
          const r = processAction(c, pi, m);
          if (r.error) { ok = false; break; }
        }
        if (ok) accepted.push(code);
      }
      results.push({
        op: 'probe', legal: accepted, snap: snap(state),
        teamStats: getTeamStats(state),
        qualifyingTeamIds: getQualifyingTeamIds(state),
        rating: calculateRatingChanges(state.players, state),
      });
    } else if (op[0] === 'apply') {
      const pi = state.currentPlayerIndex;
      const rs = [];
      let goldTaken;
      let error = null;
      for (const m of codeToMessages(op[1])) {
        const r = processAction(state, pi, m);
        if (r.error) { error = r.error; break; }
        rs.push(r.result);
        if (m.type === 'ENTER_RESERVE') goldTaken = r.result.payload.goldTaken;
      }
      const tc = state._tileClaimed;
      if (tc) delete state._tileClaimed;
      results.push({
        op: 'apply', error,
        ev: error ? null : eventOf(rs, tc, goldTaken),
        snap: snap(state),
        rating: calculateRatingChanges(state.players, state),
      });
    } else if (op[0] === 'select') {
      // incremental SELECT_GEM, one colour
      const r = processAction(state, state.currentPlayerIndex,
        { type: 'SELECT_GEM', color: op[1] });
      const tc = state._tileClaimed;
      if (tc) delete state._tileClaimed;
      results.push({
        op: 'select', error: r.error || null,
        completed: r.completed === undefined ? null : r.completed,
        payload: r.result ? r.result.payload : null,
        snap: snap(state),
      });
    } else if (op[0] === 'resign' || op[0] === 'timeout') {
      const who = op[1] === undefined ? state.currentPlayerIndex : op[1];
      processResign(state, who);
      if (op[0] === 'timeout'
        && state.numPlayers - (state.resignedPlayers || []).length < 2) {
        state.phase = 'GAME_OVER';
      }
      const tc = state._tileClaimed;
      if (tc) delete state._tileClaimed;
      results.push({
        op: op[0], snap: snap(state), tileClaimed: tc || null,
        rating: calculateRatingChanges(state.players, state),
      });
    } else {
      throw new Error(`unknown op ${JSON.stringify(op)}`);
    }
  }
  return { resolved: resolvedState, results };
}

let raw = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', d => { raw += d; });
process.stdin.on('end', () => {
  let out;
  try {
    out = run(JSON.parse(raw));
  } catch (e) {
    out = { error: String(e && e.stack || e) };
  }
  process.stdout.write(JSON.stringify(out));
});
