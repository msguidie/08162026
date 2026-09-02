#!/usr/bin/env node
/**
 * Cross-language validation: random self-play driven by the AUTHORITATIVE
 * engine (`server/gameLogic.js`), dumping every step in a form the Python
 * checker (`replay_check.py`) can replay and compare against.
 *
 * The set of legal actions is determined BLACK-BOX: every candidate action is
 * tried on a copy of the live state and kept only if `processAction` returns
 * `ok`.  Nothing about legality is re-derived here, so the dump is an
 * independent oracle for `splendor_ai.rules.engine.legal_mask`.
 *
 * Usage:
 *   node gen_trajectories.js --out data/ind2.jsonl.gz --games 2000 \
 *        --mode INDIVIDUAL --players 2 --seed 1
 *
 * Options:
 *   --out PATH          output JSONL (".gz" suffix => gzip)            [required]
 *   --games N           games to generate                              [100]
 *   --mode M            INDIVIDUAL | TEAM | ONE_V_TWO                  [INDIVIDUAL]
 *   --players N         seats (ignored for TEAM=4 / ONE_V_TWO=3)       [2]
 *   --layout L          ADJACENT | OPPOSITE (TEAM only)                [ADJACENT]
 *   --seed N            base PRNG seed                                 [1]
 *   --max-steps N       abort a game after this many steps             [800]
 *   --chaos-frac F      fraction of games that inject resign/timeout   [0.3]
 *   --resign-p P        per-step resign probability in a chaos game    [0.02]
 *   --timeout-p P       per-step timeout probability in a chaos game   [0.01]
 *   --incremental-p P   fraction of gem takes fed through SELECT_GEM   [0.3]
 *   --buy-bias P        per-step chance of preferring a buy (coverage)    [0]
 *   --orphan-hunt       right after a CHOOSE_TILE, prefer a NON-buy action so
 *                       that a still-qualifying second noble produces the
 *                       "orphaned pending choice" state (turnAction === null
 *                       while _pendingTileChoice is set)
 *   --t1-bias P         per-step chance of preferring a cheap tier-1 buy;
 *                       grows tableaus without growing scores, which is how
 *                       three-noble (and orphaned-choice) states are reached  [0]
 *   --perm-check-p P    fraction of steps that brute-force every colour
 *                       ordering of every gem multiset                 [0.02]
 *   --replay-only       emit only the replay-format record per game
 *   --quiet             no progress on stderr
 *
 * Output records (one JSON object per line):
 *   {"k":"game", ...}   setup + metadata
 *   {"k":"step", ...}   one completed turn action (see below)
 *   {"k":"end",  ...}   terminal state, ratings, full remaining decks, replay
 *
 * A "step" record:
 *   i      step index within the game
 *   actor  seat that acted
 *   a      compact replay action WITH the seat, e.g. [0,"G",[0,1,2]]
 *   legal  every accepted candidate BEFORE the action, as compact codes
 *          without the seat: ["G",[0,1,2]] | ["R",id] | ["RD",tier] |
 *          ["B",id,"b"|"r"] | ["N",tileId].  These map 1:1 onto protocol
 *          messages (see PROTOCOL below).
 *   via    "confirm" | "incremental" | "resign" | "timeout" | "stuck-resign"
 *   ev     result payload fields: {type, selected?, gemsReturned?, goldTaken?,
 *          tier?, cardId?, source?, tileId?, tileClaimed?}
 *   s      post-action snapshot (see snap())
 *
 * PROTOCOL mapping for `legal` / `a` codes:
 *   ["G",colors]  -> [{type:"TAKE_GEMS_CONFIRMED",colors}]
 *   ["R",cardId]  -> [{type:"ENTER_RESERVE"},{type:"RESERVE_CARD",cardId}]
 *   ["RD",tier]   -> [{type:"ENTER_RESERVE"},{type:"RESERVE_FROM_DECK",tier}]
 *   ["B",id,src]  -> [{type:"BUY_CARD",cardId:id,source:src==="b"?"board":"reserved"}]
 *   ["N",tileId]  -> [{type:"CHOOSE_TILE",tileId}]
 *   ["X"] / ["T"] -> processResign / eliminateTimedOutPlayer
 */

'use strict';

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

// ── seeded PRNG, installed BEFORE gameLogic.js is required ────────────────

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let RNG = mulberry32(1);
Math.random = () => RNG();

const GL = require(path.join(__dirname, '..', '..', 'server', 'gameLogic.js'));
const {
  createInitialGameState, processAction, processResign,
  calculateRatingChanges, ALL_CARDS,
} = GL;

const CARD_BY_ID = [];
for (const c of (ALL_CARDS || [])) CARD_BY_ID[c.id] = c;

// ── CLI ───────────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const o = {
    out: null, games: 100, mode: 'INDIVIDUAL', players: 2, layout: 'ADJACENT',
    seed: 1, maxSteps: 800, chaosFrac: 0.3, resignP: 0.02, timeoutP: 0.01,
    incrementalP: 0.3, permCheckP: 0.02, replayOnly: false, quiet: false,
    buyBias: 0, t1Bias: 0, orphanHunt: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    const v = argv[i + 1];
    switch (k) {
      case '--out': o.out = v; i++; break;
      case '--games': o.games = Number(v); i++; break;
      case '--mode': o.mode = v; i++; break;
      case '--players': o.players = Number(v); i++; break;
      case '--layout': o.layout = v; i++; break;
      case '--seed': o.seed = Number(v); i++; break;
      case '--max-steps': o.maxSteps = Number(v); i++; break;
      case '--chaos-frac': o.chaosFrac = Number(v); i++; break;
      case '--resign-p': o.resignP = Number(v); i++; break;
      case '--timeout-p': o.timeoutP = Number(v); i++; break;
      case '--incremental-p': o.incrementalP = Number(v); i++; break;
      case '--perm-check-p': o.permCheckP = Number(v); i++; break;
      case '--buy-bias': o.buyBias = Number(v); i++; break;
      case '--t1-bias': o.t1Bias = Number(v); i++; break;
      case '--orphan-hunt': o.orphanHunt = true; break;
      case '--replay-only': o.replayOnly = true; break;
      case '--quiet': o.quiet = true; break;
      default:
        if (k.startsWith('--')) { console.error(`unknown option ${k}`); process.exit(2); }
    }
  }
  if (!o.out) { console.error('--out is required'); process.exit(2); }
  if (o.mode === 'ONE_V_TWO') o.players = 3;
  if (o.mode === 'TEAM') o.players = 4;
  return o;
}

// ── candidate space ───────────────────────────────────────────────────────

/** All 55 colour multisets of size 1..3, each in sorted (canonical) order. */
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
  for (let t = 0; t < 3; t++) {
    for (let s = 0; s < 4; s++) {
      const c = state.board[t][s];
      if (c) out.push(['R', c.id]);
    }
  }
  for (let t = 0; t < 3; t++) out.push(['RD', t + 1]);
  for (let t = 0; t < 3; t++) {
    for (let s = 0; s < 4; s++) {
      const c = state.board[t][s];
      if (c) out.push(['B', c.id, 'b']);
    }
  }
  const p = state.players[state.currentPlayerIndex];
  for (const c of p.reserved) out.push(['B', c.id, 'r']);
  for (const t of state.bonusTiles) out.push(['N', t.id]);
  return out;
}

// ── fast structural clone ─────────────────────────────────────────────────
// Card and tile objects are never mutated by processAction, so sharing the
// references is safe and ~20x faster than structuredClone.

function cloneState(s) {
  return {
    phase: s.phase,
    board: [s.board[0].slice(), s.board[1].slice(), s.board[2].slice()],
    decks: [s.decks[0].slice(), s.decks[1].slice(), s.decks[2].slice()],
    deckCounts: s.deckCounts.slice(),
    gems: s.gems.slice(),
    bonusTiles: s.bonusTiles.slice(),
    players: s.players.map(p => ({
      username: p.username, avatarSeed: p.avatarSeed, teamId: p.teamId,
      gems: p.gems.slice(), cards: p.cards.slice(), reserved: p.reserved.slice(),
      bonusTiles: p.bonusTiles.slice(), score: p.score,
    })),
    currentPlayerIndex: s.currentPlayerIndex,
    roundStartPlayer: s.roundStartPlayer,
    turnAction: s.turnAction
      ? (s.turnAction.selected
        ? { ...s.turnAction, selected: s.turnAction.selected.slice() }
        : { ...s.turnAction })
      : s.turnAction,
    finalRoundTriggeredBy: s.finalRoundTriggeredBy,
    turnNumber: s.turnNumber,
    numPlayers: s.numPlayers,
    config: s.config,
    resignedPlayers: (s.resignedPlayers || []).slice(),
    gameMode: s.gameMode,
    teamLayout: s.teamLayout,
    teams: s.teams,
    gameResult: s.gameResult ? { ...s.gameResult } : s.gameResult,
    timeControl: null,
    _pendingTileChoice: s._pendingTileChoice ? s._pendingTileChoice.slice() : s._pendingTileChoice,
    _tileClaimed: s._tileClaimed ? { ...s._tileClaimed } : s._tileClaimed,
  };
}

/** Run a protocol message sequence on a copy; return the results or null. */
function tryMessages(state, msgs) {
  const c = cloneState(state);
  const pi = state.currentPlayerIndex;
  const results = [];
  for (const m of msgs) {
    const r = processAction(c, pi, m);
    if (r.error) return null;
    results.push(r.result);
  }
  return results;
}

// ── snapshots ─────────────────────────────────────────────────────────────

function snap(s) {
  return {
    b: [s.board[0].map(c => c.id), s.board[1].map(c => c.id), s.board[2].map(c => c.id)],
    dc: s.deckCounts.slice(),
    dt: [0, 1, 2].map(t => (s.decks[t].length ? s.decks[t][s.decks[t].length - 1].id : -1)),
    g: s.gems.slice(),
    tl: s.bonusTiles.map(t => t.id),
    p: s.players.map(p => [
      p.gems.slice(), p.cards.map(c => c.id), p.reserved.map(c => c.id),
      p.bonusTiles.map(t => t.id), p.score,
    ]),
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

/** Result payload fields the Python engine also produces. */
function eventOf(type, results, tileClaimed, goldTaken) {
  const last = results[results.length - 1];
  const pl = last.payload || {};
  const ev = { type: last.type };
  if (pl.selected !== undefined) ev.selected = pl.selected.slice();
  if (pl.gemsReturned !== undefined) ev.gemsReturned = pl.gemsReturned.slice();
  if (pl.tier !== undefined) ev.tier = pl.tier;
  if (pl.cardId !== undefined) ev.cardId = pl.cardId;
  if (pl.source !== undefined) ev.source = pl.source;
  if (pl.reward !== undefined) ev.reward = pl.reward;
  if (pl.points !== undefined) ev.points = pl.points;
  if (pl.tileId !== undefined) ev.tileId = pl.tileId;
  if (pl.fromDeck !== undefined) ev.fromDeck = pl.fromDeck;
  if (goldTaken !== undefined) ev.goldTaken = goldTaken;
  ev.tileClaimed = tileClaimed || null;
  return ev;
}

/** Emulate broadcastProcessedAction()'s consumption of _tileClaimed. */
function consumeTileClaimed(state) {
  const tc = state._tileClaimed;
  if (tc) delete state._tileClaimed;
  return tc || null;
}

// ── permutation brute force (order-independence proof) ────────────────────

function permutations(arr) {
  if (arr.length <= 1) return [arr.slice()];
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const rest = arr.slice(0, i).concat(arr.slice(i + 1));
    for (const p of permutations(rest)) out.push([arr[i]].concat(p));
  }
  return out;
}

/**
 * For every colour multiset, check that acceptance and the resulting selection
 * are identical for every ordering.  Returns the number of orderings checked;
 * throws on any disagreement (which would invalidate the canonical ordering
 * used by the Python action space).
 */
function checkTakeOrderIndependence(state) {
  const pi = state.currentPlayerIndex;
  let checked = 0;
  for (const ms of GEM_MULTISETS) {
    const base = tryMessages(state, [{ type: 'TAKE_GEMS_CONFIRMED', colors: ms.slice() }]);
    const baseOk = base !== null;
    for (const perm of permutations(ms)) {
      const r = tryMessages(state, [{ type: 'TAKE_GEMS_CONFIRMED', colors: perm }]);
      checked++;
      if ((r !== null) !== baseOk) {
        throw new Error(
          `ORDER DEPENDENCE: multiset ${JSON.stringify(ms)} accepted=${baseOk} `
          + `but ordering ${JSON.stringify(perm)} accepted=${r !== null}`);
      }
      if (r !== null) {
        const sel = r[0].payload.selected.slice().sort();
        if (JSON.stringify(sel) !== JSON.stringify(ms)) {
          throw new Error(
            `ORDER DEPENDENCE: ordering ${JSON.stringify(perm)} produced `
            + `${JSON.stringify(sel)} != ${JSON.stringify(ms)}`);
        }
      }
    }
  }
  return checked;
}

// ── one game ──────────────────────────────────────────────────────────────

const SEAT_TEAMS = {
  'ONE_V_TWO': [0, 1, 1],                 // seatOrder [[0,0],[1,0],[1,1]]
  'TEAM/ADJACENT': [0, 0, 1, 1],          // [[0,0],[0,1],[1,0],[1,1]]
  'TEAM/OPPOSITE': [0, 1, 0, 1],          // [[0,0],[1,0],[0,1],[1,1]]
};

function seatTeams(mode, layout, n) {
  if (mode === 'INDIVIDUAL') return null;
  if (mode === 'ONE_V_TWO') return SEAT_TEAMS['ONE_V_TWO'];
  return SEAT_TEAMS[`TEAM/${layout}`];
}

function shuffleInPlace(a, rnd) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function playGame(opt, gameIndex, emit, stats) {
  RNG = mulberry32((opt.seed * 1000003 + gameIndex * 7919) >>> 0);
  const n = opt.players;
  const teams = seatTeams(opt.mode, opt.layout, n);
  const playerInfos = [];
  for (let i = 0; i < n; i++) {
    playerInfos.push({
      username: `p${i}`, avatarSeed: i,
      ...(teams ? { teamId: teams[i] } : {}),
    });
  }
  const options = { gameMode: opt.mode, unlimitedTime: true };
  if (opt.mode === 'TEAM') options.teamLayout = opt.layout;
  // Exercise a random first player for non-1v2 modes.
  if (opt.mode !== 'ONE_V_TWO') options.firstPlayerIndex = Math.floor(RNG() * n);

  const state = createInitialGameState(playerInfos, options);
  state.timeControl = null;

  const setup = {
    board: [0, 1, 2].map(t => state.board[t].map(c => c.id)),
    decks: [0, 1, 2].map(t => state.decks[t].map(c => c.id)),
    tiles: state.bonusTiles.map(t => t.id),
  };
  const gameId = `gen-${opt.mode}-${opt.layout || ''}-${n}-${opt.seed}-${gameIndex}`;

  const gameRec = {
    k: 'game', id: gameId, gi: gameIndex, mode: opt.mode,
    layout: opt.mode === 'TEAM' ? opt.layout : null,
    n, teams: teams ? teams.slice() : null,
    first: state.currentPlayerIndex, setup,
  };
  if (!opt.replayOnly) emit(gameRec);

  const chaos = RNG() < opt.chaosFrac;
  const replayActions = [];
  let preferNonBuyFor = -1;
  let stepIndex = 0;
  let truncated = false;

  while (state.phase === 'PLAYING') {
    if (stepIndex >= opt.maxSteps) { truncated = true; break; }

    // Optional exhaustive ordering check on the live state.
    if (RNG() < opt.permCheckP) {
      stats.permChecks += checkTakeOrderIndependence(state);
    }

    const actor = state.currentPlayerIndex;

    // ── out-of-band events (resign / timeout) ──
    if (chaos && RNG() < opt.timeoutP) {
      // eliminateTimedOutPlayer(): always the CURRENT seat.
      processResign(state, actor);
      if (state.numPlayers - (state.resignedPlayers || []).length < 2) state.phase = 'GAME_OVER';
      const tc = consumeTileClaimed(state);
      replayActions.push([actor, 'T']);
      stats.timeouts++;
      if (!opt.replayOnly) {
        emit({
          k: 'step', i: stepIndex, actor, a: [actor, 'T'], legal: null,
          via: 'timeout',
          ev: { type: 'TIMEOUT', timedOutPlayerIndex: actor, tileClaimed: tc },
          s: snap(state),
        });
      }
      stepIndex++;
      continue;
    }
    if (chaos && RNG() < opt.resignP) {
      const active = [];
      for (let i = 0; i < n; i++) if (!(state.resignedPlayers || []).includes(i)) active.push(i);
      const who = active[Math.floor(RNG() * active.length)];
      processResign(state, who);
      const tc = consumeTileClaimed(state);
      replayActions.push([who, 'X']);
      stats.resigns++;
      if (!opt.replayOnly) {
        emit({
          k: 'step', i: stepIndex, actor: who, a: [who, 'X'], legal: null,
          via: 'resign',
          ev: { type: 'RESIGN', resignedPlayerIndex: who, tileClaimed: tc },
          s: snap(state),
        });
      }
      stepIndex++;
      continue;
    }

    // ── black-box legality ──
    const accepted = [];
    for (const code of candidateCodes(state)) {
      if (tryMessages(state, codeToMessages(code)) !== null) accepted.push(code);
    }

    if (accepted.length === 0) {
      // No action the server would accept: 10 tokens, 3 reserved, nothing
      // affordable (or an orphaned noble choice with an empty supply).
      stats.stuck++;
      processResign(state, actor);
      const tc = consumeTileClaimed(state);
      replayActions.push([actor, 'X']);
      if (!opt.replayOnly) {
        emit({
          k: 'step', i: stepIndex, actor, a: [actor, 'X'], legal: [],
          via: 'stuck-resign',
          ev: { type: 'RESIGN', resignedPlayerIndex: actor, tileClaimed: tc },
          s: snap(state),
        });
      }
      stepIndex++;
      continue;
    }

    // Action choice is UNIFORM over the accepted set by default.  The bias
    // options only steer exploration towards rarely-visited positions; every
    // legality/state assertion is unaffected by which action gets picked.
    let chosen = null;
    if (opt.orphanHunt && actor === preferNonBuyFor) {
      const nonBuy = accepted.filter(c => c[0] !== 'B' && c[0] !== 'N');
      if (nonBuy.length) chosen = nonBuy[Math.floor(RNG() * nonBuy.length)];
    }
    if (!chosen && opt.t1Bias > 0 && RNG() < opt.t1Bias) {
      const cheap = accepted.filter(c => c[0] === 'B' && CARD_BY_ID[c[1]]
        && CARD_BY_ID[c[1]].tier === 1 && CARD_BY_ID[c[1]].points === 0);
      if (cheap.length) chosen = cheap[Math.floor(RNG() * cheap.length)];
    }
    if (!chosen && opt.buyBias > 0 && RNG() < opt.buyBias) {
      const buys = accepted.filter(c => c[0] === 'B');
      if (buys.length) chosen = buys[Math.floor(RNG() * buys.length)];
    }
    if (!chosen) chosen = accepted[Math.floor(RNG() * accepted.length)];
    let results;
    let via = 'confirm';
    let goldTaken;

    if (chosen[0] === 'G' && RNG() < opt.incrementalP) {
      // Incremental desktop path: SELECT_GEM one colour at a time, in a random
      // order, on the LIVE state.  Must complete exactly on the last colour.
      via = 'incremental';
      const order = shuffleInPlace(chosen[1].slice(), RNG);
      results = [];
      for (let i = 0; i < order.length; i++) {
        const r = processAction(state, actor, { type: 'SELECT_GEM', color: order[i] });
        if (r.error) throw new Error(`SELECT_GEM rejected mid-sequence: ${r.error}`);
        results.push(r.result);
        const isLast = i === order.length - 1;
        if (r.completed !== isLast) {
          throw new Error(
            `SELECT_GEM completion mismatch at ${i}/${order.length} for `
            + `${JSON.stringify(order)} (completed=${r.completed})`);
        }
      }
      const sel = results[results.length - 1].payload.selected.slice().sort();
      if (JSON.stringify(sel) !== JSON.stringify(chosen[1])) {
        throw new Error(`incremental take produced ${JSON.stringify(sel)} != ${JSON.stringify(chosen[1])}`);
      }
      stats.incremental++;
    } else {
      const msgs = codeToMessages(chosen);
      results = [];
      for (const m of msgs) {
        const r = processAction(state, actor, m);
        if (r.error) throw new Error(`accepted candidate then failed: ${r.error}`);
        results.push(r.result);
        if (m.type === 'ENTER_RESERVE') goldTaken = r.result.payload.goldTaken;
      }
    }

    const tc = consumeTileClaimed(state);
    const ev = eventOf(chosen[0], results, tc, goldTaken);
    replayActions.push([actor].concat(chosen));
    stats.steps++;
    preferNonBuyFor = chosen[0] === 'N' ? actor : -1;
    if (chosen[0] === 'N') stats.tileChoices++;
    if (state._pendingTileChoice && !state.turnAction) stats.orphaned++;
    if (tc) stats.autoTiles++;

    if (!opt.replayOnly) {
      emit({
        k: 'step', i: stepIndex, actor, a: [actor].concat(chosen),
        legal: accepted, via, ev, s: snap(state),
      });
    }
    stepIndex++;
  }

  const ratings = calculateRatingChanges(state.players, state);
  const result = {
    scores: state.players.map(p => p.score),
    cards: state.players.map(p => p.cards.length),
    resigned: (state.resignedPlayers || []).slice(),
    winners: null,
    winningTeamIds: state.gameResult ? (state.gameResult.winningTeamIds || null) : null,
    reason: state.gameResult ? state.gameResult.reason : (state.phase === 'GAME_OVER' ? 'SCORE' : null),
    rating: ratings,
  };
  if (opt.mode === 'INDIVIDUAL') {
    const ranked = state.players.map((p, i) => ({ i, s: p.score, c: p.cards.length }));
    ranked.sort((a, b) => (b.s !== a.s ? b.s - a.s : a.c - b.c));
    const best = ranked[0];
    result.winners = ranked.filter(r => r.s === best.s && r.c === best.c).map(r => r.i).sort((a, b) => a - b);
  }

  const replayJson = {
    v: 1, id: gameId, t: 0, e: 0, mode: opt.mode,
    layout: opt.mode === 'TEAM' ? opt.layout : null,
    n, clock: false,
    players: playerInfos.map(p => ({
      u: p.username, a: p.avatarSeed,
      ...(teams ? { team: p.teamId } : {}), ai: false,
    })),
    first: gameRec.first,
    setup,
    actions: replayActions,
    result,
  };

  emit({
    k: 'end', id: gameId, gi: gameIndex, steps: stepIndex, truncated,
    chaos,
    finalDecks: [0, 1, 2].map(t => state.decks[t].map(c => c.id)),
    s: snap(state), rating: ratings, result, replay: replayJson,
  });

  stats.games++;
  if (truncated) stats.truncated++;
  if (state.phase === 'GAME_OVER') {
    const reason = state.gameResult ? state.gameResult.reason : 'SCORE';
    stats.endings[reason] = (stats.endings[reason] || 0) + 1;
  } else {
    stats.endings.TRUNCATED = (stats.endings.TRUNCATED || 0) + 1;
  }
  return stepIndex;
}

// ── main ──────────────────────────────────────────────────────────────────

function main() {
  const opt = parseArgs(process.argv.slice(2));
  fs.mkdirSync(path.dirname(path.resolve(opt.out)), { recursive: true });

  const gz = opt.out.endsWith('.gz');
  const fileStream = fs.createWriteStream(opt.out);
  const sink = gz ? zlib.createGzip({ level: 6 }) : fileStream;
  if (gz) sink.pipe(fileStream);

  const buf = [];
  let pending = 0;
  const emit = (obj) => {
    buf.push(JSON.stringify(obj));
    if (buf.length >= 512) { pending += 1; sink.write(buf.join('\n') + '\n'); buf.length = 0; }
  };

  const stats = {
    games: 0, steps: 0, resigns: 0, timeouts: 0, stuck: 0, incremental: 0,
    permChecks: 0, tileChoices: 0, autoTiles: 0, truncated: 0, orphaned: 0,
    endings: {},
  };

  const t0 = Date.now();
  for (let g = 0; g < opt.games; g++) {
    playGame(opt, g, emit, stats);
    if (!opt.quiet && (g + 1) % 100 === 0) {
      const dt = (Date.now() - t0) / 1000;
      process.stderr.write(
        `\r[${opt.mode}${opt.mode === 'TEAM' ? '/' + opt.layout : ''} n=${opt.players}] `
        + `${g + 1}/${opt.games} games  ${stats.steps} steps  ${dt.toFixed(1)}s  `
        + `${(stats.steps / dt).toFixed(0)} steps/s   `);
    }
  }
  if (buf.length) sink.write(buf.join('\n') + '\n');

  const summary = {
    k: 'summary', ...stats,
    mode: opt.mode, layout: opt.mode === 'TEAM' ? opt.layout : null,
    players: opt.players, seed: opt.seed,
    seconds: (Date.now() - t0) / 1000,
  };
  sink.write(JSON.stringify(summary) + '\n');
  sink.end();
  fileStream.on('close', () => {
    if (!opt.quiet) process.stderr.write('\n');
    process.stderr.write(JSON.stringify(summary) + '\n');
  });
}

main();
