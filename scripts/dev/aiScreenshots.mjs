#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════
// Dev/QA-only: visual verification of the lobby AI features and of a bot
// actually playing, against the REAL server (server/index.js) with the mock
// worker (scripts/dev/mockAiWorker.mjs). Writes docs/screenshots/ai-*.png.
// Not imported by the app; nothing under server/ or src/ is touched.
//
//   node scripts/dev/aiScreenshots.mjs            # full run
//   node scripts/dev/aiScreenshots.mjs --keep     # leave the servers running
//   node scripts/dev/aiScreenshots.mjs --no-replay # skip the full-game phase
//
// Layout of a run:
//   phase 1  server PORT=10013 AI_WORKER_SECRET=qa AI_MOVE_DELAY_MS=800
//            + mock worker (MOCK_AI_THINK_MS holds the bot's turn on screen
//            long enough to photograph it) + vite VITE_SERVER_URL=…:10013
//     a) fresh human created through the UI → lobby → "Add AI"      (mobile + desktop)
//        → ai-lobby-bot-mobile.png, ai-lobby-bot-desktop.png
//     b) 3 humans + 1 bot, then 2 humans + 1 bot in 1v2 with the bot
//        seated through the "AI" chip on an empty seat              (desktop)
//        → ai-lobby-4p-bot-desktop.png, ai-lobby-1v2-seated-desktop.png
//     b') Leave Lobby → Enter Lobby probe (a pre-existing re-login defect
//        that sits on the bot user's path)  → ai-defect-leave-lobby-desktop.png
//     c) human + bot game; the browser human plays through the UI so the
//        bot's turn can be caught mid-thought                       (desktop + mobile)
//        → ai-game-bot-turn-desktop.png, ai-game-bot-turn-mobile.png
//   phase 2  server restarted with AI_MOVE_DELAY_MS=10 so a whole
//            socket-driven game finishes quickly → the replay list  (desktop)
//        → ai-replay-list-desktop.png
//
// Alongside the shots it prints DOM measurements (tap targets, truncation,
// type sizes) and flags anything below the app's own conventions, so the
// findings can be quoted as numbers rather than impressions. Exit code is
// non-zero when anything was flagged.
//
// Playwright/Chromium and the local Tailwind build follow
// scripts/dev/replayScreenshots.mjs exactly (the sandbox blocks the Tailwind CDN).
// ═══════════════════════════════════════════════════════════

import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { io as ioClient } from 'socket.io-client';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');
const shotDir = join(repoRoot, 'docs', 'screenshots');

const SERVER_PORT = 10013;
const VITE_PORT = 5174;
const SECRET = 'qa';
const SLOW_DELAY_MS = 800;   // AI_MOVE_DELAY_MS for the screenshot phase
const FAST_DELAY_MS = 10;    // …and for the full-game phase
const THINK_MS = 2500;       // mock-worker "thinking" time: keeps the bot's turn on screen
const API = `http://127.0.0.1:${SERVER_PORT}`;
const BROWSER_API = `http://localhost:${SERVER_PORT}`;
const BASE_URL = `http://localhost:${VITE_PORT}`;
const settle = 500;
const stamp = Date.now().toString(36).slice(-5);

const skipReplayPhase = process.argv.includes('--no-replay');

process.env.PLAYWRIGHT_BROWSERS_PATH = process.env.PLAYWRIGHT_BROWSERS_PATH || '/opt/pw-browsers';
const { chromium } = await import('playwright');

const sleep = ms => new Promise(r => setTimeout(r, ms));
const problems = [];   // anything worth putting in the QA report
const consoleErrors = [];
const shots = [];

function note(message) {
  problems.push(message);
  console.log(`  !! ${message}`);
}

// ── Chromium / Tailwind (same tricks as replayScreenshots.mjs) ─────────────

function preinstalledChromium() {
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (!existsSync(root)) return undefined;
  const build = readdirSync(root)
    .filter(name => /^chromium-\d+$/.test(name))
    .sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]))[0];
  if (!build) return undefined;
  const binary = join(root, build, 'chrome-linux', 'chrome');
  return existsSync(binary) ? binary : undefined;
}

function buildTailwindCss() {
  const cli = join(repoRoot, 'node_modules', '.bin', 'tailwindcss');
  if (!existsSync(cli)) {
    console.warn('  tailwindcss not installed — the page would be unstyled');
    return null;
  }
  const configPath = join(tmpdir(), 'ai-shots-tailwind.config.cjs');
  const inputPath = join(tmpdir(), 'ai-shots-input.css');
  const outputPath = join(tmpdir(), 'ai-shots-output.css');
  writeFileSync(configPath, `module.exports = {
  content: [${JSON.stringify(join(repoRoot, 'index.html'))}, ${JSON.stringify(join(repoRoot, 'src/**/*.{ts,tsx}'))}],
  theme: { extend: { fontFamily: {
    sans: ['Inter', 'system-ui', 'sans-serif'],
    display: ['Space Grotesk', 'Inter', 'sans-serif'],
  } } },
};`);
  writeFileSync(inputPath, '@tailwind base;\n@tailwind components;\n@tailwind utilities;\n');
  execFileSync(cli, ['-c', configPath, '-i', inputPath, '-o', outputPath, '--minify'], { stdio: 'pipe' });
  return readFileSync(outputPath, 'utf8');
}

mkdirSync(shotDir, { recursive: true });

let tailwindShim = null;
try {
  const css = buildTailwindCss();
  if (css) {
    tailwindShim = `window.tailwind = { config: {} };
document.head.appendChild(Object.assign(document.createElement('style'), { textContent: ${JSON.stringify(css)} }));`;
    console.log(`  built local Tailwind CSS (${Math.round(css.length / 1024)} kB)`);
  }
} catch (err) {
  console.warn(`  Tailwind build failed: ${err.message}`);
}

// ── child processes ────────────────────────────────────────────────────────

spawnSync('pkill', ['-f', 'mockAiWorker.mjs']);
spawnSync('pkill', ['-f', `vite(\\.js)? --port ${VITE_PORT}`]);

const children = new Set();
const serverLog = [];
const workerLog = [];

function run(command, args, options = {}) {
  const { label = command, sink, ...rest } = options;
  const child = spawn(command, args, {
    cwd: repoRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true, // own process group so the whole tree dies with it
    ...rest,
  });
  children.add(child);
  const record = stream => chunk => {
    const text = String(chunk);
    process.stdout.write(`[${label}] ${text}`);
    if (!sink) return;
    for (const line of text.split('\n')) if (line.trim()) sink.push(`${stream}: ${line.trim()}`);
  };
  child.stdout.on('data', record('out'));
  child.stderr.on('data', record('err'));
  return child;
}

function stopChild(child) {
  if (!child) return;
  children.delete(child);
  try { process.kill(-child.pid, 'SIGTERM'); } catch { try { child.kill('SIGTERM'); } catch {} }
}

function killChildren() {
  for (const child of [...children]) stopChild(child);
  spawnSync('pkill', ['-f', 'mockAiWorker.mjs']);
  spawnSync('pkill', ['-f', `vite(\\.js)? --port ${VITE_PORT}`]);
}

for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { killChildren(); process.exit(1); });
for (const fatal of ['uncaughtException', 'unhandledRejection']) {
  process.on(fatal, err => { console.error(err); killChildren(); process.exit(1); });
}

async function apiGet(path) {
  const res = await fetch(`${API}${path}`);
  return { status: res.status, body: await res.json().catch(() => null) };
}

async function waitFor(condition, message, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let last;
  while (Date.now() < deadline) {
    try { last = await condition(); if (last) return last; } catch (err) { last = err.message; }
    await sleep(150);
  }
  throw new Error(`timed out waiting for ${message}`);
}

async function waitForHttp(url, timeoutMs = 60000) {
  await waitFor(async () => {
    try { return (await fetch(url)).ok; } catch { return false; }
  }, url, timeoutMs);
}

let serverChild = null;
let workerChild = null;

async function startServer(moveDelayMs) {
  const env = { ...process.env, PORT: String(SERVER_PORT), AI_WORKER_SECRET: SECRET, AI_MOVE_DELAY_MS: String(moveDelayMs) };
  delete env.REPLAY_GITHUB_TOKEN;
  delete env.REPLAY_GITHUB_REPO;
  delete env.RENDER_EXTERNAL_URL;
  serverChild = run('node', ['server/index.js'], { label: 'server', env, sink: serverLog });
  await waitForHttp(`${API}/health`, 60000);
  console.log(`  server up on ${SERVER_PORT} (AI_MOVE_DELAY_MS=${moveDelayMs})`);
}

async function startWorker(thinkMs) {
  workerChild = run('node', ['scripts/dev/mockAiWorker.mjs'], {
    label: 'worker',
    sink: workerLog,
    env: {
      ...process.env,
      SERVER_URL: API,
      AI_WORKER_SECRET: SECRET,
      MOCK_AI_NAME: 'qa-mock-worker',
      MOCK_AI_THINK_MS: String(thinkMs),
      MOCK_AI_VERBOSE: '1',
    },
  });
  await waitFor(async () => (await apiGet('/api/ai/status')).body?.available === true,
    'the mock worker to register', 30000);
  console.log(`  mock worker registered (MOCK_AI_THINK_MS=${thinkMs})`);
}

// ── socket.io human drivers (mirrors server/test/ai.e2e.js) ────────────────

const ACK_TIMEOUT_MS = 8000;

function emitAck(client, event, payload) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`ack timeout: ${event}`)), ACK_TIMEOUT_MS);
    const done = response => { clearTimeout(timer); resolve(response || {}); };
    if (payload === undefined) client.socket.emit(event, done);
    else client.socket.emit(event, payload, done);
  });
}

function act(client, payload) {
  return emitAck(client, 'game_action', { roomId: client.roomId, action: payload });
}

async function connectHuman(username) {
  const socket = ioClient(API, { transports: ['websocket'], forceNew: true, reconnection: false });
  const client = { username, socket, playerIndex: null, roomId: null, state: null, lobby: null, pendingTile: null };
  socket.on('lobby_update', lobby => { client.lobby = lobby; });
  socket.on('game_start', data => {
    client.roomId = data.roomId;
    client.playerIndex = data.playerIndex;
    client.state = data.gameState;
  });
  socket.on('game_state_update', state => { client.state = state; });
  socket.on('tile_choice_required', ({ tileIds }) => { client.pendingTile = tileIds; });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`connect timeout for ${username}`)), ACK_TIMEOUT_MS);
    socket.on('connect', () => { clearTimeout(timer); resolve(); });
    socket.on('connect_error', err => { clearTimeout(timer); reject(err); });
  });
  await fetch(`${API}/api/accounts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username }),
  });
  const login = await emitAck(client, 'login', { username });
  if (!login.success) throw new Error(`login failed for ${username}: ${JSON.stringify(login)}`);
  const lobby = await emitAck(client, 'enter_lobby');
  client.lobby = lobby.lobbyState;
  return client;
}

// random-legal-move policy, same shape as server/test/ai.e2e.js
function shuffled(list) {
  const copy = [...list];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function gemCandidates() {
  const out = [];
  for (let a = 0; a < 5; a++) {
    out.push([a, a]);
    out.push([a]);
    for (let b = 0; b < 5; b++) {
      if (b === a) continue;
      out.push([a, b]);
      for (let c = 0; c < 5; c++) {
        if (c === a || c === b) continue;
        out.push([a, b, c]);
      }
    }
  }
  return out;
}

function turnMoved(client, fromIndex) {
  return !client.state
    || client.state.phase !== 'PLAYING'
    || client.state.currentPlayerIndex !== fromIndex
    || !!client.pendingTile;
}

async function takeSocketTurn(client) {
  if (client.pendingTile) {
    const tileIds = client.pendingTile;
    client.pendingTile = null;
    for (const tileId of shuffled(tileIds)) {
      if (!(await act(client, { type: 'CHOOSE_TILE', tileId })).error) return true;
    }
    return false;
  }
  const from = client.state.currentPlayerIndex;
  const me = client.state.players[client.playerIndex];

  const buys = [
    ...shuffled(me.reserved.filter(card => card.id >= 0)).map(card => ({ cardId: card.id, source: 'reserved' })),
    ...shuffled(client.state.board.flat()).map(card => ({ cardId: card.id, source: 'board' })),
  ];
  for (const candidate of buys) {
    if (!(await act(client, { type: 'BUY_CARD', ...candidate })).error) return true;
  }
  for (const colors of shuffled(gemCandidates())) {
    if (!(await act(client, { type: 'TAKE_GEMS_CONFIRMED', colors })).error) return true;
  }
  if (me.reserved.length < client.state.config.maxReserved) {
    const boardCards = client.state.board.flat();
    const openDecks = [1, 2, 3].filter(tier => client.state.deckCounts[tier - 1] > 0);
    if ((boardCards.length || openDecks.length) && !(await act(client, { type: 'ENTER_RESERVE' })).error) {
      for (const card of shuffled(boardCards)) {
        if (!(await act(client, { type: 'RESERVE_CARD', cardId: card.id })).error) return true;
      }
      for (const tier of shuffled(openDecks)) {
        if (!(await act(client, { type: 'RESERVE_FROM_DECK', tier })).error) return true;
      }
    }
  }
  return turnMoved(client, from);
}

/** Plays a socket human against bot seats until GAME_OVER. */
async function playToEnd(client, timeoutMs = 240000) {
  const deadline = Date.now() + timeoutMs;
  let humanTurns = 0;
  let botTurns = 0;
  while (Date.now() < deadline) {
    if (!client.state || client.state.phase !== 'PLAYING') break;
    const seat = client.state.currentPlayerIndex;
    if (seat !== client.playerIndex && !client.pendingTile) {
      const turnBefore = client.state.turnNumber;
      await waitFor(() => !client.state
        || client.state.phase !== 'PLAYING'
        || client.state.turnNumber !== turnBefore
        || client.state.currentPlayerIndex !== seat, `bot seat ${seat} to play`, 20000);
      botTurns++;
      continue;
    }
    humanTurns++;
    if (!(await takeSocketTurn(client))) {
      client.socket.emit('resign', { roomId: client.roomId });
      await waitFor(() => client.state.resignedPlayers?.includes(client.playerIndex)
        || client.state.phase === 'GAME_OVER', 'the resignation to register');
    }
  }
  await waitFor(() => client.state?.phase === 'GAME_OVER', 'GAME_OVER', 30000);
  return { humanTurns, botTurns };
}

// ── browser plumbing ───────────────────────────────────────────────────────

const proxyServer = process.env.HTTPS_PROXY || process.env.https_proxy;
const browser = await chromium.launch({
  executablePath: preinstalledChromium(),
  ...(proxyServer ? { proxy: { server: proxyServer, bypass: 'localhost,127.0.0.1' } } : {}),
});

const MOBILE = { viewport: { width: 375, height: 812 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true };
const DESKTOP = { viewport: { width: 1280, height: 800 } };

async function newPage(label, options) {
  const context = await browser.newContext({ ignoreHTTPSErrors: true, ...options });
  if (tailwindShim) {
    await context.route('https://cdn.tailwindcss.com/**', route =>
      route.fulfill({ status: 200, contentType: 'text/javascript', body: tailwindShim }));
  }
  const page = await context.newPage();
  page.on('console', msg => {
    if (msg.type() !== 'error') return;
    const location = msg.location()?.url ?? '';
    if (/fonts\.g|cdn\.tailwindcss\.com/.test(location)) return;
    consoleErrors.push(`${label} console: ${msg.text()} ${location}`);
  });
  page.on('pageerror', err => consoleErrors.push(`${label} pageerror: ${err.message}`));
  page.on('requestfailed', request => {
    const url = request.url();
    if (url.includes('fonts.g') || url.includes('cdn.tailwindcss.com')) return;
    consoleErrors.push(`${label} requestfailed: ${url} (${request.failure()?.errorText})`);
  });
  return { context, page };
}

async function shot(page, name) {
  await sleep(settle);
  await page.screenshot({ path: join(shotDir, name) });
  shots.push(name);
  console.log(`  saved docs/screenshots/${name}`);
}

/** Creates a brand-new account through the UI; lands in the lobby. */
async function createAccountAndEnterLobby(page, username) {
  await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Create Account' }).waitFor({ timeout: 90000 });
  await page.getByRole('button', { name: 'Create Account' }).click();
  await page.getByPlaceholder('Enter username').fill(username);
  await page.getByRole('button', { name: 'Create', exact: true }).click();
  await page.getByRole('heading', { name: 'Lobby' }).waitFor({ timeout: 30000 });
}

async function addAI(page) {
  await page.getByRole('button', { name: 'Add AI' }).click();
  await page.getByText('Bot Alpha', { exact: true }).first().waitFor({ timeout: 15000 });
}

/**
 * Whose turn the board says it is, plus whether the VISIBLE bot panel carries
 * the amber current-turn ring (both the mobile and desktop panel variants are
 * in the DOM at all times, so only laid-out ones count).
 */
function boardTurn(page) {
  return page.evaluate(() => {
    const shell = document.querySelector('.game-shell');
    if (!shell) return null;
    const indicator = shell.querySelector('.absolute.top-1.left-1');
    const panels = [...document.querySelectorAll('[data-player-panel]')]
      .filter(p => p.getClientRects().length > 0);
    const botPanels = panels.filter(p => /Bot [A-Z]/.test(p.textContent || ''));
    return {
      indicator: (indicator?.textContent || '').trim(),
      myTurn: /Your turn/.test(indicator?.textContent || ''),
      botPanels: botPanels.length,
      botHighlighted: botPanels.some(p => p.innerHTML.includes('ring-amber-500')),
      gameOver: document.body.textContent.includes('Game Over'),
    };
  });
}

/** One human turn through the UI: Take Gems → three distinct gems (→ Confirm on mobile). */
async function humanGemTurn(page, mobile) {
  // Both panel variants live in the DOM; only click the laid-out one.
  const takeGems = page.locator('button:has-text("Take Gems"):visible').first();
  await takeGems.waitFor({ timeout: 15000 });
  await takeGems.click();
  let picked = 0;
  for (let i = 0; i < 3; i++) {
    const buttons = page.locator('button[aria-label*=" gem, "]:not([disabled])');
    const count = await buttons.count();
    let clicked = false;
    for (let j = 0; j < count; j++) {
      const label = await buttons.nth(j).getAttribute('aria-label');
      if (!label || label.includes('selected') || /, 0 available/.test(label)) continue;
      await buttons.nth(j).click();
      clicked = true;
      picked++;
      break;
    }
    if (!clicked) break;
    await sleep(180);
  }
  if (mobile && picked > 0) {
    const confirm = page.locator('button:has-text("Confirm"):visible').first();
    if (await confirm.count()) await confirm.click();
  }
  return picked;
}

/**
 * Plays human turns through the UI until the bot is thinking, then shoots.
 * With AI_MOVE_DELAY_MS=800 + MOCK_AI_THINK_MS the bot's turn stays on
 * screen for ~3 s, which is what makes the shot reproducible.
 */
async function shootBotTurn(page, name, mobile) {
  await page.locator('.game-shell').waitFor({ timeout: 30000 });
  let humanTurns = 0;
  for (let step = 0; step < 10; step++) {
    const state = await boardTurn(page);
    if (!state || state.gameOver) break;

    if (state.myTurn) {
      const picked = await humanGemTurn(page, mobile);
      if (picked === 0) { note(`${name}: the human could not take gems through the UI`); break; }
      humanTurns++;
      await waitFor(async () => {
        const s = await boardTurn(page);
        return s && (!s.myTurn || s.gameOver);
      }, 'the turn to pass to the bot', 25000).catch(() => null);
      continue;
    }

    // The bot opened the game: let it move first so the shot shows a played board.
    if (humanTurns === 0) {
      await waitFor(async () => {
        const s = await boardTurn(page);
        return s && (s.myTurn || s.gameOver);
      }, 'the bot to finish its opening turn', 30000).catch(() => null);
      continue;
    }

    const withRing = await waitFor(async () => {
      const s = await boardTurn(page);
      return s?.botHighlighted ? s : false;
    }, 'the bot panel to be highlighted', 4000).catch(() => null);
    if (!withRing) note(`${name}: the bot's turn is not highlighted (indicator="${state.indicator}")`);
    await page.screenshot({ path: join(shotDir, name) });
    shots.push(name);
    console.log(`  saved docs/screenshots/${name} after ${humanTurns} human turn(s)`);
    return withRing || state;
  }
  note(`${name}: never caught the bot on turn`);
  await page.screenshot({ path: join(shotDir, name) });
  shots.push(name);
  return null;
}

/**
 * Geometry + type size of a few AI-specific controls, so the report can quote
 * numbers instead of impressions (tap targets, truncation, font sizes).
 */
function measure(page, description) {
  return page.evaluate(desc => {
    const box = el => {
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      return {
        text: (el.textContent || '').trim(),
        w: Math.round(rect.width * 10) / 10,
        h: Math.round(rect.height * 10) / 10,
        fontSize: style.fontSize,
        color: style.color,
        clipped: el.scrollWidth > el.clientWidth,
        scrollW: el.scrollWidth,
        clientW: el.clientWidth,
      };
    };
    const byText = (selector, text) =>
      [...document.querySelectorAll(selector)].find(el => (el.textContent || '').trim() === text) || null;
    const out = { where: desc };
    out.remove = box(byText('button', 'Remove'));
    out.addAI = box([...document.querySelectorAll('button')].find(b => /Add AI/.test(b.textContent || '')));
    const badge = [...document.querySelectorAll('span[title="AI player"]')][0];
    out.badge = box(badge);
    // the seated-bot name inside a team card
    const nameSpan = [...document.querySelectorAll('span.truncate')].find(el => /^Bot /.test(el.textContent || ''));
    out.seatName = box(nameSpan);
    const replayAI = [...document.querySelectorAll('span')].find(el => (el.textContent || '').trim() === 'AI'
      && !el.closest('span[title="AI player"]'));
    out.replayAI = box(replayAI);
    // Does the lobby card still fit, now that the AI row + button are in it?
    const card = byText('h2', 'Lobby')?.parentElement;
    const ready = [...document.querySelectorAll('button')]
      .find(b => /^(Ready|Cancel Ready|Choose a seat first)$/.test((b.textContent || '').trim()));
    if (card) out.card = { scrollH: card.scrollHeight, clientH: card.clientHeight, viewportH: window.innerHeight };
    if (ready) out.readyBottom = Math.round(ready.getBoundingClientRect().bottom);
    return out;
  }, description);
}

/** Reads the visible lobby rows so the report can quote what was on screen. */
function lobbyRows(page) {
  return page.evaluate(() => {
    const heading = [...document.querySelectorAll('h2')].find(h => h.textContent.trim() === 'Lobby');
    if (!heading) return null;
    return (heading.parentElement.innerText || '').split('\n').map(s => s.trim()).filter(Boolean);
  });
}

// ── boot ───────────────────────────────────────────────────────────────────

await startServer(SLOW_DELAY_MS);
await startWorker(THINK_MS);

const viteBin = join(repoRoot, 'node_modules', 'vite', 'bin', 'vite.js');
if (!existsSync(viteBin)) throw new Error(`vite is not installed at ${viteBin}`);
run(process.execPath, [viteBin, '--port', String(VITE_PORT), '--strictPort'], {
  label: 'vite',
  env: { ...process.env, VITE_SERVER_URL: BROWSER_API },
});
await waitForHttp(BASE_URL, 90000);

const bootStatus = await apiGet('/api/ai/status');
console.log(`  GET /api/ai/status → ${JSON.stringify(bootStatus.body)}`);
if (bootStatus.body?.available !== true) note('the worker did not register before the run started');

// ── a) "Add AI" in the plain lobby, mobile then desktop ────────────────────

{
  const { context, page } = await newPage('mobile', MOBILE);
  await createAccountAndEnterLobby(page, `qa-m-${stamp}`);
  await addAI(page);
  console.log(`  mobile lobby: ${JSON.stringify(await lobbyRows(page))}`);
  await shot(page, 'ai-lobby-bot-mobile.png');
  const mobileMetrics = await measure(page, 'mobile lobby');
  console.log(`  mobile metrics: ${JSON.stringify(mobileMetrics)}`);
  if (mobileMetrics.remove && mobileMetrics.remove.h < 44) {
    note(`mobile: the bot row's "Remove" hit area is ${mobileMetrics.remove.w}×${mobileMetrics.remove.h} px`
      + ' — below the 44 px target every other mobile control in this app uses');
  }
  await context.close();
  await waitFor(async () => (await apiGet('/')).body?.lobby === 0, 'the lobby to empty after the human left');
}

const { context: deskCtx, page: desk } = await newPage('desktop', DESKTOP);
await createAccountAndEnterLobby(desk, `qa-d-${stamp}`);
await addAI(desk);
console.log(`  desktop lobby: ${JSON.stringify(await lobbyRows(desk))}`);
await shot(desk, 'ai-lobby-bot-desktop.png');
console.log(`  desktop metrics: ${JSON.stringify(await measure(desk, 'desktop lobby'))}`);

// The Remove control has to actually remove the bot, and Add AI has to work twice.
try {
  await desk.getByRole('button', { name: 'Remove' }).first().click();
  await waitFor(async () => (await lobbyRows(desk))?.every(line => !line.includes('Bot Alpha')),
    'the bot to leave when Remove is clicked', 10000);
  console.log('  Remove took the bot out of the lobby');
  await addAI(desk);
} catch (err) {
  note(`the lobby "Remove" control did not remove the bot: ${err.message}`);
}

// ── b) 3 humans + 1 bot, then a seated 1v2 ─────────────────────────────────

let s1 = null;
let s2 = null;
try {
  s1 = await connectHuman(`qa-s1-${stamp}`);
  s2 = await connectHuman(`qa-s2-${stamp}`);
  await desk.getByText(s2.username, { exact: true }).waitFor({ timeout: 15000 });
  console.log(`  4-player lobby: ${JSON.stringify(await lobbyRows(desk))}`);
  await shot(desk, 'ai-lobby-4p-bot-desktop.png');
  const fourMetrics = await measure(desk, '4-player lobby');
  console.log(`  4-player metrics: ${JSON.stringify(fourMetrics)}`);
  if (fourMetrics.card && fourMetrics.card.scrollH > fourMetrics.card.clientH) {
    note(`3 humans + 1 bot: the lobby card overflows (${fourMetrics.card.scrollH} px of content in`
      + ` ${fourMetrics.card.clientH} px) and the Ready button ends at y=${fourMetrics.readyBottom}`
      + ` in an ${fourMetrics.card.viewportH} px viewport, with no scroll affordance`);
  }

  // 1v2 needs three lobby members: 2 humans + the bot.
  s2.socket.disconnect();
  s2 = null;
  await waitFor(async () => (await apiGet('/')).body?.lobby === 3, 'the lobby to drop to three members');
  await sleep(400);

  const toggle = desk.locator('button[aria-label="Toggle 1v2 mode"], button[aria-label="Toggle team mode"]').first();
  await toggle.click();
  await desk.getByText('Solo', { exact: true }).first().waitFor({ timeout: 15000 });

  // Browser human takes the solo seat, the socket human takes a duo seat…
  await desk.locator('button[title="Join this team seat"]').first().click();
  await sleep(300);
  const seated = await emitAck(s1, 'select_team_seat', { teamId: 1, seatIndex: 0 });
  if (seated.error) note(`select_team_seat for ${s1.username} failed: ${seated.error}`);
  await sleep(400);

  // …and the last empty seat gets the bot through its "AI" chip.
  const chips = desk.locator('button[title="Seat an AI player here"]');
  await chips.first().waitFor({ timeout: 15000 });
  console.log(`  empty-seat AI chips on screen: ${await chips.count()}`);
  await chips.last().click();
  await waitFor(async () => {
    const rows = await lobbyRows(desk);
    return rows?.some(line => line.includes('Bot Alpha'));
  }, 'the bot to appear in a team seat');
  console.log(`  1v2 lobby: ${JSON.stringify(await lobbyRows(desk))}`);
  await desk.mouse.move(20, 20); // drop the hover state left on the Remove control
  await shot(desk, 'ai-lobby-1v2-seated-desktop.png');
  const seatMetrics = await measure(desk, '1v2 team card');
  console.log(`  1v2 metrics: ${JSON.stringify(seatMetrics)}`);
  if (seatMetrics.seatName?.clipped) {
    note(`1v2 team card: the bot name is truncated ("${seatMetrics.seatName.text}",`
      + ` ${seatMetrics.seatName.clientW} px of ${seatMetrics.seatName.scrollW} px shown)`);
  }
} catch (err) {
  note(`1v2 lobby phase failed: ${err.message}`);
}

// ── b') Leave Lobby → Enter Lobby round trip (defect probe) ────────────────
// LoginScreen re-runs wakeAndConnect() on every mount, and connectToServer()
// throws the authenticated socket away, so coming back needs a re-login the
// app never performs. Captured here because it is on the path a bot user takes.

try {
  if (s1) { s1.socket.disconnect(); s1 = null; }
  await desk.getByRole('button', { name: 'Leave Lobby' }).click();
  await desk.getByRole('button', { name: 'Enter Lobby' }).waitFor({ timeout: 15000 });
  await waitFor(async () => (await apiGet('/')).body?.lobby === 0, 'the lobby to empty');
  await sleep(1500); // let LoginScreen's mount effect finish reconnecting
  await desk.getByRole('button', { name: 'Enter Lobby' }).click();
  let toast = '';
  const outcome = await waitFor(async () => {
    if (await desk.getByRole('heading', { name: 'Lobby' }).count()) return 'lobby';
    const texts = await desk.locator('div.pointer-events-auto').allInnerTexts();
    if (texts.length) { toast = texts.join(' / '); return 'toast'; } // toasts live 4 s
    return false;
  }, 'the lobby or an error toast', 8000).catch(() => 'nothing');
  if (outcome !== 'lobby') {
    note(`Leave Lobby → Enter Lobby does not return to the lobby (toast: "${toast || outcome}")`);
    await shot(desk, 'ai-defect-leave-lobby-desktop.png');
  } else {
    console.log('  Leave Lobby → Enter Lobby round trip works');
  }
} catch (err) {
  note(`leave/enter lobby probe failed: ${err.message}`);
}
await deskCtx.close();
await sleep(500);

// ── c) a real game: browser human + bot, desktop then mobile ───────────────

for (const [label, options, name, mobile] of [
  ['desktop-game', DESKTOP, 'ai-game-bot-turn-desktop.png', false],
  ['mobile-game', MOBILE, 'ai-game-bot-turn-mobile.png', true],
]) {
  try {
    const { context, page } = await newPage(label, options);
    await createAccountAndEnterLobby(page, `qa-g${mobile ? 'm' : 'd'}-${stamp}`);
    await addAI(page);
    await page.getByRole('button', { name: 'Ready', exact: true }).click();
    await page.locator('.game-shell').waitFor({ timeout: 30000 });
    console.log(`  ${label}: game started (human + Bot Alpha)`);
    const caught = await shootBotTurn(page, name, mobile);
    console.log(`  ${label}: board turn state ${JSON.stringify(caught)}`);

    // Let the bot finish the move it was thinking about: the turn coming back
    // is the proof that the worker's action was applied to the real game.
    const back = await waitFor(async () => {
      const s = await boardTurn(page);
      return s && (s.myTurn || s.gameOver) ? s : false;
    }, 'the bot to finish its move', 20000).catch(() => null);
    if (!back) note(`${label}: the bot never completed its move`);
    else console.log(`  ${label}: turn returned to the human (${JSON.stringify(back)})`);

    // Quit so the next human meets an empty lobby.
    try {
      await page.getByRole('button', { name: 'Open game menu' }).click();
      await page.getByRole('button', { name: 'Quit Room' }).click();
      await page.getByRole('button', { name: 'Quit', exact: true }).click();
      await sleep(800);
    } catch { /* the game may already be over */ }
    await context.close();
    await waitFor(async () => (await apiGet('/')).body?.lobby === 0, 'the lobby to empty after the game', 15000)
      .catch(() => note(`${label}: the lobby did not empty after the game`));
  } catch (err) {
    note(`${label} bot game failed: ${err.message}`);
  }
}

// ── d) worker attribution + a whole game fast, then the replay list ────────

const midStatus = await apiGet('/api/ai/status');
console.log(`  GET /api/ai/status → ${JSON.stringify(midStatus.body)}`);
const workerAnswers = workerLog.filter(line => /seat \d+ →/.test(line));
console.log(`  mock worker answered ${workerAnswers.length} request(s); last: ${workerAnswers.at(-1) || 'none'}`);
if (workerAnswers.length === 0) note('the mock worker never answered a move request');
const fallbackLines = serverLog.filter(line => line.includes('[ai] fallback'));
if (fallbackLines.length) note(`server fell back instead of using the worker: ${fallbackLines.slice(0, 3).join(' | ')}`);

if (!skipReplayPhase) {
  try {
    stopChild(workerChild);
    stopChild(serverChild);
    await sleep(800);
    serverLog.length = 0;
    await startServer(FAST_DELAY_MS);
    await startWorker(0);

    const runner = await connectHuman(`qa-replay-${stamp}`);
    const added = await emitAck(runner, 'lobby_add_ai', {});
    if (added.error) throw new Error(`lobby_add_ai: ${added.error}`);
    const started = new Promise(resolve => runner.socket.once('game_start', resolve));
    runner.socket.emit('lobby_ready');
    await started;
    await waitFor(() => runner.state?.phase === 'PLAYING', 'the first state');
    const summary = await playToEnd(runner);
    console.log(`  full game finished: ${summary.humanTurns} human turns, ${summary.botTurns} bot turns`);
    const list = await apiGet('/api/replays?limit=20');
    const entry = list.body?.games?.find(game => game.id === runner.roomId);
    console.log(`  replay index entry: ${JSON.stringify(entry)}`);
    if (!entry) note('the finished game is missing from /api/replays');
    else if (!entry.ai?.some(Boolean)) note(`replay entry ${entry.id} does not mark any seat as AI`);
    runner.socket.disconnect();

    const { context, page } = await newPage('replays', DESKTOP);
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.getByRole('button', { name: 'Replays' }).first().waitFor({ timeout: 90000 });
    await page.getByRole('button', { name: 'Replays' }).first().click();
    await page.locator('button:has-text("turns")').first().waitFor({ timeout: 30000 });
    const firstRow = (await page.locator('button:has-text("turns")').first().innerText()).replace(/\n+/g, ' | ');
    console.log(`  first replay row: ${firstRow}`);
    if (!/Bot /.test(firstRow)) note('the newest replay row does not name the bot');
    await shot(page, 'ai-replay-list-desktop.png');
    const replayMetrics = await measure(page, 'replay list');
    console.log(`  replay metrics: ${JSON.stringify(replayMetrics)}`);
    if (replayMetrics.replayAI && parseFloat(replayMetrics.replayAI.fontSize) < 9) {
      note(`replay list: the "AI" marker renders at ${replayMetrics.replayAI.fontSize} in`
        + ` ${replayMetrics.replayAI.color} — smaller and fainter than the lobby's AI badge`);
    }
    await context.close();
  } catch (err) {
    note(`replay-list phase failed: ${err.message}`);
  }
}

// ── report ─────────────────────────────────────────────────────────────────

await browser.close();

const finalStatus = await apiGet('/api/ai/status').catch(() => ({ body: null }));
console.log('\n── summary ─────────────────────────────');
console.log(`  screenshots: ${shots.join(', ')}`);
console.log(`  /api/ai/status: ${JSON.stringify(finalStatus.body)}`);
const aiLines = serverLog.filter(line => line.includes('[ai]'));
console.log(`  server [ai] lines (${aiLines.length}):`);
for (const line of aiLines.slice(0, 20)) console.log(`    ${line}`);
const errorLines = serverLog.filter(line => line.startsWith('err:') || /Error|error:|ECONNREFUSED|unhandled/i.test(line));
console.log(`  server error-ish lines (${errorLines.length}):`);
for (const line of errorLines.slice(0, 20)) console.log(`    ${line}`);

if (consoleErrors.length) {
  console.log(`  ${consoleErrors.length} browser console error(s):`);
  for (const line of consoleErrors) console.log(`    ${line}`);
} else {
  console.log('  no browser console errors');
}
if (problems.length) {
  console.log(`  ${problems.length} problem(s) flagged:`);
  for (const line of problems) console.log(`    ${line}`);
} else {
  console.log('  no problems flagged');
}

if (!process.argv.includes('--keep')) {
  killChildren();
  await sleep(400);
  process.exit(problems.length || consoleErrors.length ? 1 : 0);
}
