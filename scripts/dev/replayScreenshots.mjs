// Dev-only: boots the mock replay server + vite, drives the replay UI with Playwright and
// writes docs/screenshots/*.png. Not imported by the app.
//
//   npm install --no-save playwright tailwindcss@3   # Chromium is preinstalled at /opt/pw-browsers
//   node scripts/dev/replayScreenshots.mjs
//
// Flags: --keep (leave servers running), --slow (extra settle time before each shot)
//
// Env: REAL_SERVER_URL=http://localhost:10012 points the page at an already-running real
// server (server/index.js) instead of the mock — nothing is spawned for it, and the shots
// are written under a `real-` prefix so the mock set is never overwritten.

import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');
const shotDir = join(repoRoot, 'docs', 'screenshots');
const MOCK_PORT = 10011;
const VITE_PORT = 5173;
const BASE_URL = `http://localhost:${VITE_PORT}`;
const settle = process.argv.includes('--slow') ? 1200 : 600;

// When REAL_SERVER_URL is set the mock is not started at all and the page talks to that
// server; everything else (vite, Tailwind shim, Playwright driving) is unchanged.
const REAL_SERVER_URL = (process.env.REAL_SERVER_URL || '').replace(/\/$/, '') || null;
const API_URL = REAL_SERVER_URL ?? `http://localhost:${MOCK_PORT}`;

process.env.PLAYWRIGHT_BROWSERS_PATH = process.env.PLAYWRIGHT_BROWSERS_PATH || '/opt/pw-browsers';
const { chromium } = await import('playwright');

// Chromium is preinstalled (never run `playwright install`); pin the binary when the
// installed playwright expects a different build number.
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

mkdirSync(shotDir, { recursive: true });

// This sandbox blocks cdn.tailwindcss.com, so build the same utilities locally and serve
// them to the page in place of the CDN script (Google Fonts still loads via the proxy).
function buildTailwindCss() {
  const cli = join(repoRoot, 'node_modules', '.bin', 'tailwindcss');
  if (!existsSync(cli)) {
    console.warn('  tailwindcss not installed (npm install --no-save tailwindcss@3) — page will be unstyled');
    return null;
  }
  const configPath = join(tmpdir(), 'replay-shots-tailwind.config.cjs');
  const inputPath = join(tmpdir(), 'replay-shots-input.css');
  const outputPath = join(tmpdir(), 'replay-shots-output.css');
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

// Leftovers from an interrupted run would hold the ports.
spawnSync('pkill', ['-f', 'mockReplayServer.mjs']);
spawnSync('pkill', ['-f', `vite(\\.js)? --port ${VITE_PORT}`]); // npx-style and direct-binary leftovers

const children = [];
function killChildren() {
  for (const child of children) {
    // Spawned detached, so the whole process group goes (vite's esbuild helper included).
    try { process.kill(-child.pid, 'SIGTERM'); } catch { try { child.kill('SIGTERM'); } catch {} }
  }
}
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { killChildren(); process.exit(1); });
for (const fatal of ['uncaughtException', 'unhandledRejection']) {
  process.on(fatal, err => { console.error(err); killChildren(); process.exit(1); });
}
function run(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: repoRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true, // own process group, so killChildren() takes the children down too
    ...options,
  });
  children.push(child);
  child.stdout.on('data', d => process.stdout.write(`[${options.label ?? command}] ${d}`));
  child.stderr.on('data', d => process.stderr.write(`[${options.label ?? command}] ${d}`));
  return child;
}

async function waitForHttp(url, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {}
    await new Promise(r => setTimeout(r, 400));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Boot servers ───────────────────────────────────────────────────────────
if (REAL_SERVER_URL) {
  console.log(`  using the real server at ${REAL_SERVER_URL} (mock not started)`);
  await waitForHttp(`${REAL_SERVER_URL}/health`, 30000);
} else {
  run('node', [join('scripts', 'dev', 'mockReplayServer.mjs'), '--port', String(MOCK_PORT)], { label: 'mock' });
  await waitForHttp(`http://localhost:${MOCK_PORT}/health`, 90000);
}

// Spawn vite's own binary rather than `npx vite`: npx is only a wrapper, and killing it
// would leave the real vite process (and the port) behind.
const viteBin = join(repoRoot, 'node_modules', 'vite', 'bin', 'vite.js');
if (!existsSync(viteBin)) throw new Error(`vite is not installed at ${viteBin} — run npm install`);
run(process.execPath, [viteBin, '--port', String(VITE_PORT), '--strictPort'], {
  label: 'vite',
  env: { ...process.env, VITE_SERVER_URL: API_URL },
});
await waitForHttp(BASE_URL, 60000);

// ── Drive the UI ───────────────────────────────────────────────────────────
const proxyServer = process.env.HTTPS_PROXY || process.env.https_proxy;
const browser = await chromium.launch({
  executablePath: preinstalledChromium(),
  ...(proxyServer ? { proxy: { server: proxyServer, bypass: 'localhost,127.0.0.1' } } : {}),
});
const consoleErrors = [];

async function newPage(contextOptions) {
  const context = await browser.newContext({ ignoreHTTPSErrors: true, ...contextOptions });
  if (tailwindShim) {
    await context.route('https://cdn.tailwindcss.com/**', route =>
      route.fulfill({ status: 200, contentType: 'text/javascript', body: tailwindShim }));
  }
  const page = await context.newPage();
  page.on('console', msg => {
    if (msg.type() !== 'error') return;
    const text = msg.text();
    const location = msg.location()?.url ?? '';
    if (/fonts\.g|cdn\.tailwindcss\.com/.test(location)) return; // blocked by the sandbox proxy
    consoleErrors.push(`${contextOptions.label ?? ''} console: ${text} ${location}`);
  });
  page.on('pageerror', err => consoleErrors.push(`${contextOptions.label ?? ''} pageerror: ${err.message}`));
  page.on('requestfailed', request => {
    const url = request.url();
    // Font/CDN fetches blocked by the sandbox proxy are environmental, not app errors.
    if (url.includes('fonts.g') || url.includes('cdn.tailwindcss.com')) return;
    consoleErrors.push(`${contextOptions.label ?? ''} requestfailed: ${url} (${request.failure()?.errorText})`);
  });
  return { context, page };
}

async function shot(page, name) {
  await sleep(settle);
  const file = join(shotDir, name);
  await page.screenshot({ path: file });
  console.log(`  saved docs/screenshots/${name}`);
}

/** Opens the browser list from the login screen and enters the first replay. */
async function enterFirstReplay(page, index = 0) {
  await page.getByRole('button', { name: 'Replays' }).first().click();
  await page.getByRole('heading', { name: 'Replays' }).waitFor();
  await page.locator('button:has-text("turns")').first().waitFor({ timeout: 20000 });
  await page.locator('button:has-text("turns")').nth(index).click();
  await page.locator('.game-shell').waitFor({ timeout: 30000 });
  await sleep(400);
}

/** React-friendly range input update. */
async function seekTo(page, value) {
  await page.evaluate(v => {
    const input = document.querySelector('input[type="range"]');
    if (!input) return;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(input, String(v));
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await sleep(400);
}

async function stepForward(page, times) {
  for (let i = 0; i < times; i++) {
    await page.getByRole('button', { name: 'Next frame' }).first().click();
    await sleep(250);
  }
}

// ── Mobile: login, browser, viewer ─────────────────────────────────────────
if (!REAL_SERVER_URL) {
  const { context, page } = await newPage({
    label: 'mobile',
    viewport: { width: 375, height: 812 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
  });
  await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Replays' }).first().waitFor({ timeout: 60000 });
  await shot(page, 'login-mobile.png');

  await page.getByRole('button', { name: 'Replays' }).first().click();
  await page.locator('button:has-text("turns")').first().waitFor({ timeout: 20000 });
  await shot(page, 'replay-browser-mobile.png');

  // A 4-player individual game shows the busiest layout.
  await page.locator('button:has-text("turns")').nth(2).click();
  await page.locator('.game-shell').waitFor({ timeout: 30000 });
  await stepForward(page, 12);
  await shot(page, 'replay-viewer-mobile.png');
  await context.close();
}

// ── Desktop: viewer + game over ────────────────────────────────────────────
if (!REAL_SERVER_URL) {
  const { context, page } = await newPage({
    label: 'desktop',
    viewport: { width: 1280, height: 800 },
  });
  await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Replays' }).first().waitFor({ timeout: 60000 });
  await shot(page, 'login-desktop.png');

  await enterFirstReplay(page, 2);
  await stepForward(page, 14);
  await shot(page, 'replay-viewer-desktop.png');

  // Jump to the last frame for the Game Over overlay.
  const max = await page.locator('input[type="range"]').first().getAttribute('max');
  await seekTo(page, Number(max));
  await shot(page, 'replay-gameover-desktop.png');

  // Team game: check the 2v2 summary and side panels too.
  await page.getByRole('button', { name: 'Exit Replay', exact: true }).click();
  await page.locator('button:has-text("turns")').first().waitFor({ timeout: 20000 });
  await shot(page, 'replay-browser-desktop.png');
  await page.locator('button:has-text("2v2")').first().click();
  await page.locator('.game-shell').waitFor({ timeout: 30000 });
  const teamMax = await page.locator('input[type="range"]').first().getAttribute('max');
  await seekTo(page, Number(teamMax));
  await shot(page, 'replay-gameover-team-desktop.png');
  await context.close();
}

// ── Mobile game over (control bar must stay reachable) ─────────────────────
if (!REAL_SERVER_URL) {
  const { context, page } = await newPage({
    label: 'mobile-over',
    viewport: { width: 375, height: 812 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
  });
  await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Replays' }).first().waitFor({ timeout: 60000 });
  await enterFirstReplay(page, 0);
  const max = await page.locator('input[type="range"]').first().getAttribute('max');
  await seekTo(page, Number(max));
  await shot(page, 'replay-gameover-mobile.png');
  await context.close();
}

// ── Real server: the same journey against real recorded games ──────────────
if (REAL_SERVER_URL) {
  const list = await (await fetch(`${REAL_SERVER_URL}/api/replays?limit=50`)).json();
  const games = list.games ?? [];
  console.log(`  ${games.length} replay(s) from the real server (source: ${list.source})`);
  for (const game of games) {
    console.log(`    ${game.id}  ${game.mode} n=${game.n} turns=${game.turns} `
      + `winners=${JSON.stringify(game.winners)} teams=${JSON.stringify(game.winningTeamIds)} `
      + `[${game.players.join(', ')}]`);
  }
  if (!games.length) throw new Error(`${REAL_SERVER_URL}/api/replays is empty — run scripts/dev/playRandomGames.mjs first`);

  // Busiest individual game for the board shots, and any team game for the team shot.
  const individual = [...games].filter(g => g.mode === 'INDIVIDUAL')
    .sort((a, b) => (b.n - a.n) || (b.turns - a.turns))[0] ?? games[0];
  const team = games.find(g => g.mode === 'TEAM') ?? games.find(g => g.mode === 'ONE_V_TWO') ?? games[0];

  /** Rows are unique by player name, so pick by that rather than by position. */
  async function openRow(page, game) {
    await page.locator('button:has-text("turns")').first().waitFor({ timeout: 20000 });
    const row = page.locator('button:has-text("turns")').filter({ hasText: game.players[0] }).first();
    await row.waitFor({ timeout: 20000 });
    await row.click();
    await page.locator('.game-shell').waitFor({ timeout: 30000 });
    await sleep(400);
  }

  async function toLastFrame(page) {
    const max = await page.locator('input[type="range"]').first().getAttribute('max');
    await seekTo(page, Number(max));
    return Number(max);
  }

  // Mobile: the list, then a board mid-replay.
  {
    const { context, page } = await newPage({
      label: 'real-mobile',
      viewport: { width: 375, height: 812 },
      deviceScaleFactor: 2,
      isMobile: true,
      hasTouch: true,
    });
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.getByRole('button', { name: 'Replays' }).first().waitFor({ timeout: 60000 });
    await page.getByRole('button', { name: 'Replays' }).first().click();
    await page.getByRole('heading', { name: 'Replays' }).waitFor();
    await page.locator('button:has-text("turns")').first().waitFor({ timeout: 20000 });
    await shot(page, 'real-replay-browser-mobile.png');

    await openRow(page, individual);
    await stepForward(page, 12);
    await shot(page, 'real-replay-viewer-mobile.png');
    await context.close();
  }

  // Desktop: board mid-replay, the Game Over summary, and a team game.
  {
    const { context, page } = await newPage({ label: 'real-desktop', viewport: { width: 1280, height: 800 } });
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.getByRole('button', { name: 'Replays' }).first().waitFor({ timeout: 60000 });
    await page.getByRole('button', { name: 'Replays' }).first().click();

    await openRow(page, individual);
    await stepForward(page, 14);
    console.log(`  ${individual.id} caption at frame 14: `
      + `${(await page.locator('.game-shell').first().innerText()).split('\n').slice(0, 2).join(' / ')}`);
    await shot(page, 'real-replay-viewer-desktop.png');

    const lastIndex = await toLastFrame(page);
    const summary = await page.locator('div.fixed.inset-0.z-50').first().innerText();
    console.log(`  ${individual.id} frame ${lastIndex} summary: ${JSON.stringify(summary.replace(/\n+/g, ' | '))}`);
    await shot(page, 'real-replay-gameover-desktop.png');

    await page.getByRole('button', { name: 'Exit Replay', exact: true }).click();
    await openRow(page, team);
    const teamLast = await toLastFrame(page);
    const teamSummary = await page.locator('div.fixed.inset-0.z-50').first().innerText();
    console.log(`  ${team.id} frame ${teamLast} summary: ${JSON.stringify(teamSummary.replace(/\n+/g, ' | '))}`);
    await shot(page, 'real-replay-team-desktop.png');
    await context.close();
  }
}

await browser.close();

if (consoleErrors.length) {
  console.error(`\n${consoleErrors.length} console error(s):`);
  for (const line of consoleErrors) console.error(`  ${line}`);
} else {
  console.log('\nNo console errors.');
}

if (!process.argv.includes('--keep')) {
  killChildren();
  process.exit(consoleErrors.length ? 1 : 0);
}
