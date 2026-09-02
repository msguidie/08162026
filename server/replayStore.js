// ═══════════════════════════════════════════════════════════
// Replay storage — in-memory ring of the most recent finished games,
// optional GitHub archive, and reconstruction cache.
// Contract: docs/REPLAY_FORMAT.md §1 (index.json) and §4 (REST).
//
// Every GitHub call is async, retried with backoff, logged, and swallowed:
// storage problems must never surface in the game flow.
// ═══════════════════════════════════════════════════════════

const replayEngine = require('./replayEngine');
const replayGithub = require('./replayGithub');

const MEMORY_LIMIT = 100;
const FRAMES_CACHE_LIMIT = 20;
const REMOTE_CACHE_LIMIT = 20;
const INDEX_REFRESH_MS = 60 * 1000;
const GITHUB_ATTEMPTS = 3;
const GITHUB_BACKOFF_MS = 400;

const memoryReplays = new Map(); // id → stored JSON (insertion order: oldest first)
const memoryIndex = new Map(); // id → index entry
const framesCache = new Map(); // id → { meta, frames } (LRU)
const remoteCache = new Map(); // id → stored JSON fetched from GitHub (LRU)

let githubIndex = null; // last successfully read replays/index.json games array
let githubIndexFetchedAt = 0;
let githubIndexPromise = null;
let writeQueue = Promise.resolve();

function log(message) {
  console.log(`[replay] ${message}`);
}

function logError(message, err) {
  console.error(`[replay] ${message}${err ? `: ${err.message || err}` : ''}`);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function withRetry(label, fn, attempts = GITHUB_ATTEMPTS) {
  let delay = GITHUB_BACKOFF_MS;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return { ok: true, value: await fn() };
    } catch (err) {
      logError(`${label} attempt ${attempt}/${attempts} failed`, err);
      if (attempt < attempts) {
        await sleep(delay);
        delay *= 2;
      }
    }
  }
  return { ok: false, value: null };
}

function lruGet(cache, key) {
  if (!cache.has(key)) return undefined;
  const value = cache.get(key);
  cache.delete(key);
  cache.set(key, value);
  return value;
}

function lruSet(cache, key, value, limit) {
  if (cache.has(key)) cache.delete(key);
  cache.set(key, value);
  while (cache.size > limit) {
    const oldest = cache.keys().next();
    if (oldest.done) break;
    cache.delete(oldest.value);
  }
}

// ── paths ──────────────────────────────────────────────────

function two(value) {
  return String(value).padStart(2, '0');
}

// A replay id is a room id minted by `startGame()`: `game-<startMs>-<rand>`.
// Everything reachable from the REST layer is checked against this shape, so
// no caller can steer a cache key or a GitHub path (`..%2F`, `../`, `?ref=`).
const REPLAY_ID_RE = /^game-\d+-[a-z0-9]{1,12}$/;

function isValidReplayId(id) {
  return typeof id === 'string' && REPLAY_ID_RE.test(id);
}

function gameFilePath(dir, id, timestamp) {
  if (!isValidReplayId(id)) throw new Error(`invalid replay id: ${JSON.stringify(id)}`);
  const date = new Date(Number(timestamp) || 0);
  return `${dir}/${date.getUTCFullYear()}/${two(date.getUTCMonth() + 1)}/${id}.json`;
}

function indexFilePath(dir) {
  return `${dir}/index.json`;
}

// A room id looks like `game-<startMs>-<rand>`; the timestamp lets us find
// the YYYY/MM folder without reading the index first.
function timestampFromId(id) {
  const match = /^game-(\d+)-/.exec(String(id || ''));
  return match ? Number(match[1]) : null;
}

// ── index entries ──────────────────────────────────────────

function indexEntryFor(json) {
  return {
    id: json.id,
    t: json.t ?? null,
    e: json.e ?? null,
    mode: json.mode,
    layout: json.layout ?? null,
    n: json.n ?? (json.players || []).length,
    players: (json.players || []).map(player => player.u),
    ai: (json.players || []).map(player => player.ai === true),
    // Per-seat team ids (null in INDIVIDUAL) so the browser can star a winning
    // side without loading the whole replay — team modes leave `winners` null.
    teams: (json.players || []).map(player => (player.team ?? null)),
    winners: json.result?.winners ?? null,
    winningTeamIds: json.result?.winningTeamIds ?? null,
    turns: Array.isArray(json.actions) ? json.actions.length : 0,
  };
}

function sortNewestFirst(entries) {
  return entries.sort((a, b) => {
    const at = Number(a.t) || 0;
    const bt = Number(b.t) || 0;
    if (bt !== at) return bt - at;
    const ae = Number(a.e) || 0;
    const be = Number(b.e) || 0;
    if (be !== ae) return be - ae;
    return String(a.id).localeCompare(String(b.id));
  });
}

// ── GitHub archive ─────────────────────────────────────────

function githubEnabled() {
  return replayGithub.getClient().enabled;
}

async function readGithubIndex(force = false) {
  const client = replayGithub.getClient();
  if (!client.enabled) {
    githubIndex = null;
    return null;
  }
  const now = Date.now();
  if (!force && githubIndex && now - githubIndexFetchedAt < INDEX_REFRESH_MS) return githubIndex;
  if (githubIndexPromise) return githubIndexPromise;

  githubIndexPromise = (async () => {
    const read = await withRetry('read index.json', () => client.getFile(indexFilePath(client.dir)));
    if (!read.ok) return githubIndex; // keep the previous snapshot on failure
    const games = Array.isArray(read.value?.json?.games) ? read.value.json.games : [];
    githubIndex = games;
    githubIndexFetchedAt = Date.now();
    return githubIndex;
  })();

  try {
    return await githubIndexPromise;
  } finally {
    githubIndexPromise = null;
  }
}

async function pushGameToGithub(json) {
  const client = replayGithub.getClient();
  if (!client.enabled) return;

  const path = gameFilePath(client.dir, json.id, json.t);
  const put = await withRetry(
    `write ${path}`,
    () => client.putFile(path, json, `replay: ${json.id}`),
  );
  if (put.ok && put.value && put.value.ok === false && put.value.conflict) {
    log(`${path} already exists, keeping the stored copy`);
  } else if (put.ok) {
    log(`stored ${path}`);
  }

  const entry = indexEntryFor(json);
  const indexPath = indexFilePath(client.dir);
  const update = await withRetry(
    `update ${indexPath}`,
    () => client.updateJson(indexPath, current => {
      const games = Array.isArray(current?.games) ? current.games : [];
      const without = games.filter(game => game && game.id !== entry.id);
      return { v: 1, games: [entry, ...without] };
    }, `replay index: ${json.id}`, { attempts: 3, backoffMs: GITHUB_BACKOFF_MS }),
    // updateJson retries sha conflicts itself; this retries transport errors.
  );
  if (!update.ok || !update.value?.ok) {
    logError(`could not update ${indexPath} for ${json.id}`);
  }

  // Our own write invalidates the cached index snapshot.
  githubIndexFetchedAt = 0;
}

function enqueueGithubWrite(json) {
  writeQueue = writeQueue
    .then(() => pushGameToGithub(json))
    .catch(err => logError(`GitHub archive failed for ${json.id}`, err));
  return writeQueue;
}

// ── public API ─────────────────────────────────────────────

// Called by replayRecorder.finish(); returns immediately, GitHub I/O is async.
function add(json) {
  if (!json || !json.id) return null;
  lruSet(memoryReplays, json.id, json, MEMORY_LIMIT);
  lruSet(memoryIndex, json.id, indexEntryFor(json), MEMORY_LIMIT);
  framesCache.delete(json.id);
  remoteCache.delete(json.id);
  log(`recorded ${json.id} (${json.mode}, ${json.actions.length} actions)`);
  if (githubEnabled()) enqueueGithubWrite(json);
  return json;
}

async function listGames(options = {}) {
  const limit = Math.min(Math.max(Number(options.limit) || 50, 1), 200);
  const offset = Math.max(Number(options.offset) || 0, 0);

  let remote = null;
  if (githubEnabled()) remote = await readGithubIndex();

  const merged = new Map();
  if (Array.isArray(remote)) {
    for (const entry of remote) {
      if (entry && entry.id) merged.set(entry.id, entry);
    }
  }
  // Memory wins over the archived copy of the same game.
  for (const entry of memoryIndex.values()) merged.set(entry.id, entry);

  const games = sortNewestFirst([...merged.values()]);
  return {
    games: games.slice(offset, offset + limit),
    total: games.length,
    source: Array.isArray(remote) ? 'github' : 'memory',
  };
}

async function getReplay(id) {
  if (!isValidReplayId(id)) return null;
  const local = memoryReplays.get(id);
  if (local) return local;

  const cached = lruGet(remoteCache, id);
  if (cached) return cached;

  const client = replayGithub.getClient();
  if (!client.enabled) return null;

  let timestamp = timestampFromId(id);
  if (!timestamp) {
    const remote = await readGithubIndex();
    timestamp = (remote || []).find(entry => entry && entry.id === id)?.t || null;
  }
  if (!timestamp) return null;

  const path = gameFilePath(client.dir, id, timestamp);
  const read = await withRetry(`read ${path}`, () => client.getFile(path));
  if (!read.ok || !read.value?.json) return null;
  lruSet(remoteCache, id, read.value.json, REMOTE_CACHE_LIMIT);
  return read.value.json;
}

// Reconstructs through replayEngine; throws ReplayCorruptError for a
// replay the rules engine rejects (the REST layer answers 422).
async function getFrames(id) {
  if (!isValidReplayId(id)) return null;
  const cached = lruGet(framesCache, id);
  if (cached) return cached;

  const json = await getReplay(id);
  if (!json) return null;

  const reconstructed = replayEngine.reconstruct(json);
  lruSet(framesCache, id, reconstructed, FRAMES_CACHE_LIMIT);
  return reconstructed;
}

function status() {
  return { github: githubEnabled(), memory: memoryReplays.size };
}

// Resolves once every queued GitHub write has settled (tests, shutdown).
function flush() {
  return writeQueue.catch(() => {});
}

// Test helper — clears every cache and the in-memory ring.
function reset() {
  memoryReplays.clear();
  memoryIndex.clear();
  framesCache.clear();
  remoteCache.clear();
  githubIndex = null;
  githubIndexFetchedAt = 0;
  writeQueue = Promise.resolve();
}

module.exports = {
  add,
  listGames,
  getReplay,
  getFrames,
  status,
  flush,
  reset,
  indexEntryFor,
  gameFilePath,
  indexFilePath,
  timestampFromId,
  isValidReplayId,
  MEMORY_LIMIT,
  FRAMES_CACHE_LIMIT,
};
