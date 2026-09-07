// ═══════════════════════════════════════════════════════════
// Minimal GitHub Contents API client for replay storage
// Contract: docs/REPLAY_FORMAT.md §1 / §4 (env vars)
//
// Nothing in here is allowed to affect gameplay: every caller
// (replayStore.js) treats failures as "GitHub is unavailable".
// ═══════════════════════════════════════════════════════════

const API_ROOT = 'https://api.github.com';

class GithubApiError extends Error {
  constructor(status, message, body) {
    super(message);
    this.name = 'GithubApiError';
    this.status = status;
    this.body = body;
  }
}

function encodeJson(json) {
  return Buffer.from(JSON.stringify(json), 'utf8').toString('base64');
}

function decodeJson(base64Content) {
  // GitHub wraps base64 payloads at 60 chars; Buffer ignores the newlines.
  return JSON.parse(Buffer.from(String(base64Content || ''), 'base64').toString('utf8'));
}

function encodePath(path) {
  return String(path)
    .split('/')
    .filter(segment => segment.length > 0)
    .map(segment => encodeURIComponent(segment))
    .join('/');
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// `fetch` and `env` are injectable so tests can run without network access.
function createClient(options = {}) {
  const env = options.env || process.env;
  const fetchImpl = options.fetch || (typeof globalThis.fetch === 'function' ? globalThis.fetch : null);
  const token = env.REPLAY_GITHUB_TOKEN || '';
  const repo = env.REPLAY_GITHUB_REPO || '';
  const branch = env.REPLAY_GITHUB_BRANCH || 'main';
  const dir = env.REPLAY_GITHUB_DIR || 'replays';
  const enabled = !!(token && repo && fetchImpl);

  function headers() {
    return {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'splendor-replay-store',
      'Content-Type': 'application/json',
    };
  }

  async function readBody(response) {
    try {
      return await response.json();
    } catch (err) {
      return null;
    }
  }

  // → { sha, json } when the file exists, null on 404 (missing).
  async function getFile(path) {
    if (!enabled) return null;
    const url = `${API_ROOT}/repos/${repo}/contents/${encodePath(path)}?ref=${encodeURIComponent(branch)}`;
    const response = await fetchImpl(url, { method: 'GET', headers: headers() });
    if (response.status === 404) return null;
    if (!response.ok) {
      const body = await readBody(response);
      throw new GithubApiError(response.status, `GET ${path} failed with ${response.status}`, body);
    }
    const body = await readBody(response);
    if (!body || typeof body.content !== 'string') {
      throw new GithubApiError(response.status, `GET ${path} returned no content`, body);
    }
    return { sha: body.sha, json: decodeJson(body.content) };
  }

  // → { ok: true, sha } on success,
  //   { ok: false, conflict: true, status } on 409/422 (stale or missing sha).
  async function putFile(path, json, message, sha) {
    if (!enabled) return { ok: false, disabled: true };
    const url = `${API_ROOT}/repos/${repo}/contents/${encodePath(path)}`;
    const payload = {
      message: message || `chore(replay): update ${path}`,
      content: encodeJson(json),
      branch,
    };
    if (sha) payload.sha = sha;
    const response = await fetchImpl(url, {
      method: 'PUT',
      headers: headers(),
      body: JSON.stringify(payload),
    });
    if (response.status === 409 || response.status === 422) {
      const body = await readBody(response);
      return { ok: false, conflict: true, status: response.status, body };
    }
    if (!response.ok) {
      const body = await readBody(response);
      throw new GithubApiError(response.status, `PUT ${path} failed with ${response.status}`, body);
    }
    const body = await readBody(response);
    return { ok: true, sha: body?.content?.sha || null };
  }

  // Read → mutate → write with sha, re-reading on 409/422 conflicts.
  // `mutate(currentJson | null)` returns the JSON to store (or null to skip).
  async function updateJson(path, mutate, message, config = {}) {
    if (!enabled) return { ok: false, disabled: true };
    const attempts = Number.isInteger(config.attempts) ? config.attempts : 3;
    const backoffMs = Number.isInteger(config.backoffMs) ? config.backoffMs : 250;
    let lastConflict = null;
    for (let attempt = 0; attempt < attempts; attempt++) {
      const existing = await getFile(path);
      const next = await mutate(existing ? existing.json : null);
      if (next === null || next === undefined) return { ok: false, skipped: true };
      const result = await putFile(path, next, message, existing ? existing.sha : undefined);
      if (result.ok) return result;
      if (!result.conflict) return result;
      lastConflict = result;
      if (attempt < attempts - 1) await sleep(backoffMs * (attempt + 1));
    }
    return lastConflict || { ok: false, conflict: true };
  }

  return {
    enabled,
    repo,
    branch,
    dir,
    getFile,
    putFile,
    updateJson,
  };
}

let defaultClient = null;

// Lazily built so environment variables set after require() are honoured.
function getClient() {
  if (!defaultClient) defaultClient = createClient();
  return defaultClient;
}

function resetClient() {
  defaultClient = null;
}

module.exports = {
  createClient,
  getClient,
  resetClient,
  GithubApiError,
  encodeJson,
  decodeJson,
};
