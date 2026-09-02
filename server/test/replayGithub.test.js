// Unit tests for server/replayGithub.js using an injected fake fetch —
// no network access, no token required.

const { suite, test, assert, assertEqual, assertThrows } = require('./harness');
const { createClient, getClient, resetClient, GithubApiError } = require('../replayGithub');

const ENV = {
  REPLAY_GITHUB_TOKEN: 'ghp_test_token',
  REPLAY_GITHUB_REPO: 'owner/replays',
  REPLAY_GITHUB_BRANCH: 'main',
  REPLAY_GITHUB_DIR: 'replays',
};

function response(status, body) {
  return {
    status,
    ok: status >= 200 && status < 300,
    json: async () => {
      if (body === undefined) throw new Error('no body');
      return body;
    },
  };
}

function contentsResponse(json, sha) {
  return response(200, {
    sha,
    content: Buffer.from(JSON.stringify(json), 'utf8').toString('base64'),
  });
}

// Fake fetch driven by a queue of handlers; records every call.
function fakeFetch(handlers) {
  const calls = [];
  const queue = [...handlers];
  const impl = async (url, init) => {
    const body = init && init.body ? JSON.parse(init.body) : null;
    calls.push({ url, method: (init && init.method) || 'GET', headers: (init && init.headers) || {}, body });
    const handler = queue.shift();
    if (!handler) throw new Error(`unexpected fetch call: ${init?.method} ${url}`);
    return handler({ url, init, body });
  };
  impl.calls = calls;
  impl.remaining = () => queue.length;
  return impl;
}

async function run() {
  suite('replayGithub — configuration');

  await test('a missing token disables the client', async () => {
    const client = createClient({ env: { REPLAY_GITHUB_REPO: 'owner/replays' }, fetch: fakeFetch([]) });
    assertEqual(client.enabled, false);
    assertEqual(await client.getFile('replays/index.json'), null);
    assertEqual(await client.putFile('replays/x.json', {}, 'msg'), { ok: false, disabled: true });
    assertEqual(await client.updateJson('replays/index.json', () => ({}), 'msg'), { ok: false, disabled: true });
  });

  await test('a missing repo disables the client', () => {
    const client = createClient({ env: { REPLAY_GITHUB_TOKEN: 'x' }, fetch: fakeFetch([]) });
    assertEqual(client.enabled, false);
  });

  await test('defaults branch to main and dir to replays', () => {
    const client = createClient({
      env: { REPLAY_GITHUB_TOKEN: 'x', REPLAY_GITHUB_REPO: 'owner/repo' },
      fetch: fakeFetch([]),
    });
    assertEqual(client.enabled, true);
    assertEqual(client.branch, 'main');
    assertEqual(client.dir, 'replays');
  });

  await test('honours REPLAY_GITHUB_BRANCH and REPLAY_GITHUB_DIR', () => {
    const client = createClient({
      env: { ...ENV, REPLAY_GITHUB_BRANCH: 'archive', REPLAY_GITHUB_DIR: 'games' },
      fetch: fakeFetch([]),
    });
    assertEqual(client.branch, 'archive');
    assertEqual(client.dir, 'games');
  });

  await test('the process-wide client is disabled without env vars', () => {
    resetClient();
    const client = getClient();
    assertEqual(client.enabled, !!(process.env.REPLAY_GITHUB_TOKEN && process.env.REPLAY_GITHUB_REPO));
  });

  suite('replayGithub — getFile');

  await test('reads and decodes a JSON file', async () => {
    const fetchImpl = fakeFetch([() => contentsResponse({ v: 1, games: [{ id: 'g1' }] }, 'sha-1')]);
    const client = createClient({ env: ENV, fetch: fetchImpl });
    const file = await client.getFile('replays/2026/09/game-1.json');
    assertEqual(file.sha, 'sha-1');
    assertEqual(file.json, { v: 1, games: [{ id: 'g1' }] });

    const call = fetchImpl.calls[0];
    assertEqual(call.method, 'GET');
    assertEqual(call.url,
      'https://api.github.com/repos/owner/replays/contents/replays/2026/09/game-1.json?ref=main');
    assertEqual(call.headers.Authorization, 'Bearer ghp_test_token');
    assertEqual(call.headers.Accept, 'application/vnd.github+json');
  });

  await test('returns null for a missing file (404)', async () => {
    const client = createClient({ env: ENV, fetch: fakeFetch([() => response(404, { message: 'Not Found' })]) });
    assertEqual(await client.getFile('replays/index.json'), null);
  });

  await test('throws a GithubApiError for other failures', async () => {
    const client = createClient({ env: ENV, fetch: fakeFetch([() => response(500, { message: 'boom' })]) });
    const err = await assertThrows(() => client.getFile('replays/index.json'));
    assert(err instanceof GithubApiError, 'expected GithubApiError');
    assertEqual(err.status, 500);
  });

  await test('decodes newline-wrapped base64 and UTF-8 content', async () => {
    const json = { u: 'Ünïcødé 玩家', n: 3 };
    const wrapped = Buffer.from(JSON.stringify(json), 'utf8').toString('base64').replace(/(.{6})/g, '$1\n');
    const client = createClient({ env: ENV, fetch: fakeFetch([() => response(200, { sha: 's', content: wrapped })]) });
    assertEqual((await client.getFile('replays/x.json')).json, json);
  });

  suite('replayGithub — putFile');

  await test('creates a new file without a sha', async () => {
    const fetchImpl = fakeFetch([() => response(201, { content: { sha: 'new-sha' } })]);
    const client = createClient({ env: ENV, fetch: fetchImpl });
    const result = await client.putFile('replays/2026/09/g.json', { id: 'g', u: 'Ünïcødé' }, 'replay: g');
    assertEqual(result, { ok: true, sha: 'new-sha' });

    const call = fetchImpl.calls[0];
    assertEqual(call.method, 'PUT');
    assertEqual(call.body.message, 'replay: g');
    assertEqual(call.body.branch, 'main');
    assert(!('sha' in call.body), 'a create must not send a sha');
    assertEqual(JSON.parse(Buffer.from(call.body.content, 'base64').toString('utf8')),
      { id: 'g', u: 'Ünïcødé' }, 'content is UTF-8 base64');
  });

  await test('updates an existing file with its sha', async () => {
    const fetchImpl = fakeFetch([() => response(200, { content: { sha: 'sha-2' } })]);
    const client = createClient({ env: ENV, fetch: fetchImpl });
    const result = await client.putFile('replays/index.json', { v: 1 }, 'index', 'sha-1');
    assertEqual(result, { ok: true, sha: 'sha-2' });
    assertEqual(fetchImpl.calls[0].body.sha, 'sha-1');
  });

  await test('reports 409 and 422 as conflicts rather than errors', async () => {
    for (const status of [409, 422]) {
      const client = createClient({ env: ENV, fetch: fakeFetch([() => response(status, { message: 'conflict' })]) });
      const result = await client.putFile('replays/index.json', { v: 1 }, 'index', 'stale');
      assertEqual(result.ok, false);
      assertEqual(result.conflict, true);
      assertEqual(result.status, status);
    }
  });

  await test('throws for other write failures', async () => {
    const client = createClient({ env: ENV, fetch: fakeFetch([() => response(403, { message: 'forbidden' })]) });
    const err = await assertThrows(() => client.putFile('replays/index.json', { v: 1 }, 'index'));
    assertEqual(err.status, 403);
  });

  suite('replayGithub — updateJson (read → mutate → write)');

  await test('creates index.json when it does not exist yet', async () => {
    const fetchImpl = fakeFetch([
      () => response(404, { message: 'Not Found' }),
      () => response(201, { content: { sha: 'sha-1' } }),
    ]);
    const client = createClient({ env: ENV, fetch: fetchImpl });
    const result = await client.updateJson('replays/index.json', current => {
      assertEqual(current, null, 'mutator sees null for a missing file');
      return { v: 1, games: [{ id: 'g1' }] };
    }, 'index: g1', { backoffMs: 0 });
    assertEqual(result.ok, true);
    assertEqual(JSON.parse(Buffer.from(fetchImpl.calls[1].body.content, 'base64').toString('utf8')),
      { v: 1, games: [{ id: 'g1' }] });
  });

  await test('prepends to an existing index and sends its sha', async () => {
    const fetchImpl = fakeFetch([
      () => contentsResponse({ v: 1, games: [{ id: 'old' }] }, 'sha-old'),
      () => response(200, { content: { sha: 'sha-new' } }),
    ]);
    const client = createClient({ env: ENV, fetch: fetchImpl });
    const result = await client.updateJson('replays/index.json',
      current => ({ v: 1, games: [{ id: 'new' }, ...current.games] }),
      'index: new', { backoffMs: 0 });
    assertEqual(result.ok, true);
    assertEqual(fetchImpl.calls[1].body.sha, 'sha-old');
    assertEqual(JSON.parse(Buffer.from(fetchImpl.calls[1].body.content, 'base64').toString('utf8')).games,
      [{ id: 'new' }, { id: 'old' }]);
  });

  await test('re-reads the sha and retries on a conflict', async () => {
    const fetchImpl = fakeFetch([
      () => contentsResponse({ v: 1, games: [{ id: 'a' }] }, 'sha-1'),
      () => response(409, { message: 'conflict' }),
      () => contentsResponse({ v: 1, games: [{ id: 'b' }, { id: 'a' }] }, 'sha-2'),
      () => response(200, { content: { sha: 'sha-3' } }),
    ]);
    const client = createClient({ env: ENV, fetch: fetchImpl });
    const result = await client.updateJson('replays/index.json',
      current => ({ v: 1, games: [{ id: 'mine' }, ...current.games] }),
      'index: mine', { backoffMs: 0 });
    assertEqual(result, { ok: true, sha: 'sha-3' });
    assertEqual(fetchImpl.calls.length, 4);
    assertEqual(fetchImpl.calls[3].body.sha, 'sha-2', 'retry uses the freshly read sha');
    assertEqual(JSON.parse(Buffer.from(fetchImpl.calls[3].body.content, 'base64').toString('utf8')).games,
      [{ id: 'mine' }, { id: 'b' }, { id: 'a' }], 'the concurrent write is preserved');
  });

  await test('gives up after three conflicting attempts', async () => {
    const handlers = [];
    for (let i = 0; i < 3; i++) {
      handlers.push(() => contentsResponse({ v: 1, games: [] }, `sha-${i}`));
      handlers.push(() => response(422, { message: 'conflict' }));
    }
    const fetchImpl = fakeFetch(handlers);
    const client = createClient({ env: ENV, fetch: fetchImpl });
    const result = await client.updateJson('replays/index.json', () => ({ v: 1, games: [] }), 'index',
      { attempts: 3, backoffMs: 0 });
    assertEqual(result.ok, false);
    assertEqual(result.conflict, true);
    assertEqual(fetchImpl.calls.length, 6, 'three read/write rounds');
    assertEqual(fetchImpl.remaining(), 0);
  });

  await test('a mutator returning null skips the write', async () => {
    const fetchImpl = fakeFetch([() => contentsResponse({ v: 1, games: [] }, 'sha')]);
    const client = createClient({ env: ENV, fetch: fetchImpl });
    const result = await client.updateJson('replays/index.json', () => null, 'index', { backoffMs: 0 });
    assertEqual(result, { ok: false, skipped: true });
    assertEqual(fetchImpl.calls.length, 1);
  });
}

module.exports = { run };
