// Unit tests for server/replayStore.js — memory ring, GitHub archive,
// merged listing and the reconstruction cache. The GitHub client is faked.

const { suite, test, assert, assertEqual, assertThrows } = require('./harness');
const store = require('../replayStore');
const github = require('../replayGithub');
const fixtures = require('./fixtures');

const realGetClient = github.getClient;

function fakeReplay(id, t, overrides = {}) {
  const base = fixtures.clone(fixtures.reserves);
  return {
    ...base,
    id,
    t,
    e: t + 1000,
    ...overrides,
  };
}

// A tiny in-memory stand-in for the Contents API.
function fakeGithub(options = {}) {
  const files = new Map(Object.entries(options.files || {}));
  const client = {
    enabled: options.enabled !== false,
    repo: 'owner/replays',
    branch: 'main',
    dir: 'replays',
    calls: [],
    failGetOnce: options.failGetOnce || 0,
    async getFile(path) {
      client.calls.push(['get', path]);
      if (client.failGetOnce > 0) {
        client.failGetOnce--;
        throw new Error('network down');
      }
      if (!files.has(path)) return null;
      return { sha: `sha-${path}`, json: JSON.parse(files.get(path)) };
    },
    async putFile(path, json) {
      client.calls.push(['put', path]);
      files.set(path, JSON.stringify(json));
      return { ok: true, sha: `sha-${path}` };
    },
    async updateJson(path, mutate, message) {
      client.calls.push(['update', path]);
      const current = files.has(path) ? JSON.parse(files.get(path)) : null;
      const next = await mutate(current);
      if (next == null) return { ok: false, skipped: true };
      files.set(path, JSON.stringify(next));
      return { ok: true, sha: `sha-${path}` };
    },
    files,
  };
  return client;
}

function useGithub(client) {
  github.getClient = () => client;
}

function useNoGithub() {
  github.getClient = () => ({ enabled: false, dir: 'replays', branch: 'main', repo: '' });
}

async function run() {
  suite('replayStore — in-memory ring');

  await test('stores, lists and reads back a replay without GitHub', async () => {
    useNoGithub();
    store.reset();
    const json = fakeReplay('game-1000000000000-aaaa', 1000000000000);
    assertEqual(store.add(json), json);
    assertEqual(await store.getReplay(json.id), json);

    const list = await store.listGames({});
    assertEqual(list.source, 'memory');
    assertEqual(list.total, 1);
    assertEqual(list.games[0], {
      id: json.id,
      t: json.t,
      e: json.e,
      mode: 'INDIVIDUAL',
      layout: null,
      n: 2,
      players: ['alice', 'bob'],
      ai: [false, false],
      teams: [null, null],
      winners: [0],
      winningTeamIds: null,
      turns: json.actions.length,
    });
    assertEqual(store.status(), { github: false, memory: 1 });
  });

  await test('keeps only the most recent 100 replays', async () => {
    useNoGithub();
    store.reset();
    for (let i = 0; i < 105; i++) {
      store.add(fakeReplay(`game-${1000000000000 + i}-r${i}`, 1000000000000 + i));
    }
    assertEqual(store.status().memory, store.MEMORY_LIMIT);
    assertEqual(await store.getReplay('game-1000000000000-r0'), null, 'the oldest fell out of the ring');
    assert(await store.getReplay('game-1000000000104-r104'), 'the newest is kept');
    const list = await store.listGames({ limit: 200 });
    assertEqual(list.total, 100);
    assertEqual(list.games[0].id, 'game-1000000000104-r104', 'newest first');
  });

  await test('a team entry carries per-seat team ids and the layout', async () => {
    useNoGithub();
    store.reset();
    const team = fixtures.clone(fixtures.teamGame);
    store.add(team);
    const list = await store.listGames({});
    const entry = list.games[0];
    assertEqual(entry.teams, [0, 1, 0, 1], 'seat → teamId, so the browser can star a winning side');
    assertEqual(entry.layout, 'OPPOSITE');
    assertEqual(entry.winners, null, 'team modes have no individual winners');
    assertEqual(entry.winningTeamIds, team.result.winningTeamIds);
  });

  await test('paginates with limit and offset', async () => {
    useNoGithub();
    store.reset();
    for (let i = 0; i < 5; i++) store.add(fakeReplay(`game-${2000000000000 + i}-p${i}`, 2000000000000 + i));
    const page = await store.listGames({ limit: 2, offset: 1 });
    assertEqual(page.total, 5);
    assertEqual(page.games.map(game => game.id), ['game-2000000000003-p3', 'game-2000000000002-p2']);
    assertEqual((await store.listGames({ limit: 0 })).games.length, 5, 'an unusable limit falls back to the default');
    assertEqual((await store.listGames({ limit: -3 })).games.length, 1, 'a negative limit is clamped to 1');
    assertEqual((await store.listGames({ limit: 9999 })).games.length, 5, 'limit is clamped to 200');
  });

  suite('replayStore — reconstruction cache');

  await test('getFrames reconstructs and caches', async () => {
    useNoGithub();
    store.reset();
    const json = fakeReplay('game-3000000000000-frm', 3000000000000);
    store.add(json);
    const first = await store.getFrames(json.id);
    assertEqual(first.frames.length, json.actions.length + 1);
    const second = await store.getFrames(json.id);
    assert(first === second, 'the second call comes from the cache');
    assertEqual(await store.getFrames('game-0000000000000-none'), null);
  });

  await test('getFrames surfaces ReplayCorruptError for a broken replay', async () => {
    useNoGithub();
    store.reset();
    const json = fakeReplay('game-3000000000001-bad', 3000000000001);
    json.actions[4] = [0, 'B', 71, 'b'];
    store.add(json);
    const err = await assertThrows(() => store.getFrames(json.id));
    assertEqual(err.name, 'ReplayCorruptError');
    assertEqual(err.actionIndex, 4);
  });

  suite('replayStore — GitHub archive');

  await test('writes the game file and prepends to index.json', async () => {
    const client = fakeGithub();
    useGithub(client);
    store.reset();
    const json = fakeReplay('game-1725280000000-gh1', 1725280000000);
    store.add(json);
    await store.flush();

    const path = 'replays/2024/09/game-1725280000000-gh1.json';
    assert(client.files.has(path), `expected ${path}, got ${[...client.files.keys()].join(', ')}`);
    assertEqual(JSON.parse(client.files.get(path)), json);
    const index = JSON.parse(client.files.get('replays/index.json'));
    assertEqual(index.v, 1);
    assertEqual(index.games.map(game => game.id), [json.id]);
    assertEqual(store.status().github, true);
  });

  await test('a second game is prepended ahead of the first', async () => {
    const client = fakeGithub();
    useGithub(client);
    store.reset();
    store.add(fakeReplay('game-1725280000000-gh1', 1725280000000));
    await store.flush();
    store.add(fakeReplay('game-1725290000000-gh2', 1725290000000));
    await store.flush();
    const index = JSON.parse(client.files.get('replays/index.json'));
    assertEqual(index.games.map(game => game.id), ['game-1725290000000-gh2', 'game-1725280000000-gh1']);
  });

  await test('merges the GitHub index with memory, newest first, without duplicates', async () => {
    const client = fakeGithub({
      files: {
        'replays/index.json': JSON.stringify({
          v: 1,
          games: [
            { id: 'game-1725290000000-remote', t: 1725290000000, e: 1, mode: 'TEAM', n: 4, players: [], ai: [], winners: null, winningTeamIds: [0], turns: 12 },
            { id: 'game-1725270000000-shared', t: 1725270000000, e: 1, mode: 'INDIVIDUAL', n: 2, players: [], ai: [], winners: [0], winningTeamIds: null, turns: 3 },
          ],
        }),
      },
    });
    useGithub(client);
    store.reset();
    store.add(fakeReplay('game-1725280000000-local', 1725280000000));
    store.add(fakeReplay('game-1725270000000-shared', 1725270000000));
    await store.flush();

    const list = await store.listGames({ limit: 50 });
    assertEqual(list.source, 'github');
    assertEqual(list.games.map(game => game.id), [
      'game-1725290000000-remote',
      'game-1725280000000-local',
      'game-1725270000000-shared',
    ]);
    assertEqual(list.total, 3);
    const shared = list.games.find(game => game.id === 'game-1725270000000-shared');
    assertEqual(shared.turns, fixtures.reserves.actions.length, 'the in-memory entry wins');
  });

  await test('falls back to memory when the index cannot be read', async () => {
    const client = fakeGithub({ failGetOnce: 99 });
    useGithub(client);
    store.reset();
    store.add(fakeReplay('game-1725280000000-off', 1725280000000));
    const list = await store.listGames({});
    assertEqual(list.source, 'memory', 'a failed index read degrades to memory');
    assertEqual(list.games.length, 1);
  });

  await test('fetches an archived replay by its id timestamp', async () => {
    const archived = fakeReplay('game-1725280000000-old', 1725280000000);
    const client = fakeGithub({
      files: { 'replays/2024/09/game-1725280000000-old.json': JSON.stringify(archived) },
    });
    useGithub(client);
    store.reset();
    const found = await store.getReplay(archived.id);
    assertEqual(found, archived);
    assertEqual(client.calls[0], ['get', 'replays/2024/09/game-1725280000000-old.json']);
    const again = await store.getReplay(archived.id);
    assertEqual(again, archived);
    assertEqual(client.calls.filter(call => call[0] === 'get').length, 1, 'the fetch is cached');
    const frames = await store.getFrames(archived.id);
    assertEqual(frames.frames.length, archived.actions.length + 1);
  });

  await test('an id the archive does not hold is null, never an exception', async () => {
    const client = fakeGithub({ failGetOnce: 99 });
    useGithub(client);
    store.reset();
    assertEqual(await store.getReplay('game-1725280000000-gone'), null);
    assertEqual(await store.getFrames('game-1725280000000-gone'), null);
    assertEqual(await store.getReplay('not-a-game-id'), null);
  });

  await test('a GitHub outage never throws into add()', async () => {
    const client = fakeGithub();
    client.putFile = async () => { throw new Error('502'); };
    client.updateJson = async () => { throw new Error('502'); };
    useGithub(client);
    store.reset();
    const json = fakeReplay('game-1725280000000-err', 1725280000000);
    assertEqual(store.add(json), json, 'add stays synchronous and successful');
    await store.flush();
    assertEqual(await store.getReplay(json.id), json, 'memory still serves the replay');
  });

  suite('replayStore — paths');

  await test('derives YYYY/MM paths in UTC', () => {
    assertEqual(store.gameFilePath('replays', 'game-1', Date.UTC(2026, 0, 5)), 'replays/2026/01/game-1.json');
    assertEqual(store.gameFilePath('games', 'game-2', Date.UTC(2026, 11, 31)), 'games/2026/12/game-2.json');
    assertEqual(store.indexFilePath('replays'), 'replays/index.json');
    assertEqual(store.timestampFromId('game-1725280000000-ab12'), 1725280000000);
    assertEqual(store.timestampFromId('nonsense'), null);
  });

  // Leave the process exactly as it was for the end-to-end suite.
  github.getClient = realGetClient;
  store.reset();
}

module.exports = { run };
