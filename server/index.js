const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const {
  createInitialGameState,
  clientViewForPlayer,
  processAction,
  processResign,
  calculateRatingChanges,
  updateTimeControl,
  consumeTurnTime,
  addTimeIncrement,
  startTurnTimeControl,
  pauseTurnTimeControl,
} = require('./gameLogic');
const replayRecorder = require('./replayRecorder');
const replayStore = require('./replayStore');
const aiBridge = require('./aiBridge');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: '*', methods: ['GET', 'POST'] },
  pingTimeout: 30000,
  pingInterval: 10000,
  transports: ['websocket', 'polling'],
  allowUpgrades: true,
  connectTimeout: 45000,
});

app.use(cors());
app.use(express.json());

// ── In-memory stores ──
const accounts = new Map();
const socketToAccount = new Map();
const accountToSocket = new Map();
const gameRooms = new Map();

// Single global lobby
let lobbyQueue = []; // [{ socketId, username, ready, avatarSeed }]
let lobbySettings = createDefaultLobbySettings();

let avatarCounter = 1;
function nextAvatarSeed() { return avatarCounter++; }

// ── Replay recording (additive: a recorder failure never affects the game) ──
function replaySafe(fn) {
  try {
    fn();
  } catch (err) {
    console.error('[replay] hook failed:', err?.message || err);
  }
}

function replayFinishIfGameOver(room) {
  if (room?.gameState?.phase === 'GAME_OVER') replaySafe(() => replayRecorder.finish(room));
}

// ── Helpers ──

function safeDeleteAccountSocket(username, socketId) {
  if (accountToSocket.get(username) === socketId) accountToSocket.delete(username);
}

function createDefaultLobbySettings() {
  return {
    teamMode: false,
    teamFormat: null,
    teamLayout: 'ADJACENT',
    teamSeats: [[null, null], [null, null]],
    unlimitedTime: false,
  };
}

function resetLobbySettings(resetReady = false, preserveUnlimitedTime = false) {
  const unlimitedTime = preserveUnlimitedTime ? lobbySettings.unlimitedTime : false;
  lobbySettings = { ...createDefaultLobbySettings(), unlimitedTime };
  if (resetReady) resetLobbyReady();
}

// Bots are always ready (docs/AI_BRIDGE.md §3); humans go back to "not ready".
function resetLobbyReady() {
  lobbyQueue.forEach(player => { player.ready = isBotEntry(player); });
}

function lobbyState() {
  return {
    players: lobbyQueue.map(p => ({
      socketId: p.socketId,
      username: p.username,
      ready: p.ready,
      wantsFirst: !!p.wantsFirst,
      avatarSeed: p.avatarSeed,
      isAI: p.isAI === true,
    })),
    teamMode: lobbySettings.teamMode,
    teamFormat: lobbySettings.teamFormat,
    teamLayout: lobbySettings.teamLayout,
    teamSeats: lobbySettings.teamSeats.map(team => [...team]),
    unlimitedTime: lobbySettings.unlimitedTime,
    aiAvailable: aiBridge.isAvailable(),
  };
}

function isPlayerSeated(username) {
  return lobbySettings.teamSeats.some(team => team.includes(username));
}

function requiredTeamSeatCoordinates() {
  if (lobbySettings.teamFormat === 'ONE_V_TWO') return [[0, 0], [1, 0], [1, 1]];
  if (lobbySettings.teamFormat === 'TWO_V_TWO') return [[0, 0], [0, 1], [1, 0], [1, 1]];
  return [];
}

function requiredTeamPlayerCount() {
  return requiredTeamSeatCoordinates().length;
}

function hasCompleteTeamSeats() {
  const usernames = requiredTeamSeatCoordinates().map(([teamId, seatIndex]) =>
    lobbySettings.teamSeats[teamId][seatIndex]);
  return usernames.every(Boolean)
    && usernames.length === requiredTeamPlayerCount()
    && new Set(usernames).size === usernames.length
    && usernames.every(username => lobbyQueue.some(player => player.username === username));
}

// ── AI bots in the lobby (docs/AI_BRIDGE.md §3) ──
// Bot accounts are ordinary accounts, auto-created on first use and rated
// like humans. A bot lobby entry carries `isAI: true` and a synthetic
// socket id, so every existing lookup by socket id simply never matches it.
const BOT_USERNAMES = ['Bot Alpha', 'Bot Beta', 'Bot Gamma', 'Bot Delta'];

function isBotEntry(entry) {
  return entry?.isAI === true;
}

function humanLobbyCount() {
  return lobbyQueue.filter(player => !isBotEntry(player)).length;
}

function ensureBotAccount(username) {
  let account = accounts.get(username);
  if (!account) {
    account = { username, rating: 1000, gamesPlayed: 0, wins: 0, avatarSeed: nextAvatarSeed(), created: Date.now() };
    accounts.set(username, account);
  }
  return account;
}

function lobbyCapacity() {
  return lobbySettings.teamMode ? requiredTeamPlayerCount() : 6;
}

function addBotToLobby() {
  if (!aiBridge.isEnabled()) return { error: 'AI is not enabled on this server' };
  if (!aiBridge.isAvailable()) return { error: 'No AI worker is connected right now' };
  if (humanLobbyCount() === 0) return { error: 'A human player has to be in the lobby first' };
  if (lobbyQueue.length >= lobbyCapacity()) return { error: 'This lobby is full' };
  const username = BOT_USERNAMES.find(name => !lobbyQueue.some(player => player.username === name));
  if (!username) return { error: 'No more AI players are available' };
  const account = ensureBotAccount(username);
  lobbyQueue.push({
    socketId: `ai:${username}`,
    username,
    ready: true,
    wantsFirst: false,
    avatarSeed: account.avatarSeed,
    isAI: true,
  });
  return { ok: true, username };
}

function removeBotFromLobby(username) {
  const bot = lobbyQueue.find(player => player.username === username && isBotEntry(player));
  if (!bot) return { error: 'That AI player is not in this lobby' };
  lobbyQueue = lobbyQueue.filter(player => player !== bot);
  lobbySettings.teamSeats = lobbySettings.teamSeats.map(team =>
    team.map(name => name === username ? null : name));
  if (lobbySettings.teamMode && lobbyQueue.length !== requiredTeamPlayerCount()) resetLobbySettings(true, true);
  return { ok: true };
}

// Bots never hold a lobby on their own.
function dropBotsWithoutHumans() {
  if (lobbyQueue.length === 0 || humanLobbyCount() > 0) return false;
  lobbyQueue = [];
  resetLobbySettings();
  return true;
}

// ── REST endpoints ──
app.get('/', (_req, res) => {
  res.json({
    status: 'running',
    accounts: accounts.size,
    lobby: lobbyQueue.length,
    games: gameRooms.size,
    connections: io.engine.clientsCount,
  });
});

app.get('/health', (_req, res) => res.json({ status: 'healthy', timestamp: Date.now() }));

app.get('/api/accounts', (_req, res) => {
  res.json(Array.from(accounts.values()).map(a => ({
    username: a.username, rating: a.rating, gamesPlayed: a.gamesPlayed,
    wins: a.wins, avatarSeed: a.avatarSeed,
  })).sort((a, b) => b.rating - a.rating));
});

app.post('/api/accounts', (req, res) => {
  const { username } = req.body;
  if (!username || typeof username !== 'string' || !username.trim()) return res.status(400).json({ error: 'Username required' });
  const name = username.trim();
  if (accounts.has(name)) return res.status(409).json({ error: 'Username already exists' });
  const account = { username: name, rating: 1000, gamesPlayed: 0, wins: 0, avatarSeed: nextAvatarSeed(), created: Date.now() };
  accounts.set(name, account);
  res.json(account);
});

// ── Replay REST API (docs/REPLAY_FORMAT.md §4) ──
app.get('/api/replays', async (req, res) => {
  try {
    res.json(await replayStore.listGames({ limit: req.query.limit, offset: req.query.offset }));
  } catch (err) {
    console.error('[replay] list failed:', err?.message || err);
    res.status(500).json({ error: 'Failed to list replays' });
  }
});

app.get('/api/replays/status', (_req, res) => res.json(replayStore.status()));

app.get('/api/replays/:id', async (req, res) => {
  try {
    const data = await replayStore.getFrames(req.params.id);
    if (!data) { res.status(404).json({ error: 'Replay not found' }); return; }
    res.json({ id: req.params.id, meta: data.meta, frames: data.frames });
  } catch (err) {
    if (err?.name === 'ReplayCorruptError') {
      res.status(422).json({ error: err.message, actionIndex: err.actionIndex });
      return;
    }
    console.error('[replay] reconstruct failed:', err?.message || err);
    res.status(500).json({ error: 'Failed to reconstruct replay' });
  }
});

app.get('/api/replays/:id/raw', async (req, res) => {
  try {
    const json = await replayStore.getReplay(req.params.id);
    if (!json) { res.status(404).json({ error: 'Replay not found' }); return; }
    res.json(json);
  } catch (err) {
    console.error('[replay] raw fetch failed:', err?.message || err);
    res.status(500).json({ error: 'Failed to read replay' });
  }
});

// ── AI bridge status (docs/AI_BRIDGE.md §5) ──
app.get('/api/ai/status', (_req, res) => res.json(aiBridge.status()));

// Self-ping to keep Render awake
const RENDER_URL = process.env.RENDER_EXTERNAL_URL;
if (RENDER_URL) {
  setInterval(() => { fetch(`${RENDER_URL}/health`).catch(() => {}); }, 5 * 60 * 1000);
}

// ── Periodic cleanup ──
// Only clean up rooms where NO player has done ANYTHING for 10 minutes.
// We track lastPlayerAction (updated on actual game actions AND heartbeats from room players).
const INACTIVITY_TIMEOUT = 10 * 60 * 1000;
const ROOM_TTL = 2 * 60 * 60 * 1000;

setInterval(() => {
  const now = Date.now();
  for (const [id, room] of gameRooms) {
    if (room.gameState?.phase === 'GAME_OVER' && now - room.lastPlayerAction > 5 * 60 * 1000) {
      replaySafe(() => replayRecorder.discard(room));
      aiBridge.clearRoom(id);
      gameRooms.delete(id);
      continue;
    }
    if (now - room.lastPlayerAction > INACTIVITY_TIMEOUT) {
      console.log(`Auto-abandoning inactive game ${id}`);
      for (const p of room.playerSockets) {
        const sock = io.sockets.sockets.get(p.socketId);
        if (sock) { sock.emit('game_abandoned', { reason: 'Game abandoned due to inactivity.' }); sock.leave(id); }
      }
      replaySafe(() => replayRecorder.discard(room));
      aiBridge.clearRoom(id);
      gameRooms.delete(id);
      continue;
    }
    if (now - room.created > ROOM_TTL) {
      replaySafe(() => replayRecorder.discard(room));
      aiBridge.clearRoom(id);
      gameRooms.delete(id);
    }
  }
}, 60 * 1000);

// ── Lobby helpers ──
function broadcastLobby() {
  const data = lobbyState();
  lobbyQueue.forEach(p => {
    io.to(p.socketId).emit('lobby_update', data);
  });
}

function removeFromLobby(socketId) {
  const leavingPlayer = lobbyQueue.find(p => p.socketId === socketId);
  const before = lobbyQueue.length;
  lobbyQueue = lobbyQueue.filter(p => p.socketId !== socketId);
  if (lobbyQueue.length !== before) {
    if (leavingPlayer) {
      lobbySettings.teamSeats = lobbySettings.teamSeats.map(team =>
        team.map(username => username === leavingPlayer.username ? null : username));
    }
    if (dropBotsWithoutHumans()) { broadcastLobby(); return; }
    if (lobbySettings.teamMode && lobbyQueue.length !== requiredTeamPlayerCount()) resetLobbySettings(true, true);
    broadcastLobby();
  }
}

// Auto-start when all >= 2 are ready
function checkAutoStart() {
  // A lobby of bots alone never starts a game.
  if (humanLobbyCount() === 0) return;
  if (lobbySettings.teamMode) {
    if (lobbyQueue.length !== requiredTeamPlayerCount() || !hasCompleteTeamSeats()) return;
    if (!lobbyQueue.every(p => p.ready)) return;
    startGame();
    return;
  }
  if (lobbyQueue.length < 2) return;
  if (!lobbyQueue.every(p => p.ready)) return;
  startGame();
}

function startGame() {
  const roomId = `game-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
  let playerInfos;
  const gameOptions = lobbySettings.teamMode
    ? {
      gameMode: lobbySettings.teamFormat === 'ONE_V_TWO' ? 'ONE_V_TWO' : 'TEAM',
      teamLayout: lobbySettings.teamFormat === 'TWO_V_TWO' ? lobbySettings.teamLayout : null,
      unlimitedTime: lobbySettings.unlimitedTime,
    }
    : { gameMode: 'INDIVIDUAL', unlimitedTime: lobbySettings.unlimitedTime };

  if (lobbySettings.teamMode) {
    const seatOrder = lobbySettings.teamFormat === 'ONE_V_TWO'
      ? [[0, 0], [1, 0], [1, 1]]
      : lobbySettings.teamLayout === 'OPPOSITE'
        ? [[0, 0], [1, 0], [0, 1], [1, 1]]
        : [[0, 0], [0, 1], [1, 0], [1, 1]];
    playerInfos = seatOrder.map(([teamId, seatIndex]) => {
      const username = lobbySettings.teamSeats[teamId][seatIndex];
      const player = lobbyQueue.find(p => p.username === username);
      return {
        username: player.username,
        socketId: player.socketId,
        avatarSeed: player.avatarSeed,
        teamId,
        isAI: isBotEntry(player),
      };
    });
  } else {
    playerInfos = lobbyQueue.map(p => ({
      username: p.username,
      socketId: p.socketId,
      avatarSeed: p.avatarSeed,
      isAI: isBotEntry(p),
    }));
  }

  // Multiple volunteers are resolved randomly; with no volunteer, everyone
  // remains eligible. The 1v2 solo-seat rule takes precedence in game logic.
  if (gameOptions.gameMode !== 'ONE_V_TWO') {
    const preferredIndices = playerInfos
      .map((player, index) => lobbyQueue.find(entry => entry.username === player.username)?.wantsFirst ? index : -1)
      .filter(index => index >= 0);
    if (preferredIndices.length > 0) {
      gameOptions.firstPlayerIndex = preferredIndices[Math.floor(Math.random() * preferredIndices.length)];
    }
  }

  // Create server-authoritative game state
  const gameState = createInitialGameState(playerInfos, gameOptions);

  const room = {
    id: roomId,
    playerSockets: playerInfos.map((p, i) => (p.isAI
      ? { socketId: `ai:${i}`, username: p.username, playerIndex: i, isAI: true }
      : { socketId: p.socketId, username: p.username, playerIndex: i })),
    gameState,
    lastPlayerAction: Date.now(),
    created: Date.now(),
    resultsApplied: false,
    ratingChanges: null,
  };
  gameRooms.set(roomId, room);
  replaySafe(() => replayRecorder.begin(room));

  // Seats played by bots, so every client can mark them in-game.
  const aiSeats = playerInfos.reduce((seats, p, i) => (p.isAI ? [...seats, i] : seats), []);

  // Notify each player
  playerInfos.forEach((p, idx) => {
    const s = io.sockets.sockets.get(p.socketId);
    if (s) s.join(roomId);
    io.to(p.socketId).emit('game_start', {
      roomId,
      playerIndex: idx,
      gameState: clientViewForPlayer(gameState, idx),
      aiSeats,
    });
  });

  // Turn driver: hands over to the AI bridge when seat 0 to move is a bot.
  aiBridge.maybeAct(room);

  lobbyQueue = [];
  resetLobbySettings();
  broadcastLobby();
  console.log(`Game ${roomId} started with ${playerInfos.length} players`);
}

// Find active game for a username
function findActiveGame(username) {
  for (const [roomId, room] of gameRooms) {
    if (room.gameState?.phase === 'GAME_OVER') continue;
    const ps = room.playerSockets.find(p => p.username === username);
    if (ps) return { roomId, room, playerSocket: ps };
  }
  return null;
}

// Send game state to all players in a room
function broadcastGameState(room) {
  for (const ps of room.playerSockets) {
    if (!ps.socketId) continue;
    io.to(ps.socketId).emit('game_state_update', clientViewForPlayer(room.gameState, ps.playerIndex));
  }
}

function applyRoomRatings(room, excludeResigned = false) {
  if (room.resultsApplied) return room.ratingChanges || [];
  const ratingChanges = calculateRatingChanges(room.gameState.players, room.gameState);
  for (let i = 0; i < room.playerSockets.length; i++) {
    if (excludeResigned && room.gameState.resignedPlayers.includes(i)) continue;
    const account = accounts.get(room.playerSockets[i].username);
    if (!account) continue;
    account.rating += ratingChanges[i];
    account.gamesPlayed += 1;
    if (ratingChanges[i] >= 5) account.wins += 1;
  }
  room.resultsApplied = true;
  room.ratingChanges = ratingChanges;
  return ratingChanges;
}

function broadcastProcessedAction(room, actionResult, excludeResignedFromRatings = false) {
  const result = actionResult || {
    type: 'TIMEOUT',
    actingPlayer: room.gameState.currentPlayerIndex,
    payload: {},
  };

  const tileClaimed = room.gameState._tileClaimed;
  if (tileClaimed) {
    delete room.gameState._tileClaimed;
    result.tileClaimed = tileClaimed;
  }

  if (room.gameState.phase === 'GAME_OVER') {
    result.ratingChanges = applyRoomRatings(room, excludeResignedFromRatings);
  }

  for (const playerSocket of room.playerSockets) {
    if (!playerSocket.socketId) continue;
    io.to(playerSocket.socketId).emit(
      'game_state_update',
      clientViewForPlayer(room.gameState, playerSocket.playerIndex),
    );
    io.to(playerSocket.socketId).emit('action_result', result);
  }

  const pendingTileChoice = room.gameState._pendingTileChoice;
  if (pendingTileChoice && room.gameState.phase === 'PLAYING') {
    const activeSocket = room.playerSockets[room.gameState.currentPlayerIndex]?.socketId;
    if (activeSocket) io.to(activeSocket).emit('tile_choice_required', { tileIds: pendingTileChoice });
  }

  // Turn driver (docs/AI_BRIDGE.md §3): no-op unless the seat to move is a bot.
  aiBridge.maybeAct(room);
}

function isPlayerConnected(room, playerIndex) {
  return !!room.playerSockets[playerIndex]?.socketId;
}

function startRoomTurnClock(room, now = Date.now()) {
  startTurnTimeControl(room.gameState, now, !isPlayerConnected(room, room.gameState.currentPlayerIndex));
}

function eliminateTimedOutPlayer(room, now = Date.now()) {
  const timedOutPlayer = room.gameState.currentPlayerIndex;
  processResign(room.gameState, timedOutPlayer);
  replaySafe(() => replayRecorder.onTimeout(room, timedOutPlayer));

  const activeCount = room.gameState.numPlayers - (room.gameState.resignedPlayers?.length || 0);
  if (activeCount < 2) room.gameState.phase = 'GAME_OVER';
  if (room.gameState.phase === 'PLAYING') startRoomTurnClock(room, now);
  room.lastPlayerAction = now;
  broadcastProcessedAction(room, {
    type: 'TIMEOUT',
    actingPlayer: timedOutPlayer,
    payload: { timedOutPlayerIndex: timedOutPlayer },
  }, room.gameState.gameMode === 'INDIVIDUAL');
  replayFinishIfGameOver(room);
  return true;
}

// ── Shared game-action path ──
// The body of the `game_action` socket handler, so bot seats driven by the AI
// bridge go through exactly the same code (timers, replay hook, broadcast).
function applyGameAction(room, playerIndex, action) {
  const now = Date.now();
  const timerStatus = updateTimeControl(room.gameState, now);
  if (timerStatus.expired) {
    eliminateTimedOutPlayer(room, now);
    return { error: 'The active player ran out of time.' };
  }

  const actingPlayerIndex = room.gameState.currentPlayerIndex;
  const previousTurnNumber = room.gameState.turnNumber;
  const result = processAction(room.gameState, playerIndex, action);

  if (result.error) return { error: result.error };

  replaySafe(() => replayRecorder.onActionResult(room, result.result));
  aiBridge.onActionResult(room, result.result);

  consumeTurnTime(room.gameState, actingPlayerIndex, now);
  const turnCompleted = room.gameState.turnNumber !== previousTurnNumber || room.gameState.phase === 'GAME_OVER';
  if (turnCompleted) addTimeIncrement(room.gameState, actingPlayerIndex, now);
  if (room.gameState.phase === 'PLAYING' && room.gameState.turnNumber !== previousTurnNumber) {
    startRoomTurnClock(room, now);
  }
  room.lastPlayerAction = now;
  broadcastProcessedAction(room, result.result);
  replayFinishIfGameOver(room);

  return { ok: true };
}

// The body of the `resign` socket handler, shared with the AI bridge
// (a worker may answer RESIGN, and a stuck bot resigns like a stuck human).
function resignPlayer(room, playerIndex) {
  if (!room || room.gameState.phase !== 'PLAYING') return { error: 'Game is over' };

  const now = Date.now();
  const previousCurrentPlayer = room.gameState.currentPlayerIndex;
  if (playerIndex === previousCurrentPlayer) consumeTurnTime(room.gameState, playerIndex, now);
  processResign(room.gameState, playerIndex);
  replaySafe(() => replayRecorder.onResign(room, playerIndex));

  const activeCount = room.gameState.numPlayers - (room.gameState.resignedPlayers?.length || 0);
  if (activeCount < 2) room.gameState.phase = 'GAME_OVER';
  if (room.gameState.phase === 'PLAYING' && room.gameState.currentPlayerIndex !== previousCurrentPlayer) {
    startRoomTurnClock(room, now);
  }
  room.lastPlayerAction = now;
  broadcastProcessedAction(room, {
    type: 'RESIGN',
    actingPlayer: playerIndex,
    payload: { resignedPlayerIndex: playerIndex },
  }, room.gameState.gameMode === 'INDIVIDUAL');
  replayFinishIfGameOver(room);

  return { ok: true };
}

// ── AI bridge wiring (docs/AI_BRIDGE.md) ──
aiBridge.init({
  getRoom: roomId => gameRooms.get(roomId) || null,
  applyGameAction,
  resignPlayer,
});

const TIMER_POLL_INTERVAL = 250;
setInterval(() => {
  const now = Date.now();
  for (const [, room] of gameRooms) {
    if (!room.gameState.timeControl || room.gameState.phase !== 'PLAYING') continue;
    const timerStatus = updateTimeControl(room.gameState, now);
    if (timerStatus.expired) eliminateTimedOutPlayer(room, now);
  }
}, TIMER_POLL_INTERVAL);

// ── Socket.io ──
io.on('connection', (socket) => {
  console.log(`Connected: ${socket.id}`);
  let currentUsername = null;

  // AI worker events (`ai_worker_register`, `ai_move_response`) — inert for
  // browser clients and when AI_WORKER_SECRET is unset.
  aiBridge.attach(socket);

  socket.on('login', ({ username }, cb) => {
    const account = accounts.get(username);
    if (!account) { cb?.({ error: 'Account not found' }); return; }
    currentUsername = username;
    socketToAccount.set(socket.id, username);
    accountToSocket.set(username, socket.id);
    cb?.({ success: true, account });
  });

  socket.on('enter_lobby', (cb) => {
    const username = currentUsername || socketToAccount.get(socket.id);
    if (!username) { cb?.({ action: 'error', error: 'Login required' }); return; }

    // Check for active game → auto-reconnect
    const activeGame = findActiveGame(username);
    if (activeGame) {
      const { roomId, room, playerSocket } = activeGame;
      const now = Date.now();
      const wasDisconnected = !playerSocket.socketId;
      // Update socket id
      playerSocket.socketId = socket.id;
      socket.join(roomId);
      socketToAccount.set(socket.id, username);
      accountToSocket.set(username, socket.id);

      const isStillPlaying = !room.gameState.resignedPlayers.includes(playerSocket.playerIndex);
      if (wasDisconnected && isStillPlaying && room.gameState.timeControl) {
        addTimeIncrement(room.gameState, playerSocket.playerIndex, now);
        if (room.gameState.currentPlayerIndex === playerSocket.playerIndex) {
          startRoomTurnClock(room, now);
        }
      }

      socket.emit('game_start', {
        roomId,
        playerIndex: playerSocket.playerIndex,
        gameState: clientViewForPlayer(room.gameState, playerSocket.playerIndex),
        aiSeats: room.playerSockets.reduce((seats, ps, i) => (ps.isAI ? [...seats, i] : seats), []),
        isReconnect: true,
      });
      if (room.gameState.currentPlayerIndex === playerSocket.playerIndex && room.gameState._pendingTileChoice) {
        socket.emit('tile_choice_required', { tileIds: room.gameState._pendingTileChoice });
      }
      socket.to(roomId).emit('player_reconnected', { username });
      broadcastGameState(room);
      room.lastPlayerAction = now;
      cb?.({ action: 'rejoin_game' });
      return;
    }

    // Remove any stale lobby entry for this username while preserving a team
    // seat that is keyed by username.
    lobbyQueue = lobbyQueue.filter(p => p.username !== username);
    if (lobbySettings.teamMode && lobbyQueue.length >= requiredTeamPlayerCount()) {
      cb?.({ action: 'lobby_full', error: 'This team lobby is full.' });
      return;
    }
    const account = accounts.get(username);
    lobbyQueue.push({ socketId: socket.id, username, ready: false, wantsFirst: false, avatarSeed: account?.avatarSeed ?? 0 });

    cb?.({ action: 'lobby', lobbyState: lobbyState() });
    broadcastLobby();
  });

  socket.on('leave_lobby', () => {
    removeFromLobby(socket.id);
  });

  socket.on('lobby_ready', () => {
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (player) {
      if (isBotEntry(player)) return; // bots are always ready
      if (lobbySettings.teamMode && !isPlayerSeated(player.username)) return;
      player.ready = !player.ready;
      broadcastLobby();
      checkAutoStart();
    }
  });

  socket.on('set_go_first', (data = {}, cb) => {
    const { enabled } = data;
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (!player) { cb?.({ error: 'Not in lobby' }); return; }
    if (typeof enabled !== 'boolean') { cb?.({ error: 'Invalid first-player preference' }); return; }
    if (lobbySettings.teamMode && lobbySettings.teamFormat === 'ONE_V_TWO') {
      cb?.({ error: 'The solo player always goes first in 1v2 mode' });
      return;
    }
    player.wantsFirst = enabled;
    broadcastLobby();
    cb?.({ ok: true });
  });

  // Any lobby member may switch the shared team configuration.
  socket.on('set_team_mode', (data = {}, cb) => {
    const { enabled } = data;
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (!player) { cb?.({ error: 'Not in lobby' }); return; }
    if (typeof enabled !== 'boolean') { cb?.({ error: 'Invalid team mode setting' }); return; }
    if (enabled && lobbyQueue.length !== 3 && lobbyQueue.length !== 4) {
      cb?.({ error: 'Team mode requires exactly three or four players' });
      return;
    }
    const unlimitedTime = lobbySettings.unlimitedTime;
    lobbySettings = enabled ? {
      teamMode: true,
      teamFormat: lobbyQueue.length === 3 ? 'ONE_V_TWO' : 'TWO_V_TWO',
      teamLayout: 'ADJACENT',
      teamSeats: [[null, null], [null, null]],
      unlimitedTime,
    } : { ...createDefaultLobbySettings(), unlimitedTime };
    lobbyQueue.forEach(p => {
      p.ready = isBotEntry(p);
      if (lobbySettings.teamFormat === 'ONE_V_TWO') p.wantsFirst = false;
    });
    broadcastLobby();
    cb?.({ ok: true });
  });

  socket.on('set_unlimited_time', (data = {}, cb) => {
    const { enabled } = data;
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (!player) { cb?.({ error: 'Not in lobby' }); return; }
    if (typeof enabled !== 'boolean') { cb?.({ error: 'Invalid time setting' }); return; }
    if (lobbyQueue.length !== 3 && lobbyQueue.length !== 4) {
      cb?.({ error: 'The time setting is available in three- and four-player lobbies' });
      return;
    }
    lobbySettings.unlimitedTime = enabled;
    resetLobbyReady();
    broadcastLobby();
    cb?.({ ok: true });
  });

  socket.on('set_team_layout', (data = {}, cb) => {
    const { layout } = data;
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (!player || !lobbySettings.teamMode || lobbySettings.teamFormat !== 'TWO_V_TWO') {
      cb?.({ error: 'The seat layout toggle is only available in 2v2 mode' });
      return;
    }
    if (layout !== 'ADJACENT' && layout !== 'OPPOSITE') { cb?.({ error: 'Invalid team layout' }); return; }
    lobbySettings.teamLayout = layout;
    resetLobbyReady();
    broadcastLobby();
    cb?.({ ok: true });
  });

  // ── AI lobby members (docs/AI_BRIDGE.md §3) ──
  socket.on('lobby_add_ai', (data, cb) => {
    const ack = typeof data === 'function' ? data : cb;
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (!player) { ack?.({ error: 'Not in lobby' }); return; }
    const result = addBotToLobby();
    if (result.error) { ack?.({ error: result.error }); return; }
    broadcastLobby();
    ack?.({ ok: true, username: result.username });
    checkAutoStart();
  });

  socket.on('lobby_remove_ai', (data, cb) => {
    const ack = typeof data === 'function' ? data : cb;
    const payload = typeof data === 'function' || !data ? {} : data;
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (!player) { ack?.({ error: 'Not in lobby' }); return; }
    const result = removeBotFromLobby(payload.username);
    if (result.error) { ack?.({ error: result.error }); return; }
    broadcastLobby();
    ack?.({ ok: true });
  });

  socket.on('select_team_seat', (data = {}, cb) => {
    const { teamId, seatIndex, forUsername } = data;
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (!player || !lobbySettings.teamMode) { cb?.({ error: 'Team mode is not active' }); return; }
    if (![0, 1].includes(teamId) || ![0, 1].includes(seatIndex)) { cb?.({ error: 'Invalid team seat' }); return; }
    if (lobbySettings.teamFormat === 'ONE_V_TWO' && teamId === 0 && seatIndex !== 0) {
      cb?.({ error: 'The solo side has only one seat' });
      return;
    }

    // Bots cannot click a seat themselves: any lobby member may seat one for
    // them. `forUsername` is only ever accepted for a bot in this lobby.
    const target = forUsername === undefined || forUsername === null
      ? player
      : lobbyQueue.find(p => p.username === forUsername && isBotEntry(p));
    if (!target) { cb?.({ error: 'Only AI players can be seated by someone else' }); return; }

    const targetOccupant = lobbySettings.teamSeats[teamId][seatIndex];
    if (targetOccupant && targetOccupant !== target.username) {
      cb?.({ error: 'That seat is already occupied' });
      return;
    }

    const wasOwnSeat = targetOccupant === target.username;
    lobbySettings.teamSeats = lobbySettings.teamSeats.map(team =>
      team.map(username => username === target.username ? null : username));
    if (!wasOwnSeat) lobbySettings.teamSeats[teamId][seatIndex] = target.username;
    target.ready = isBotEntry(target);
    broadcastLobby();
    cb?.({ ok: true });
    // Seating a bot cannot be followed by the bot readying up.
    if (isBotEntry(target)) checkAutoStart();
  });

  // ── Game actions — server processes all logic ──
  socket.on('game_action', (data = {}, ack) => {
    const { roomId, action } = data;
    const room = gameRooms.get(roomId);
    if (!room) { ack?.({ error: 'Room not found' }); return; }

    const ps = room.playerSockets.find(p => p.socketId === socket.id);
    if (!ps) { ack?.({ error: 'Not in room' }); return; }

    ack?.(applyGameAction(room, ps.playerIndex, action));
  });

  // ── Heartbeat — keeps room alive ──
  socket.on('ping', () => {
    socket.emit('pong');
    // Update lastPlayerAction for any room this player is in
    const username = currentUsername || socketToAccount.get(socket.id);
    if (username) {
      for (const [, room] of gameRooms) {
        if (room.playerSockets.some(p => p.username === username)) {
          room.lastPlayerAction = Date.now();
        }
      }
    }
  });

  // ── Resign ──
  socket.on('resign', (data = {}) => {
    const { roomId } = data;
    const room = gameRooms.get(roomId);
    if (!room || room.gameState.phase !== 'PLAYING') return;
    const ps = room.playerSockets.find(p => p.socketId === socket.id);
    if (!ps) return;

    resignPlayer(room, ps.playerIndex);

    // Remove resigned player from room
    socket.leave(roomId);
  });

  // ── Return to lobby ──
  socket.on('return_to_lobby', ({ roomId }, ack) => {
    socket.leave(roomId);
    ack?.({ ok: true });
  });

  // ── Quit room (everyone out) ──
  socket.on('quit_room', ({ roomId }) => {
    const room = gameRooms.get(roomId);
    if (!room) return;
    for (const p of room.playerSockets) {
      const s = io.sockets.sockets.get(p.socketId);
      if (s) { s.emit('game_quit'); s.leave(roomId); }
    }
    replaySafe(() => replayRecorder.discard(room));
    aiBridge.clearRoom(roomId);
    gameRooms.delete(roomId);
  });

  socket.on('disconnect', () => {
    const username = currentUsername || socketToAccount.get(socket.id);
    console.log(`Disconnected: ${socket.id} (${username || 'unknown'})`);
    removeFromLobby(socket.id);

    // Mark player as disconnected in game rooms but don't remove them
    if (username) {
      for (const [roomId, room] of gameRooms) {
        const ps = room.playerSockets.find(p => p.username === username && p.socketId === socket.id);
        if (ps) {
          const now = Date.now();
          const timerStatus = updateTimeControl(room.gameState, now);
          if (timerStatus.expired) {
            eliminateTimedOutPlayer(room, now);
          } else {
            pauseTurnTimeControl(room.gameState, ps.playerIndex, now);
          }
          ps.socketId = null;
          pauseTurnTimeControl(room.gameState, ps.playerIndex, now);
          socket.to(roomId).emit('player_disconnected', { username });
          broadcastGameState(room);
        }
      }
    }

    socketToAccount.delete(socket.id);
    if (username) safeDeleteAccountSocket(username, socket.id);
    currentUsername = null;
  });
});

const PORT = process.env.PORT || 10000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
