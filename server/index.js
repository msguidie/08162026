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
  if (resetReady) lobbyQueue.forEach(player => { player.ready = false; });
}

function lobbyState() {
  return {
    players: lobbyQueue.map(p => ({
      socketId: p.socketId,
      username: p.username,
      ready: p.ready,
      avatarSeed: p.avatarSeed,
    })),
    teamMode: lobbySettings.teamMode,
    teamFormat: lobbySettings.teamFormat,
    teamLayout: lobbySettings.teamLayout,
    teamSeats: lobbySettings.teamSeats.map(team => [...team]),
    unlimitedTime: lobbySettings.unlimitedTime,
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
      gameRooms.delete(id);
      continue;
    }
    if (now - room.lastPlayerAction > INACTIVITY_TIMEOUT) {
      console.log(`Auto-abandoning inactive game ${id}`);
      for (const p of room.playerSockets) {
        const sock = io.sockets.sockets.get(p.socketId);
        if (sock) { sock.emit('game_abandoned', { reason: 'Game abandoned due to inactivity.' }); sock.leave(id); }
      }
      gameRooms.delete(id);
      continue;
    }
    if (now - room.created > ROOM_TTL) gameRooms.delete(id);
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
    if (lobbySettings.teamMode && lobbyQueue.length !== requiredTeamPlayerCount()) resetLobbySettings(true, true);
    broadcastLobby();
  }
}

// Auto-start when all >= 2 are ready
function checkAutoStart() {
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
      };
    });
  } else {
    playerInfos = lobbyQueue.map(p => ({
      username: p.username,
      socketId: p.socketId,
      avatarSeed: p.avatarSeed,
    }));
  }

  // Create server-authoritative game state
  const gameState = createInitialGameState(playerInfos, gameOptions);

  const room = {
    id: roomId,
    playerSockets: playerInfos.map((p, i) => ({ socketId: p.socketId, username: p.username, playerIndex: i })),
    gameState,
    lastPlayerAction: Date.now(),
    created: Date.now(),
    resultsApplied: false,
    ratingChanges: null,
  };
  gameRooms.set(roomId, room);

  // Notify each player
  playerInfos.forEach((p, idx) => {
    const s = io.sockets.sockets.get(p.socketId);
    if (s) s.join(roomId);
    io.to(p.socketId).emit('game_start', {
      roomId,
      playerIndex: idx,
      gameState: clientViewForPlayer(gameState, idx),
    });
  });

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

  const activeCount = room.gameState.numPlayers - (room.gameState.resignedPlayers?.length || 0);
  if (activeCount < 2) room.gameState.phase = 'GAME_OVER';
  if (room.gameState.phase === 'PLAYING') startRoomTurnClock(room, now);
  room.lastPlayerAction = now;
  broadcastProcessedAction(room, {
    type: 'TIMEOUT',
    actingPlayer: timedOutPlayer,
    payload: { timedOutPlayerIndex: timedOutPlayer },
  }, room.gameState.gameMode === 'INDIVIDUAL');
  return true;
}

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
    lobbyQueue.push({ socketId: socket.id, username, ready: false, avatarSeed: account?.avatarSeed ?? 0 });

    cb?.({ action: 'lobby', lobbyState: lobbyState() });
    broadcastLobby();
  });

  socket.on('leave_lobby', () => {
    removeFromLobby(socket.id);
  });

  socket.on('lobby_ready', () => {
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (player) {
      if (lobbySettings.teamMode && !isPlayerSeated(player.username)) return;
      player.ready = !player.ready;
      broadcastLobby();
      checkAutoStart();
    }
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
    lobbyQueue.forEach(p => { p.ready = false; });
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
    lobbyQueue.forEach(p => { p.ready = false; });
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
    lobbyQueue.forEach(p => { p.ready = false; });
    broadcastLobby();
    cb?.({ ok: true });
  });

  socket.on('select_team_seat', (data = {}, cb) => {
    const { teamId, seatIndex } = data;
    const player = lobbyQueue.find(p => p.socketId === socket.id);
    if (!player || !lobbySettings.teamMode) { cb?.({ error: 'Team mode is not active' }); return; }
    if (![0, 1].includes(teamId) || ![0, 1].includes(seatIndex)) { cb?.({ error: 'Invalid team seat' }); return; }
    if (lobbySettings.teamFormat === 'ONE_V_TWO' && teamId === 0 && seatIndex !== 0) {
      cb?.({ error: 'The solo side has only one seat' });
      return;
    }

    const targetOccupant = lobbySettings.teamSeats[teamId][seatIndex];
    if (targetOccupant && targetOccupant !== player.username) {
      cb?.({ error: 'That seat is already occupied' });
      return;
    }

    const wasOwnSeat = targetOccupant === player.username;
    lobbySettings.teamSeats = lobbySettings.teamSeats.map(team =>
      team.map(username => username === player.username ? null : username));
    if (!wasOwnSeat) lobbySettings.teamSeats[teamId][seatIndex] = player.username;
    player.ready = false;
    broadcastLobby();
    cb?.({ ok: true });
  });

  // ── Game actions — server processes all logic ──
  socket.on('game_action', (data = {}, ack) => {
    const { roomId, action } = data;
    const room = gameRooms.get(roomId);
    if (!room) { ack?.({ error: 'Room not found' }); return; }

    const ps = room.playerSockets.find(p => p.socketId === socket.id);
    if (!ps) { ack?.({ error: 'Not in room' }); return; }

    const now = Date.now();
    const timerStatus = updateTimeControl(room.gameState, now);
    if (timerStatus.expired) {
      eliminateTimedOutPlayer(room, now);
      ack?.({ error: 'The active player ran out of time.' });
      return;
    }

    const actingPlayerIndex = room.gameState.currentPlayerIndex;
    const previousTurnNumber = room.gameState.turnNumber;
    const result = processAction(room.gameState, ps.playerIndex, action);

    if (result.error) {
      ack?.({ error: result.error });
      return;
    }

    consumeTurnTime(room.gameState, actingPlayerIndex, now);
    const turnCompleted = room.gameState.turnNumber !== previousTurnNumber || room.gameState.phase === 'GAME_OVER';
    if (turnCompleted) addTimeIncrement(room.gameState, actingPlayerIndex, now);
    if (room.gameState.phase === 'PLAYING' && room.gameState.turnNumber !== previousTurnNumber) {
      startRoomTurnClock(room, now);
    }
    room.lastPlayerAction = now;
    broadcastProcessedAction(room, result.result);

    ack?.({ ok: true });
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

    const now = Date.now();
    const previousCurrentPlayer = room.gameState.currentPlayerIndex;
    if (ps.playerIndex === previousCurrentPlayer) consumeTurnTime(room.gameState, ps.playerIndex, now);
    processResign(room.gameState, ps.playerIndex);

    const activeCount = room.gameState.numPlayers - (room.gameState.resignedPlayers?.length || 0);
    if (activeCount < 2) room.gameState.phase = 'GAME_OVER';
    if (room.gameState.phase === 'PLAYING' && room.gameState.currentPlayerIndex !== previousCurrentPlayer) {
      startRoomTurnClock(room, now);
    }
    room.lastPlayerAction = now;
    broadcastProcessedAction(room, {
      type: 'RESIGN',
      actingPlayer: ps.playerIndex,
      payload: { resignedPlayerIndex: ps.playerIndex },
    }, room.gameState.gameMode === 'INDIVIDUAL');

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
