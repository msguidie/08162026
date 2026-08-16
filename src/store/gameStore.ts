import { create } from 'zustand';
import { io, Socket } from 'socket.io-client';
import type {
  AppPhase, ConnectionStatus, GameState, LobbyPlayer,
  Account, ActionMode, ActionResult, BonusTile, LobbyState,
  LobbyTeamFormat, TeamId, TeamLayout, TeamSeats,
} from '../types';
import { SERVER_URL, CONNECTION_CONFIG } from '../constants';

export interface Toast {
  id: string;
  message: string;
  type: 'info' | 'warn' | 'error';
}

interface GameStore {
  appPhase: AppPhase;
  connectionStatus: ConnectionStatus;
  socket: Socket | null;
  myAccount: Account | null;
  roomId: string | null;
  playerIndex: number;

  lobbyPlayers: LobbyPlayer[];
  lobbyTeamMode: boolean;
  lobbyTeamFormat: LobbyTeamFormat;
  lobbyTeamLayout: TeamLayout;
  lobbyTeamSeats: TeamSeats;
  gameState: GameState | null;
  actionMode: ActionMode;
  pendingTileChoice: number[] | null; // tile IDs to choose from

  lastActionResult: ActionResult | null;
  reconnectAttempts: number;
  disconnectedPlayers: Set<string>;
  toasts: Toast[];

  connectToServer: () => Promise<void>;
  login: (username: string) => Promise<boolean>;
  enterLobby: () => void;
  leaveLobby: () => void;
  toggleReady: () => void;
  toggleTeamMode: () => void;
  toggleTeamLayout: () => void;
  selectTeamSeat: (teamId: TeamId, seatIndex: 0 | 1) => void;
  setActionMode: (mode: ActionMode) => void;
  sendAction: (action: Record<string, unknown>) => void;
  chooseBonusTile: (tileId: number) => void;
  returnToLobby: () => void;
  quitRoom: () => void;
  resign: () => void;
  disconnect: () => void;
  addToast: (message: string, type?: Toast['type']) => void;
  removeToast: (id: string) => void;
}

interface LobbyResponse {
  action: string;
  lobbyState?: LobbyState;
  error?: string;
}

const EMPTY_TEAM_SEATS: TeamSeats = [[null, null], [null, null]];

const useGameStore = create<GameStore>((set, get) => {
  let heartbeatInterval: ReturnType<typeof setInterval> | null = null;

  function applyLobbyState(lobbyState?: LobbyState) {
    if (!lobbyState) return;
    set({
      lobbyPlayers: lobbyState.players,
      lobbyTeamMode: lobbyState.teamMode,
      lobbyTeamFormat: lobbyState.teamFormat,
      lobbyTeamLayout: lobbyState.teamLayout,
      lobbyTeamSeats: lobbyState.teamSeats,
    });
  }

  function handleLobbyResponse(result: LobbyResponse) {
    if (result?.action === 'rejoin_game') return;
    if (result?.action === 'lobby_full' || result?.action === 'error') {
      showToast(result.error || 'Unable to enter the lobby', 'warn');
      set({ appPhase: 'LOGIN', connectionStatus: 'connected' });
      return;
    }
    applyLobbyState(result?.lobbyState);
    set({ appPhase: 'WAITING_ROOM', connectionStatus: 'in_lobby' });
  }

  function showToast(message: string, type: Toast['type'] = 'info') {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 5)}`;
    set(s => ({ toasts: [...s.toasts, { id, message, type }] }));
    setTimeout(() => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })), 4000);
  }

  function setupSocket(socket: Socket) {
    socket.on('lobby_update', (lobbyState: LobbyState) => {
      applyLobbyState(lobbyState);
    });

    socket.on('game_start', (data: {
      roomId: string;
      playerIndex: number;
      gameState: GameState;
      isReconnect?: boolean;
    }) => {
      const restoredActionMode: ActionMode =
        data.gameState.currentPlayerIndex === data.playerIndex && data.gameState.turnAction?.type === 'RESERVE'
          ? 'RESERVE'
          : data.gameState.currentPlayerIndex === data.playerIndex && data.gameState.turnAction?.type === 'TAKE_GEMS'
            ? 'TAKE_GEMS'
            : null;
      set({
        appPhase: 'GAME',
        roomId: data.roomId,
        playerIndex: data.playerIndex,
        gameState: data.gameState,
        actionMode: restoredActionMode,
        pendingTileChoice: null,
        disconnectedPlayers: new Set(),
        connectionStatus: 'connected',
        reconnectAttempts: 0,
      });
    });

    socket.on('game_state_update', (gameState: GameState) => {
      set(s => {
        let nextActionMode = s.actionMode;
        const isMyActiveTurn = gameState.phase === 'PLAYING' && gameState.currentPlayerIndex === s.playerIndex;
        if (!isMyActiveTurn) nextActionMode = null;
        else if (gameState.turnAction?.type === 'RESERVE') nextActionMode = 'RESERVE';
        else if (gameState.turnAction?.type === 'TAKE_GEMS') nextActionMode = 'TAKE_GEMS';
        return { gameState, actionMode: nextActionMode };
      });
    });

    socket.on('action_result', (result: ActionResult) => {
      set({ lastActionResult: result });
      if (result.payload?.forced) {
        const username = get().gameState?.players[result.actingPlayer]?.username || 'A player';
        showToast(`${username} timed out; the system completed the turn.`, 'warn');
      }
      // If the action was completed (turn changed), reset action mode
      if (result.type === 'BUY_CARD' || result.type === 'RESERVE_CARD' ||
          result.type === 'RESERVE_FROM_DECK' || result.type === 'RESIGN' ||
          result.type === 'CHOOSE_TILE' || result.type === 'AUTO_PASS') {
        set({ actionMode: null, pendingTileChoice: null });
      }
      // If gems were fully taken, reset mode
      if (result.type === 'SELECT_GEM' && result.payload?.completed !== false) {
        set({ actionMode: null });
      }
    });

    socket.on('tile_choice_required', ({ tileIds }: { tileIds: number[] }) => {
      set({ pendingTileChoice: tileIds });
    });

    socket.on('player_disconnected', ({ username }: { username: string }) => {
      set(s => ({ disconnectedPlayers: new Set([...s.disconnectedPlayers, username]) }));
      showToast(`${username} disconnected`, 'warn');
    });

    socket.on('player_reconnected', ({ username }: { username: string }) => {
      set(s => {
        const dp = new Set(s.disconnectedPlayers);
        dp.delete(username);
        return { disconnectedPlayers: dp };
      });
      showToast(`${username} reconnected`, 'info');
    });

    socket.on('game_quit', () => {
      set({ gameState: null, actionMode: null, roomId: null, pendingTileChoice: null, disconnectedPlayers: new Set() });
      const sock = get().socket;
      if (sock) {
        sock.emit('enter_lobby', (result: LobbyResponse) => handleLobbyResponse(result));
      }
    });

    socket.on('game_abandoned', (data?: { reason?: string }) => {
      showToast(data?.reason || 'Game was abandoned.', 'warn');
      set({ gameState: null, actionMode: null, roomId: null, pendingTileChoice: null, disconnectedPlayers: new Set() });
      const sock = get().socket;
      if (sock) {
        sock.emit('enter_lobby', (result: LobbyResponse) => handleLobbyResponse(result));
      }
    });

    socket.on('disconnect', (reason: string) => {
      if (reason === 'io client disconnect') return;
      if (get().appPhase === 'GAME') {
        set({ connectionStatus: 'reconnecting' });
        attemptReconnect();
      } else {
        set({ connectionStatus: 'disconnected' });
      }
    });

    socket.on('pong', () => {});
  }

  function attemptReconnect() {
    const currentAttempts = get().reconnectAttempts;
    const myAccount = get().myAccount;
    if (currentAttempts >= CONNECTION_CONFIG.MAX_RECONNECT_ATTEMPTS) {
      set({ connectionStatus: 'error' });
      showToast('Reconnection failed. Please refresh.', 'error');
      return;
    }
    const delay = Math.min(
      CONNECTION_CONFIG.RECONNECT_DELAY_BASE * Math.pow(2, currentAttempts),
      CONNECTION_CONFIG.RECONNECT_DELAY_MAX,
    );
    setTimeout(async () => {
      set({ reconnectAttempts: currentAttempts + 1 });
      try {
        await get().connectToServer();
        const socket = get().socket;
        if (socket && myAccount) {
          socket.emit('login', { username: myAccount.username }, () => {});
          // enter_lobby will auto-reconnect to active game
          socket.emit('enter_lobby', (result: LobbyResponse) => {
            if (result?.action === 'rejoin_game') {
              set({ connectionStatus: 'connected', reconnectAttempts: 0 });
              showToast('Reconnected!', 'info');
            } else {
              set({ connectionStatus: 'error', appPhase: 'LOGIN' });
              showToast('Game no longer available.', 'warn');
            }
          });
        }
      } catch {
        attemptReconnect();
      }
    }, delay);
  }

  return {
    appPhase: 'LOGIN',
    connectionStatus: 'idle',
    socket: null,
    myAccount: null,
    roomId: null,
    playerIndex: -1,
    lobbyPlayers: [],
    gameState: null,
    actionMode: null,
    pendingTileChoice: null,
    lastActionResult: null,
    reconnectAttempts: 0,
    disconnectedPlayers: new Set(),
    toasts: [],
    lobbyTeamMode: false,
    lobbyTeamFormat: null,
    lobbyTeamLayout: 'ADJACENT',
    lobbyTeamSeats: EMPTY_TEAM_SEATS,

    connectToServer: async () => {
      const existingSocket = get().socket;
      if (existingSocket) { existingSocket.removeAllListeners(); existingSocket.disconnect(); }
      if (heartbeatInterval) clearInterval(heartbeatInterval);
      set({ connectionStatus: 'connecting' });
      return new Promise<void>((resolve, reject) => {
        const socket = io(SERVER_URL, {
          transports: ['websocket', 'polling'],
          timeout: CONNECTION_CONFIG.CONNECTION_TIMEOUT,
          reconnection: false,
          forceNew: true,
        });
        const connectTimeout = setTimeout(() => {
          socket.disconnect(); set({ connectionStatus: 'error' }); reject(new Error('Connection timeout'));
        }, CONNECTION_CONFIG.CONNECTION_TIMEOUT);
        socket.on('connect', () => {
          clearTimeout(connectTimeout);
          set({ socket, connectionStatus: 'connected' });
          setupSocket(socket);
          heartbeatInterval = setInterval(() => socket.emit('ping'), CONNECTION_CONFIG.HEARTBEAT_INTERVAL);
          resolve();
        });
        socket.on('connect_error', (err) => { clearTimeout(connectTimeout); set({ connectionStatus: 'error' }); reject(err); });
      });
    },

    login: (username: string): Promise<boolean> => {
      return new Promise((resolve) => {
        const { socket } = get();
        if (!socket) { resolve(false); return; }
        socket.emit('login', { username }, (result: { success: boolean; account: Account }) => {
          if (result?.success) {
            set({ myAccount: result.account });
            resolve(true);
          } else { resolve(false); }
        });
        setTimeout(() => resolve(false), 5000);
      });
    },

    enterLobby: () => {
      const { socket } = get();
      if (!socket) return;
      set({ connectionStatus: 'entering_lobby' });
      socket.emit('enter_lobby', (result: LobbyResponse) => {
        handleLobbyResponse(result);
      });
      setTimeout(() => {
        if (get().connectionStatus === 'entering_lobby') {
          set({ appPhase: 'WAITING_ROOM', connectionStatus: 'in_lobby' });
        }
      }, 3000);
    },

    leaveLobby: () => {
      get().socket?.emit('leave_lobby');
      set({
        appPhase: 'LOGIN', lobbyPlayers: [], lobbyTeamMode: false, lobbyTeamFormat: null,
        lobbyTeamLayout: 'ADJACENT', lobbyTeamSeats: EMPTY_TEAM_SEATS,
      });
    },

    toggleReady: () => {
      get().socket?.emit('lobby_ready');
    },

    toggleTeamMode: () => {
      const { socket } = get();
      if (!socket) return;
      socket.emit('set_team_mode', { enabled: !get().lobbyTeamMode }, (res: { error?: string }) => {
        if (res?.error) showToast(res.error, 'warn');
      });
    },

    toggleTeamLayout: () => {
      const { socket, lobbyTeamLayout } = get();
      if (!socket) return;
      const layout: TeamLayout = lobbyTeamLayout === 'ADJACENT' ? 'OPPOSITE' : 'ADJACENT';
      socket.emit('set_team_layout', { layout }, (res: { error?: string }) => {
        if (res?.error) showToast(res.error, 'warn');
      });
    },

    selectTeamSeat: (teamId, seatIndex) => {
      const { socket } = get();
      if (!socket) return;
      socket.emit('select_team_seat', { teamId, seatIndex }, (res: { error?: string }) => {
        if (res?.error) showToast(res.error, 'warn');
      });
    },

    setActionMode: (mode: ActionMode) => {
      const { gameState, playerIndex, actionMode: currentMode, socket, roomId } = get();
      if (!gameState || gameState.currentPlayerIndex !== playerIndex || gameState.phase !== 'PLAYING') return;
      // Cannot cancel reserve mode (gold already taken)
      if (currentMode === 'RESERVE' || gameState.turnAction?.type === 'RESERVE') {
        set({ actionMode: 'RESERVE' });
        return;
      }
      if (gameState.turnAction?.type === 'TAKE_GEMS' && mode !== 'TAKE_GEMS' && mode !== null) {
        showToast('Cancel the current gem selection first', 'warn');
        return;
      }

      if (mode === null || mode === currentMode) {
        // Cancel current mode
        if (currentMode === 'TAKE_GEMS' && socket && roomId) {
          socket.emit('game_action', { roomId, action: { type: 'CANCEL_GEMS' } });
        }
        set({ actionMode: null });
        return;
      }

      if (mode === 'RESERVE') {
        // Send ENTER_RESERVE action to server (takes gold)
        if (socket && roomId) {
          socket.emit('game_action', { roomId, action: { type: 'ENTER_RESERVE' } }, (res: { error?: string }) => {
            if (res?.error) { showToast(res.error, 'error'); return; }
            set({ actionMode: 'RESERVE' });
          });
        }
      } else {
        set({ actionMode: mode });
      }
    },

    sendAction: (action: Record<string, unknown>) => {
      const { socket, roomId } = get();
      if (!socket || !roomId) return;
      socket.emit('game_action', { roomId, action }, (res: { error?: string }) => {
        if (res?.error) showToast(res.error, 'error');
      });
    },

    chooseBonusTile: (tileId: number) => {
      const { socket, roomId } = get();
      if (!socket || !roomId) return;
      socket.emit('game_action', { roomId, action: { type: 'CHOOSE_TILE', tileId } }, (res: { error?: string }) => {
        if (res?.error) showToast(res.error, 'error');
      });
      set({ pendingTileChoice: null });
    },

    returnToLobby: () => {
      const { socket, roomId } = get();
      set({ gameState: null, actionMode: null, roomId: null, pendingTileChoice: null, disconnectedPlayers: new Set() });
      if (socket && roomId) {
        socket.emit('return_to_lobby', { roomId }, () => {
          if (socket.connected) {
            socket.emit('enter_lobby', (result: LobbyResponse) => handleLobbyResponse(result));
          } else {
            set({ appPhase: 'LOGIN' });
          }
        });
      } else if (socket) {
        socket.emit('enter_lobby', (result: LobbyResponse) => handleLobbyResponse(result));
      } else {
        set({ appPhase: 'LOGIN' });
      }
    },

    quitRoom: () => {
      const { socket, roomId } = get();
      if (socket && roomId) socket.emit('quit_room', { roomId });
      set({ gameState: null, actionMode: null, roomId: null, pendingTileChoice: null, disconnectedPlayers: new Set() });
      if (socket) {
        socket.emit('enter_lobby', (result: LobbyResponse) => handleLobbyResponse(result));
      } else {
        set({ appPhase: 'LOGIN' });
      }
    },

    resign: () => {
      const { socket, roomId } = get();
      if (!socket || !roomId) return;
      socket.emit('resign', { roomId });
      set({ gameState: null, actionMode: null, roomId: null, pendingTileChoice: null, disconnectedPlayers: new Set() });
      socket.emit('enter_lobby', (result: LobbyResponse) => handleLobbyResponse(result));
    },

    disconnect: () => {
      const { socket } = get();
      if (socket) socket.disconnect();
      if (heartbeatInterval) clearInterval(heartbeatInterval);
      set({ socket: null, connectionStatus: 'idle', appPhase: 'LOGIN', gameState: null, disconnectedPlayers: new Set(), toasts: [] });
    },

    addToast: (message, type = 'info') => showToast(message, type),
    removeToast: (id) => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })),
  };
});

export default useGameStore;
