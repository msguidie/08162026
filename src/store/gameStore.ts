import { create } from 'zustand';
import { io, Socket } from 'socket.io-client';
import type {
  AppPhase, ConnectionStatus, GameState, LobbyPlayer,
  Account, ActionMode, ActionResult, BonusTile, LobbyState,
  LobbyTeamFormat, TeamId, TeamLayout, TeamSeats,
  ReplayData, ReplayIndexEntry,
} from '../types';
import { SERVER_URL, CONNECTION_CONFIG } from '../constants';
import { fetchReplay, fetchReplayList } from '../replay/replayApi';

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
  lobbyUnlimitedTime: boolean;
  /** AI is enabled on this server and a worker is connected (docs/AI_BRIDGE.md §3). */
  lobbyAiAvailable: boolean;
  gameState: GameState | null;
  actionMode: ActionMode;
  pendingTileChoice: number[] | null; // tile IDs to choose from

  lastActionResult: ActionResult | null;
  reconnectAttempts: number;
  disconnectedPlayers: Set<string>;
  toasts: Toast[];

  // ── Replays (additive; see docs/REPLAY_FORMAT.md) ──
  replayList: ReplayIndexEntry[];
  replayListLoading: boolean;
  replayListError: string | null;
  currentReplay: ReplayData | null;
  replayLoading: boolean;
  replayError: string | null;

  connectToServer: () => Promise<void>;
  login: (username: string) => Promise<boolean>;
  enterLobby: () => void;
  leaveLobby: () => void;
  toggleReady: () => void;
  toggleGoFirst: () => void;
  toggleTeamMode: () => void;
  toggleTeamLayout: () => void;
  toggleUnlimitedTime: () => void;
  selectTeamSeat: (teamId: TeamId, seatIndex: 0 | 1) => void;
  selectTeamSeatFor: (teamId: TeamId, seatIndex: 0 | 1, username: string) => void;
  addAI: () => void;
  removeAI: (username: string) => void;
  setActionMode: (mode: ActionMode) => void;
  sendAction: (action: Record<string, unknown>, onComplete?: (success: boolean) => void) => void;
  chooseBonusTile: (tileId: number) => void;
  returnToLobby: () => void;
  quitRoom: () => void;
  resign: () => void;
  disconnect: () => void;
  addToast: (message: string, type?: Toast['type']) => void;
  removeToast: (id: string) => void;

  openReplayBrowser: () => void;
  refreshReplayList: () => Promise<void>;
  openReplay: (id: string) => Promise<void>;
  closeReplayViewer: () => void;
  closeReplayBrowser: () => void;
}

interface LobbyResponse {
  action: string;
  lobbyState?: LobbyState;
  error?: string;
}

const EMPTY_TEAM_SEATS: TeamSeats = [[null, null], [null, null]];

const useGameStore = create<GameStore>((set, get) => {
  let heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  // Tokens so a slow replay response cannot overwrite a newer one.
  let replayListRequest = 0;
  let replayRequest = 0;

  function applyLobbyState(lobbyState?: LobbyState) {
    if (!lobbyState) return;
    set({
      lobbyPlayers: lobbyState.players,
      lobbyTeamMode: lobbyState.teamMode,
      lobbyTeamFormat: lobbyState.teamFormat,
      lobbyTeamLayout: lobbyState.teamLayout,
      lobbyTeamSeats: lobbyState.teamSeats,
      lobbyUnlimitedTime: lobbyState.unlimitedTime,
      lobbyAiAvailable: lobbyState.aiAvailable === true,
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
      // If the action was completed (turn changed), reset action mode
      if (result.type === 'BUY_CARD' || result.type === 'RESERVE_CARD' ||
          result.type === 'RESERVE_FROM_DECK' || result.type === 'RESIGN' ||
          result.type === 'CHOOSE_TILE' || result.type === 'TIMEOUT') {
        set({ actionMode: null, pendingTileChoice: null });
      }
      // If gems were fully taken, reset mode
      if (result.type === 'SELECT_GEM' && result.payload?.completed !== false) {
        set({ actionMode: null });
      }
      if (result.type === 'TAKE_GEMS_CONFIRMED') {
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
    lobbyUnlimitedTime: false,
    lobbyAiAvailable: false,
    replayList: [],
    replayListLoading: false,
    replayListError: null,
    currentReplay: null,
    replayLoading: false,
    replayError: null,

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
        lobbyUnlimitedTime: false, lobbyAiAvailable: false,
        lobbyTeamLayout: 'ADJACENT', lobbyTeamSeats: EMPTY_TEAM_SEATS,
      });
    },

    toggleReady: () => {
      get().socket?.emit('lobby_ready');
    },

    toggleGoFirst: () => {
      const { socket, myAccount, lobbyPlayers } = get();
      if (!socket || !myAccount) return;
      const me = lobbyPlayers.find(player => player.username === myAccount.username);
      socket.emit('set_go_first', { enabled: !me?.wantsFirst }, (res: { error?: string }) => {
        if (res?.error) showToast(res.error, 'warn');
      });
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

    toggleUnlimitedTime: () => {
      const { socket, lobbyUnlimitedTime } = get();
      if (!socket) return;
      socket.emit('set_unlimited_time', { enabled: !lobbyUnlimitedTime }, (res: { error?: string }) => {
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

    // ── AI bots (docs/AI_BRIDGE.md §3) ──
    // Bots cannot click a seat themselves, so any lobby member seats them.
    selectTeamSeatFor: (teamId, seatIndex, username) => {
      const { socket } = get();
      if (!socket) return;
      socket.emit('select_team_seat', { teamId, seatIndex, forUsername: username }, (res: { error?: string }) => {
        if (res?.error) showToast(res.error, 'warn');
      });
    },

    addAI: () => {
      const { socket } = get();
      if (!socket) return;
      socket.emit('lobby_add_ai', {}, (res: { error?: string; username?: string }) => {
        if (res?.error) showToast(res.error, 'warn');
      });
    },

    removeAI: (username: string) => {
      const { socket } = get();
      if (!socket) return;
      socket.emit('lobby_remove_ai', { username }, (res: { error?: string }) => {
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

    sendAction: (action: Record<string, unknown>, onComplete?: (success: boolean) => void) => {
      const { socket, roomId } = get();
      if (!socket || !roomId) { onComplete?.(false); return; }
      socket.emit('game_action', { roomId, action }, (res: { error?: string }) => {
        if (res?.error) showToast(res.error, 'error');
        onComplete?.(!res?.error);
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

    // ── Replays: read-only REST, independent of the socket session ──
    openReplayBrowser: () => {
      set({ appPhase: 'REPLAY_BROWSER' });
      void get().refreshReplayList();
    },

    refreshReplayList: async () => {
      const token = ++replayListRequest;
      set({ replayListLoading: true, replayListError: null });
      try {
        const { games } = await fetchReplayList();
        if (token !== replayListRequest) return;
        set({ replayList: games, replayListLoading: false });
      } catch (err) {
        if (token !== replayListRequest) return;
        set({
          replayListLoading: false,
          replayListError: err instanceof Error ? err.message : 'Failed to load replays',
        });
      }
    },

    openReplay: async (id: string) => {
      const token = ++replayRequest;
      set({ appPhase: 'REPLAY_VIEWER', currentReplay: null, replayLoading: true, replayError: null });
      try {
        const replay = await fetchReplay(id);
        if (token !== replayRequest || get().appPhase !== 'REPLAY_VIEWER') return;
        set({ currentReplay: replay, replayLoading: false });
      } catch (err) {
        if (token !== replayRequest) return;
        set({
          replayLoading: false,
          replayError: err instanceof Error ? err.message : 'Failed to load this replay',
        });
      }
    },

    closeReplayViewer: () => {
      replayRequest++;
      set({ appPhase: 'REPLAY_BROWSER', currentReplay: null, replayLoading: false, replayError: null });
    },

    closeReplayBrowser: () => {
      set({ appPhase: 'LOGIN' });
    },
  };
});

export default useGameStore;
