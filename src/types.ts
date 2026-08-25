export type GemColor = 0 | 1 | 2 | 3 | 4;
export type GemOrGold = 0 | 1 | 2 | 3 | 4 | 5;
export type Cost = [number, number, number, number, number];
export type Tier = 1 | 2 | 3;
export type GameMode = 'INDIVIDUAL' | 'TEAM' | 'ONE_V_TWO';
export type TeamId = 0 | 1;
export type TeamLayout = 'ADJACENT' | 'OPPOSITE';
export type LobbyTeamFormat = 'TWO_V_TWO' | 'ONE_V_TWO' | null;

export const GEM_NAMES = ['Indigo', 'Jade', 'Amber', 'Rose', 'Violet'] as const;
export const GEM_COLORS_HEX = ['#5B7CC4', '#4DAA8D', '#D4944C', '#C75B7A', '#8E6FBF'] as const;
export const GEM_COLORS_LIGHT = ['#B3C4E6', '#A8D8C4', '#EDCDA3', '#E6ABBC', '#C9B8E0'] as const;
export const GOLD_HEX = '#C9A84C';

export const GEM_KEYS = ['a', 's', 'd', 'f', 'g'] as const;

export interface Card {
  id: number;
  tier: Tier;
  reward: GemColor;
  points: number;
  cost: Cost;
}

export interface BonusTile {
  id: number;
  points: 3;
  requirement: Cost;
}

export interface Player {
  username: string;
  gems: [number, number, number, number, number, number];
  cards: Card[];
  reserved: Card[];
  bonusTiles: BonusTile[];
  score: number;
  avatarSeed: number;
  teamId?: TeamId;
}

export interface TeamDefinition {
  id: TeamId;
  playerIndices: number[];
}

export interface GameResult {
  reason: 'SCORE' | 'FORFEIT';
  winningTeamIds?: TeamId[];
  forfeitingTeamId?: TeamId;
}

export interface TimeControlState {
  mainTimeMs: number;
  incrementMs: number;
  playerTimeRemainingMs: number[];
  activeSince: number | null;
  serverNow: number;
}

export interface GameState {
  phase: 'PLAYING' | 'GAME_OVER';
  board: [Card[], Card[], Card[]];
  deckCounts: [number, number, number];
  gems: [number, number, number, number, number, number];
  bonusTiles: BonusTile[];
  players: Player[];
  currentPlayerIndex: number;
  roundStartPlayer: number;
  turnAction: TurnAction | null;
  finalRoundTriggeredBy: number | null;
  turnNumber: number;
  numPlayers: number;
  config: GameConfig;
  resignedPlayers: number[];
  gameMode: GameMode;
  teamLayout: TeamLayout | null;
  teams: TeamDefinition[];
  gameResult: GameResult | null;
  timeControl: TimeControlState | null;
}

export interface GameConfig {
  tokensPerColor: number;
  wildTokens: number;
  revealedTiles: number;
  cardsPerRow: number;
  maxTokensInHand: number;
  maxReserved: number;
  winThreshold: number;
  take2MinStack: number;
}

export type TurnAction =
  | { type: 'TAKE_GEMS'; selected: number[] }
  | { type: 'RESERVE'; goldTaken: boolean; cardPicked: boolean }
  | { type: 'BUY' };

export type ActionMode = 'TAKE_GEMS' | 'RESERVE' | 'BUY' | null;

export interface Account {
  username: string;
  rating: number;
  gamesPlayed: number;
  wins: number;
  avatarSeed: number;
  created: number;
}

export type ConnectionStatus =
  | 'idle' | 'waking_server' | 'connecting' | 'connected'
  | 'entering_lobby' | 'in_lobby' | 'reconnecting' | 'disconnected' | 'error';

export type AppPhase = 'LOGIN' | 'WAITING_ROOM' | 'GAME';

export interface LobbyPlayer {
  socketId: string;
  username: string;
  ready: boolean;
  wantsFirst: boolean;
  avatarSeed: number;
}

export type TeamSeats = [
  [string | null, string | null],
  [string | null, string | null],
];

export interface LobbyState {
  players: LobbyPlayer[];
  teamMode: boolean;
  teamFormat: LobbyTeamFormat;
  teamLayout: TeamLayout;
  teamSeats: TeamSeats;
  unlimitedTime: boolean;
}

// Server -> client: what just happened (for animations)
export interface ActionResult {
  type: string;
  payload: Record<string, unknown>;
  actingPlayer: number;
}
