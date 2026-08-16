import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Maximize, Minimize, MoreVertical } from 'lucide-react';
import useGameStore from '../store/gameStore';
import CardView, { DeckView } from './CardView';
import NobleTiles from './NobleTiles';
import GemSupply from './GemSupply';
import PlayerInfo from './PlayerInfo';
import type { BonusTile, Card, GameState, TeamId, TimeControlState } from '../types';
import { GEM_COLORS_HEX, GEM_KEYS } from '../types';
import { getRewardCounts } from '../utils/gameHelpers';

interface GemDelta {
  color: number;
  delta: number;
  key: string;
}

export default function GameBoard() {
  const {
    gameState, playerIndex, actionMode, pendingTileChoice,
    setActionMode, sendAction, chooseBonusTile,
    returnToLobby, quitRoom, resign, lastActionResult,
    disconnectedPlayers,
  } = useGameStore();

  const [showMenu, setShowMenu] = useState(false);
  const [showQuitConfirm, setShowQuitConfirm] = useState(false);
  const [showResignConfirm, setShowResignConfirm] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [nobleClaim, setNobleClaim] = useState(false);
  const [clockNow, setClockNow] = useState(Date.now());
  const [serverClockOffset, setServerClockOffset] = useState(0);

  // Animation state
  const [gemDeltas, setGemDeltas] = useState<GemDelta[]>([]);
  const [opponentCardDeltas, setOpponentCardDeltas] = useState<Record<number, Record<number, boolean>>>({});

  // Track previous reward counts for +1 animation
  const prevRewardsRef = useRef<Record<number, number[]>>({});

  // ── Fullscreen toggle ──
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  }, []);

  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  useEffect(() => {
    const serverNow = gameState?.timeControl?.serverNow;
    if (serverNow !== undefined) setServerClockOffset(serverNow - Date.now());
  }, [gameState?.timeControl?.serverNow]);

  useEffect(() => {
    if (!gameState?.timeControl) return;
    setClockNow(Date.now());
    const interval = setInterval(() => setClockNow(Date.now()), 100);
    return () => clearInterval(interval);
  }, [!!gameState?.timeControl]);

  // ── Noble claim notification ──
  const prevTileCountRef = useRef<number | null>(null);
  useEffect(() => {
    if (!gameState) return;
    const me = gameState.players[playerIndex];
    const myTileCount = me.bonusTiles.length;
    if (prevTileCountRef.current !== null && myTileCount > prevTileCountRef.current) {
      setNobleClaim(true);
      setTimeout(() => setNobleClaim(false), 2500);
    }
    prevTileCountRef.current = myTileCount;
  }, [gameState?.players[playerIndex]?.bonusTiles.length]);

  // ── Handle remote action animations ──
  const lastActionIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!lastActionResult || !gameState) return;
    const actionKey = `${lastActionResult.type}-${lastActionResult.actingPlayer}-${Date.now()}`;
    if (actionKey === lastActionIdRef.current) return;
    lastActionIdRef.current = actionKey;

    const { type, payload, actingPlayer } = lastActionResult;

    // Gem deltas for supply animation
    if (type === 'SELECT_GEM' && payload.completed !== false) {
      const selected = payload.selected as number[];
      const deltaMap: Record<number, number> = {};
      for (const c of selected) deltaMap[c] = (deltaMap[c] || 0) - 1;
      setGemDeltas(Object.entries(deltaMap).map(([c, d]) => ({ color: +c, delta: d, key: `${c}-${Date.now()}` })));
      setTimeout(() => setGemDeltas([]), 2200);
    }

    if (type === 'ENTER_RESERVE' && payload.goldTaken) {
      setGemDeltas([{ color: 5, delta: -1, key: `5-${Date.now()}` }]);
      setTimeout(() => setGemDeltas([]), 2200);
    }

    // Opponent card purchase +1 animation
    if (type === 'BUY_CARD' && actingPlayer !== playerIndex) {
      const reward = payload.reward as number;
      setOpponentCardDeltas(prev => ({
        ...prev,
        [actingPlayer]: { ...prev[actingPlayer], [reward]: true },
      }));
      setTimeout(() => {
        setOpponentCardDeltas(prev => {
          const updated = { ...prev };
          if (updated[actingPlayer]) {
            updated[actingPlayer] = { ...updated[actingPlayer] };
            delete updated[actingPlayer][reward];
          }
          return updated;
        });
      }, 2500);
    }
  }, [lastActionResult, playerIndex]);

  // ── Persistent keyboard handler for gem selection (ref-based, no useEffect re-registration) ──
  const actionModeRef = useRef(actionMode);
  const gameStateRef = useRef(gameState);
  const playerIndexRef = useRef(playerIndex);
  actionModeRef.current = actionMode;
  gameStateRef.current = gameState;
  playerIndexRef.current = playerIndex;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const gs = gameStateRef.current;
      const pi = playerIndexRef.current;
      const am = actionModeRef.current;

      if (!gs || gs.phase !== 'PLAYING' || gs.currentPlayerIndex !== pi) return;
      if (am !== 'TAKE_GEMS') return;

      const idx = GEM_KEYS.indexOf(e.key.toLowerCase() as typeof GEM_KEYS[number]);
      if (idx !== -1) {
        e.preventDefault();
        useGameStore.getState().sendAction({ type: 'SELECT_GEM', color: idx });
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []); // Mount once, never re-register

  if (!gameState) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-slate-400 text-sm">Waiting for game state...</div>
      </div>
    );
  }

  const me = gameState.players[playerIndex];
  const isMyTurn = gameState.currentPlayerIndex === playerIndex && gameState.phase === 'PLAYING';
  const currentPlayerName = gameState.players[gameState.currentPlayerIndex]?.username ?? '';
  const resignedPlayers = gameState.resignedPlayers || [];

  // Opponents in clockwise order
  const others: { player: typeof me; idx: number; resigned: boolean }[] = [];
  for (let offset = 1; offset < gameState.numPlayers; offset++) {
    const si = (playerIndex + offset) % gameState.numPlayers;
    others.push({ player: gameState.players[si], idx: si, resigned: resignedPlayers.includes(si) });
  }

  // Get selected gems for display
  const selectedGems = gameState.turnAction?.type === 'TAKE_GEMS' ? gameState.turnAction.selected : [];
  const isTeamGame = gameState.gameMode !== 'INDIVIDUAL';
  const isOneVsTwo = gameState.gameMode === 'ONE_V_TWO';
  const teamSummaries = isTeamGame ? ([0, 1] as TeamId[]).map(teamId => {
    const members = gameState.players.filter(player => player.teamId === teamId);
    const scores = members.map(player => player.score).sort((a, b) => b - a);
    return {
      teamId,
      members,
      total: scores.reduce((sum, score) => sum + score, 0),
      secondScore: scores[1] ?? 0,
    };
  }) : [];
  const effectiveServerNow = clockNow + serverClockOffset;
  const activeClockPlayerIndex = gameState.phase === 'PLAYING' ? gameState.currentPlayerIndex : -1;
  const playerClocks = gameState.players.map((_, index) =>
    getPlayerClock(gameState.timeControl, index, activeClockPlayerIndex, effectiveServerNow));

  // Layout: side players or top players
  const leftPlayer = others.length >= 2 ? others[0] : null;
  const rightPlayer = others.length >= 2 ? others[others.length - 1] : null;
  const topPlayers = others.length >= 2 ? others.slice(1, -1) : others;
  const useSideLayout = others.length >= 2;

  return (
    <div className="h-screen flex flex-col p-1 max-w-6xl mx-auto relative overflow-hidden">
      {/* ── Top-right controls: fullscreen + menu ── */}
      <div className="absolute top-1 right-1 z-30 flex items-center gap-1">
        {/* Turn indicator - subtle */}
        {gameState.phase === 'PLAYING' && (
          <div className="text-[10px] text-slate-400 mr-2 font-display">
            T{gameState.turnNumber + 1}
            {gameState.finalRoundTriggeredBy !== null && <span className="text-red-400 ml-1">Final</span>}
          </div>
        )}
        <button onClick={toggleFullscreen} className="p-1 text-slate-400 hover:text-slate-600 transition" title="Toggle fullscreen">
          {isFullscreen ? <Minimize size={14} /> : <Maximize size={14} />}
        </button>
        <div className="relative">
          <button onClick={() => setShowMenu(!showMenu)} className="p-1 text-slate-400 hover:text-slate-600 transition">
            <MoreVertical size={14} />
          </button>
          {showMenu && (
            <div className="absolute right-0 top-6 bg-white/90 backdrop-blur-md rounded-lg shadow-lg border border-white/70 py-1 w-32 z-50">
              <button onClick={() => { setShowQuitConfirm(true); setShowMenu(false); }} className="w-full text-left px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-100">Quit Room</button>
              {gameState.phase === 'PLAYING' && (
                <button onClick={() => { setShowResignConfirm(true); setShowMenu(false); }} className="w-full text-left px-3 py-1.5 text-xs text-red-500 hover:bg-red-50">Resign</button>
              )}
            </div>
          )}
        </div>
      </div>

      {isTeamGame && (
        <div className="absolute top-1 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1.5 text-[9px] font-display">
          {teamSummaries.map(team => (
            <div key={team.teamId} className={`px-2 py-0.5 rounded-lg border backdrop-blur-sm ${
              team.teamId === 0
                ? 'bg-[#5B8C6A]/10 border-[#5B8C6A]/20 text-[#4C795A]'
                : 'bg-[#7B6FA0]/10 border-[#7B6FA0]/20 text-[#6B5F91]'
            }`}>
              {isOneVsTwo ? (
                <>{team.teamId === 0 ? 'Solo' : 'Duo'} {team.total}/{team.teamId === 0 ? 15 : 32}</>
              ) : (
                <>
                  Team {team.teamId === 0 ? 'A' : 'B'} {team.total}
                  <span className="opacity-60 ml-1">2nd {team.secondScore}</span>
                </>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Click-away to close menu */}
      {showMenu && <div className="fixed inset-0 z-20" onClick={() => setShowMenu(false)} />}

      {/* ── Confirm modals ── */}
      <AnimatePresence>
        {showQuitConfirm && (
          <ConfirmModal title="Quit Room?" message="This game will be abandoned." confirmLabel="Quit" confirmColor="bg-slate-500"
            onConfirm={() => { quitRoom(); setShowQuitConfirm(false); }} onCancel={() => setShowQuitConfirm(false)} />
        )}
      </AnimatePresence>
      <AnimatePresence>
        {showResignConfirm && (
          <ConfirmModal title="Resign?" message={isTeamGame
            ? 'Resigning forfeits the game for your entire side.'
            : 'Your gems return to supply and cards are discarded.'} confirmLabel="Resign" confirmColor="bg-red-500"
            onConfirm={() => { resign(); setShowResignConfirm(false); }} onCancel={() => setShowResignConfirm(false)} />
        )}
      </AnimatePresence>

      {/* ── Game Over overlay ── */}
      <AnimatePresence>
        {gameState.phase === 'GAME_OVER' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center backdrop-blur-sm">
            <motion.div initial={{ scale: 0.8 }} animate={{ scale: 1 }}
              className="bg-white/80 backdrop-blur-md rounded-2xl p-8 max-w-md w-full space-y-4 border border-white/70 shadow-xl">
              {isTeamGame ? (
                <TeamGameOver gameState={gameState} />
              ) : (
                <>
                  <h2 className="text-2xl font-display font-bold text-center text-amber-600">Game Over</h2>
                  <div className="space-y-2">
                    {[...gameState.players].sort((a, b) => b.score - a.score || a.cards.length - b.cards.length).map((p, rank) => (
                      <div key={p.username} className={`flex items-center justify-between px-4 py-2 rounded-xl ${rank === 0 ? 'bg-amber-50 border border-amber-200' : 'bg-white/50'}`}>
                        <span className="font-medium text-sm text-slate-700">{rank === 0 && '★ '}{p.username}</span>
                        <span>
                          <span className="text-amber-600 font-display font-bold">{p.score}</span>
                          <span className="text-slate-400 text-xs ml-2">({p.cards.length} cards)</span>
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              )}
              <button onClick={returnToLobby} className="w-full py-3 bg-[#7EA68A] hover:bg-[#6B9477] text-white rounded-xl font-semibold transition text-sm shadow-sm">
                Return to Lobby
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Noble claim notification ── */}
      <AnimatePresence>
        {nobleClaim && (
          <motion.div initial={{ opacity: 0, y: -40 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -40 }}
            className="fixed top-4 left-1/2 -translate-x-1/2 z-50 bg-[#7B6FA0]/90 backdrop-blur-sm text-white px-5 py-2.5 rounded-xl shadow-lg">
            <span className="text-sm font-display font-semibold">Noble acquired! +3 points</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Bonus tile choice modal ── */}
      <AnimatePresence>
        {pendingTileChoice && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="fixed inset-0 bg-black/30 z-40 flex items-center justify-center backdrop-blur-sm">
            <motion.div initial={{ scale: 0.8 }} animate={{ scale: 1 }} className="bg-white/80 backdrop-blur-md rounded-2xl p-6 space-y-4 border border-white/70 shadow-xl">
              <h3 className="text-lg font-display font-bold text-center text-[#7B6FA0]">Choose a Noble</h3>
              <div className="flex gap-3 justify-center">
                {gameState.bonusTiles.filter(t => pendingTileChoice.includes(t.id)).map(tile => (
                  <TileChoice key={tile.id} tile={tile} onPick={() => chooseBonusTile(tile.id)} />
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Turn indicator (minimal, no vertical space) ── */}
      {gameState.phase === 'PLAYING' && (
        <div className="absolute top-1 left-1 z-10">
          {isMyTurn ? (
            <span className="text-[11px] text-amber-600 font-semibold font-display bg-amber-50/80 px-2 py-0.5 rounded-lg">Your turn</span>
          ) : (
            <span className="text-[10px] text-slate-400">
              <span className="text-slate-500 font-medium">{currentPlayerName}</span>
              {disconnectedPlayers.has(currentPlayerName) && <span className="text-red-400 ml-1">(offline)</span>}
            </span>
          )}
        </div>
      )}

      {/* ── Top opponents (for 2-player or 4+ middle players) ── */}
      {topPlayers.length > 0 && (
        <div className="flex gap-3 justify-center mt-5 mb-1 flex-wrap flex-shrink-0">
          {topPlayers.map(({ player, idx, resigned }) => (
            <div key={idx} data-player-panel={idx}
              className={`flex-1 min-w-[220px] max-w-[280px] ${resigned ? 'opacity-30 pointer-events-none' : ''}`}>
              {resigned && <div className="text-[9px] text-red-400/70 text-center mb-0.5 font-display">Resigned</div>}
              <PlayerInfo player={player} isMe={false} isCurrentTurn={!resigned && gameState.currentPlayerIndex === idx}
                compact isPulsing={false} isDisconnected={disconnectedPlayers.has(player.username)}
                gameMode={gameState.gameMode}
                clockLabel={playerClocks[idx]?.label} clockUrgent={playerClocks[idx]?.urgent}
                cardDeltas={opponentCardDeltas[idx] || {}} />
            </div>
          ))}
        </div>
      )}

      {/* ── Main 3-column layout ── */}
      <div className="flex-1 flex items-start gap-1 min-h-0">
        {/* Left player */}
        {useSideLayout && leftPlayer && (
          <div data-player-panel={leftPlayer.idx} className={`w-[230px] flex flex-col items-center pt-1 flex-shrink-0 ${leftPlayer.resigned ? 'opacity-30' : ''}`}>
            <PlayerInfo player={leftPlayer.player} isMe={false} isCurrentTurn={!leftPlayer.resigned && gameState.currentPlayerIndex === leftPlayer.idx}
              compact isDisconnected={disconnectedPlayers.has(leftPlayer.player.username)}
              gameMode={gameState.gameMode}
              clockLabel={playerClocks[leftPlayer.idx]?.label} clockUrgent={playerClocks[leftPlayer.idx]?.urgent}
              cardDeltas={opponentCardDeltas[leftPlayer.idx] || {}} />
          </div>
        )}

        {/* Center: board + gems */}
        <div className="flex-1 flex flex-col items-center gap-1 min-h-0">
          <div className="flex gap-3 justify-center items-start">
            {/* Card board */}
            <div className="flex flex-col gap-1 items-center">
              {[2, 1, 0].map(tierIdx => (
                <div key={tierIdx} className="flex gap-1 items-center">
                  <DeckView tier={tierIdx + 1} count={gameState.deckCounts[tierIdx]}
                    clickable={isMyTurn && actionMode === 'RESERVE' && me.reserved.length < gameState.config.maxReserved}
                    onClick={() => sendAction({ type: 'RESERVE_FROM_DECK', tier: tierIdx + 1 })} />
                  <AnimatePresence mode="popLayout">
                    {gameState.board[tierIdx].map(card => (
                      <motion.div key={card.id} layout
                        initial={{ scaleX: 0, opacity: 0.4 }} animate={{ scaleX: 1, opacity: 1 }}
                        exit={{ scale: 0.4, opacity: 0, transition: { duration: 0.28 } }}
                        transition={{ duration: 0.22 }} data-card-id={card.id}>
                        <CardView card={card}
                          clickable={isMyTurn && (actionMode === 'BUY' || actionMode === 'RESERVE')}
                          onClick={() => {
                            if (actionMode === 'BUY') sendAction({ type: 'BUY_CARD', cardId: card.id, source: 'board' });
                            else if (actionMode === 'RESERVE') sendAction({ type: 'RESERVE_CARD', cardId: card.id });
                          }} />
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  {Array.from({ length: Math.max(0, 4 - gameState.board[tierIdx].length) }).map((_, i) => (
                    <div key={`e${tierIdx}-${i}`} className="w-[88px] h-[122px] rounded-lg border border-slate-200/40 border-dashed" />
                  ))}
                </div>
              ))}
            </div>

            {/* Noble tiles */}
            <NobleTiles tiles={gameState.bonusTiles} />
          </div>

          {/* Gem supply */}
          <div className="mt-1 flex-shrink-0">
            <GemSupply gems={gameState.gems as [number, number, number, number, number, number]}
              gemDeltas={gemDeltas} selectedGems={selectedGems} />
          </div>
        </div>

        {/* Right player */}
        {useSideLayout && rightPlayer && (
          <div data-player-panel={rightPlayer.idx} className={`w-[230px] flex flex-col items-center pt-1 flex-shrink-0 ${rightPlayer.resigned ? 'opacity-30' : ''}`}>
            <PlayerInfo player={rightPlayer.player} isMe={false} isCurrentTurn={!rightPlayer.resigned && gameState.currentPlayerIndex === rightPlayer.idx}
              compact isDisconnected={disconnectedPlayers.has(rightPlayer.player.username)}
              gameMode={gameState.gameMode}
              clockLabel={playerClocks[rightPlayer.idx]?.label} clockUrgent={playerClocks[rightPlayer.idx]?.urgent}
              cardDeltas={opponentCardDeltas[rightPlayer.idx] || {}} />
          </div>
        )}
      </div>

      {/* ── Bottom: self panel (action buttons integrated) ── */}
      <div className="flex-shrink-0 mt-1" data-player-panel={playerIndex}>
        <PlayerInfo player={me} isMe={true} isCurrentTurn={isMyTurn}
          actionMode={actionMode} gameState={gameState} playerIndex={playerIndex}
          gameMode={gameState.gameMode}
          clockLabel={playerClocks[playerIndex]?.label} clockUrgent={playerClocks[playerIndex]?.urgent}
          onSetActionMode={setActionMode}
          onClickCard={(card, source) => {
            if (actionMode === 'BUY') sendAction({ type: 'BUY_CARD', cardId: card.id, source });
          }} />
      </div>
    </div>
  );
}

function getPlayerClock(
  timeControl: TimeControlState | null,
  playerIndex: number,
  currentPlayerIndex: number,
  now: number,
): { label: string; urgent: boolean } | null {
  if (!timeControl) return null;
  const storedMainTime = Math.max(0, timeControl.playerTimeRemainingMs[playerIndex] ?? 0);
  const isActive = playerIndex === currentPlayerIndex;

  if (isActive && timeControl.countdownDeadline !== null) {
    const countdown = Math.max(0, timeControl.countdownDeadline - now);
    return { label: `${(countdown / 1000).toFixed(1)}s`, urgent: true };
  }

  if (isActive) {
    const elapsed = Math.max(0, now - timeControl.activeSince);
    const mainRemaining = Math.max(0, storedMainTime - elapsed);
    if (mainRemaining <= 0) {
      const impliedCountdownDeadline = timeControl.activeSince + storedMainTime + timeControl.countdownMs;
      const countdown = Math.max(0, impliedCountdownDeadline - now);
      return { label: `${(countdown / 1000).toFixed(1)}s`, urgent: true };
    }
    return { label: formatMainTime(mainRemaining), urgent: false };
  }

  return { label: formatMainTime(storedMainTime), urgent: false };
}

function formatMainTime(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.ceil(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function TeamGameOver({ gameState }: { gameState: GameState }) {
  const winningTeamIds = gameState.gameResult?.winningTeamIds ?? [];
  const isDraw = winningTeamIds.length > 1;
  const winner = winningTeamIds.length === 1 ? winningTeamIds[0] : null;
  const isOneVsTwo = gameState.gameMode === 'ONE_V_TWO';
  const sideName = (teamId: TeamId) => isOneVsTwo
    ? (teamId === 0 ? 'Solo' : 'Duo')
    : `Team ${teamId === 0 ? 'A' : 'B'}`;

  return (
    <>
      <div className="text-center">
        <h2 className="text-2xl font-display font-bold text-amber-600">
          {isDraw ? (isOneVsTwo ? '1v2 Draw' : 'Team Draw') : winner !== null ? `${sideName(winner)} Wins` : 'Game Over'}
        </h2>
        {gameState.gameResult?.reason === 'FORFEIT' && (
          <p className="text-xs text-slate-400 mt-1">
            {sideName(gameState.gameResult.forfeitingTeamId ?? 0)} forfeited
          </p>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3">
        {([0, 1] as TeamId[]).map(teamId => {
          const members = gameState.players.filter(player => player.teamId === teamId);
          const total = members.reduce((sum, player) => sum + player.score, 0);
          const secondScore = [...members].sort((a, b) => b.score - a.score)[1]?.score ?? 0;
          const won = winningTeamIds.includes(teamId);
          const threshold = teamId === 0 ? 15 : 32;
          return (
            <div key={teamId} className={`rounded-xl p-3 border ${
              won ? 'bg-amber-50 border-amber-200' : 'bg-white/45 border-white/60'
            }`}>
              <div className={`text-sm font-display font-bold mb-2 ${teamId === 0 ? 'text-[#5B8C6A]' : 'text-[#7B6FA0]'}`}>
                {won && '★ '}{sideName(teamId)} · {total}
              </div>
              <div className="space-y-1">
                {members.map(player => (
                  <div key={player.username} className="flex justify-between gap-2 text-xs text-slate-600">
                    <span className="truncate">{player.username}</span>
                    <span className="font-display font-semibold">{player.score}</span>
                  </div>
                ))}
              </div>
              <div className="text-[9px] text-slate-400 mt-2">
                {isOneVsTwo
                  ? `Threshold: ${threshold} · Excess: ${Math.max(0, total - threshold)}`
                  : `Second score: ${secondScore}`}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

function ConfirmModal({ title, message, confirmLabel, confirmColor, onConfirm, onCancel }: {
  title: string; message: string; confirmLabel: string; confirmColor: string;
  onConfirm: () => void; onCancel: () => void;
}) {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center backdrop-blur-sm">
      <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} exit={{ scale: 0.9 }}
        className="bg-white/80 backdrop-blur-md rounded-2xl p-6 w-80 space-y-4 border border-white/70 shadow-xl">
        <h3 className="text-lg font-display font-bold text-center text-slate-800">{title}</h3>
        <p className="text-sm text-slate-500 text-center">{message}</p>
        <div className="flex gap-2">
          <button onClick={onCancel} className="flex-1 py-2 bg-white/60 hover:bg-white/80 border border-slate-200 rounded-xl text-sm transition text-slate-600">Cancel</button>
          <button onClick={onConfirm} className={`flex-1 py-2 ${confirmColor} hover:opacity-80 text-white rounded-xl text-sm font-semibold transition shadow-sm`}>{confirmLabel}</button>
        </div>
      </motion.div>
    </motion.div>
  );
}

function TileChoice({ tile, onPick }: { tile: BonusTile; onPick: () => void }) {
  return (
    <button onClick={onPick}
      className="w-20 h-20 bg-white/60 border-2 border-[#9B8EC4]/50 hover:border-amber-500 rounded-xl flex flex-col items-center justify-center p-1 transition shadow-sm">
      <span className="text-sm font-display font-bold text-[#7B6FA0] mb-1">+3</span>
      <div className="flex gap-0.5 flex-wrap justify-center">
        {tile.requirement.map((r, i) => r > 0 ? (
          <div key={i} className="flex items-center gap-0.5">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: GEM_COLORS_HEX[i] }} />
            <span className="text-[10px] text-slate-500">{r}</span>
          </div>
        ) : null)}
      </div>
    </button>
  );
}
