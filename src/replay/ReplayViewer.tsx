import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Gauge, Pause, Play, SkipBack, SkipForward, X } from 'lucide-react';
import useGameStore from '../store/gameStore';
import CardView, { DeckView } from '../components/CardView';
import NobleTiles from '../components/NobleTiles';
import GemSupply from '../components/GemSupply';
import PlayerInfo from '../components/PlayerInfo';
import Avatar from '../components/Avatar';
import { TeamGameOver } from '../components/GameBoard';
import type { ReplayData } from '../types';
import { useReplayPlayer } from './useReplayPlayer';
import type { ReplayPlayer } from './useReplayPlayer';
import { frameCaption, framePendingTiles } from './replayText';

interface GemDelta {
  color: number;
  delta: number;
  key: string;
}

export default function ReplayViewer() {
  const { currentReplay, replayLoading, replayError, closeReplayViewer } = useGameStore();

  if (replayError) {
    return (
      <div className="min-h-[100dvh] flex items-center justify-center p-4">
        <div className="bg-white/60 backdrop-blur-md rounded-2xl p-6 w-80 max-w-full text-center space-y-4 border border-white/70 shadow-lg">
          <h2 className="text-lg font-display font-bold text-slate-800">Replay unavailable</h2>
          <p className="text-sm text-red-500">{replayError}</p>
          <button onClick={closeReplayViewer}
            className="w-full min-h-11 py-2.5 bg-[#8B9DAF] hover:bg-[#7A8D9F] text-white rounded-xl font-semibold transition shadow-sm text-sm">
            Back to replays
          </button>
        </div>
      </div>
    );
  }

  if (replayLoading || !currentReplay || currentReplay.frames.length === 0) {
    return (
      <div className="min-h-[100dvh] flex items-center justify-center p-4">
        <div className="text-center space-y-3">
          <span className="inline-block animate-spin h-6 w-6 border-2 border-slate-400 border-t-transparent rounded-full" />
          <p className="text-slate-500 text-sm">Loading replay...</p>
        </div>
      </div>
    );
  }

  return <ReplayStage key={currentReplay.id} replay={currentReplay} onExit={closeReplayViewer} />;
}

// ── The board itself: same structure and classes as GameBoard, read-only ──

function ReplayStage({ replay, onExit }: { replay: ReplayData; onExit: () => void }) {
  const isMobileLayout = useMobileLayout();
  const frames = replay.frames;
  const seatCount = frames[0].state.players.length;
  const player = useReplayPlayer(frames.length, seatCount);
  const frame = frames[Math.min(player.index, frames.length - 1)];
  const gameState = frame.state;
  const perspective = Math.min(player.perspective, seatCount - 1);

  const [gemDeltas, setGemDeltas] = useState<GemDelta[]>([]);
  const [opponentCardDeltas, setOpponentCardDeltas] = useState<Record<number, Record<number, boolean>>>({});
  const [nobleClaim, setNobleClaim] = useState(false);
  const [overlayHidden, setOverlayHidden] = useState(false);

  const perspectiveRef = useRef(perspective);
  perspectiveRef.current = perspective;
  const prevIndexRef = useRef(player.index);

  // ── Animations, driven by the frame's ActionResult (same mapping as GameBoard) ──
  useEffect(() => {
    const previous = prevIndexRef.current;
    prevIndexRef.current = player.index;
    // Only animate when moving forward; scrubbing back just resets.
    if (player.index <= previous) {
      setGemDeltas([]);
      setOpponentCardDeltas({});
      return;
    }

    const result = frame.result;
    if (!result) return;
    const { type, payload, actingPlayer } = result;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const stamp = `${player.index}`;

    const deltaMap: Record<number, number> = {};
    if ((type === 'SELECT_GEM' || type === 'TAKE_GEMS_CONFIRMED') && payload.completed !== false) {
      for (const color of (payload.selected as number[] | undefined) ?? []) {
        deltaMap[color] = (deltaMap[color] || 0) - 1;
      }
    }
    if (type === 'BUY_CARD') {
      const returned = (payload.gemsReturned as number[] | undefined) ?? [];
      returned.forEach((count, color) => {
        if (count > 0) deltaMap[color] = (deltaMap[color] || 0) + count;
      });
    }
    if ((type === 'RESERVE_CARD' || type === 'RESERVE_FROM_DECK' || type === 'ENTER_RESERVE') && payload.goldTaken) {
      deltaMap[5] = (deltaMap[5] || 0) - 1;
    }

    const deltas = Object.entries(deltaMap)
      .filter(([, delta]) => delta !== 0)
      .map(([color, delta]) => ({ color: +color, delta, key: `${color}-${stamp}` }));
    if (deltas.length > 0) {
      setGemDeltas(deltas);
      timers.push(setTimeout(() => setGemDeltas([]), 2200));
    }

    // Noble toast for the watched seat (single-step advances only, not scrubs)
    if (player.index === previous + 1) {
      const seat = perspectiveRef.current;
      const before = frames[previous]?.state.players[seat]?.bonusTiles.length ?? 0;
      const after = frame.state.players[seat]?.bonusTiles.length ?? 0;
      if (after > before) {
        setNobleClaim(true);
        timers.push(setTimeout(() => setNobleClaim(false), 2500));
      }
    }

    if (type === 'BUY_CARD' && actingPlayer !== perspectiveRef.current) {
      const reward = payload.reward as number;
      setOpponentCardDeltas(prev => ({
        ...prev,
        [actingPlayer]: { ...prev[actingPlayer], [reward]: true },
      }));
      timers.push(setTimeout(() => {
        setOpponentCardDeltas(prev => {
          const updated = { ...prev };
          if (updated[actingPlayer]) {
            updated[actingPlayer] = { ...updated[actingPlayer] };
            delete updated[actingPlayer][reward];
          }
          return updated;
        });
      }, 2500));
    }

    return () => timers.forEach(clearTimeout);
  }, [player.index, frame, frames]);

  // The Game Over card comes back when the viewer scrubs away and returns.
  useEffect(() => {
    if (gameState.phase !== 'GAME_OVER') setOverlayHidden(false);
  }, [gameState.phase]);

  const me = gameState.players[perspective];
  const resignedPlayers = gameState.resignedPlayers || [];
  const isPlaying = gameState.phase === 'PLAYING';
  const currentPlayerName = gameState.players[gameState.currentPlayerIndex]?.username ?? '';
  const isTeamGame = gameState.gameMode !== 'INDIVIDUAL';
  const names = gameState.players.map(p => p.username);
  const caption = frameCaption(frame, names);
  const pendingTiles = framePendingTiles(frame);
  const lastIndex = frames.length - 1;

  // Opponents clockwise from the watched seat — identical to GameBoard
  const others: { player: typeof me; idx: number; resigned: boolean }[] = [];
  for (let offset = 1; offset < gameState.numPlayers; offset++) {
    const si = (perspective + offset) % gameState.numPlayers;
    others.push({ player: gameState.players[si], idx: si, resigned: resignedPlayers.includes(si) });
  }
  const leftPlayer = others.length >= 2 ? others[0] : null;
  const rightPlayer = others.length >= 2 ? others[others.length - 1] : null;
  const topPlayers = others.length >= 2 ? others.slice(1, -1) : others;
  const useSideLayout = others.length >= 2;

  return (
    <div
      className="game-shell flex flex-col max-w-6xl mx-auto relative overflow-x-hidden overflow-y-auto md:overflow-hidden md:p-1"
      style={isMobileLayout ? { paddingBottom: 'calc(104px + env(safe-area-inset-bottom))' } : undefined}
    >
      {/* ── Header: turn indicator + (desktop) progress and controls ── */}
      <div className="flex items-start gap-2 flex-shrink-0 px-1 pt-0.5 md:relative md:z-[60] md:items-center md:gap-3 md:pt-0 md:min-h-8">
        <div className="min-w-0 flex-1 leading-tight">
          {isPlaying ? (
            <div className="text-[11px] md:text-xs text-slate-500 truncate">
              <span className="font-display font-semibold text-amber-600">{currentPlayerName}</span>
              <span className="text-slate-400">&rsquo;s turn</span>
            </div>
          ) : (
            <div className="text-[11px] md:text-xs font-display font-semibold text-amber-600">Game over</div>
          )}
          <div className="text-[10px] text-slate-400 truncate">
            {pendingTiles?.length ? 'choosing a noble…' : (caption ?? 'Game start')}
          </div>
        </div>

        <div className="hidden md:flex items-center gap-2 w-[300px] flex-shrink-0">
          <span className="text-[10px] font-display text-slate-400 tabular-nums whitespace-nowrap">
            Turn {player.index} / {lastIndex}
          </span>
          <ScrubBar player={player} lastIndex={lastIndex} />
        </div>

        <div className="hidden md:flex items-center gap-1 flex-shrink-0">
          <ControlButtons player={player} />
          <PerspectiveSwitcher
            players={gameState.players} perspective={perspective}
            onSelect={player.setPerspective} />
          <button onClick={onExit} title="Exit replay" aria-label="Exit replay"
            className="ml-1 px-2 py-1 flex items-center gap-1 rounded-lg text-[11px] font-display text-slate-500 hover:text-slate-700 hover:bg-white/60 border border-white/60 transition">
            <X className="w-3.5 h-3.5" /> Exit
          </button>
        </div>
      </div>

      {/* ── Game Over overlay (same card as GameBoard) ── */}
      <AnimatePresence>
        {gameState.phase === 'GAME_OVER' && !overlayHidden && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-3 pb-[116px] md:pb-3 backdrop-blur-sm">
            <motion.div initial={{ scale: 0.8 }} animate={{ scale: 1 }}
              className="relative bg-white/80 backdrop-blur-md rounded-2xl p-4 md:p-8 max-w-md w-full max-h-[calc(100dvh-9rem)] md:max-h-[calc(100dvh-1.5rem)] overflow-y-auto space-y-4 border border-white/70 shadow-xl">
              <button
                onClick={() => { setOverlayHidden(true); if (player.playing) player.togglePlay(); }}
                aria-label="Close summary"
                className="absolute top-2 right-2 w-8 h-8 flex items-center justify-center rounded-lg text-slate-400 hover:text-slate-600 hover:bg-white/70 transition">
                <X className="w-4 h-4" />
              </button>
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
              <button onClick={onExit} className="w-full py-3 bg-[#7EA68A] hover:bg-[#6B9477] text-white rounded-xl font-semibold transition text-sm shadow-sm">
                Exit Replay
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

      {/* ── Top opponents (mobile) ── */}
      {others.length > 0 && (
        <div
          className="grid md:hidden gap-1 mt-1 mb-1 flex-shrink-0"
          style={{ gridTemplateColumns: `repeat(${others.length}, minmax(0, 1fr))` }}
        >
          {others.map(({ player: seat, idx, resigned }) => (
            <div key={idx} data-player-panel={idx} className={`relative min-w-0 ${resigned ? 'opacity-35' : ''}`}>
              {resigned && <div className="absolute top-0.5 right-0.5 z-10 text-[7px] leading-none text-red-500 font-display">OUT</div>}
              <PlayerInfo player={seat} isMe={false} mobile
                isAI={replay.meta.players[idx]?.isAI === true}
                isCurrentTurn={!resigned && isPlaying && gameState.currentPlayerIndex === idx}
                gameMode={gameState.gameMode}
                cardDeltas={opponentCardDeltas[idx] || {}} />
            </div>
          ))}
        </div>
      )}

      {/* ── Top opponents (desktop, 2P or 4P middle seat) ── */}
      {topPlayers.length > 0 && (
        <div className="hidden md:flex gap-3 justify-center mt-1 mb-1 flex-wrap flex-shrink-0">
          {topPlayers.map(({ player: seat, idx, resigned }) => (
            <div key={idx} data-player-panel={idx}
              className={`flex-1 min-w-[220px] max-w-[280px] ${resigned ? 'opacity-30' : ''}`}>
              {resigned && <div className="text-[9px] text-red-400/70 text-center mb-0.5 font-display">Resigned</div>}
              <PlayerInfo player={seat} isMe={false}
                isAI={replay.meta.players[idx]?.isAI === true}
                isCurrentTurn={!resigned && isPlaying && gameState.currentPlayerIndex === idx}
                compact isPulsing={false}
                gameMode={gameState.gameMode}
                cardDeltas={opponentCardDeltas[idx] || {}} />
            </div>
          ))}
        </div>
      )}

      {/* ── Main 3-column layout ── */}
      <div className="w-full flex-none md:flex-1 flex items-start gap-1 md:min-h-0">
        {useSideLayout && leftPlayer && (
          <div data-player-panel={leftPlayer.idx} className={`hidden md:flex w-[230px] flex-col items-center pt-1 flex-shrink-0 ${leftPlayer.resigned ? 'opacity-30' : ''}`}>
            <PlayerInfo player={leftPlayer.player} isMe={false}
              isAI={replay.meta.players[leftPlayer.idx]?.isAI === true}
              isCurrentTurn={!leftPlayer.resigned && isPlaying && gameState.currentPlayerIndex === leftPlayer.idx}
              compact gameMode={gameState.gameMode}
              cardDeltas={opponentCardDeltas[leftPlayer.idx] || {}} />
          </div>
        )}

        {/* Centre: board + gems */}
        <div className="w-full flex-none md:flex-1 flex flex-col items-center gap-1 md:min-h-0">
          <div className="mobile-market-width flex flex-col gap-1 items-center md:w-auto md:flex-row md:gap-3 md:justify-center md:items-start">
            <div className="order-2 md:order-1 mobile-market-width flex flex-col gap-1 items-center md:w-auto">
              {[2, 1, 0].map(tierIdx => (
                <div key={tierIdx} className="w-full grid grid-cols-5 gap-[3px] items-stretch md:w-auto md:flex md:gap-1 md:items-center">
                  <DeckView tier={tierIdx + 1} count={gameState.deckCounts[tierIdx]} size="market" clickable={false} />
                  <AnimatePresence mode="popLayout">
                    {gameState.board[tierIdx].map(card => (
                      <motion.div key={card.id} layout
                        initial={{ scaleX: 0, opacity: 0.4 }} animate={{ scaleX: 1, opacity: 1 }}
                        exit={{ scale: 0.4, opacity: 0, transition: { duration: 0.28 } }}
                        transition={{ duration: 0.22 }} data-card-id={card.id}>
                        <CardView card={card} size="market" clickable={false} />
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  {Array.from({ length: Math.max(0, 4 - gameState.board[tierIdx].length) }).map((_, i) => (
                    <div key={`e${tierIdx}-${i}`} className="market-card-size rounded-lg border border-slate-200/40 border-dashed" />
                  ))}
                </div>
              ))}
            </div>

            <div className="order-1 md:order-2 w-full md:w-auto">
              <NobleTiles tiles={gameState.bonusTiles} />
            </div>
          </div>

          <div className="mt-1 w-full flex justify-center flex-shrink-0">
            <GemSupply gems={gameState.gems as [number, number, number, number, number, number]}
              gemDeltas={gemDeltas} selectable={false} />
          </div>
        </div>

        {useSideLayout && rightPlayer && (
          <div data-player-panel={rightPlayer.idx} className={`hidden md:flex w-[230px] flex-col items-center pt-1 flex-shrink-0 ${rightPlayer.resigned ? 'opacity-30' : ''}`}>
            <PlayerInfo player={rightPlayer.player} isMe={false}
              isAI={replay.meta.players[rightPlayer.idx]?.isAI === true}
              isCurrentTurn={!rightPlayer.resigned && isPlaying && gameState.currentPlayerIndex === rightPlayer.idx}
              compact gameMode={gameState.gameMode}
              cardDeltas={opponentCardDeltas[rightPlayer.idx] || {}} />
          </div>
        )}
      </div>

      {/* ── Bottom: watched seat ── */}
      <div className="md:hidden flex-shrink-0 mt-auto pt-1 w-full max-w-[600px] mx-auto" data-player-panel={perspective}>
        <PlayerInfo player={me} isMe mobile
          isAI={replay.meta.players[perspective]?.isAI === true}
          isCurrentTurn={isPlaying && gameState.currentPlayerIndex === perspective}
          gameState={gameState} playerIndex={perspective} gameMode={gameState.gameMode} />
      </div>
      <div className="hidden md:block flex-shrink-0 mt-1" data-player-panel={perspective}>
        <PlayerInfo player={me} isMe
          isAI={replay.meta.players[perspective]?.isAI === true}
          isCurrentTurn={isPlaying && gameState.currentPlayerIndex === perspective}
          gameState={gameState} playerIndex={perspective} gameMode={gameState.gameMode} />
      </div>

      {/* ── Mobile control bar (above everything, safe-area aware) ── */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 z-[60] bg-white/85 backdrop-blur-md border-t border-white/70 shadow-[0_-2px_12px_rgba(0,0,0,0.08)]"
        style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}>
        <div className="flex items-center gap-2 px-2 pt-1">
          <span className="text-[10px] font-display text-slate-500 tabular-nums whitespace-nowrap">
            Turn {player.index} / {lastIndex}
          </span>
          <ScrubBar player={player} lastIndex={lastIndex} />
          <PerspectiveSwitcher mobile players={gameState.players} perspective={perspective}
            onSelect={player.setPerspective} />
        </div>
        <div className="flex items-center gap-1 px-2 pb-1">
          <ControlButtons mobile player={player} />
          <div className="flex-1" />
          <button onClick={onExit} aria-label="Exit replay"
            className="min-h-11 px-3 flex items-center gap-1 rounded-xl text-xs font-display font-medium text-slate-600 bg-white/70 border border-white/70 active:scale-95 transition touch-manipulation">
            <X className="w-4 h-4" /> Exit
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Control bar pieces ──

function ScrubBar({ player, lastIndex }: { player: ReplayPlayer; lastIndex: number }) {
  return (
    <input
      type="range" min={0} max={Math.max(1, lastIndex)} value={player.index}
      onChange={e => player.seek(Number(e.target.value))}
      aria-label="Replay position"
      className="flex-1 min-w-0 h-1.5 md:h-1 accent-[#7EA68A] cursor-pointer touch-manipulation"
    />
  );
}

function ControlButtons({ player, mobile = false }: { player: ReplayPlayer; mobile?: boolean }) {
  const shape = mobile
    ? 'min-h-11 min-w-11 flex items-center justify-center rounded-xl active:scale-95 transition touch-manipulation'
    : 'w-7 h-7 flex items-center justify-center rounded-lg transition';
  const secondary = mobile
    ? 'bg-white/70 border border-white/70 text-slate-600'
    : 'text-slate-500 hover:text-slate-700 hover:bg-white/60 border border-white/60';
  const primary = 'bg-[#7EA68A] hover:bg-[#6B9477] text-white border border-transparent shadow-sm';
  const icon = mobile ? 'w-5 h-5' : 'w-3.5 h-3.5';

  return (
    <>
      <button onClick={player.stepBack} disabled={player.index === 0}
        aria-label="Previous frame" title="Previous frame (Left arrow)"
        className={`${shape} ${secondary} disabled:opacity-35`}>
        <SkipBack className={icon} />
      </button>
      <button onClick={player.togglePlay}
        aria-label={player.playing ? 'Pause' : 'Play'} title={player.playing ? 'Pause (Space)' : 'Play (Space)'}
        className={`${shape} ${primary} ${mobile ? 'px-4' : ''}`}>
        {player.playing ? <Pause className={icon} /> : <Play className={icon} />}
      </button>
      <button onClick={player.stepForward} disabled={player.atEnd}
        aria-label="Next frame" title="Next frame (Right arrow)"
        className={`${shape} ${secondary} disabled:opacity-35`}>
        <SkipForward className={icon} />
      </button>
      <button onClick={player.cycleSpeed} aria-label={`Playback speed ${player.speed}x`} title="Playback speed"
        className={`${shape} ${secondary} ${mobile ? 'px-2.5 gap-1 text-xs' : 'w-auto px-1.5 gap-1 text-[11px]'} font-display font-semibold tabular-nums`}>
        <Gauge className={mobile ? 'w-4 h-4' : 'w-3 h-3'} />
        {player.speed}&times;
      </button>
    </>
  );
}

function PerspectiveSwitcher({ players, perspective, onSelect, mobile = false }: {
  players: { username: string; avatarSeed: number }[];
  perspective: number;
  onSelect: (seat: number) => void;
  mobile?: boolean;
}) {
  return (
    <div className={`flex items-center gap-0.5 min-w-0 ${mobile ? 'overflow-x-auto' : ''}`}>
      {players.map((seat, index) => (
        <button
          key={`${seat.username}-${index}`}
          onClick={() => onSelect(index)}
          title={`Watch as ${seat.username}`}
          aria-label={`Watch as ${seat.username}`}
          aria-pressed={index === perspective}
          className={`${mobile ? 'min-h-11 min-w-9' : 'w-7 h-7'} flex items-center justify-center rounded-lg transition touch-manipulation flex-shrink-0 ${
            index === perspective ? '' : 'opacity-55 hover:opacity-90'
          }`}
        >
          <Avatar seed={seat.avatarSeed} size={mobile ? 26 : 20} highlight={index === perspective} />
        </button>
      ))}
    </div>
  );
}

function useMobileLayout(): boolean {
  const [isMobile, setIsMobile] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia('(max-width: 767px)').matches);

  useEffect(() => {
    const media = window.matchMedia('(max-width: 767px)');
    const update = () => setIsMobile(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  return isMobile;
}
