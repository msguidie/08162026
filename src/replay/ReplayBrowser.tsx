import React, { useEffect } from 'react';
import { RefreshCw } from 'lucide-react';
import useGameStore from '../store/gameStore';
import type { ReplayIndexEntry } from '../types';
import { formatDuration, formatReplayDate, modeLabel } from './replayText';

export default function ReplayBrowser() {
  const {
    replayList, replayListLoading, replayListError,
    refreshReplayList, openReplay, closeReplayBrowser,
  } = useGameStore();

  // openReplayBrowser() already kicks off a load; this covers a direct mount.
  useEffect(() => {
    if (replayList.length === 0 && !replayListLoading && !replayListError) void refreshReplayList();
  }, []);

  return (
    <div className="min-h-[100dvh] flex items-center justify-center p-2 sm:p-4">
      <div className="bg-white/60 backdrop-blur-md rounded-2xl p-4 sm:p-6 w-[34rem] max-w-full space-y-4 border border-white/70 shadow-lg max-h-[calc(100dvh-1rem)] overflow-hidden flex flex-col">
        <div className="flex items-center justify-between gap-3 flex-shrink-0">
          <h2 className="text-xl font-display font-bold text-slate-800">Replays</h2>
          <button
            onClick={() => void refreshReplayList()}
            disabled={replayListLoading}
            className="min-h-9 px-3 py-1.5 flex items-center gap-1.5 bg-white/60 hover:bg-white/85 border border-white/70 rounded-xl text-xs font-medium text-slate-600 transition disabled:opacity-50 touch-manipulation"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${replayListLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto space-y-2 -mx-1 px-1 max-h-[calc(100dvh-12rem)]">
          {replayListLoading && replayList.length === 0 && (
            <div className="py-10 text-center space-y-2">
              <span className="inline-block animate-spin h-5 w-5 border-2 border-slate-400 border-t-transparent rounded-full" />
              <p className="text-sm text-slate-500">Loading replays...</p>
            </div>
          )}

          {!replayListLoading && replayListError && (
            <div className="py-8 text-center space-y-3">
              <p className="text-sm text-red-500">{replayListError}</p>
              <button
                onClick={() => void refreshReplayList()}
                className="px-4 py-2 bg-[#8B9DAF] hover:bg-[#7A8D9F] text-white rounded-xl text-sm font-semibold transition shadow-sm"
              >
                Try again
              </button>
            </div>
          )}

          {!replayListLoading && !replayListError && replayList.length === 0 && (
            <p className="text-slate-400 text-center text-sm py-10">No finished games recorded yet</p>
          )}

          {replayList.map(game => (
            <ReplayRow key={game.id} game={game} onOpen={() => void openReplay(game.id)} />
          ))}
        </div>

        <button
          onClick={closeReplayBrowser}
          className="w-full min-h-11 py-2.5 text-slate-400 hover:text-slate-600 text-sm transition flex-shrink-0"
        >
          Back
        </button>
      </div>
    </div>
  );
}

function ReplayRow({ game, onOpen }: { game: ReplayIndexEntry; onOpen: () => void }) {
  const winners = winnerSeats(game);

  return (
    <button
      onClick={onOpen}
      className="w-full text-left px-3 py-2.5 bg-white/50 hover:bg-white/85 rounded-xl transition border border-white/60 shadow-sm touch-manipulation"
    >
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <span className="text-[11px] text-slate-400 truncate">{formatReplayDate(game.t)}</span>
        <span className="text-[10px] font-display font-bold px-2 py-0.5 rounded-full bg-[#7EA68A]/15 text-[#5B8C6A] whitespace-nowrap flex-shrink-0">
          {modeLabel(game.mode, game.n)}
        </span>
      </div>

      <div className="flex items-center gap-1.5 flex-wrap">
        {game.players.map((name, seat) => (
          <span
            key={`${game.id}-${seat}`}
            className={`inline-flex items-center gap-1 pl-1 pr-2 py-0.5 rounded-full border text-[11px] max-w-[10rem] ${
              winners.has(seat)
                ? 'bg-amber-50 border-amber-200 text-amber-700 font-medium'
                : 'bg-white/60 border-white/70 text-slate-600'
            }`}
          >
            <span className="w-4 h-4 rounded-full bg-slate-200/90 text-[8px] font-display font-bold text-slate-500 flex items-center justify-center flex-shrink-0">
              {initials(name)}
            </span>
            <span className="truncate">{winners.has(seat) && '★ '}{name}</span>
            {game.ai?.[seat] && (
              <span
                title="AI player"
                className="text-[9px] px-1 rounded bg-[#7B6FA0]/10 text-[#7B6FA0] font-display font-bold flex-shrink-0"
              >
                AI
              </span>
            )}
          </span>
        ))}
      </div>

      <div className="mt-1.5 flex items-center justify-between gap-2 text-[10px] text-slate-400 font-display">
        <span>
          {game.turns} turns
          {formatDuration(game.t, game.e) && <span> &middot; {formatDuration(game.t, game.e)}</span>}
        </span>
        {teamOutcome(game) && <span className="text-[#7B6FA0]">{teamOutcome(game)}</span>}
      </div>
    </button>
  );
}

function initials(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) return '?';
  const parts = trimmed.split(/[\s_-]+/).filter(Boolean);
  if (parts.length > 1) return (parts[0][0] + parts[1][0]).toUpperCase();
  return trimmed.slice(0, 2).toUpperCase();
}

/** Team modes carry the outcome on the side, not on the seats. */
function teamOutcome(game: ReplayIndexEntry): string | null {
  if (game.mode === 'INDIVIDUAL' || !game.winningTeamIds) return null;
  const isOneVsTwo = game.mode === 'ONE_V_TWO';
  const sideName = (teamId: number) => isOneVsTwo
    ? (teamId === 0 ? 'Solo' : 'Duo')
    : `Team ${teamId === 0 ? 'A' : 'B'}`;
  if (game.winningTeamIds.length === 0) return 'No winner';
  if (game.winningTeamIds.length > 1) return 'Draw';
  return `${sideName(game.winningTeamIds[0])} wins`;
}

/** Seats to star: individual winners, or every member of a winning team. */
function winnerSeats(game: ReplayIndexEntry): Set<number> {
  const seats = new Set<number>();
  if (game.winners?.length) {
    for (const seat of game.winners) seats.add(seat);
    return seats;
  }
  // Team modes leave `winners` null, so map the winning side onto seats
  // through the per-seat team ids the index entry carries.
  if (game.winningTeamIds?.length && game.teams) {
    const winningTeams = new Set<number>(game.winningTeamIds);
    game.teams.forEach((team, seat) => {
      if (team != null && winningTeams.has(team)) seats.add(seat);
    });
  }
  return seats;
}
