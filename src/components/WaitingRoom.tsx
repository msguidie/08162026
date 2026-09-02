import React from 'react';
import { Bot } from 'lucide-react';
import useGameStore from '../store/gameStore';
import Avatar from './Avatar';
import type { LobbyPlayer, TeamId } from '../types';

export default function WaitingRoom() {
  const {
    lobbyPlayers, lobbyTeamMode, lobbyTeamFormat, lobbyTeamLayout, lobbyTeamSeats, lobbyUnlimitedTime,
    myAccount, toggleReady, toggleGoFirst, toggleTeamMode, toggleTeamLayout, toggleUnlimitedTime,
    selectTeamSeat, leaveLobby,
    lobbyAiAvailable, addAI, removeAI, selectTeamSeatFor,
  } = useGameStore();

  const me = lobbyPlayers.find(p => p.username === myAccount?.username);
  const activeSeatValues = lobbyTeamFormat === 'ONE_V_TWO'
    ? [lobbyTeamSeats[0][0], lobbyTeamSeats[1][0], lobbyTeamSeats[1][1]]
    : lobbyTeamSeats.flat();
  const seatedNames = activeSeatValues.filter((name): name is string => !!name);
  const meIsSeated = !!myAccount && seatedNames.includes(myAccount.username);
  const requiredSeatCount = lobbyTeamFormat === 'ONE_V_TWO' ? 3 : 4;
  const allSeatsFilled = seatedNames.length === requiredSeatCount && new Set(seatedNames).size === requiredSeatCount;
  const allReady = lobbyTeamMode
    ? lobbyPlayers.length === requiredSeatCount && allSeatsFilled && lobbyPlayers.every(p => p.ready)
    : lobbyPlayers.length >= 2 && lobbyPlayers.every(p => p.ready);
  const waitingCount = lobbyPlayers.filter(p => !p.ready).length;
  const emptySeatCount = requiredSeatCount - seatedNames.length;
  // ── AI bots (docs/AI_BRIDGE.md §3) ──
  const bots = lobbyPlayers.filter(p => p.isAI);
  const unseatedBots = bots.filter(bot => !seatedNames.includes(bot.username));
  const lobbyCapacity = lobbyTeamMode ? requiredSeatCount : 6;
  const canAddAI = lobbyAiAvailable && lobbyPlayers.length < lobbyCapacity && bots.length < 4;
  const seatFirstUnseatedBot = (teamId: TeamId, seatIndex: 0 | 1) => {
    const bot = unseatedBots[0];
    if (bot) selectTeamSeatFor(teamId, seatIndex, bot.username);
  };

  const canShowTeamToggle = lobbyPlayers.length === 3 || lobbyPlayers.length === 4 || lobbyTeamMode;
  const canShowTimeToggle = lobbyPlayers.length === 3 || lobbyPlayers.length === 4 || lobbyTeamMode;
  const isOneVsTwo = lobbyTeamFormat === 'ONE_V_TWO';

  return (
    <div className="min-h-[100dvh] flex items-center justify-center p-2 sm:p-4">
      <div className={`bg-white/60 backdrop-blur-md rounded-2xl p-4 sm:p-8 space-y-4 sm:space-y-5 border border-white/70 shadow-lg transition-all max-h-[calc(100dvh-1rem)] overflow-y-auto ${
        lobbyTeamMode ? 'w-[34rem] max-w-full' : 'w-96 max-w-full'
      }`}>
        <h2 className="text-xl font-display font-bold text-center text-slate-800">Lobby</h2>

        {!isOneVsTwo && (
          <div className="flex items-center justify-between gap-3 rounded-xl bg-white/45 border border-white/60 px-4 py-2.5">
            <div className="min-w-0">
              <div className="text-sm font-display font-semibold text-slate-700">I WANT TO GO FIRST</div>
              <div className="text-[10px] leading-tight text-slate-400">
                Multiple volunteers are resolved randomly; otherwise everyone is eligible
              </div>
            </div>
            <Switch checked={!!me?.wantsFirst} onClick={toggleGoFirst} label="Toggle first-player preference" />
          </div>
        )}

        {canShowTeamToggle && (
          <div className="flex items-center justify-between rounded-xl bg-white/45 border border-white/60 px-4 py-3">
            <div>
              <div className="text-sm font-display font-semibold text-slate-700">
                {lobbyPlayers.length === 3 || isOneVsTwo ? '1v2 Mode' : 'Team Mode'}
              </div>
              <div className="text-[10px] text-slate-400">
                Shared setting for this {lobbyPlayers.length === 3 || isOneVsTwo ? 'three' : 'four'}-player lobby
              </div>
            </div>
            <Switch
              checked={lobbyTeamMode}
              onClick={toggleTeamMode}
              label={lobbyPlayers.length === 3 || isOneVsTwo ? 'Toggle 1v2 mode' : 'Toggle team mode'}
            />
          </div>
        )}

        {canShowTimeToggle && (
          <div className="flex items-center justify-between rounded-xl bg-white/45 border border-white/60 px-4 py-3">
            <div>
              <div className="text-sm font-display font-semibold text-slate-700">Unlimited Time</div>
              <div className="text-[10px] text-slate-400">Otherwise each player uses a 3 min + 10 sec clock</div>
            </div>
            <Switch
              checked={lobbyUnlimitedTime}
              onClick={toggleUnlimitedTime}
              label="Toggle unlimited time"
            />
          </div>
        )}

        {lobbyTeamMode ? (
          <div className="space-y-3">
            {!isOneVsTwo && (
              <button
                type="button"
                role="switch"
                aria-checked={lobbyTeamLayout === 'OPPOSITE'}
                onClick={toggleTeamLayout}
                className="w-full rounded-xl bg-white/45 border border-white/60 p-1 flex items-center text-xs font-display transition"
                title="Choose whether teammates sit next to or across from each other"
              >
                <span className={`flex-1 py-2 rounded-lg transition ${
                  lobbyTeamLayout === 'ADJACENT' ? 'bg-white text-[#5B8C6A] shadow-sm font-semibold' : 'text-slate-400'
                }`}>Adjacent</span>
                <span className={`flex-1 py-2 rounded-lg transition ${
                  lobbyTeamLayout === 'OPPOSITE' ? 'bg-white text-[#7B6FA0] shadow-sm font-semibold' : 'text-slate-400'
                }`}>Opposite</span>
              </button>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {([0, 1] as TeamId[]).map(teamId => (
                <TeamCard
                  key={teamId}
                  teamId={teamId}
                  seats={lobbyTeamSeats[teamId]}
                  title={isOneVsTwo ? (teamId === 0 ? 'Solo' : 'Duo') : `Team ${teamId === 0 ? 'A' : 'B'}`}
                  seatCount={isOneVsTwo && teamId === 0 ? 1 : 2}
                  players={lobbyPlayers}
                  myUsername={myAccount?.username}
                  onSelectSeat={selectTeamSeat}
                  onSeatAI={seatFirstUnseatedBot}
                  onRemoveAI={removeAI}
                  canSeatAI={unseatedBots.length > 0}
                />
              ))}
            </div>
            <p className="text-[10px] text-slate-400 text-center">
              {isOneVsTwo
                ? 'The solo player always goes first. Solo needs 15 points; the duo needs 32 combined.'
                : lobbyTeamLayout === 'ADJACENT'
                  ? 'Teammates will sit next to each other around the board.'
                  : 'Teammates will sit across from each other around the board.'}
            </p>
          </div>
        ) : (
          <div className="space-y-1.5">
            {lobbyPlayers.map(p => (
              <div
                key={p.socketId}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition border ${
                  p.username === myAccount?.username
                    ? 'bg-white/70 border-[#7EA68A]/30'
                    : 'bg-white/40 border-white/50'
                }`}
              >
                <Avatar seed={p.avatarSeed} size={36} />
                <div className="flex-1 min-w-0 flex items-center gap-1.5">
                  <span className="font-medium text-sm text-slate-700 truncate">{p.username}</span>
                  {p.isAI && <BotBadge />}
                </div>
                <div className="flex flex-col items-end gap-0.5 flex-shrink-0">
                  {p.wantsFirst && <span className="text-[9px] font-display font-semibold text-amber-600">First</span>}
                  {p.isAI ? (
                    <button
                      type="button"
                      onClick={() => removeAI(p.username)}
                      className="text-[10px] font-display font-semibold text-slate-400 hover:text-rose-500 transition"
                    >
                      Remove
                    </button>
                  ) : (
                    <span className={`text-xs font-medium ${p.ready ? 'text-[#5B8C6A]' : 'text-slate-400'}`}>
                      {p.ready ? '✓ Ready' : 'Not Ready'}
                    </span>
                  )}
                </div>
              </div>
            ))}
            {lobbyPlayers.length === 0 && (
              <p className="text-slate-400 text-center py-4 text-sm">Waiting for players...</p>
            )}
          </div>
        )}

        {lobbyAiAvailable && (
          <button
            type="button"
            onClick={addAI}
            disabled={!canAddAI}
            title={canAddAI ? 'Add an AI player to this lobby' : 'This lobby has no room for another AI player'}
            className={`w-full py-2.5 rounded-xl border border-dashed text-sm font-display font-semibold flex items-center justify-center gap-2 transition ${
              canAddAI
                ? 'border-[#7B6FA0]/40 text-[#7B6FA0] hover:bg-white/60 hover:border-[#7B6FA0]/70'
                : 'border-white/60 text-slate-300 cursor-not-allowed'
            }`}
          >
            <Bot size={16} />
            Add AI
          </button>
        )}

        <div className="text-center min-h-[20px]">
          {allReady ? (
            <span className="text-amber-600 font-display text-sm font-semibold animate-pulse">Starting game...</span>
          ) : lobbyTeamMode && emptySeatCount > 0 ? (
            <span className="text-slate-400 text-xs">Choose the remaining {emptySeatCount} team seat{emptySeatCount > 1 ? 's' : ''}</span>
          ) : lobbyPlayers.length < 2 ? (
            <span className="text-slate-400 text-xs">Need at least 2 players</span>
          ) : waitingCount > 0 ? (
            <span className="text-slate-400 text-xs">Waiting for {waitingCount} player{waitingCount > 1 ? 's' : ''} to ready up</span>
          ) : null}
        </div>

        <div className="space-y-2">
          <button
            onClick={toggleReady}
            disabled={lobbyTeamMode && !meIsSeated}
            className={`w-full py-3 rounded-xl font-semibold transition text-sm shadow-sm ${
              lobbyTeamMode && !meIsSeated
                ? 'bg-white/30 text-slate-300 cursor-not-allowed'
                : me?.ready
                  ? 'bg-slate-200 hover:bg-slate-300 text-slate-600'
                  : 'bg-[#7EA68A] hover:bg-[#6B9477] text-white'
            }`}
          >
            {lobbyTeamMode && !meIsSeated ? 'Choose a seat first' : me?.ready ? 'Cancel Ready' : 'Ready'}
          </button>
          <button onClick={leaveLobby} className="w-full py-2 text-slate-400 hover:text-slate-600 text-sm">Leave Lobby</button>
        </div>

        <p className="text-[10px] text-slate-400 text-center">
          {lobbyPlayers.length} / {lobbyTeamMode ? requiredSeatCount : 6} players
        </p>
      </div>
    </div>
  );
}

function Switch({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-label={label}
      aria-checked={checked}
      onClick={onClick}
      className="w-11 h-11 sm:h-6 flex-shrink-0 flex items-center justify-center touch-manipulation"
    >
      <span className={`relative block w-11 h-6 rounded-full transition-colors ${checked ? 'bg-[#7EA68A]' : 'bg-slate-300'}`}>
        <span className={`absolute left-1 top-1 w-4 h-4 bg-white rounded-full shadow-sm transition-transform ${
          checked ? 'translate-x-5' : 'translate-x-0'
        }`} />
      </span>
    </button>
  );
}

function BotBadge() {
  return (
    <span
      title="AI player"
      className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md bg-[#7B6FA0]/10 text-[#7B6FA0] text-[9px] font-display font-semibold flex-shrink-0"
    >
      <Bot size={10} />
      AI
    </span>
  );
}

function TeamCard({
  teamId, seats, title, seatCount, players, myUsername, onSelectSeat, onSeatAI, onRemoveAI, canSeatAI,
}: {
  teamId: TeamId;
  seats: [string | null, string | null];
  title: string;
  seatCount: 1 | 2;
  players: LobbyPlayer[];
  myUsername?: string;
  onSelectSeat: (teamId: TeamId, seatIndex: 0 | 1) => void;
  onSeatAI: (teamId: TeamId, seatIndex: 0 | 1) => void;
  onRemoveAI: (username: string) => void;
  canSeatAI: boolean;
}) {
  const accent = teamId === 0 ? '#5B8C6A' : '#7B6FA0';
  return (
    <div className="rounded-xl bg-white/40 border border-white/60 p-3 space-y-2">
      <div className="text-xs font-display font-bold text-center" style={{ color: accent }}>{title}</div>
      {seats.slice(0, seatCount).map((username, seatIndex) => {
        const player = username ? players.find(p => p.username === username) : undefined;
        const isMe = username === myUsername;
        const unavailable = !!username && !isMe;

        // A bot seat is a plain box (it never reacts to a click) with its own
        // Remove control, so no button is ever nested inside another.
        if (player?.isAI) {
          return (
            <div
              key={seatIndex}
              className="w-full min-h-[58px] rounded-xl border border-white/60 bg-white/55 flex items-center gap-2 px-3 py-2"
              title={`${player.username} occupies this seat`}
            >
              <Avatar seed={player.avatarSeed} size={32} />
              <span className="flex-1 min-w-0 flex items-center gap-1 text-left">
                <span className="text-xs font-medium text-slate-700 truncate">{player.username}</span>
                <BotBadge />
              </span>
              <button
                type="button"
                onClick={() => onRemoveAI(player.username)}
                className="text-[10px] font-display font-semibold text-slate-400 hover:text-rose-500 transition flex-shrink-0"
              >
                Remove
              </button>
            </div>
          );
        }

        const seatButton = (
          <button
            type="button"
            disabled={unavailable}
            onClick={() => onSelectSeat(teamId, seatIndex as 0 | 1)}
            className={`w-full min-h-[58px] rounded-xl border flex items-center gap-2 px-3 py-2 transition ${
              isMe
                ? 'bg-white/80 ring-2 ring-[#7EA68A]/30'
                : player
                  ? 'bg-white/55 cursor-default'
                  : 'bg-white/25 border-dashed hover:bg-white/55 hover:border-[#7EA68A]/50'
            }`}
            style={{ borderColor: isMe ? accent : undefined }}
            title={isMe ? 'Click to leave this seat' : player ? `${player.username} occupies this seat` : 'Join this team seat'}
          >
            {player ? (
              <>
                <Avatar seed={player.avatarSeed} size={32} />
                <span className="flex-1 text-left text-xs font-medium text-slate-700 truncate">{player.username}</span>
                <span className="flex flex-col items-end gap-0.5">
                  {player.wantsFirst && <span className="text-[8px] font-display font-semibold text-amber-600">First</span>}
                  <span className={`text-[10px] ${player.ready ? 'text-[#5B8C6A]' : 'text-slate-400'}`}>
                    {player.ready ? 'Ready' : 'Waiting'}
                  </span>
                </span>
              </>
            ) : (
              <span className="w-full text-center text-2xl font-light text-slate-300">+</span>
            )}
          </button>
        );

        // An empty seat also offers to seat a waiting bot (it cannot click).
        if (!player && canSeatAI) {
          return (
            <div key={seatIndex} className="relative">
              {seatButton}
              <button
                type="button"
                onClick={() => onSeatAI(teamId, seatIndex as 0 | 1)}
                title="Seat an AI player here"
                className="absolute right-2 top-1/2 -translate-y-1/2 px-2 py-1 rounded-lg bg-white/70 border border-white/70 text-[9px] font-display font-semibold text-[#7B6FA0] hover:border-[#7B6FA0]/50 hover:bg-white transition flex items-center gap-1"
              >
                <Bot size={10} />
                AI
              </button>
            </div>
          );
        }

        return <React.Fragment key={seatIndex}>{seatButton}</React.Fragment>;
      })}
    </div>
  );
}
