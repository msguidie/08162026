import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Player, GameState, ActionMode, GameMode } from '../types';
import { GEM_COLORS_HEX, GEM_COLORS_LIGHT, GEM_NAMES, GOLD_HEX } from '../types';
import { getRewardCounts } from '../utils/gameHelpers';
import Avatar from './Avatar';
import CardView from './CardView';

interface PlayerInfoProps {
  player: Player;
  isMe: boolean;
  isCurrentTurn: boolean;
  mobile?: boolean;
  compact?: boolean;
  isPulsing?: boolean;
  isDisconnected?: boolean;
  clockLabel?: string;
  clockUrgent?: boolean;
  gameMode?: GameMode;
  // For +1 animation: map from color index to remaining display time
  cardDeltas?: Record<number, boolean>;
  // For self panel: action buttons and interaction
  actionMode?: ActionMode;
  gameState?: GameState;
  playerIndex?: number;
  onSetActionMode?: (mode: ActionMode) => void;
  onClickCard?: (card: { id: number; tier: number; reward: number; points: number; cost: number[] }, source: 'board' | 'reserved') => void;
  onConfirmGems?: () => void;
  canConfirmGems?: boolean;
  isConfirmingGems?: boolean;
}

function MiniCardTile({ color, size = 'md' }: { color: number; size?: 'xs' | 'sm' | 'md' }) {
  const w = size === 'xs' ? 10 : size === 'sm' ? 16 : 20;
  const h = size === 'xs' ? 14 : size === 'sm' ? 20 : 26;
  return (
    <div
      style={{
        width: w, height: h, borderRadius: 3,
        background: `linear-gradient(to top right, ${GEM_COLORS_LIGHT[color]}, ${GEM_COLORS_HEX[color]})`,
        boxShadow: '0 1px 2px rgba(0,0,0,0.12)',
      }}
    />
  );
}

export default function PlayerInfo({
  player, isMe, isCurrentTurn, mobile = false, compact = false, isPulsing = false,
  isDisconnected = false, clockLabel, clockUrgent = false, gameMode = 'INDIVIDUAL', cardDeltas = {},
  actionMode, gameState, playerIndex, onSetActionMode, onClickCard,
  onConfirmGems, canConfirmGems = false, isConfirmingGems = false,
}: PlayerInfoProps) {
  const rewards = getRewardCounts(player);

  if (!isMe) {
    // ── Opponent panel ──
    return (
      <motion.div
        animate={isPulsing ? { scale: [1, 1.04, 1], transition: { duration: 0.5 } } : {}}
        className={`${mobile ? 'rounded-lg px-1 py-1' : 'rounded-xl px-4 py-2.5'} transition-all border shadow-sm w-full ${
          isCurrentTurn
            ? 'ring-2 ring-amber-500/70 bg-white/70 border-amber-400/30'
            : 'bg-white/40 border-white/60'
        }`}
      >
        {mobile ? (
          <div className="mb-1 min-w-0">
            <div className="flex items-start gap-1 min-w-0">
              <div className="relative flex-shrink-0">
                <Avatar seed={player.avatarSeed} size={20} highlight={isCurrentTurn} />
                {isDisconnected && (
                  <div className="absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full bg-red-400 border border-white" title="Disconnected" />
                )}
              </div>
              <div className={`min-w-0 flex-1 text-[8px] font-medium break-all leading-[9px] ${isDisconnected ? 'text-slate-400' : 'text-slate-700'}`}>
                {player.username}
              </div>
              {player.teamId !== undefined && <TeamBadge teamId={player.teamId} gameMode={gameMode} mobile />}
            </div>
            <div className="mt-0.5 flex items-center justify-between gap-0.5 font-display leading-none whitespace-nowrap">
              <span className="text-[10px] font-bold text-amber-600">{player.score} pts</span>
              {clockLabel && (
                <span className={`text-[8px] tabular-nums ${clockUrgent ? 'text-red-500 font-bold animate-pulse' : 'text-slate-400'}`}>
                  {clockLabel}
                </span>
              )}
              <span className="text-[8px] text-[#7B6FA0]">★{player.bonusTiles.length}</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center min-w-0 gap-2 mb-2">
            <div className="relative flex-shrink-0">
              <Avatar seed={player.avatarSeed} size={28} highlight={isCurrentTurn} />
              {isDisconnected && (
                <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-red-400 border border-white" title="Disconnected" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1 min-w-0">
                <div className={`text-xs font-medium truncate leading-tight ${isDisconnected ? 'text-slate-400' : 'text-slate-700'}`}>
                  {player.username}
                </div>
                {player.teamId !== undefined && <TeamBadge teamId={player.teamId} gameMode={gameMode} />}
              </div>
            </div>
            {clockLabel && (
              <div className={`text-[10px] font-display tabular-nums whitespace-nowrap ${
                clockUrgent ? 'text-red-500 font-bold animate-pulse' : 'text-slate-400'
              }`}>{clockLabel}</div>
            )}
            <div className="text-base font-display font-bold text-amber-600 leading-none">{player.score}</div>
            {player.bonusTiles.length > 0 && (
              <div className="text-[10px] text-[#7B6FA0] font-display leading-none">+{player.bonusTiles.length * 3}</div>
            )}
          </div>
        )}

        <div className={`grid ${mobile ? 'gap-y-px gap-x-0' : 'gap-y-[4px] gap-x-2'}`} style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
          {/* Row 1: gem dots */}
          {[0, 1, 2, 3, 4].map(i => (
            <div key={`dot${i}`} className="flex justify-center">
              <div className={`${mobile ? 'w-2.5 h-2.5' : 'w-4 h-4'} rounded-full ring-1 ring-black/5`} style={{ backgroundColor: GEM_COLORS_HEX[i] }} title={GEM_NAMES[i]} />
            </div>
          ))}
          <div className="flex justify-center">
            <div className={`${mobile ? 'w-2.5 h-2.5' : 'w-4 h-4'} rounded-full ring-1 ring-black/5`} style={{ backgroundColor: GOLD_HEX }} title="Gold" />
          </div>

          {/* Row 2: gem counts */}
          {[0, 1, 2, 3, 4].map(i => (
            <div key={`gcn${i}`} className={`text-center ${mobile ? 'text-[9px] leading-none' : 'text-xs'} font-display ${player.gems[i] > 0 ? 'text-slate-700 font-medium' : 'text-slate-300'}`}>
              {player.gems[i]}
            </div>
          ))}
          <div className={`text-center ${mobile ? 'text-[9px] leading-none' : 'text-xs'} font-display ${player.gems[5] > 0 ? 'text-amber-600 font-medium' : 'text-slate-300'}`}>
            {player.gems[5]}
          </div>

          {/* Row 3: card tiles */}
          {[0, 1, 2, 3, 4].map(i => (
            <div key={`ct${i}`} className="flex justify-center items-end py-px">
              <MiniCardTile color={i} size={mobile ? 'xs' : 'sm'} />
            </div>
          ))}
          <div className="flex justify-center items-center">
            {mobile || player.reserved.length > 0 ? (
              <span className={`${mobile ? 'text-[8px]' : 'text-[10px]'} text-slate-400 font-display font-medium whitespace-nowrap`}>{player.reserved.length}R</span>
            ) : null}
          </div>

          {/* Row 4: card counts with +1 animation */}
          {[0, 1, 2, 3, 4].map(i => (
            <div key={`ccn${i}`} className="text-center relative">
              <span className={`${mobile ? 'text-[9px] leading-none' : 'text-xs'} font-display font-medium ${rewards[i] > 0 ? 'text-slate-700' : 'text-slate-300'}`}>
                {rewards[i] || '·'}
              </span>
              <AnimatePresence>
                {cardDeltas[i] && (
                  <motion.span
                    initial={{ opacity: 1, y: 0, scale: 1.2 }}
                    animate={{ opacity: 0, y: -14, scale: 0.8 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 2.2, ease: 'easeOut' }}
                    className="absolute -top-1 left-1/2 -translate-x-1/2 text-[10px] font-bold text-green-600 pointer-events-none whitespace-nowrap"
                  >
                    +1
                  </motion.span>
                )}
              </AnimatePresence>
            </div>
          ))}
          <div className="flex justify-center items-center">
            {mobile ? (
              <span className="text-[8px] text-[#7B6FA0] font-display whitespace-nowrap">★{player.bonusTiles.length}</span>
            ) : player.bonusTiles.length > 0 ? (
              <span className="text-[10px] text-[#7B6FA0] font-display">★</span>
            ) : null}
          </div>
        </div>
      </motion.div>
    );
  }

  // ── Self panel ──
  const myTurn = isMe && isCurrentTurn;
  const pendingTileChoice = false; // handled at GameBoard level

  if (mobile) {
    const maxReserved = gameState?.config.maxReserved ?? 3;
    const reserveLocked = actionMode === 'RESERVE';

    return (
      <div className={`rounded-lg px-1.5 py-1.5 transition-all border shadow-sm ${
        isCurrentTurn
          ? 'ring-2 ring-amber-500/70 bg-white/75 border-amber-400/30'
          : 'bg-white/45 border-white/60'
      }`}>
        <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-1 items-center">
          <div className="min-w-0 flex items-center gap-1.5">
            <Avatar seed={player.avatarSeed} size={30} highlight={isCurrentTurn} />
            <div className="min-w-0">
              <div className="flex items-center gap-1 min-w-0">
                <span className="text-[10px] font-medium break-all leading-tight text-slate-700">{player.username}</span>
                {player.teamId !== undefined && <TeamBadge teamId={player.teamId} gameMode={gameMode} mobile />}
              </div>
              <div className="flex items-center gap-1.5 text-[9px] font-display whitespace-nowrap">
                <span className="text-amber-600 font-bold">{player.score} pts</span>
                <span className="text-[#7B6FA0]">★{player.bonusTiles.length}</span>
                {clockLabel && (
                  <span className={`tabular-nums ${clockUrgent ? 'text-red-500 font-bold animate-pulse' : 'text-slate-400'}`}>
                    {clockLabel}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="min-w-0">
            <div className="text-[8px] leading-none text-right text-slate-400 uppercase tracking-wide mb-0.5">
              Reserved {player.reserved.length}/{maxReserved}
            </div>
            <div className={`flex justify-end gap-0.5 ${player.reserved.length > 0 ? 'min-h-[61px]' : 'min-h-0'}`}>
              {player.reserved.map(card => (
                <CardView
                  key={card.id}
                  card={card as any}
                  size="xs"
                  clickable={myTurn && actionMode === 'BUY'}
                  onClick={() => onClickCard?.(card as any, 'reserved')}
                />
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 mt-1 border-t border-white/60 pt-1">
          <div className="min-w-0">
            <div className="text-[8px] leading-none text-slate-400 uppercase tracking-wide mb-0.5">Cards</div>
            <div className="grid grid-cols-5 gap-x-1">
              {[0, 1, 2, 3, 4].map(i => (
                <div key={`mcd${i}`} className="flex justify-center"><MiniCardTile color={i} size="xs" /></div>
              ))}
              {[0, 1, 2, 3, 4].map(i => (
                <div key={`mcc${i}`} className={`text-center text-[10px] leading-none font-display font-medium ${rewards[i] > 0 ? 'text-slate-700' : 'text-slate-400'}`}>
                  {rewards[i]}
                </div>
              ))}
            </div>
          </div>

          <div className="min-w-0">
            <div className="text-[8px] leading-none text-slate-400 uppercase tracking-wide mb-0.5">Gems</div>
            <div className="grid grid-cols-6 gap-x-0.5">
              {[0, 1, 2, 3, 4].map(i => (
                <div key={`mgd${i}`} className="flex justify-center">
                  <div className="w-3 h-3 rounded-full ring-1 ring-black/5" style={{ backgroundColor: GEM_COLORS_HEX[i] }} title={GEM_NAMES[i]} />
                </div>
              ))}
              <div className="flex justify-center">
                <div className="w-3 h-3 rounded-full ring-1 ring-black/5" style={{ backgroundColor: GOLD_HEX }} title="Gold" />
              </div>
              {player.gems.map((count, i) => (
                <div key={`mgc${i}`} className={`text-center text-[10px] leading-none font-display font-medium ${
                  i === 5 ? (count > 0 ? 'text-amber-600' : 'text-slate-400') : (count > 0 ? 'text-slate-700' : 'text-slate-400')
                }`}>
                  {count}
                </div>
              ))}
            </div>
          </div>
        </div>

        {myTurn && onSetActionMode && !pendingTileChoice && (
          <div className="grid grid-cols-3 gap-1 mt-1 border-t border-white/60 pt-1">
            <ActionBtn mobile
              label={actionMode === 'TAKE_GEMS'
                ? (isConfirmingGems ? 'Confirming…' : 'Confirm')
                : 'Take Gems'}
              active={actionMode === 'TAKE_GEMS'}
              disabled={reserveLocked || (actionMode === 'TAKE_GEMS' && (!canConfirmGems || isConfirmingGems))}
              onClick={() => actionMode === 'TAKE_GEMS'
                ? onConfirmGems?.()
                : onSetActionMode('TAKE_GEMS')} />
            <ActionBtn mobile label="Reserve" active={reserveLocked}
              disabled={reserveLocked || isConfirmingGems || player.reserved.length >= maxReserved}
              onClick={() => onSetActionMode('RESERVE')} />
            <ActionBtn mobile label="Buy Card" active={actionMode === 'BUY'}
              disabled={reserveLocked || isConfirmingGems}
              onClick={() => onSetActionMode(actionMode === 'BUY' ? null : 'BUY')} />
            {reserveLocked && (
              <div className="col-span-3 text-center text-[9px] leading-none text-amber-600/80 font-display">
                Pick a card or deck to reserve
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`rounded-xl px-3 py-2.5 transition-all border shadow-sm ${
      isCurrentTurn
        ? 'ring-2 ring-amber-500/70 bg-white/70 border-amber-400/30'
        : 'bg-white/40 border-white/60'
    }`}>
      <div className="flex items-center gap-3 justify-center flex-wrap">
        {/* Avatar + Score */}
        <div className="flex flex-col items-center flex-shrink-0">
          <Avatar seed={player.avatarSeed} size={40} highlight={isCurrentTurn} />
          <div className="flex items-center gap-1 mt-0.5 max-w-[90px]">
            <div className="text-xs font-medium truncate text-slate-700">{player.username}</div>
            {player.teamId !== undefined && <TeamBadge teamId={player.teamId} gameMode={gameMode} />}
          </div>
          <div className="text-sm font-display">
            <span className="text-amber-600 font-bold">{player.score}</span>
            {player.bonusTiles.length > 0 && (
              <span className="text-[#7B6FA0] ml-0.5 text-[11px]">+{player.bonusTiles.length * 3}</span>
            )}
          </div>
          {clockLabel && (
            <div className={`text-[10px] font-display tabular-nums leading-none ${
              clockUrgent ? 'text-red-500 font-bold animate-pulse' : 'text-slate-400'
            }`}>{clockLabel}</div>
          )}
        </div>

        <div className="w-px h-12 bg-slate-200 flex-shrink-0" />

        {/* Cards 2×5 */}
        <div className="flex-shrink-0">
          <div className="text-[9px] text-slate-400 uppercase tracking-wider mb-0.5">Cards</div>
          <div className="grid gap-x-3" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
            {[0, 1, 2, 3, 4].map(i => (
              <div key={`cd${i}`} className="flex justify-center"><MiniCardTile color={i} /></div>
            ))}
            {[0, 1, 2, 3, 4].map(i => (
              <div key={`cc${i}`} className={`text-center text-sm font-display font-medium ${rewards[i] > 0 ? 'text-slate-700' : 'text-slate-400'}`}>
                {rewards[i] || '·'}
              </div>
            ))}
          </div>
        </div>

        <div className="w-px h-12 bg-slate-200 flex-shrink-0" />

        {/* Gems 2×6 */}
        <div className="flex-shrink-0">
          <div className="text-[9px] text-slate-400 uppercase tracking-wider mb-0.5">Gems</div>
          <div className="grid gap-x-2.5" style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
            {[0, 1, 2, 3, 4].map(i => (
              <div key={`gd${i}`} className="flex justify-center">
                <div className="w-5 h-5 rounded-full ring-1 ring-black/5" style={{ backgroundColor: GEM_COLORS_HEX[i] }} title={GEM_NAMES[i]} />
              </div>
            ))}
            <div className="flex justify-center">
              <div className="w-5 h-5 rounded-full ring-1 ring-black/5" style={{ backgroundColor: GOLD_HEX }} />
            </div>
            {[0, 1, 2, 3, 4].map(i => (
              <div key={`gc${i}`} className={`text-center text-base font-display font-bold ${player.gems[i] > 0 ? 'text-slate-700' : 'text-slate-400'}`}>
                {player.gems[i] || '·'}
              </div>
            ))}
            <div className={`text-center text-base font-display font-bold ${player.gems[5] > 0 ? 'text-amber-600' : 'text-slate-400'}`}>
              {player.gems[5] || '·'}
            </div>
          </div>
        </div>

        {/* Reserved cards */}
        {player.reserved.length > 0 && (
          <>
            <div className="w-px h-12 bg-slate-200 flex-shrink-0" />
            <div className="flex-shrink-0">
              <div className="text-[9px] text-slate-400 uppercase tracking-wider mb-0.5">
                Reserved {player.reserved.length}/{gameState?.config.maxReserved ?? 3}
              </div>
              <div className="flex gap-1">
                {player.reserved.map(card => (
                  <CardView
                    key={card.id}
                    card={card as any}
                    size="sm"
                    clickable={myTurn && actionMode === 'BUY'}
                    onClick={() => onClickCard?.(card as any, 'reserved')}
                  />
                ))}
              </div>
            </div>
          </>
        )}

        {/* Action buttons integrated into panel */}
        {myTurn && onSetActionMode && !pendingTileChoice && actionMode !== 'RESERVE' && (
          <>
            <div className="w-px h-12 bg-slate-200 flex-shrink-0" />
            <div className="flex flex-col gap-1 flex-shrink-0">
              <ActionBtn label="Take Gems" active={actionMode === 'TAKE_GEMS'} onClick={() => onSetActionMode(actionMode === 'TAKE_GEMS' ? null : 'TAKE_GEMS')} />
              <ActionBtn label="Reserve" active={false} onClick={() => onSetActionMode('RESERVE')} disabled={player.reserved.length >= (gameState?.config.maxReserved ?? 3)} />
              <ActionBtn label="Buy Card" active={actionMode === 'BUY'} onClick={() => onSetActionMode(actionMode === 'BUY' ? null : 'BUY')} />
            </div>
          </>
        )}
        {actionMode === 'RESERVE' && (
          <>
            <div className="w-px h-12 bg-slate-200 flex-shrink-0" />
            <div className="text-xs text-amber-600/80 font-display flex-shrink-0">Pick a card<br/>to reserve...</div>
          </>
        )}
      </div>
    </div>
  );
}

function ActionBtn({ label, active, onClick, disabled, mobile = false }: {
  label: string; active: boolean; onClick: () => void; disabled?: boolean; mobile?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${mobile ? 'min-h-11 px-1 py-1 text-[10px]' : 'px-3 py-1 text-[11px]'} rounded-lg font-medium transition font-display shadow-sm touch-manipulation disabled:opacity-40 ${
        active
          ? 'bg-amber-500/20 text-amber-700 border border-amber-400/40'
          : disabled
            ? 'bg-white/30 text-slate-300 cursor-not-allowed'
            : 'bg-white/50 text-slate-600 hover:bg-white/70 border border-white/60'
      }`}
    >
      {label}
    </button>
  );
}

function TeamBadge({ teamId, gameMode, mobile = false }: { teamId: 0 | 1; gameMode: GameMode; mobile?: boolean }) {
  const label = gameMode === 'ONE_V_TWO' ? (teamId === 0 ? 'Solo' : 'Duo') : (teamId === 0 ? 'A' : 'B');
  return (
    <span className={`${mobile ? 'text-[7px] px-0.5' : 'text-[8px] px-1'} leading-none py-0.5 rounded font-display font-bold flex-shrink-0 ${
      teamId === 0 ? 'bg-[#5B8C6A]/15 text-[#5B8C6A]' : 'bg-[#7B6FA0]/15 text-[#7B6FA0]'
    }`}>
      {label}
    </span>
  );
}
