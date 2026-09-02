import type { ReplayActionEntry, ReplayActionResult, ReplayFrame } from '../types';
import { GEM_NAMES } from '../types';

const GOLD_NAME = 'Gold';

function gemName(color: number): string {
  return color === 5 ? GOLD_NAME : (GEM_NAMES[color] ?? 'gem');
}

function describeGems(selected: number[]): string {
  if (!selected.length) return 'took gems';
  const counts = new Map<number, number>();
  for (const color of selected) counts.set(color, (counts.get(color) ?? 0) + 1);
  const parts = [...counts.entries()].map(([color, count]) =>
    count > 1 ? `${count} ${gemName(color)}` : gemName(color));
  return `took ${parts.join(', ')}`;
}

/** Short verb phrase for the action a frame just played, e.g. "bought a 3-point Amber card". */
export function describeAction(
  result: ReplayActionResult | null,
  action: ReplayActionEntry | null,
): string | null {
  if (!result) {
    // Fall back to the stored tuple if a frame carries no ActionResult.
    if (!action) return null;
    const [, code, arg] = action;
    if (code === 'G') return describeGems(Array.isArray(arg) ? arg : []);
    if (code === 'R') return 'reserved a card';
    if (code === 'RD') return `reserved from the tier-${arg} deck`;
    if (code === 'B') return 'bought a card';
    if (code === 'N') return 'chose a noble';
    if (code === 'X') return 'resigned';
    if (code === 'T') return 'timed out';
    return null;
  }

  const payload = (result.payload ?? {}) as Record<string, unknown>;

  switch (result.type) {
    case 'SELECT_GEM':
    case 'TAKE_GEMS_CONFIRMED':
      return describeGems(Array.isArray(payload.selected) ? (payload.selected as number[]) : []);
    case 'RESERVE_CARD':
      return 'reserved a card';
    case 'RESERVE_FROM_DECK':
      return `reserved from the tier-${payload.tier ?? '?'} deck`;
    case 'BUY_CARD': {
      const points = typeof payload.points === 'number' ? payload.points : 0;
      const reward = typeof payload.reward === 'number' ? payload.reward : 0;
      const fromReserve = payload.source === 'reserved' || payload.source === 'r';
      const name = gemName(reward);
      const card = points > 0
        ? `a ${points}-point ${name} card`
        : `${/^[AEIOU]/.test(name) ? 'an' : 'a'} ${name} card`;
      return `bought ${card}${fromReserve ? ' from reserve' : ''}`;
    }
    case 'CHOOSE_TILE':
      return 'chose a noble';
    case 'RESIGN':
      return 'resigned';
    case 'TIMEOUT':
      return 'timed out';
    default:
      return null;
  }
}

/** Whether this frame left the actor choosing between several nobles. */
export function framePendingTiles(frame: ReplayFrame | undefined): number[] | null {
  if (!frame) return null;
  return frame.pendingTileChoice
    ?? frame.state.pendingTileChoice
    ?? frame.state._pendingTileChoice
    ?? null;
}

/** Full caption line, e.g. "Bob took 2 Rose" — or the pending-noble hint. */
export function frameCaption(frame: ReplayFrame | undefined, names: string[]): string | null {
  if (!frame) return null;
  if (framePendingTiles(frame)?.length) {
    const chooser = frame.actor !== null ? names[frame.actor] : null;
    return chooser ? `${chooser} is choosing a noble…` : 'choosing a noble…';
  }
  const phrase = describeAction(frame.result, frame.action);
  if (!phrase) return null;
  const actor = frame.actor ?? frame.result?.actingPlayer ?? null;
  const name = actor !== null ? names[actor] : null;
  const claimed = frame.result?.tileClaimed ? ' · claimed a noble' : '';
  return name ? `${name} ${phrase}${claimed}` : `${phrase}${claimed}`;
}

export function formatReplayDate(ms: number): string {
  if (!ms) return '';
  const d = new Date(ms);
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

export function formatDuration(startMs: number, endMs: number): string | null {
  if (!startMs || !endMs || endMs <= startMs) return null;
  // Short games (bot runs, quick resignations) are seconds, not "1 min".
  const totalSeconds = Math.round((endMs - startMs) / 1000);
  if (totalSeconds < 60) return `${Math.max(1, totalSeconds)} s`;
  const totalMinutes = Math.round(totalSeconds / 60);
  if (totalMinutes < 60) return `${Math.max(1, totalMinutes)} min`;
  const hours = Math.floor(totalMinutes / 60);
  return `${hours} h ${totalMinutes % 60} min`;
}

/** "4P · 2v2", "3P · 1v2", "2P · Individual" */
export function modeLabel(mode: string, n: number): string {
  const suffix = mode === 'TEAM' ? '2v2' : mode === 'ONE_V_TWO' ? '1v2' : 'Individual';
  return `${n}P · ${suffix}`;
}
