import type { Player, Cost } from '../types';

export function getRewardCounts(player: Player): Cost {
  const d: Cost = [0, 0, 0, 0, 0];
  for (const card of player.cards) d[card.reward]++;
  return d;
}

export function totalGems(player: Player): number {
  return player.gems.reduce((a, b) => a + b, 0);
}
