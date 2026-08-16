import type { Card, BonusTile, Cost, GemColor, Tier } from '../types';

export const ALL_CARDS: Card[] = [];
export const ALL_BONUS_TILES: BonusTile[] = [];

let nextId = 0;

function addCycle(tier: Tier, points: number, template: Cost) {
  for (let i = 0; i < 5; i++) {
    const cost: Cost = [0, 0, 0, 0, 0];
    for (let j = 0; j < 5; j++) {
      const relativePos = (j - i + 5) % 5;
      cost[j] = template[relativePos];
    }
    ALL_CARDS.push({ id: nextId++, tier, reward: i as GemColor, points, cost });
  }
}

// TIER 1: 40 Cards
addCycle(1, 0, [1, 1, 0, 1, 1]);
addCycle(1, 0, [1, 2, 1, 0, 1]);
addCycle(1, 0, [0, 2, 1, 0, 2]);
addCycle(1, 0, [3, 0, 0, 1, 0]);
addCycle(1, 0, [0, 0, 2, 2, 0]);
addCycle(1, 0, [0, 0, 0, 0, 3]);
addCycle(1, 0, [0, 2, 1, 1, 1]);
addCycle(1, 1, [0, 4, 0, 0, 0]);

// TIER 2: 30 Cards
addCycle(2, 3, [6, 0, 0, 0, 0]);
addCycle(2, 1, [0, 2, 0, 3, 3]);
addCycle(2, 2, [0, 0, 0, 0, 5]);
addCycle(2, 2, [0, 4, 0, 1, 3]);
addCycle(2, 2, [0, 1, 4, 2, 0]);
addCycle(2, 3, [0, 0, 0, 0, 6]);

// TIER 3: 20 Cards
addCycle(3, 3, [0, 3, 3, 3, 5]);
addCycle(3, 4, [0, 0, 0, 7, 0]);
addCycle(3, 4, [0, 0, 0, 3, 6]);
addCycle(3, 5, [0, 3, 0, 0, 7]);

// NOBLE TILES: 10
let tileId = 0;

for (let i = 0; i < 5; i++) {
  const req: Cost = [0, 0, 0, 0, 0];
  req[i] = 4;
  req[(i + 1) % 5] = 4;
  ALL_BONUS_TILES.push({ id: tileId++, points: 3, requirement: req });
}

for (let i = 0; i < 5; i++) {
  const req: Cost = [0, 0, 0, 0, 0];
  req[i] = 3;
  req[(i + 2) % 5] = 3;
  req[(i + 4) % 5] = 3;
  ALL_BONUS_TILES.push({ id: tileId++, points: 3, requirement: req });
}
