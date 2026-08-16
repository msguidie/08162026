import React from 'react';
import { motion } from 'framer-motion';
import type { Card } from '../types';
import { GEM_COLORS_HEX, GEM_COLORS_LIGHT, GEM_NAMES } from '../types';

interface CardViewProps {
  card: Card;
  onClick?: () => void;
  clickable?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export default function CardView({ card, onClick, clickable = false, size = 'md' }: CardViewProps) {
  const dims = {
    sm: { w: 'w-16', h: 'h-[88px]', text: 'text-[10px]', pt: 'text-sm', dot: 'w-3 h-3', cdot: 'w-2.5 h-2.5', gap: 'gap-px' },
    md: { w: 'w-[88px]', h: 'h-[122px]', text: 'text-xs', pt: 'text-lg', dot: 'w-4 h-4', cdot: 'w-3 h-3', gap: 'gap-0.5' },
    lg: { w: 'w-28', h: 'h-[154px]', text: 'text-sm', pt: 'text-xl', dot: 'w-5 h-5', cdot: 'w-3.5 h-3.5', gap: 'gap-0.5' },
  };
  const d = dims[size];

  const baseColor = GEM_COLORS_HEX[card.reward];
  const lightColor = GEM_COLORS_LIGHT[card.reward];
  const bg = `linear-gradient(to top right, ${lightColor}, ${baseColor})`;

  return (
    <motion.div
      layout
      whileHover={clickable ? { scale: 1.06, y: -3 } : undefined}
      whileTap={clickable ? { scale: 0.95 } : undefined}
      onClick={clickable ? onClick : undefined}
      className={`${d.w} ${d.h} rounded-lg flex flex-col overflow-hidden shadow-md ring-1 ring-black/5 ${
        clickable ? 'cursor-pointer hover:ring-2 hover:ring-amber-500/80 hover:shadow-lg' : ''
      }`}
      style={{ background: bg }}
    >
      <div className="flex items-center justify-between px-1.5 py-1">
        {card.points > 0 ? (
          <span className={`font-bold text-white/90 ${d.pt} leading-none drop-shadow`}>{card.points}</span>
        ) : (
          <span />
        )}
        <div
          className={`${d.dot} rounded-full border-2 border-white/50`}
          style={{ backgroundColor: baseColor }}
          title={GEM_NAMES[card.reward]}
        />
      </div>
      <div className={`flex-1 flex flex-col justify-end px-1.5 pb-1.5 ${d.gap}`}>
        {card.cost.map((c, i) =>
          c > 0 ? (
            <div key={i} className="flex items-center gap-0.5">
              <div
                className={`${d.cdot} rounded-full flex-shrink-0 border border-white/30`}
                style={{ backgroundColor: GEM_COLORS_HEX[i] }}
              />
              <span className={`text-white/90 font-semibold ${d.text} drop-shadow`}>{c}</span>
            </div>
          ) : null
        )}
      </div>
    </motion.div>
  );
}

export function DeckView({ tier, count, onClick, clickable }: {
  tier: number; count: number; onClick?: () => void; clickable?: boolean;
}) {
  const tierColors = ['#4DAA8D', '#5B7CC4', '#8E6FBF'];
  const color = tierColors[tier - 1] || '#666';

  return (
    <motion.div
      whileHover={clickable ? { scale: 1.05 } : undefined}
      whileTap={clickable ? { scale: 0.95 } : undefined}
      onClick={clickable ? onClick : undefined}
      className={`w-[88px] h-[122px] rounded-lg border-2 border-dashed flex flex-col items-center justify-center ${
        clickable && count > 0 ? 'cursor-pointer hover:border-amber-500/70' : 'opacity-30'
      }`}
      style={{ borderColor: `${color}60`, background: `${color}0A` }}
    >
      <span className="text-base font-bold" style={{ color }}>{count}</span>
      <span className="text-xs mt-0.5" style={{ color: `${color}80` }}>L{tier}</span>
    </motion.div>
  );
}
