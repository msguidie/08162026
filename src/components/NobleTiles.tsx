import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { BonusTile } from '../types';
import { GEM_COLORS_HEX } from '../types';

interface NobleTilesProps {
  tiles: BonusTile[];
}

export default function NobleTiles({ tiles }: NobleTilesProps) {
  return (
    <div className="flex flex-col gap-1.5 items-center">
      <div className="text-[10px] text-slate-400 font-medium tracking-wide uppercase">Nobles</div>
      <AnimatePresence>
        {tiles.map(tile => (
          <motion.div
            key={tile.id}
            layout
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5, y: 30, transition: { duration: 0.4 } }}
            className="w-16 bg-white/50 backdrop-blur-sm border border-[#9B8EC4]/30 rounded-lg flex flex-col items-center py-1.5 px-1 shadow-sm"
          >
            <span className="text-xs font-bold text-[#7B6FA0] mb-0.5">+{tile.points}</span>
            <div className="flex gap-0.5 flex-wrap justify-center">
              {tile.requirement.map((r, i) =>
                r > 0 ? (
                  <div key={i} className="flex items-center gap-px">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: GEM_COLORS_HEX[i] }} />
                    <span className="text-[9px] text-slate-500">{r}</span>
                  </div>
                ) : null
              )}
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
