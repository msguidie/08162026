import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GEM_COLORS_HEX, GEM_NAMES, GOLD_HEX } from '../types';

interface GemDelta {
  color: number;
  delta: number;
  key: string;
}

interface GemSupplyProps {
  gems: [number, number, number, number, number, number];
  gemDeltas?: GemDelta[];
  selectedGems?: number[];
}

export default function GemSupply({ gems, gemDeltas = [], selectedGems = [] }: GemSupplyProps) {
  return (
    <div className="flex items-center justify-center gap-4 py-2 px-4 bg-white/30 backdrop-blur-sm rounded-xl border border-white/50">
      {gems.slice(0, 5).map((count, i) => {
        const delta = gemDeltas.find(d => d.color === i);
        const selectedCount = selectedGems.filter(c => c === i).length;
        return (
          <div key={i} className="flex flex-col items-center gap-0.5 relative">
            <div
              className="w-7 h-7 rounded-full shadow-sm ring-1 ring-black/5"
              style={{ backgroundColor: GEM_COLORS_HEX[i] }}
              title={GEM_NAMES[i]}
            />
            <span className={`text-sm font-display w-6 text-center font-medium ${
              count === 0 ? 'text-slate-300' : 'text-slate-600'
            }`}>
              {count}
            </span>
            {selectedCount > 0 && (
              <div
                className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold text-white shadow"
                style={{ backgroundColor: GEM_COLORS_HEX[i] }}
              >
                {selectedCount}
              </div>
            )}
            <AnimatePresence>
              {delta && (
                <motion.span
                  key={delta.key}
                  initial={{ opacity: 1, y: 0, scale: 1 }}
                  animate={{ opacity: 0, y: -22, scale: 0.85 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 1.4, ease: 'easeOut' }}
                  className="absolute -top-3 text-xs font-bold text-red-500 pointer-events-none"
                  style={{ left: '50%', transform: 'translateX(-50%)' }}
                >
                  {delta.delta}
                </motion.span>
              )}
            </AnimatePresence>
          </div>
        );
      })}
      <div className="w-px h-10 bg-slate-200/80" />
      {(() => {
        const goldDelta = gemDeltas.find(d => d.color === 5);
        return (
          <div className="flex flex-col items-center gap-0.5 relative">
            <div className="w-7 h-7 rounded-full shadow-sm ring-1 ring-black/5" style={{ backgroundColor: GOLD_HEX }} title="Gold" />
            <span className={`text-sm font-display w-6 text-center font-medium ${gems[5] === 0 ? 'text-slate-300' : 'text-amber-600'}`}>
              {gems[5]}
            </span>
            <AnimatePresence>
              {goldDelta && (
                <motion.span
                  key={goldDelta.key}
                  initial={{ opacity: 1, y: 0 }}
                  animate={{ opacity: 0, y: -22 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 1.4, ease: 'easeOut' }}
                  className="absolute -top-3 text-xs font-bold text-red-500 pointer-events-none"
                  style={{ left: '50%', transform: 'translateX(-50%)' }}
                >
                  {goldDelta.delta}
                </motion.span>
              )}
            </AnimatePresence>
          </div>
        );
      })()}
    </div>
  );
}
