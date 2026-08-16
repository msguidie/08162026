import React from 'react';
import { renderAvatarSVG } from '../utils/avatar';

interface AvatarProps {
  seed: number;
  size?: number;
  highlight?: boolean;
}

export default function Avatar({ seed, size = 40, highlight = false }: AvatarProps) {
  const svg = renderAvatarSVG(seed, size);
  return (
    <div
      className={`inline-block rounded-lg border-2 ${
        highlight ? 'border-amber-500 ring-2 ring-amber-500/30 ring-offset-1 ring-offset-white/50' : 'border-slate-300'
      }`}
      style={{ width: size, height: size }}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
