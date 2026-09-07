import React from 'react';
import { Bot } from 'lucide-react';

/**
 * Shared "this seat is an AI player" marker (docs/AI_BRIDGE.md §3).
 * Used by the lobby, the in-game player panels and the replay viewer so the
 * marker looks the same everywhere. `iconOnly` drops the "AI" text where the
 * row is tight and the username must not be truncated.
 */
export default function BotBadge({ iconOnly = false }: { iconOnly?: boolean }) {
  return (
    <span
      title="AI player"
      aria-label="AI player"
      className={`inline-flex items-center gap-0.5 rounded-md bg-[#7B6FA0]/10 text-[#7B6FA0] text-[9px] font-display font-semibold flex-shrink-0 ${
        iconOnly ? 'p-0.5' : 'px-1.5 py-0.5'
      }`}
    >
      <Bot size={10} />
      {!iconOnly && 'AI'}
    </span>
  );
}
