import { useCallback, useEffect, useRef, useState } from 'react';

export const REPLAY_SPEEDS = [0.5, 1, 2, 4] as const;
export type ReplaySpeed = typeof REPLAY_SPEEDS[number];

/** Base pace: one frame every 2 s at 1×. */
export const REPLAY_FRAME_MS = 2000;

export interface ReplayPlayer {
  index: number;
  playing: boolean;
  speed: ReplaySpeed;
  perspective: number;
  atEnd: boolean;
  frameCount: number;
  stepForward: () => void;
  stepBack: () => void;
  seek: (index: number) => void;
  togglePlay: () => void;
  cycleSpeed: () => void;
  setPerspective: (seat: number) => void;
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return true;
  if (target instanceof HTMLSelectElement) return true;
  return target.isContentEditable;
}

export function useReplayPlayer(frameCount: number, seatCount: number): ReplayPlayer {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<ReplaySpeed>(1);
  // Random seat on load — a replay is watched from somebody's chair.
  const [perspective, setPerspective] = useState(() =>
    (seatCount > 0 ? Math.floor(Math.random() * seatCount) : 0));

  const lastIndex = Math.max(0, frameCount - 1);
  const atEnd = index >= lastIndex;

  const seek = useCallback((next: number) => {
    setIndex(current => {
      const clamped = Math.min(Math.max(0, Math.round(next)), Math.max(0, frameCount - 1));
      return clamped === current ? current : clamped;
    });
  }, [frameCount]);

  const stepForward = useCallback(() => {
    setPlaying(false);
    setIndex(current => Math.min(current + 1, Math.max(0, frameCount - 1)));
  }, [frameCount]);

  const stepBack = useCallback(() => {
    setPlaying(false);
    setIndex(current => Math.max(0, current - 1));
  }, []);

  const togglePlay = useCallback(() => {
    setPlaying(current => {
      if (current) return false;
      // Replay again from the top when the game is already over.
      setIndex(i => (i >= Math.max(0, frameCount - 1) ? 0 : i));
      return true;
    });
  }, [frameCount]);

  const cycleSpeed = useCallback(() => {
    setSpeed(current => REPLAY_SPEEDS[(REPLAY_SPEEDS.indexOf(current) + 1) % REPLAY_SPEEDS.length]);
  }, []);

  // ── Playback timer: auto-pauses on the last frame ──
  useEffect(() => {
    if (!playing) return;
    if (index >= lastIndex) { setPlaying(false); return; }
    const timer = setTimeout(
      () => setIndex(current => Math.min(current + 1, lastIndex)),
      REPLAY_FRAME_MS / speed,
    );
    return () => clearTimeout(timer);
  }, [playing, index, speed, lastIndex]);

  // ── Keyboard: Space toggles, ← / → step ──
  const handlersRef = useRef({ togglePlay, stepForward, stepBack });
  handlersRef.current = { togglePlay, stepForward, stepBack };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        handlersRef.current.togglePlay();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        handlersRef.current.stepForward();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        handlersRef.current.stepBack();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return {
    index: Math.min(index, lastIndex),
    playing, speed, perspective, atEnd, frameCount,
    stepForward, stepBack, seek, togglePlay, cycleSpeed, setPerspective,
  };
}
