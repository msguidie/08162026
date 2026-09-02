import { SERVER_URL } from '../constants';
import type { ReplayData, ReplayIndexEntry } from '../types';

// Plain fetch against the REST routes from docs/REPLAY_FORMAT.md §4 — no socket needed.

export interface ReplayListResponse {
  games: ReplayIndexEntry[];
  total: number;
  source?: 'github' | 'memory';
}

async function getJson<T>(path: string, timeoutMs = 15000): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${SERVER_URL}${path}`, { signal: AbortSignal.timeout(timeoutMs) });
  } catch {
    throw new Error('Cannot reach the server. Try again in a moment.');
  }
  if (!res.ok) {
    if (res.status === 404) throw new Error('This replay is no longer available.');
    if (res.status === 422) throw new Error('This replay could not be reconstructed.');
    throw new Error(`Server error (${res.status})`);
  }
  try {
    return (await res.json()) as T;
  } catch {
    throw new Error('The server returned an unreadable response.');
  }
}

export async function fetchReplayList(limit = 50, offset = 0): Promise<ReplayListResponse> {
  const data = await getJson<Partial<ReplayListResponse>>(`/api/replays?limit=${limit}&offset=${offset}`);
  return {
    games: Array.isArray(data.games) ? data.games : [],
    total: typeof data.total === 'number' ? data.total : (data.games?.length ?? 0),
    source: data.source,
  };
}

export async function fetchReplay(id: string): Promise<ReplayData> {
  const data = await getJson<ReplayData>(`/api/replays/${encodeURIComponent(id)}`);
  if (!data?.frames?.length || !data.meta) throw new Error('This replay contains no frames.');
  return data;
}
