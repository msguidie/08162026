import React, { useState, useEffect, useRef } from 'react';
import useGameStore from '../store/gameStore';
import { SERVER_URL, CONNECTION_CONFIG } from '../constants';
import Avatar from './Avatar';
import type { Account } from '../types';

export default function LoginScreen() {
  const { connectionStatus, myAccount, connectToServer, login, enterLobby, openReplayBrowser } = useGameStore();
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [view, setView] = useState<'main' | 'create' | 'login'>('main');
  const [newName, setNewName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [wakeStatus, setWakeStatus] = useState('');
  const wakeAttempted = useRef(false);

  const connected = connectionStatus === 'connected' || connectionStatus === 'in_lobby';

  useEffect(() => {
    if (wakeAttempted.current) return;
    wakeAttempted.current = true;
    wakeAndConnect();
  }, []);

  async function wakeAndConnect() {
    setLoading(true);
    setError('');
    setWakeStatus('Waking server...');

    let healthy = false;
    for (let i = 0; i < CONNECTION_CONFIG.MAX_HEALTH_CHECKS; i++) {
      try {
        const res = await fetch(`${SERVER_URL}/health`, { signal: AbortSignal.timeout(5000) });
        if (res.ok) { healthy = true; break; }
      } catch {}
      setWakeStatus(`Waking server... (${i + 1}/${CONNECTION_CONFIG.MAX_HEALTH_CHECKS})`);
      await new Promise(r => setTimeout(r, CONNECTION_CONFIG.HEALTH_CHECK_RETRY));
    }

    if (!healthy) {
      setError('Server unreachable. Try again in a minute.');
      setWakeStatus('');
      setLoading(false);
      return;
    }

    setWakeStatus('Connecting...');
    try {
      await connectToServer();
      setWakeStatus('');
      fetchAccounts();
    } catch {
      setError('Failed to connect. Try again.');
      setWakeStatus('');
    }
    setLoading(false);
  }

  async function fetchAccounts() {
    try {
      const res = await fetch(`${SERVER_URL}/api/accounts`);
      const data = await res.json();
      setAccounts(data);
    } catch {}
  }

  async function createAccount() {
    if (!newName.trim()) return;
    setError('');
    try {
      const res = await fetch(`${SERVER_URL}/api/accounts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: newName.trim() }),
      });
      const data = await res.json();
      if (res.ok) {
        const ok = await login(data.username);
        if (ok) { setView('main'); setNewName(''); enterLobby(); }
        else setError('Login failed after account creation');
      } else {
        setError(data.error || 'Failed to create account');
      }
    } catch {
      setError('Network error');
    }
  }

  async function handleLogin(username: string) {
    const ok = await login(username);
    if (ok) enterLobby();
    else setError('Login failed');
  }

  useEffect(() => {
    if (connected && view === 'login') fetchAccounts();
  }, [connected, view]);

  if (myAccount) {
    return (
      <div className="min-h-[100dvh] flex items-center justify-center p-4">
        <div className="bg-white/60 backdrop-blur-md rounded-2xl p-5 sm:p-8 w-80 max-w-full text-center space-y-4 border border-white/70 shadow-lg">
          <Avatar seed={myAccount.avatarSeed} size={64} />
          <h2 className="text-xl font-display font-bold text-slate-800">{myAccount.username}</h2>
          <p className="text-sm text-slate-500">Rating: {myAccount.rating} | Wins: {myAccount.wins}</p>
          <button onClick={enterLobby} className="w-full py-3 bg-[#7EA68A] hover:bg-[#6B9477] text-white rounded-xl font-semibold transition shadow-sm">
            Enter Lobby
          </button>
          <button onClick={openReplayBrowser} className="w-full py-3 bg-[#8B9DAF] hover:bg-[#7A8D9F] text-white rounded-xl font-semibold transition shadow-sm">
            Replays
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[100dvh] flex items-center justify-center p-4">
      <div className="bg-white/60 backdrop-blur-md rounded-2xl p-5 sm:p-8 w-80 max-w-full space-y-5 border border-white/70 shadow-lg">
        <h1 className="text-3xl font-display font-bold text-center tracking-tight text-slate-800">Splendor</h1>

        {!connected ? (
          <div className="space-y-3">
            {loading ? (
              <div className="text-center space-y-2 py-4">
                <span className="flex items-center justify-center gap-2">
                  <span className="animate-spin h-5 w-5 border-2 border-slate-400 border-t-transparent rounded-full" />
                  <span className="text-slate-500 text-sm">{wakeStatus || 'Connecting...'}</span>
                </span>
              </div>
            ) : (
              <button onClick={wakeAndConnect} className="w-full py-3 bg-[#8B9DAF] hover:bg-[#7A8D9F] text-white rounded-xl font-semibold transition shadow-sm">
                Connect to Server
              </button>
            )}
            {error && <p className="text-red-500 text-sm text-center">{error}</p>}
          </div>
        ) : view === 'main' ? (
          <div className="space-y-3">
            <button onClick={() => setView('create')} className="w-full py-3 bg-[#7EA68A] hover:bg-[#6B9477] text-white rounded-xl font-semibold transition shadow-sm">
              Create Account
            </button>
            <button onClick={() => { setView('login'); fetchAccounts(); }} className="w-full py-3 bg-[#8B9DAF] hover:bg-[#7A8D9F] text-white rounded-xl font-semibold transition shadow-sm">
              Login
            </button>
            <button onClick={openReplayBrowser} className="w-full py-3 bg-[#8B9DAF] hover:bg-[#7A8D9F] text-white rounded-xl font-semibold transition shadow-sm">
              Replays
            </button>
          </div>
        ) : view === 'create' ? (
          <div className="space-y-3">
            <input
              type="text" placeholder="Enter username" value={newName}
              onChange={e => setNewName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && createAccount()}
              className="w-full px-4 py-2.5 bg-white/70 border border-slate-200 rounded-xl outline-none focus:ring-2 ring-[#7EA68A]/50 placeholder-slate-400 text-slate-700 text-sm"
              autoFocus
            />
            <button onClick={createAccount} className="w-full py-3 bg-[#7EA68A] hover:bg-[#6B9477] text-white rounded-xl font-semibold transition shadow-sm">Create</button>
            <button onClick={() => setView('main')} className="w-full py-2 text-slate-400 hover:text-slate-600 text-sm">Back</button>
            {error && <p className="text-red-500 text-sm text-center">{error}</p>}
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-slate-500 text-center">Select account:</p>
            <div className="max-h-60 overflow-y-auto space-y-1.5">
              {accounts.map(acc => (
                <button key={acc.username} onClick={() => handleLogin(acc.username)}
                  className="w-full flex items-center gap-3 px-3 py-2.5 bg-white/50 hover:bg-white/80 rounded-xl transition border border-white/60">
                  <Avatar seed={acc.avatarSeed} size={32} />
                  <div className="text-left flex-1">
                    <div className="font-medium text-sm text-slate-700">{acc.username}</div>
                    <div className="text-xs text-slate-400">Rating: {acc.rating}</div>
                  </div>
                </button>
              ))}
              {accounts.length === 0 && <p className="text-slate-400 text-center text-sm py-4">No accounts yet</p>}
            </div>
            <button onClick={() => setView('main')} className="w-full py-2 text-slate-400 hover:text-slate-600 text-sm">Back</button>
          </div>
        )}
      </div>
    </div>
  );
}
