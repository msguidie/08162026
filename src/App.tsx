import React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import useGameStore from './store/gameStore';
import LoginScreen from './components/LoginScreen';
import WaitingRoom from './components/WaitingRoom';
import GameBoard from './components/GameBoard';

export default function App() {
  const { appPhase, connectionStatus, reconnectAttempts, toasts, removeToast } = useGameStore();

  return (
    <div className="min-h-screen font-sans">
      {connectionStatus === 'reconnecting' && (
        <div className="fixed top-0 left-0 right-0 bg-amber-500/90 text-white text-center py-1.5 text-xs z-50 backdrop-blur-sm shadow-sm flex items-center justify-center gap-2">
          <span className="animate-spin h-3 w-3 border-2 border-white border-t-transparent rounded-full" />
          Reconnecting{reconnectAttempts > 0 ? ` (attempt ${reconnectAttempts})` : ''}...
        </div>
      )}
      {connectionStatus === 'error' && (
        <div className="fixed top-0 left-0 right-0 bg-red-500/90 text-white text-center py-1.5 text-xs z-50 backdrop-blur-sm shadow-sm flex items-center justify-center gap-3">
          <span>Connection lost.</span>
          <button onClick={() => window.location.reload()} className="px-3 py-0.5 bg-white/20 hover:bg-white/30 rounded-md text-[11px] font-medium transition">Refresh</button>
        </div>
      )}
      {connectionStatus === 'entering_lobby' && (
        <div className="fixed top-0 left-0 right-0 bg-slate-500/80 text-white text-center py-1 text-xs z-50 backdrop-blur-sm shadow-sm flex items-center justify-center gap-2">
          <span className="animate-spin h-3 w-3 border-2 border-white border-t-transparent rounded-full" />
          Connecting to lobby...
        </div>
      )}

      <div className="fixed top-8 left-1/2 -translate-x-1/2 z-[60] flex flex-col items-center gap-1.5 pointer-events-none">
        <AnimatePresence>
          {toasts.map(toast => (
            <motion.div key={toast.id}
              initial={{ opacity: 0, y: -20, scale: 0.9 }} animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.9 }} transition={{ duration: 0.25 }}
              onClick={() => removeToast(toast.id)}
              className={`pointer-events-auto cursor-pointer px-4 py-2 rounded-xl shadow-lg backdrop-blur-md text-xs font-medium ${
                toast.type === 'error' ? 'bg-red-500/90 text-white'
                  : toast.type === 'warn' ? 'bg-amber-500/90 text-white'
                  : 'bg-white/80 text-slate-700 border border-white/70'
              }`}>
              {toast.message}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {appPhase === 'LOGIN' && <LoginScreen />}
      {appPhase === 'WAITING_ROOM' && <WaitingRoom />}
      {appPhase === 'GAME' && <GameBoard />}
    </div>
  );
}
