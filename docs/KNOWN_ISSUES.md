# Pre-existing issues found during this work (not changed — reported for your decision)

These were discovered while validating the replay and AI features. None were introduced by this branch, and
none were fixed unless noted, to respect the "additive only" rule.

1. **Orphaned noble choice (server rule gap).** `server/gameLogic.js`: `CHOOSE_TILE` requires `turnAction.type === 'BUY'`,
   but `advanceTurn` runs after every action. If a gem take or a reserve makes a player qualify for two or more nobles
   (possible after an earlier three-noble choice), `_pendingTileChoice` is set, no tile choice is accepted, the turn does
   not advance, and the player can keep acting until a buy resolves it. Hit 16 times in 16,000 random games.
   Suggested one-line fix: accept `CHOOSE_TILE` whenever `_pendingTileChoice` contains the tile (drop the `BUY` check).
   The Python engine mirrors the current behaviour exactly (so the AI is trained on the real rules).
2. **Resigned seat can receive the winner's rating delta.** `calculateRatingChanges` ranks all seats by score/cards and
   ignores `resignedPlayers`. Accounts are unaffected when the game ends *on* the resign (the server excludes resigned
   seats there), but if an INDIVIDUAL game ends by score after an earlier resignation the resigned account is credited.
   The replay record now stores 0 for resigned seats. Suggested fix: filter resigned indices before ranking.
3. **No legal move state.** A seat with 10 tokens, 3 reserved cards and nothing affordable has no legal action
   (no pass exists). Under random play this happens in 12–42% of games depending on mode; strong play avoids it,
   and the AI worker resigns in that case (the server has no other option). Consider adding a server-side PASS
   action if you want humans to have a way out; the Python engine has no pass by design so the AI never relies on one.
4. **Mobile opponent panel shows the noble count twice** (`src/components/PlayerInfo.tsx` lines ~87 and ~174). Cosmetic.
5. **Lobby card overflows a 1280×800 viewport with 4+ entries** (`WaitingRoom.tsx`), scrolls but without an affordance. Cosmetic.
6. **"Leave Lobby" → "Enter Lobby" failed with "Login required"** (fresh socket without re-login). *Fixed on this branch*
   by re-sending `login` when a new socket connects for a logged-in account (`gameStore.connectToServer`).
