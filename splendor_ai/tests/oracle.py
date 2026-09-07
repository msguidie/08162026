"""Test helper: drive the authoritative Node engine from pytest.

``validation/probe_state.js`` materialises an arbitrary position inside
``server/gameLogic.js`` and reports (a) exactly which actions it accepts and
(b) what applying one does.  Every rule test in this suite is therefore
checked against the real server rules, not against the Python port's own
opinion of them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Sequence

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_PKG, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from splendor_ai.rules import engine as E                        # noqa: E402
from splendor_ai.rules.actions import action_name                # noqa: E402

PROBE = os.path.join(_PKG, "validation", "probe_state.js")

HAS_NODE = shutil.which("node") is not None and os.path.exists(PROBE)
requires_node = pytest.mark.skipif(
    not HAS_NODE, reason="node and validation/probe_state.js are required")


# ── Node side ─────────────────────────────────────────────────────────────

def node_probe(spec: Dict[str, Any]) -> Dict[str, Any]:
    out = subprocess.run(
        ["node", PROBE], input=json.dumps(spec), capture_output=True,
        text=True, check=True)
    data = json.loads(out.stdout)
    if "error" in data:
        raise AssertionError("probe_state.js failed:\n" + data["error"])
    return data


# ── Python side ───────────────────────────────────────────────────────────

def py_state_from_resolved(res: Dict[str, Any]) -> E.GameState:
    """Build the identical Python state from ``probe_state.js``'s ``resolved``."""
    teams = res.get("teams")
    if teams and teams[0] is None:
        teams = None
    s = E.new_game(
        res["n"], res["mode"], res.get("layout"), team_ids=teams,
        setup={"board": res["board"], "decks": res["decks"],
               "tiles": res["tiles"], "first": res["current"]},
    )
    s.gems = list(res["gems"])
    for i, pp in enumerate(res["players"]):
        p = s.players[i]
        p.gems = list(pp["gems"])
        p.cards = list(pp["cards"])
        p.reserved = list(pp["reserved"])
        p.reserved_public = [True] * len(p.reserved)
        p.tiles = list(pp["tiles"])
        p.score = pp["score"]
        d = [0, 0, 0, 0, 0]
        from splendor_ai.rules.cards import CARD_REWARD
        for cid in p.cards:
            d[CARD_REWARD[cid]] += 1
        p.discount = d
    s.current_player = res["current"]
    s.round_start_player = res["roundStart"]
    s.turn_number = res["turnNumber"]
    s.phase = res["phase"]
    s.final_round_triggered_by = res["finalRoundTriggeredBy"]
    s.resigned = list(res["resigned"])
    s.turn_action = res["turnAction"]
    s.pending_tile_choice = res["pendingTileChoice"]
    s.game_result = res["gameResult"]
    return s


def py_snap(s: E.GameState) -> Dict[str, Any]:
    from splendor_ai.validation.replay_check import py_snap as _snap
    return _snap(s)


# ── comparison ────────────────────────────────────────────────────────────

def _codes_to_indices(state: E.GameState, codes: Sequence[Sequence[Any]]) -> set:
    return {E.from_replay_code(state, c) for c in codes}


class Position:
    """One probed position: the Node answers plus the matching Python state."""

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.data = node_probe(spec)
        self.resolved = self.data["resolved"]
        self.results = self.data["results"]
        self.state = py_state_from_resolved(self.resolved)

    # -- legality ------------------------------------------------------
    def node_legal(self, i: int = 0) -> set:
        return _codes_to_indices(self.state, self.results[i]["legal"])

    def py_legal(self) -> set:
        return set(E.legal_actions(self.state))

    def assert_legal_agrees(self, i: int = 0) -> set:
        node = self.node_legal(i)
        py = self.py_legal()
        if node != py:
            only_py = sorted(py - node)
            only_node = sorted(node - py)
            raise AssertionError(
                "legal action sets differ\n"
                f"  only python : {[action_name(a) for a in only_py]}\n"
                f"  only node   : {[action_name(a) for a in only_node]}\n"
                f"  state       : {py_snap(self.state)}")
        return py

    # -- application ---------------------------------------------------
    def do_resign(self, op: Sequence[Any], i: int) -> None:
        node = self.results[i]
        who = op[1] if len(op) > 1 else self.state.current_player
        if op[0] == "resign":
            ev = E.resign(self.state, who)
        else:
            ev = E.timeout(self.state, who)
        got, want = py_snap(self.state), node["snap"]
        assert got == want, (
            f"post-state differs after {op[0]}({who})\n"
            + "\n".join(f"  {k}: python={got[k]!r} node={want[k]!r}"
                        for k in want if got.get(k) != want[k]))
        assert ev["tileClaimed"] == node["tileClaimed"]
        assert E.rating_changes(self.state) == node["rating"]

    def run_all(self) -> "Position":
        """Walk the op list, mirroring every op in Python and comparing."""
        for i, op in enumerate(self.spec["ops"]):
            if op[0] == "probe":
                self.assert_legal_agrees(i)
            elif op[0] == "apply":
                self.apply_code(op[1], i)
            elif op[0] in ("resign", "timeout"):
                self.do_resign(op, i)
            elif op[0] == "select":
                pass          # handled explicitly by the incremental-take test
            else:                                        # pragma: no cover
                raise AssertionError(f"unhandled op {op!r}")
        return self

    def apply_code(self, code: Sequence[Any], i: int) -> None:
        """Apply ``code`` in Python and require the Node result at ``results[i]``."""
        node = self.results[i]
        assert node["op"] == "apply"
        idx = E.from_replay_code(self.state, code)
        if node["error"]:
            with pytest.raises(E.IllegalAction):
                E.apply(self.state, idx)
            return
        ev = E.apply(self.state, idx)
        got, want = py_snap(self.state), node["snap"]
        assert got == want, (
            f"post-state differs after {action_name(idx)}\n"
            + "\n".join(f"  {k}: python={got[k]!r} node={want[k]!r}"
                        for k in want if got.get(k) != want[k]))
        for key, value in node["ev"].items():
            if key == "type":
                continue
            mine = ev["tileClaimed"] if key == "tileClaimed" else ev["payload"].get(key)
            if key == "selected":
                mine = sorted(mine)
                value = sorted(value)
            assert mine == value, f"event.{key}: python={mine!r} node={value!r}"
        assert E.rating_changes(self.state) == node["rating"]


def probe(**spec: Any) -> Position:
    spec.setdefault("ops", [["probe"]])
    return Position(spec)


__all__ = ["Position", "probe", "node_probe", "py_state_from_resolved",
           "py_snap", "requires_node", "HAS_NODE", "PROBE"]
