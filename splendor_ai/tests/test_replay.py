"""End-to-end: generate trajectories with the Node engine and check them.

This is a miniature of the full validation run (see ``validation/README.md``);
it keeps the go/no-go gate wired into the normal test run.
"""

import os
import subprocess

import pytest

from splendor_ai.rules import engine as E
from splendor_ai.tests.oracle import requires_node
from splendor_ai.validation import replay_check as RC

_HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(_HERE, "..", "validation", "gen_trajectories.js")

CONFIGS = [
    ("INDIVIDUAL", 2, None),
    ("INDIVIDUAL", 3, None),
    ("INDIVIDUAL", 4, None),
    ("ONE_V_TWO", 3, None),
    ("TEAM", 4, "ADJACENT"),
    ("TEAM", 4, "OPPOSITE"),
]


@requires_node
@pytest.mark.parametrize("mode,players,layout", CONFIGS)
def test_generated_trajectories_match(tmp_path, mode, players, layout):
    out = str(tmp_path / f"{mode}{players}.jsonl")
    cmd = ["node", GEN, "--out", out, "--games", "12", "--mode", mode,
           "--players", str(players), "--seed", str(17 + players),
           "--chaos-frac", "0.5", "--quiet"]
    if layout:
        cmd += ["--layout", layout]
    subprocess.run(cmd, check=True, capture_output=True)
    stats = RC.Stats()
    RC.check_file(out, stats, progress=False)      # raises RC.Mismatch on any diff
    assert stats.games == 12
    assert stats.steps > 0
    assert stats.legal_checks > 0
    assert stats.replay_checks == 12


@requires_node
def test_replay_format_file_round_trips(tmp_path):
    """A stored replay file (docs/REPLAY_FORMAT.md §1) rebuilds to the same
    final state the live game reached."""
    import json

    out = str(tmp_path / "r.jsonl")
    subprocess.run(["node", GEN, "--out", out, "--games", "6",
                    "--mode", "ONE_V_TWO", "--seed", "5", "--quiet"],
                   check=True, capture_output=True)
    seen = 0
    for rec in RC.open_jsonl(out):
        if rec.get("k") != "end":
            continue
        rp = rec["replay"]
        assert rp["v"] == 1
        state = E.replay(rp)
        assert RC.py_snap(state) == rec["s"]
        assert E.rating_changes(state) == rp["result"]["rating"]
        seen += 1
    assert seen == 6


@requires_node
def test_orphaned_noble_states_survive_a_full_game(tmp_path):
    """The noble-stress generator setting reaches the awkward
    ``_pendingTileChoice`` states; they must replay bit-exactly too."""
    out = str(tmp_path / "orphan.jsonl")
    subprocess.run(["node", GEN, "--out", out, "--games", "40",
                    "--mode", "INDIVIDUAL", "--players", "4", "--seed", "91",
                    "--t1-bias", "0.85", "--chaos-frac", "0",
                    "--orphan-hunt", "--perm-check-p", "0", "--quiet"],
                   check=True, capture_output=True)
    stats = RC.Stats()
    RC.check_file(out, stats, progress=False)
    assert stats.games == 40
    assert stats.auto_tiles > 0
