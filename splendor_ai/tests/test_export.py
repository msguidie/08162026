"""The deployment bundle: round trip, worker layout, manifest, TorchScript.

The bundle is what the Windows worker actually loads (``docs/AI_DESIGN.md``
§1.9), so the tests here are about *compatibility*, not about the network:

* ``shared.pt`` comes back through :func:`splendor_ai.model.load_checkpoint`
  with identical weights and the version gate intact;
* ``MODEL_DIR/<mode>.pt`` and ``MODEL_DIR/shared.pt`` are exactly what
  ``worker/config.py::checkpoint_candidates`` looks for;
* everything in ``meta`` survives a ``weights_only=True`` load (a
  ``TorchVersion`` object in there would brick the worker);
* the TorchScript trace agrees with eager, and a tracing failure degrades to a
  warning instead of losing the bundle.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest
import torch

from splendor_ai import export as export_mod
from splendor_ai.encode import OBS_DIM, OBS_VERSION
from splendor_ai.export import (BUNDLE_VERSION, MODEL_MODES, TS_OUTPUTS,
                                export_bundle, read_elo_table, sha256_file)
from splendor_ai.model import (ACTION_VERSION, NetConfig, SplendorNet,
                               load_checkpoint, save_checkpoint)
from splendor_ai.worker.config import WorkerConfig


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """A small but real checkpoint (width 32 keeps the whole file ~140 kB)."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "latest.pt"
    model = SplendorNet(NetConfig(width=32, blocks=1))
    save_checkpoint(str(path), model,
                    {"step": 4242, "generation": 12,
                     "meta": {"run": "nscc", "learner": "az"}})
    return str(path)


@pytest.fixture(scope="module")
def bundle(tiny_ckpt, tmp_path_factory):
    out = tmp_path_factory.mktemp("bundle")
    manifest = export_bundle(tiny_ckpt, str(out), modes="all",
                             notes="unit test")
    return manifest, str(out)


# ── round trip ────────────────────────────────────────────────────────────

def test_shared_pt_round_trips_through_load_checkpoint(bundle, tiny_ckpt):
    _manifest, out = bundle
    source, _ = load_checkpoint(tiny_ckpt)
    model, ckpt = load_checkpoint(os.path.join(out, "shared.pt"))
    assert ckpt["obs_version"] == OBS_VERSION
    assert ckpt["action_version"] == ACTION_VERSION
    assert ckpt["generation"] == 12 and ckpt["step"] == 4242
    assert model.cfg.to_dict() == source.cfg.to_dict()
    a, b = source.state_dict(), model.state_dict()
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_meta_is_weights_only_safe_and_keeps_provenance(bundle, tiny_ckpt):
    _manifest, out = bundle
    # weights_only=True is exactly how the worker reads it
    raw = torch.load(os.path.join(out, "shared.pt"), map_location="cpu",
                     weights_only=True)
    meta = raw["meta"]
    assert meta["run"] == "nscc" and meta["learner"] == "az"   # carried over
    assert meta["bundle_version"] == BUNDLE_VERSION
    assert meta["source_checkpoint"] == os.path.abspath(tiny_ckpt)
    assert meta["source_sha256"] == sha256_file(tiny_ckpt)
    assert isinstance(meta["torch_version"], str)
    assert meta["notes"] == "unit test"


def test_per_mode_copies_are_what_the_worker_looks_for(bundle):
    _manifest, out = bundle
    config = WorkerConfig(model_dir=out)
    for key in MODEL_MODES:
        first, fallback = config.checkpoint_candidates(key)
        assert first.name == f"{key}.pt" and first.exists()
        assert fallback.name == "shared.pt" and fallback.exists()
        model, ckpt = load_checkpoint(str(first))
        assert ckpt["generation"] == 12
    shared = open(os.path.join(out, "shared.pt"), "rb").read()
    assert open(os.path.join(out, "ovt.pt"), "rb").read() == shared


def test_shared_only_bundle(tiny_ckpt, tmp_path):
    manifest = export_bundle(tiny_ckpt, str(tmp_path / "b"), modes=None,
                             torchscript=False)
    names = [f["name"] for f in manifest["files"]]
    assert names == ["shared.pt"]
    assert manifest["torchscript"]["ok"] is False
    config = WorkerConfig(model_dir=str(tmp_path / "b"))
    first, fallback = config.checkpoint_candidates("ind2")
    assert not first.exists() and fallback.exists()   # falls back to shared


def test_unknown_mode_key_is_refused(tiny_ckpt, tmp_path):
    with pytest.raises(ValueError, match="ind5"):
        export_bundle(tiny_ckpt, str(tmp_path / "b"), modes=["ind5"])


# ── manifest ──────────────────────────────────────────────────────────────

def test_manifest_describes_the_bundle(bundle):
    manifest, out = bundle
    on_disk = json.loads(open(os.path.join(out, "manifest.json"),
                              encoding="utf-8").read())
    assert on_disk["bundle_version"] == BUNDLE_VERSION
    assert on_disk["obs_version"] == OBS_VERSION
    assert on_disk["obs_dim"] == OBS_DIM
    assert on_disk["action_version"] == ACTION_VERSION
    assert on_disk["generation"] == 12 and on_disk["step"] == 4242
    assert on_disk["net"]["width"] == 32
    assert on_disk["parameters"] > 0
    assert on_disk["loader"] == "splendor_ai.model.load_checkpoint"
    assert on_disk["worker"]["mode_keys"] == list(MODEL_MODES)
    listed = {f["name"] for f in on_disk["files"]}
    assert {"shared.pt", "ind2.pt", "team.pt"} <= listed
    for entry in on_disk["files"]:
        path = os.path.join(out, entry["name"])
        assert entry["bytes"] == os.path.getsize(path)
        assert entry["sha256"] == sha256_file(path)


# ── TorchScript ───────────────────────────────────────────────────────────

def test_torchscript_matches_eager(bundle):
    manifest, out = bundle
    status = manifest["torchscript"]
    assert status["ok"] and status["file"] == "shared.ts"
    assert status["outputs"] == list(TS_OUTPUTS)

    model, _ckpt = load_checkpoint(os.path.join(out, "shared.pt"))
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.load(os.path.join(out, "shared.ts"))
    rng = np.random.default_rng(5)
    obs = torch.from_numpy(rng.standard_normal((7, OBS_DIM)).astype(np.float32))
    mask_np = rng.random((7, 65)) < 0.35
    mask_np[:, 3] = True
    mask = torch.from_numpy(mask_np)
    with torch.no_grad():
        eager = model(obs, mask)
        scripted = traced(obs, mask)
    for i, key in enumerate(TS_OUTPUTS):
        assert torch.allclose(eager[key], scripted[i], atol=1e-5), key
    # and the mask still zeroes the illegal actions after a softmax
    probs = torch.softmax(scripted[0], dim=-1)
    assert float(probs[~mask].max()) < 1e-12


def test_a_tracing_failure_only_costs_the_torchscript(tiny_ckpt, tmp_path,
                                                      monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("tracing exploded")

    monkeypatch.setattr(torch.jit, "trace", boom)
    with pytest.warns(RuntimeWarning, match="TorchScript export skipped"):
        manifest = export_bundle(tiny_ckpt, str(tmp_path / "b"),
                                 modes=["ind2"])
    assert manifest["torchscript"]["ok"] is False
    assert "tracing exploded" in manifest["torchscript"]["reason"]
    assert not os.path.exists(os.path.join(str(tmp_path / "b"), "shared.ts"))
    model, _ = load_checkpoint(os.path.join(str(tmp_path / "b"), "shared.pt"))
    assert model.cfg.width == 32


# ── version gate ──────────────────────────────────────────────────────────

def test_export_refuses_a_stale_observation_version(tiny_ckpt, tmp_path):
    """The §1.5 gate must fire here, on a workstation — not on the AI box."""
    payload = torch.load(tiny_ckpt, map_location="cpu", weights_only=True)
    payload["obs_version"] = OBS_VERSION + 99
    stale = str(tmp_path / "stale.pt")
    torch.save(payload, stale)
    with pytest.raises(RuntimeError, match="obs_version"):
        export_bundle(stale, str(tmp_path / "b"))


# ── Elo metadata ──────────────────────────────────────────────────────────

def test_elo_table_from_an_arena_report_reaches_the_bundle(tiny_ckpt,
                                                           tmp_path):
    from splendor_ai.arena import run_matches, write_reports

    results = run_matches({"random": "random", "greedy": "greedy"},
                          ["ind2", "ovt"], games_per_pairing=2, seed=0,
                          workers=1)
    _md, js = write_reports(results, str(tmp_path / "arena.md"), bootstrap=20)

    table = read_elo_table(js)
    assert table["anchor"] == "random"
    assert table["ratings"]["greedy"]["elo"] > table["ratings"]["random"]["elo"]
    assert set(table["per_mode_elo"]) == {"ind2", "ovt"}

    manifest = export_bundle(tiny_ckpt, str(tmp_path / "b"), modes=["ind2"],
                             elo=js, torchscript=False)
    assert manifest["elo"]["ratings"]["greedy"]["elo"] > 0
    raw = torch.load(os.path.join(str(tmp_path / "b"), "shared.pt"),
                     map_location="cpu", weights_only=True)
    assert raw["meta"]["elo"]["anchor"] == "random"
    assert raw["meta"]["elo"]["ratings"]["greedy"]["elo"] > 0


def test_elo_already_in_the_checkpoint_is_kept(tmp_path):
    model = SplendorNet(NetConfig(width=32, blocks=1))
    src = str(tmp_path / "src.pt")
    save_checkpoint(src, model, {"meta": {"elo": {"ratings": {"x": 123.0}}}})
    manifest = export_bundle(src, str(tmp_path / "b"), torchscript=False)
    assert manifest["elo"]["ratings"]["x"] == 123.0


# ── CLI ───────────────────────────────────────────────────────────────────

def test_cli_exports_a_bundle(tiny_ckpt, tmp_path, capsys):
    out = tmp_path / "dist" / "model"
    rc = export_mod.main(["--ckpt", tiny_ckpt, "--out", str(out),
                          "--modes", "all", "--no-torchscript"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "shared.pt" in printed and "generation 12" in printed
    for key in MODEL_MODES:
        assert (out / f"{key}.pt").exists()
    assert (out / "manifest.json").exists()
    assert not (out / "shared.ts").exists()
