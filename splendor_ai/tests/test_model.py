"""Policy / value network, loss, checkpoints and evaluator (§1.5)."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Tiny CPU models are dominated by intra-op thread synchronisation (a
# Linear(860, 64) forward measures 22 ms on four threads in this container and
# 0.11 ms on one); pin the tests to one thread and let the caller override.
torch.set_num_threads(int(os.environ.get("SPLENDOR_TORCH_THREADS", "1")))

from splendor_ai.encode import OBS_DIM, OBS_VERSION       # noqa: E402
from splendor_ai import model as M                        # noqa: E402
from splendor_ai import values as V                       # noqa: E402


def _masks(batch, rng, min_legal=1):
    mask = rng.random((batch, 65)) < 0.25
    for row in range(batch):
        if mask[row].sum() < min_legal:
            mask[row, rng.integers(0, 65)] = True
    return mask


def _batch(batch=16, seed=0, n_seats=4):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((batch, OBS_DIM)).astype(np.float32)
    mask = _masks(batch, rng)
    target = mask.astype(np.float32)
    target /= target.sum(1, keepdims=True)
    z = rng.integers(-1, 2, size=(batch, 4)).astype(np.float32)
    valid = np.stack([V.z_valid_mask(n_seats)] * batch)
    return {
        "obs": torch.from_numpy(obs),
        "mask": torch.from_numpy(mask),
        "policy_target": torch.from_numpy(target),
        "z": torch.from_numpy(z * valid),
        "z_valid": torch.from_numpy(valid),
        "score_target": torch.from_numpy(
            rng.random((batch, 4)).astype(np.float32) * valid),
        "stuck_target": torch.from_numpy(
            (rng.random((batch, 4)) < 0.2).astype(np.float32) * valid),
    }


def _smoke_net(seed=0):
    torch.manual_seed(seed)
    return M.SplendorNet(M.SMOKE_CONFIG)


# ── architecture ──────────────────────────────────────────────────────────

def test_head_shapes_and_value_range():
    net = _smoke_net()
    b = _batch(8)
    out = net(b["obs"], b["mask"])
    assert set(out) == {"logits", "value", "score", "stuck"}
    assert out["logits"].shape == (8, 65)
    for key in ("value", "score", "stuck"):
        assert out[key].shape == (8, 4)
    assert out["value"].abs().max() <= 1.0            # tanh


def test_illegal_actions_are_masked_additively_inside_forward():
    net = _smoke_net()
    b = _batch(8, seed=2)
    out = net(b["obs"], b["mask"])
    logits = out["logits"]
    illegal = ~b["mask"]
    assert torch.all(logits[illegal] < -1e8)
    assert torch.allclose(logits[illegal],
                          torch.full_like(logits[illegal], M.MASK_FILL),
                          rtol=1e-6, atol=1.0)
    probs = torch.softmax(logits.detach(), dim=-1)
    assert float(probs[illegal].max()) == 0.0
    assert torch.allclose(probs.sum(-1), torch.ones(8), atol=1e-5)
    # unmasked logits are the raw head output
    raw = net(b["obs"])["logits"]
    assert torch.allclose(raw[b["mask"]], logits[b["mask"]], atol=1e-6)
    assert raw.abs().max() < 1e3


def test_forward_rejects_bad_shapes():
    net = _smoke_net()
    with pytest.raises(ValueError):
        net(torch.zeros(OBS_DIM))
    with pytest.raises(ValueError):
        net(torch.zeros(2, OBS_DIM), torch.zeros(2, 64, dtype=torch.bool))


def test_parameter_counts(capsys):
    smoke = M.SplendorNet(M.SMOKE_CONFIG)
    default = M.SplendorNet(M.NetConfig())
    n_smoke, n_default = M.count_params(smoke), M.count_params(default)
    with capsys.disabled():
        print(f"\n  params: default(width=768, blocks=10) {n_default:,} | "
              f"smoke(width=128, blocks=2) {n_smoke:,}")
    assert 11e6 < n_default < 14e6            # "~13M" in §1.5
    assert n_smoke < 300_000
    assert M.count_params(smoke, trainable_only=True) == n_smoke


def test_net_config_round_trip():
    cfg = M.NetConfig(width=64, blocks=1, aux_score=False, aux_stuck=False)
    assert M.NetConfig.from_dict(cfg.to_dict()) == cfg
    net = M.SplendorNet(cfg)
    out = net(torch.zeros(2, OBS_DIM), torch.ones(2, 65, dtype=torch.bool))
    assert set(out) == {"logits", "value"}
    with pytest.raises(RuntimeError):
        M.NetConfig.from_dict({"width": 64, "wibble": 3})


# ── loss ──────────────────────────────────────────────────────────────────

def test_loss_is_finite_and_has_every_term():
    net = _smoke_net()
    b = _batch(16, seed=3)
    loss, parts = M.compute_loss(net(b["obs"], b["mask"]), b)
    assert torch.isfinite(loss)
    assert set(parts) == {"policy", "value", "score", "stuck", "total"}
    for value in parts.values():
        assert torch.isfinite(value)
    assert float(parts["policy"]) > 0.0


def test_masked_seats_do_not_contribute_to_the_value_loss():
    net = _smoke_net()
    b = _batch(16, seed=4, n_seats=2)
    base, _ = M.compute_loss(net(b["obs"], b["mask"]), b)
    poisoned = dict(b)
    z = b["z"].clone()
    z[:, 2:] = 99.0                    # seats a 2p game does not have
    poisoned["z"] = z
    score = b["score_target"].clone()
    score[:, 2:] = -50.0
    poisoned["score_target"] = score
    other, _ = M.compute_loss(net(b["obs"], b["mask"]), poisoned)
    assert torch.allclose(base, other, atol=1e-6)


def test_z_weight_scales_the_value_term():
    net = _smoke_net()
    b = _batch(16, seed=5)
    _, full = M.compute_loss(net(b["obs"], b["mask"]), b)
    weighted = dict(b)
    weighted["z_weight"] = torch.full((16,), 0.3)
    _, scaled = M.compute_loss(net(b["obs"], b["mask"]), weighted)
    # a constant weight cancels in the normalised mean
    assert torch.allclose(full["value"], scaled["value"], atol=1e-5)


def test_loss_decreases_on_a_tiny_overfit(capsys):
    torch.manual_seed(7)
    net = M.SplendorNet(M.NetConfig(width=64, blocks=2))
    b = _batch(24, seed=6)
    # a peaked policy target (one legal action per row) so the cross entropy
    # floor is ~0 instead of the entropy of a uniform target
    target = torch.zeros_like(b["policy_target"])
    best = torch.argmax(b["mask"].float()
                        * torch.rand(b["mask"].shape), dim=-1)
    target[torch.arange(len(best)), best] = 1.0
    b["policy_target"] = target
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    first = None
    for step in range(200):
        loss, parts = M.compute_loss(net(b["obs"], b["mask"]), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
    last = float(loss.detach())
    with capsys.disabled():
        print(f"\n  overfit 200 steps: {first:.3f} -> {last:.3f}")
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first * 0.25
    with torch.no_grad():
        out = net(b["obs"], b["mask"])
    assert float(((out["value"] - b["z"]) ** 2).mean()) < 0.2
    picked = out["logits"].argmax(-1)
    assert bool(b["mask"][torch.arange(24), picked].all())   # legal
    assert bool((picked == best).all())                      # and the target


# ── checkpoints ───────────────────────────────────────────────────────────

def test_checkpoint_round_trip(tmp_path):
    net = _smoke_net(seed=11)
    path = str(tmp_path / "weights" / "latest.pt")
    M.save_checkpoint(path, net, {"step": 1234, "generation": 7,
                                  "meta": {"note": "smoke"}, "elo": 42})
    loaded, ckpt = M.load_checkpoint(path)
    assert ckpt["step"] == 1234 and ckpt["generation"] == 7
    assert ckpt["meta"]["note"] == "smoke" and ckpt["meta"]["elo"] == 42
    assert ckpt["obs_version"] == OBS_VERSION
    assert ckpt["action_version"] == M.ACTION_VERSION
    assert loaded.cfg == net.cfg
    b = _batch(8, seed=12)
    net.eval()
    loaded.eval()
    with torch.no_grad():
        a = net(b["obs"], b["mask"])
        c = loaded(b["obs"], b["mask"])
    for key in a:
        assert torch.allclose(a[key], c[key], atol=1e-6)


def test_obs_version_gate_names_the_mismatch(tmp_path):
    net = _smoke_net()
    path = str(tmp_path / "old.pt")
    M.save_checkpoint(path, net)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["obs_version"] = OBS_VERSION + 1
    torch.save(payload, path)
    with pytest.raises(RuntimeError) as excinfo:
        M.load_checkpoint(path)
    message = str(excinfo.value)
    assert "obs_version" in message and str(OBS_VERSION + 1) in message


def test_action_version_gate(tmp_path):
    net = _smoke_net()
    path = str(tmp_path / "old_actions.pt")
    M.save_checkpoint(path, net)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["action_version"] = 99
    torch.save(payload, path)
    with pytest.raises(RuntimeError) as excinfo:
        M.load_checkpoint(path)
    assert "action_version" in str(excinfo.value) and "99" in str(excinfo.value)


def test_obs_dim_mismatch_is_refused(tmp_path):
    net = M.SplendorNet(M.NetConfig(width=32, blocks=1))
    path = str(tmp_path / "narrow.pt")
    M.save_checkpoint(path, net)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["cfg"]["obs_dim"] = OBS_DIM - 1
    torch.save(payload, path)
    with pytest.raises(RuntimeError) as excinfo:
        M.load_checkpoint(path)
    assert "obs_dim" in str(excinfo.value)


# ── evaluator ─────────────────────────────────────────────────────────────

def test_evaluator_priors_are_normalised_over_legal_actions():
    net = _smoke_net(seed=13)
    ev = M.NetEvaluator(net, "cpu")
    rng = np.random.default_rng(3)
    obs = rng.standard_normal((32, OBS_DIM)).astype(np.float32)
    mask = _masks(32, rng)
    priors, values = ev.evaluate(obs, mask)
    assert priors.shape == (32, 65) and priors.dtype == np.float32
    assert values.shape == (32, 4) and values.dtype == np.float32
    assert np.isfinite(priors).all() and np.isfinite(values).all()
    assert np.allclose(priors.sum(1), 1.0, atol=1e-5)
    assert float(priors[~mask].max()) == 0.0
    assert np.abs(values).max() <= 1.0


def test_evaluator_matches_a_plain_forward_pass():
    net = _smoke_net(seed=14)
    ev = M.NetEvaluator(net, "cpu")
    rng = np.random.default_rng(4)
    obs = rng.standard_normal((6, OBS_DIM)).astype(np.float32)
    mask = _masks(6, rng)
    priors, values = ev.evaluate(obs, mask)
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(obs), torch.from_numpy(mask))
    expected = torch.softmax(out["logits"], -1).numpy()
    assert np.allclose(priors, expected / expected.sum(1, keepdims=True),
                       atol=1e-6)
    assert np.allclose(values, out["value"].numpy(), atol=1e-6)


def test_evaluator_is_batch_only():
    ev = M.NetEvaluator(_smoke_net(), "cpu")
    with pytest.raises(ValueError):
        ev.evaluate(np.zeros(OBS_DIM, np.float32), np.ones(65, np.bool_))
    with pytest.raises(ValueError):
        ev.evaluate(np.zeros((2, OBS_DIM), np.float32),
                    np.ones((3, 65), np.bool_))


def test_evaluator_handles_a_stuck_row():
    ev = M.NetEvaluator(_smoke_net(), "cpu")
    obs = np.zeros((2, OBS_DIM), np.float32)
    mask = np.zeros((2, 65), np.bool_)
    mask[0, 3] = True
    priors, values = ev.evaluate(obs, mask)
    assert priors[0, 3] == pytest.approx(1.0)
    assert float(priors[1].sum()) == 0.0          # no legal action at all
    assert np.isfinite(values).all()
