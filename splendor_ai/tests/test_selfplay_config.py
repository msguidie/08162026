"""Config tree, YAML loading and ``--set`` overrides (selfplay/config.py)."""

import os

import pytest
import yaml

from splendor_ai.selfplay.config import (MODE_SPECS, RunConfig, apply_overrides,
                                         config_to_dict, dump_config,
                                         load_config, normalise_mixture)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "configs")


def test_defaults_are_valid():
    cfg = load_config(None, [])
    assert cfg.selfplay.actors >= 1
    assert cfg.selfplay.pcr_full_prob == 0.25
    assert cfg.replay.window_start == 4 and cfg.replay.window_end == 20


@pytest.mark.parametrize("name", ["smoke_cpu", "smoke_short", "nscc_4xa100"])
def test_shipped_configs_load(name):
    cfg = load_config(os.path.join(CONFIG_DIR, f"{name}.yaml"))
    assert cfg.net.width > 0
    assert cfg.search_full.sims >= cfg.search_fast.sims
    # every mixture key resolves to a real (players, mode, layout)
    for phase in [cfg.phase_for(0), cfg.phase_for(10**9)]:
        for key in phase.mixture:
            assert key in MODE_SPECS


def test_nscc_layout_matches_the_design_doc():
    cfg = load_config(os.path.join(CONFIG_DIR, "nscc_4xa100.yaml"))
    assert cfg.learner.device == "cuda:0"
    assert cfg.inference.mode == "server"
    assert cfg.inference.devices == ["cuda:1", "cuda:2", "cuda:3"]
    assert (cfg.selfplay.actors, cfg.selfplay.games_per_actor) == (56, 24)
    assert (cfg.search_full.sims, cfg.search_fast.sims) == (600, 120)
    assert cfg.search_full.universes == 6
    assert (cfg.net.width, cfg.net.blocks) == (768, 10)
    assert cfg.learner.batch == 4096
    assert (cfg.replay.window_start, cfg.replay.window_end) == (4, 20)
    assert cfg.selfplay.mixed_game_frac == 0.25
    assert cfg.selfplay.win_threshold is None      # never for a real run
    # curriculum: 2p only until 300k games, mixture afterwards
    assert cfg.phase_for(0).mixture == {"ind2": 1.0}
    assert cfg.phase_for(299_999).mixture == {"ind2": 1.0}
    assert set(cfg.phase_for(500_000).mixture) > {"ind2"}


def test_smoke_cpu_is_the_g3_configuration():
    cfg = load_config(os.path.join(CONFIG_DIR, "smoke_cpu.yaml"))
    assert (cfg.net.width, cfg.net.blocks) == (128, 2)
    assert cfg.inference.mode == "inproc"
    assert (cfg.selfplay.actors, cfg.selfplay.games_per_actor) == (2, 8)
    assert (cfg.search_full.sims, cfg.search_fast.sims) == (48, 12)
    assert cfg.search_full.universes == 2
    assert cfg.selfplay.win_threshold == 8         # documented smoke shortcut
    assert cfg.selfplay.augment_rotations == 5
    assert cfg.learner.batch == 256


def test_set_overrides_parse_types():
    cfg = load_config(None, ["learner.batch=512", "learner.bf16=false",
                             "selfplay.pcr_full_prob=0.5",
                             "inference.devices=[cuda:1, cuda:2]",
                             "selfplay.mode_mixture={ind2: 0.5, ovt: 0.5}",
                             "run_dir=/tmp/x"])
    assert cfg.learner.batch == 512 and cfg.learner.bf16 is False
    assert cfg.selfplay.pcr_full_prob == 0.5
    assert cfg.inference.devices == ["cuda:1", "cuda:2"]
    assert cfg.selfplay.mode_mixture == {"ind2": 0.5, "ovt": 0.5}
    assert cfg.run_dir == "/tmp/x"


def test_unknown_key_raises():
    with pytest.raises(ValueError) as exc:
        load_config(None, ["learner.bacth=512"])
    assert "bacth" in str(exc.value)
    with pytest.raises(ValueError):
        load_config(None, ["selfplay.mode_mixture={ind9: 1.0}"])


def test_unknown_set_key_names_the_flag():
    # `scripts/nscc_train.pbs` used to pass `--set job_id=...`; a scheduler id
    # is not a config leaf and the failure has to say so, naming the override.
    with pytest.raises(ValueError) as exc:
        load_config(None, ["job_id=12345.pbs101"])
    message = str(exc.value)
    assert "--set" in message and "job_id" in message and "RunConfig" in message
    assert "run_dir" in message                    # the known keys are listed

    with pytest.raises(ValueError) as exc:
        load_config(None, ["selfplay.actorz=4"])
    assert "SelfPlayConfig" in str(exc.value) and "actorz" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        load_config(None, ["run_dir.sub=1"])
    assert "not a section" in str(exc.value)

    # …while the shapes the PBS script really uses keep working.
    cfg = load_config(None, ["run_dir=runs/nscc",
                             "selfplay.mode_mixture={ind2: 1.0}"])
    assert cfg.run_dir == "runs/nscc"


def test_job_id_comes_from_the_environment(monkeypatch):
    from splendor_ai.selfplay import train as train_mod

    monkeypatch.delenv(train_mod.JOB_ID_ENV, raising=False)
    assert train_mod.job_id() is None
    monkeypatch.setenv(train_mod.JOB_ID_ENV, "  12345.pbs101  ")
    assert train_mod.job_id() == "12345.pbs101"
    monkeypatch.setenv(train_mod.JOB_ID_ENV, "")
    assert train_mod.job_id() is None


def test_win_threshold_is_individual_only():
    with pytest.raises(ValueError):
        load_config(None, ["selfplay.win_threshold=8",
                           "selfplay.mode_mixture={team_adj: 1.0}"])


def test_paths_and_roundtrip(tmp_path):
    cfg = load_config(None, [f"run_dir={tmp_path}/run"])
    cfg.make_dirs()
    assert os.path.isdir(cfg.weights_dir) and os.path.isdir(cfg.checkpoints_dir)
    path = dump_config(cfg, os.path.join(cfg.run_dir, "config.yaml"))
    again = load_config(path, [])
    assert config_to_dict(again) == config_to_dict(cfg)


def test_normalise_mixture_drops_zero_weights():
    names, weights = normalise_mixture({"ind2": 3.0, "ind3": 1.0, "ovt": 0.0})
    assert set(names) == {"ind2", "ind3"}
    assert abs(sum(weights) - 1.0) < 1e-9
    assert abs(dict(zip(names, weights))["ind2"] - 0.75) < 1e-9


def test_curriculum_phase_selection():
    cfg = load_config(None, [
        "selfplay.phases=[{until_games: 10, mixture: {ind2: 1.0}, sims_full: 8}, "
        "{until_games: null, mixture: {ind4: 1.0}}]"])
    assert cfg.phase_for(0).sims_full == 8
    assert cfg.phase_for(9).mixture == {"ind2": 1.0}
    assert cfg.phase_for(10).mixture == {"ind4": 1.0}
