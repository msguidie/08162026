"""Worker configuration — environment / ``.env`` driven.

The deployment worker is the only part of ``splendor_ai`` that runs on the
user's own machine (a Windows box with an RTX 3060, in the reference setup),
so everything it needs is a plain env var and there is no YAML.  The keys are
the ones named in ``docs/AI_BRIDGE.md`` §1 plus a few operational knobs; see
``splendor_ai/.env.example`` for the annotated list.

``load_config()`` reads ``.env`` (first hit of ``$SPLENDOR_WORKER_ENV``, the
current directory, then the ``splendor_ai/`` project directory), overlays the
real environment on top of it and returns a frozen :class:`WorkerConfig`.
Nothing here imports torch, so ``--help`` and the unit tests stay instant.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

__all__ = [
    "WorkerConfig", "load_config", "find_env_file", "mode_key",
    "MODEL_MODES", "WORKER_VERSION", "SUPPORTED_MODES",
]

#: Reported to the server in ``ai_worker_register``.
WORKER_VERSION = "1.0.0"

#: The three game modes this worker can play (contract §1).
SUPPORTED_MODES: Tuple[str, ...] = ("INDIVIDUAL", "ONE_V_TWO", "TEAM")

#: Checkpoint basenames looked up under ``MODEL_DIR`` before ``shared.pt``.
MODEL_MODES: Tuple[str, ...] = ("ind2", "ind3", "ind4", "ovt", "team")

#: ``.../splendor_ai/splendor_ai/worker/config.py`` → the project directory.
PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]


def mode_key(game_mode: Optional[str], num_players: Optional[int]) -> str:
    """The checkpoint key for a position: ``ind2|ind3|ind4|ovt|team``.

    Unknown / malformed input degrades to the 2-player individual key, which
    is the one a single-mode specialist is most likely to have been trained
    for; the caller always falls back to ``shared.pt`` anyway.
    """
    mode = (game_mode or "INDIVIDUAL").upper()
    if mode == "ONE_V_TWO":
        return "ovt"
    if mode == "TEAM":
        return "team"
    n = int(num_players or 2)
    if n <= 2:
        return "ind2"
    return "ind3" if n == 3 else "ind4"


# ── .env ──────────────────────────────────────────────────────────────────

def find_env_file(explicit: Optional[str] = None) -> Optional[Path]:
    """First existing candidate of ``explicit`` / ``$SPLENDOR_WORKER_ENV`` /
    ``./.env`` / ``<project>/.env``."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    from_env = os.environ.get("SPLENDOR_WORKER_ENV")
    if from_env:
        candidates.append(Path(from_env))
    candidates.append(Path.cwd() / ".env")
    candidates.append(PROJECT_DIR / ".env")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _read_env_file(path: Path) -> Dict[str, str]:
    """``python-dotenv`` when available, otherwise a small KEY=VALUE reader.

    The fallback keeps the worker importable (and the unit tests runnable)
    on a machine where only numpy/torch were installed.
    """
    try:
        from dotenv import dotenv_values           # type: ignore
        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    except Exception:
        out: Dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:]
            key, _, value = line.partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            out[key.strip()] = value
        return out


def _int(env: Mapping[str, str], key: str, default: int,
         minimum: int = 0) -> int:
    raw = str(env.get(key, "")).strip()
    if not raw:
        return default
    try:
        value = int(float(raw))
    except ValueError:
        return default
    return value if value >= minimum else default


def _flag(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = str(env.get(key, "")).strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# ── config ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkerConfig:
    """Everything the worker reads from the environment."""

    # -- connection ------------------------------------------------------
    server_url: str = "http://127.0.0.1:10000"
    secret: str = ""
    worker_name: str = "splendor-worker"
    version: str = WORKER_VERSION
    modes: Tuple[str, ...] = SUPPORTED_MODES
    reconnect_delay_ms: int = 1000
    reconnect_delay_max_ms: int = 30000

    # -- model -----------------------------------------------------------
    model_dir: str = "models"
    device: str = "auto"

    # -- search ----------------------------------------------------------
    search_sims: int = 4000            # hard cap; the clock usually bites first
    time_budget_ms: int = 1500         # soft: stop starting new simulations
    hard_budget_ms: int = 2500         # hard: never exceed, whatever happens
    universes: int = 16                # K determinizations per search
    root_ensemble: bool = False        # C5 colour ensemble on the root eval
    #: Milliseconds kept in hand before the server's ``deadlineMs``.
    deadline_margin_ms: int = 400
    #: Simulations between two clock reads (a clock read is ~1 % of a sim).
    sim_chunk: int = 8

    # -- operations ------------------------------------------------------
    log_dir: str = "logs"
    log_level: str = "INFO"
    seed: int = 0                      # 0 → entropy from the OS

    env_file: Optional[str] = None

    # -- derived ---------------------------------------------------------
    @property
    def model_path_dir(self) -> Path:
        return Path(self.model_dir).expanduser()

    @property
    def log_path_dir(self) -> Path:
        return Path(self.log_dir).expanduser()

    @property
    def moves_log(self) -> Path:
        return self.log_path_dir / "moves.jsonl"

    def checkpoint_candidates(self, key: str) -> List[Path]:
        """``MODEL_DIR/<mode>.pt`` then ``MODEL_DIR/shared.pt`` (§1.9)."""
        base = self.model_path_dir
        return [base / f"{key}.pt", base / "shared.pt"]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["modes"] = list(self.modes)
        data["secret"] = "<set>" if self.secret else "<missing>"
        return data


def load_config(env: Optional[Mapping[str, str]] = None,
                env_file: Optional[str] = None,
                use_dotenv: bool = True) -> WorkerConfig:
    """Build a :class:`WorkerConfig`.

    Precedence: real environment (or ``env``) > ``.env`` file > defaults.
    """
    source: Dict[str, str] = {}
    found: Optional[Path] = None
    if use_dotenv:
        found = find_env_file(env_file)
        if found is not None:
            source.update(_read_env_file(found))
    real = os.environ if env is None else env
    for key, value in real.items():
        if value is not None and str(value) != "":
            source[key] = str(value)

    modes_raw = source.get("WORKER_MODES", "")
    modes = tuple(m.strip().upper() for m in modes_raw.split(",") if m.strip()) \
        or SUPPORTED_MODES

    return WorkerConfig(
        server_url=source.get("SERVER_URL", "http://127.0.0.1:10000").rstrip("/"),
        secret=source.get("AI_WORKER_SECRET", ""),
        worker_name=source.get("WORKER_NAME", "splendor-worker"),
        modes=modes,
        reconnect_delay_ms=_int(source, "RECONNECT_DELAY_MS", 1000, 100),
        reconnect_delay_max_ms=_int(source, "RECONNECT_DELAY_MAX_MS", 30000, 1000),
        model_dir=source.get("MODEL_DIR", "models"),
        device=source.get("DEVICE", "auto").strip().lower() or "auto",
        search_sims=_int(source, "SEARCH_SIMS", 4000, 1),
        time_budget_ms=_int(source, "TIME_BUDGET_MS", 1500, 1),
        hard_budget_ms=_int(source, "HARD_BUDGET_MS", 2500, 1),
        universes=_int(source, "UNIVERSES", 16, 1),
        root_ensemble=_flag(source, "ROOT_ENSEMBLE", False),
        deadline_margin_ms=_int(source, "DEADLINE_MARGIN_MS", 400, 0),
        sim_chunk=_int(source, "SIM_CHUNK", 8, 1),
        log_dir=source.get("LOG_DIR", "logs"),
        log_level=source.get("LOG_LEVEL", "INFO").upper(),
        seed=_int(source, "SEED", 0),
        env_file=str(found) if found is not None else None,
    )
