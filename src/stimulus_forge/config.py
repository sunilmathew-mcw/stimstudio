"""Load application settings and resolve local data paths."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_FILE = PACKAGE_ROOT / "resources" / "default_config.json"


def _load_env_file() -> None:
    """Optionally load KEY=VALUE pairs from a project-level .env file."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        with env_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


_load_env_file()


def _installed_data_root() -> Path:
    """Return a conventional per-user data directory for an installed build."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Stimulus Forge"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Stimulus Forge"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "stimstudio"


def _default_data_root() -> Path:
    source_data = PROJECT_ROOT / "data"
    return source_data if source_data.is_dir() else _installed_data_root()


def _absolute_path(value: str | os.PathLike[str], base: Path) -> str:
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


with CONFIG_FILE.open("r", encoding="utf-8") as config_handle:
    config = json.load(config_handle)

data_root = Path(
    _absolute_path(
        os.environ.get("STIMULUS_FORGE_DATA_DIR", _default_data_root()),
        PROJECT_ROOT,
    )
)

runtime_path_overrides = {
    "experiment_folder": "STIMULUS_FORGE_EXPERIMENT_FOLDER",
    "pics_main_folder": "STIMULUS_FORGE_PICS_FOLDER",
    "task_images_folder": "STIMULUS_FORGE_TASK_IMAGES_FOLDER",
}

for config_key, environment_key in runtime_path_overrides.items():
    value = os.environ.get(environment_key, config[config_key])
    config[config_key] = _absolute_path(value, data_root)

concepts_override = os.environ.get("STIMULUS_FORGE_CONCEPTS_FILE")
if concepts_override:
    config["concepts_file"] = _absolute_path(concepts_override, data_root)
else:
    config["concepts_file"] = _absolute_path(
        config["concepts_file"], PACKAGE_ROOT / "resources"
    )
