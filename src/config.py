from pathlib import Path
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "training.toml"


def load_training_config(config_path: Path | None = None):
    path = config_path or CONFIG_PATH
    with path.open("rb") as f:
        return tomllib.load(f)
