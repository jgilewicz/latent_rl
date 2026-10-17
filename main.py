import argparse
from pathlib import Path
from typing import Any

import crafter
import yaml

from crafter_jepa.trainers.ppo_trainer import PPOTrainer
from crafter_jepa.utils.logger import setup_logging

CONFIG_DIR = Path(__file__).parent / "config"
ALGORITHMS: dict[str, type[PPOTrainer]] = {"ppo": PPOTrainer}


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str.lower,
        default="ppo",
        choices=sorted(ALGORITHMS),
        help="algorithm to train; config read from config/<algo>_config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    algo = parse_args().algo
    cfg = load_config(CONFIG_DIR / f"{algo}_config.yaml")
    env = crafter.Env(**cfg["env"])
    trainer = ALGORITHMS[algo](**cfg["trainer"])
    trainer.train(env)


if __name__ == "__main__":
    main()
