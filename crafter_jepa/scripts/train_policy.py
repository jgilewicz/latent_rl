import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from crafter_jepa.envs.crafter_env import make_vec_env
from crafter_jepa.trainers.jepa_trainer import JEPATrainer
from crafter_jepa.trainers.ppo_trainer import PPOTrainer
from crafter_jepa.trainers.sae_trainer import SAETrainer
from crafter_jepa.trainers.vae_trainer import VAETrainer
from crafter_jepa.utils.logger import finish_wandb, init_wandb, setup_logging

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
ALGORITHMS: dict[str, type[PPOTrainer]] = {
    "ppo": PPOTrainer,
    "jepa": JEPATrainer,
    "vae": VAETrainer,
    "sae": SAETrainer,
}


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
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    algo, seed = args.algo, args.seed
    cfg = load_config(CONFIG_DIR / f"{algo}_config.yaml")
    env_cfg = dict(cfg["env"])
    n_envs = env_cfg.pop("n_envs")
    trainer_cfg = dict(cfg["trainer"])
    trainer_cfg["checkpoint_dir"] = str(
        Path(trainer_cfg["checkpoint_dir"]) / f"seed{seed}"
    )
    torch.manual_seed(seed)

    envs = make_vec_env(n_envs, seed, **env_cfg)
    init_wandb(
        **cfg["wandb"],
        name=f"{algo}-seed{seed}",
        group=algo,
        config={**cfg, "seed": seed},
    )

    trainer = ALGORITHMS[algo](**trainer_cfg)
    try:
        trainer.train(envs, n_envs)
    finally:
        envs.close()
        finish_wandb()


if __name__ == "__main__":
    main()
