import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from latent_rl.envs.crafter_env import make_vec_env
from latent_rl.trainers.jepa_trainer import JEPATrainer
from latent_rl.trainers.ppo_trainer import PPOTrainer
from latent_rl.trainers.sae_trainer import SAETrainer
from latent_rl.trainers.vae_trainer import VAETrainer
from latent_rl.utils.logger import finish_wandb, init_wandb, setup_logging

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
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=None,
        help="override trainer.n_iterations, e.g. for a smoke test",
    )
    parser.add_argument(
        "--tag",
        default="dev",
        help="W&B experiment tag, e.g. main-5m or smoke",
    )
    return parser.parse_args()


def budget_label(trainer_cfg: dict[str, Any], n_envs: int) -> str:
    steps = trainer_cfg["n_iterations"] * trainer_cfg["n_steps"] * n_envs
    return f"{round(steps / 1e6, 1):g}M"


def main() -> None:
    setup_logging()
    args = parse_args()
    algo, seed = args.algo, args.seed
    cfg = load_config(CONFIG_DIR / f"{algo}_config.yaml")
    env_cfg = dict(cfg["env"])
    n_envs = env_cfg.pop("n_envs")
    trainer_cfg = dict(cfg["trainer"])
    if args.n_iterations is not None:
        trainer_cfg["n_iterations"] = args.n_iterations
    trainer_cfg["checkpoint_dir"] = str(
        Path(trainer_cfg["checkpoint_dir"]) / f"seed{seed}"
    )
    run_config = {**cfg, "trainer": trainer_cfg, "algo": algo, "seed": seed}
    torch.manual_seed(seed)

    envs = make_vec_env(n_envs, seed, **env_cfg)
    init_wandb(
        **cfg["wandb"],
        name=f"{algo}_crafter_s{seed}",
        group=algo,
        config=run_config,
        tags=[args.tag, algo, "crafter", budget_label(trainer_cfg, n_envs)],
    )

    trainer = ALGORITHMS[algo](
        **trainer_cfg,
        artifact_name=f"ckpt-{algo}-s{seed}",
        checkpoint_meta={"algo": algo, "seed": seed, "config": run_config},
    )
    try:
        trainer.train(envs, n_envs)
    finally:
        envs.close()
        finish_wandb()


if __name__ == "__main__":
    main()
