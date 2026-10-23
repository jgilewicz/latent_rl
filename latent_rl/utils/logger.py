import logging
from pathlib import Path
from typing import Any

import wandb

LOGGER_NAME = "latent_rl"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def init_wandb(
    entity: str,
    project: str,
    name: str,
    group: str,
    config: dict[str, Any],
    tags: list[str],
) -> None:
    wandb.init(
        entity=entity,
        project=project,
        name=name,
        group=group,
        config=config,
        tags=tags,
    )


def log_model_artifact(path: Path, name: str, aliases: list[str]) -> None:
    artifact = wandb.Artifact(name, type="model")
    artifact.add_file(str(path))
    wandb.log_artifact(artifact, aliases=aliases)


def finish_wandb() -> None:
    wandb.finish()


def log_metrics(step: int, metrics: dict[str, float]) -> None:
    wandb.log(metrics, step=step)
    formatted = " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    logging.getLogger(LOGGER_NAME).info("step=%d %s", step, formatted)
