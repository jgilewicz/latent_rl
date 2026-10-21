import logging
from typing import Any

import wandb

LOGGER_NAME = "crafter_jepa"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def init_wandb(
    entity: str, project: str, name: str, group: str, config: dict[str, Any]
) -> None:
    wandb.init(entity=entity, project=project, name=name, group=group, config=config)


def finish_wandb() -> None:
    wandb.finish()


def log_metrics(step: int, metrics: dict[str, float]) -> None:
    wandb.log(metrics, step=step)
    formatted = " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    logging.getLogger(LOGGER_NAME).info("step=%d %s", step, formatted)
