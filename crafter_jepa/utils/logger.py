import logging

LOGGER_NAME = "crafter_jepa"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def log_metrics(step: int, metrics: dict[str, float]) -> None:
    formatted = " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    logging.getLogger(LOGGER_NAME).info("step=%d %s", step, formatted)
