import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from latent_rl.utils.eval_buffer import EvalBuffer
from latent_rl.utils.logger import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

Statistic = Callable[[np.ndarray], np.ndarray]


def bootstrap(
    buffer: EvalBuffer, statistic: Statistic, name: str, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    full = statistic(np.arange(len(buffer)))
    # same seed for every statistic: all of them see the same resamples
    rng = np.random.default_rng(cfg["seed"])
    samples = []
    for rep in range(cfg["reps"]):
        samples.append(statistic(buffer.resample_episodes(rng)))
        if (rep + 1) % 250 == 0:
            logger.info("%s bootstrap %d/%d", name, rep + 1, cfg["reps"])
    return full, np.stack(samples)


def summarize(full: np.ndarray, samples: np.ndarray) -> dict[str, np.ndarray]:
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    return {"full": full, "mean": samples.mean(axis=0), "ci_low": low, "ci_high": high}


def metric_table(
    buffer: EvalBuffer,
    algos: list[str],
    statistics: dict[str, Statistic],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for metric, statistic in statistics.items():
        stats = summarize(*bootstrap(buffer, statistic, metric, cfg["bootstrap"]))
        rows += [
            {"algo": algo, "metric": metric, **{k: v[i] for k, v in stats.items()}}
            for i, algo in enumerate(algos)
        ]
    return pd.DataFrame(rows)
