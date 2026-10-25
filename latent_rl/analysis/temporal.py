import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from latent_rl.analysis.bootstrap import Statistic, metric_table
from latent_rl.envs.crafter_env import N_ACTIONS
from latent_rl.utils.eval_buffer import EvalBuffer
from latent_rl.utils.logger import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def pixel_distances(
    frames: np.ndarray, others: np.ndarray, partner: np.ndarray, chunk: int = 4096
) -> np.ndarray:
    # chunked: the int16 difference of all frames would take ~1 GB
    return np.concatenate(
        [
            np.abs(
                others[partner[i : i + chunk]].astype(np.int16) - frames[i : i + chunk]
            ).mean(axis=(1, 2, 3))
            for i in range(0, len(frames), chunk)
        ]
    )


def dynamics_errors(buffer: EvalBuffer, algo: str, test: np.ndarray) -> np.ndarray:
    z, z_next = buffer.latents(algo), buffer.latents(algo, next_step=True)
    x = np.c_[z, np.eye(N_ACTIONS)[buffer["action"]], np.ones(len(z))]
    weights = np.linalg.lstsq(x[~test], z_next[~test], rcond=None)[0]
    return ((x @ weights - z_next) ** 2).sum(axis=1)


def r_squared(errors: np.ndarray, targets: np.ndarray) -> float:
    return float(1 - errors.sum() / ((targets - targets.mean(axis=0)) ** 2).sum())


def temporal(buffer: EvalBuffer, algos: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    # a random partner per frame gives the typical distance between frames
    partner = np.random.default_rng(cfg["bootstrap"]["seed"]).permutation(len(buffer))
    obs = buffer["obs"]
    pixels = pixel_distances(obs, buffer["next_obs"], np.arange(len(buffer)))
    logger.info(
        "pixel reference: step / random pair = %.3f",
        pixels.mean() / pixel_distances(obs, obs, partner).mean(),
    )
    test = buffer["episode"] % cfg["holdout_every"] == 0
    z = [buffer.latents(algo) for algo in algos]
    z_next = [buffer.latents(algo, next_step=True) for algo in algos]
    steps = [np.linalg.norm(b - a, axis=1) for a, b in zip(z, z_next, strict=True)]
    spreads = [np.linalg.norm(a[partner] - a, axis=1) for a in z]
    errors = [dynamics_errors(buffer, algo, test) for algo in algos]

    def r2(idx: np.ndarray, delta: bool) -> np.ndarray:
        # the model is fit once; only held-out episodes are scored
        rows = idx[test[idx]]
        return np.array(
            [
                r_squared(e[rows], b[rows] - a[rows] if delta else b[rows])
                for a, b, e in zip(z, z_next, errors, strict=True)
            ]
        )

    statistics: dict[str, Statistic] = {
        "latent_step": lambda idx: np.array(
            [s[idx].mean() / d[idx].mean() for s, d in zip(steps, spreads, strict=True)]
        ),
        "step_correlation": lambda idx: np.array(
            [spearmanr(s[idx], pixels[idx]).statistic for s in steps]
        ),
        "r2_next": lambda idx: r2(idx, delta=False),
        "r2_delta": lambda idx: r2(idx, delta=True),
    }
    return metric_table(buffer, algos, statistics, cfg)
