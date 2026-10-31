import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from latent_rl.utils.eval_buffer import EvalBuffer
from latent_rl.utils.logger import LOGGER_NAME
from latent_rl.analysis.bootstrap import Statistic, metric_table

logger = logging.getLogger(LOGGER_NAME)


def player_state(buffer: EvalBuffer, names: list[str]) -> np.ndarray:
    available = buffer.meta["inventory_names"]
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"unknown probe targets {unknown}; inventory has {available}")
    columns = [available.index(n) for n in names]
    return buffer["inventory"][:, columns].astype(np.float64)


def fit_probe(
    model: RegressorMixin, z: np.ndarray, y: np.ndarray, test: np.ndarray
) -> Pipeline:
    pipeline = make_pipeline(StandardScaler(), clone(model))
    return pipeline.fit(z[~test], y[~test])


def probe_errors(
    model: RegressorMixin, z: np.ndarray, y: np.ndarray, test: np.ndarray
) -> np.ndarray:
    return (y - fit_probe(model, z, y, test).predict(z)) ** 2


def r2_statistic(
    errors: list[np.ndarray], y: np.ndarray, test: np.ndarray, j: int
) -> Statistic:
    def statistic(idx: np.ndarray) -> np.ndarray:
        rows = idx[test[idx]]
        target = y[rows, j]
        sst = ((target - target.mean()) ** 2).sum()
        return np.array([1 - e[rows, j].sum() / sst for e in errors])

    return statistic


def frame_types(obs: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    # encoder-agnostic "types": clusters of 16x16 frames (mean of 4x4 blocks)
    x = obs.reshape(len(obs), 16, 4, 16, 4, 3).mean(axis=(2, 4), dtype=np.float32)
    kmeans = MiniBatchKMeans(**cfg["kmeans"])
    return kmeans.fit_predict(x.reshape(len(obs), -1))


def control_targets(y: np.ndarray, types: np.ndarray, seed: int) -> np.ndarray:
    # one random row of y per type, shared by all its frames (Hewitt & Liang)
    rng = np.random.default_rng(seed)
    table = y[rng.integers(len(y), size=types.max() + 1)]
    return table[types]


def selectivity(real: Statistic, control: Statistic) -> Statistic:
    # control R² < 0 (worse than the mean) counts as chance level, as in H&L
    def statistic(idx: np.ndarray) -> np.ndarray:
        return real(idx) - np.maximum(control(idx), 0)

    return statistic


def probe(buffer: EvalBuffer, algos: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    probe_cfg = cfg["probes"]
    targets = probe_cfg["targets"]
    y = player_state(buffer, targets)
    test = buffer["episode"] % probe_cfg["holdout_every"] == 0
    types = frame_types(buffer["obs"], probe_cfg["control"])
    tasks = {
        "real": y,
        "control": control_targets(y, types, probe_cfg["control"]["seed"]),
    }
    models: dict[str, RegressorMixin] = {
        "linear": Ridge(**probe_cfg["ridge"]),
        "mlp": MLPRegressor(**probe_cfg["mlp"]),
    }

    errors: dict[str, dict[str, list[np.ndarray]]] = {
        task: {name: [] for name in models} for task in tasks
    }
    for algo in algos:
        z = buffer.latents(algo)
        for name, model in models.items():
            for task, labels in tasks.items():
                errors[task][name].append(probe_errors(model, z, labels, test))
                logger.info("fitted %s/%s probe on the %s task", algo, name, task)

    statistics: dict[str, Statistic] = {}
    for name in models:
        for j, target in enumerate(targets):
            real = r2_statistic(errors["real"][name], y, test, j)
            control = r2_statistic(errors["control"][name], tasks["control"], test, j)
            statistics[f"{name}_{target}"] = real
            statistics[f"{name}_control_{target}"] = control
            statistics[f"{name}_selectivity_{target}"] = selectivity(real, control)
    return metric_table(buffer, algos, statistics, cfg)
