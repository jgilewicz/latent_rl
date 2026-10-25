from typing import Any

import numpy as np
import pandas as pd
import torch

from latent_rl.analysis.bootstrap import Statistic, metric_table
from latent_rl.utils.diagnostics import (
    effective_rank,
    nn_distance_ratios,
    two_nn_dimension,
)
from latent_rl.utils.eval_buffer import EvalBuffer


def two_nn_ratios(buffer: EvalBuffer, algo: str, theiler_window: int) -> np.ndarray:
    z = buffer.latents(algo)
    # identical frames (e.g. every policy's first frame per env) give r1 = 0
    keep = np.sort(np.unique(z, axis=0, return_index=True)[1])
    ratios = np.full(len(z), np.nan)
    ratios[keep] = nn_distance_ratios(
        torch.as_tensor(z[keep]),
        torch.as_tensor(buffer["episode"][keep]),
        torch.as_tensor(buffer["t"][keep]),
        theiler_window,
    ).numpy()
    return ratios


def dimensions(
    buffer: EvalBuffer, algos: list[str], cfg: dict[str, Any]
) -> pd.DataFrame:
    latents = [torch.as_tensor(buffer.latents(algo)) for algo in algos]
    ratios = [two_nn_ratios(buffer, algo, cfg["theiler_window"]) for algo in algos]
    statistics: dict[str, Statistic] = {
        "effective_rank": lambda idx: np.array(
            [effective_rank(z[torch.as_tensor(idx)]) for z in latents]
        ),
        "two_nn": lambda idx: np.array([two_nn_dimension(r[idx]) for r in ratios]),
    }
    return metric_table(buffer, algos, statistics, cfg)
