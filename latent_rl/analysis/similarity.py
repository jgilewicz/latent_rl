import itertools
from typing import Any

import numpy as np
import pandas as pd
import torch

from latent_rl.analysis.bootstrap import bootstrap, summarize
from latent_rl.utils.diagnostics import linear_cka_matrix
from latent_rl.utils.eval_buffer import EvalBuffer


def cka(buffer: EvalBuffer, algos: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    latents = [torch.as_tensor(buffer.latents(algo)) for algo in algos]

    def statistic(idx: np.ndarray) -> np.ndarray:
        rows = torch.as_tensor(idx)
        return linear_cka_matrix([z[rows] for z in latents]).numpy()

    stats = summarize(*bootstrap(buffer, statistic, "CKA", cfg["bootstrap"]))
    return pd.DataFrame(
        [
            {"x": algos[i], "y": algos[j], **{k: v[i, j] for k, v in stats.items()}}
            for i, j in itertools.permutations(range(len(algos)), 2)
        ]
    )
