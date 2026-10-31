import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from latent_rl.analysis.bootstrap import Statistic, metric_table
from latent_rl.modules import ActorCritic
from latent_rl.utils.device import resolve_device
from latent_rl.utils.eval_buffer import EvalBuffer
from latent_rl.utils.logger import LOGGER_NAME
from latent_rl.xai.occlusion import load_policies
from latent_rl.xai.probe import fit_probe, player_state

logger = logging.getLogger(LOGGER_NAME)


@torch.no_grad()
def policy_logits(actor_critic: ActorCritic, z: np.ndarray, batch: int) -> np.ndarray:

    device = next(actor_critic.parameters()).device
    chunks = [
        actor_critic.actor(torch.as_tensor(z[i : i + batch], device=device))
        .cpu()
        .numpy()
        for i in range(0, len(z), batch)
    ]

    return np.concatenate(chunks)


def mean_ablate(z: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = z - z.mean(axis=0)

    coords = centered @ basis
    projection = coords @ basis.T

    return (z - projection).astype(z.dtype)


def policy_kl(logits: np.ndarray, ablated: np.ndarray) -> np.ndarray:
    log_p = torch.log_softmax(torch.as_tensor(logits), dim=1).numpy()
    log_q = torch.log_softmax(torch.as_tensor(ablated), dim=1).numpy()

    p = np.exp(log_p)
    return np.sum(p * (log_p - log_q), axis=1)


def probe_basis(probe: Pipeline) -> np.ndarray:
    s = probe[0].scale_
    W = probe[-1].coef_

    directions = (W / s).T

    q, _ = np.linalg.qr(directions)

    return q


def pca_basis(z: np.ndarray, k: int) -> np.ndarray:
    centered = z - z.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    return vt[:k].T


def random_basis(d: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    matrix = rng.standard_normal((d, r))
    q, _ = np.linalg.qr(matrix)

    return q


def subspaces(
    z: np.ndarray, probe: Pipeline, cfg: dict[str, Any]
) -> dict[str, np.ndarray]:
    vitals = probe_basis(probe)
    bases = {
        "vitals": vitals,
        # control for `vitals`: same number of directions, unrelated to vitals
        "random": random_basis(z.shape[1], vitals.shape[1], cfg["seed"]),
    }
    for k in cfg["pca_dims"]:
        bases[f"pca_{k}"] = pca_basis(z, k)
    return bases


def mean_statistic(values: list[np.ndarray]) -> Statistic:
    def statistic(idx: np.ndarray) -> np.ndarray:
        return np.array([v[idx].mean() for v in values])

    return statistic


def ablation(buffer: EvalBuffer, algos: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    abl_cfg, probe_cfg = cfg["ablation"], cfg["probes"]
    device = resolve_device(abl_cfg["device"], "ablation.device")
    policies = load_policies(buffer, algos, device)
    y = player_state(buffer, probe_cfg["targets"])
    test = buffer["episode"] % probe_cfg["holdout_every"] == 0

    kl: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for algo, (_, actor_critic) in policies.items():
        z = buffer.latents(algo)
        probe = fit_probe(Ridge(**probe_cfg["ridge"]), z, y, test)
        logits = policy_logits(actor_critic, z, abl_cfg["batch"])
        for name, basis in subspaces(z, probe, abl_cfg).items():
            ablated = policy_logits(
                actor_critic, mean_ablate(z, basis), abl_cfg["batch"]
            )
            kl[f"kl_{name}"].append(policy_kl(logits, ablated))
        logger.info("ablated %d subspaces of %s latents", len(kl), algo)

    statistics = {name: mean_statistic(values) for name, values in kl.items()}
    return metric_table(buffer, algos, statistics, cfg)
