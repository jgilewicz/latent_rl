import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from latent_rl.envs.crafter_env import CELL_PX
from latent_rl.modules import ImpalaEncoder
from latent_rl.scripts.collect_eval_buffer import (
    Policy,
    build_policy,
    encode,
    load_checkpoint,
)
from latent_rl.utils.device import resolve_device
from latent_rl.utils.eval_buffer import EvalBuffer
from latent_rl.utils.logger import LOGGER_NAME
from latent_rl.xai.probe import fit_probe, player_state

logger = logging.getLogger(LOGGER_NAME)

# rows: 7 world + 2 inventory (HUD with the vitals); cols: 9 world
GRID = (9, 9)

Maps = dict[str, np.ndarray]


def load_policies(
    buffer: EvalBuffer, algos: list[str], device: torch.device
) -> dict[str, Policy]:
    min_env_steps = buffer.meta["config"]["min_env_steps"]
    policies = {}
    for algo in algos:
        path = Path(buffer.meta["sources"][algo]["checkpoint"])
        policy = build_policy(
            load_checkpoint(path, algo, min_env_steps, device), device
        )
        # post-hoc XAI is only meaningful for the policy that made the buffer
        z = encode(policy[0], buffer["obs"][:64], 64, device)
        if not np.allclose(z, buffer.latents(algo)[:64], atol=1e-4):
            raise ValueError(
                f"{path} does not reproduce z_{algo} of the eval buffer; "
                "the checkpoint changed since `just eval-buffer`, rerun it"
            )
        policies[algo] = policy
        logger.info("loaded %s policy from %s", algo, path)
    return policies


def mean_frame(obs: np.ndarray) -> np.ndarray:
    return np.round(obs.mean(axis=0, dtype=np.float32)).astype(np.uint8)


def occlude(frames: np.ndarray, row: int, col: int, fill: np.ndarray) -> np.ndarray:
    occluded = frames.copy()
    rows = slice(row * CELL_PX, (row + 1) * CELL_PX)
    cols = slice(col * CELL_PX, (col + 1) * CELL_PX)
    occluded[:, rows, cols] = fill[rows, cols]

    return occluded


def occlusion_maps(
    encoder: ImpalaEncoder,
    probe: Pipeline,
    frames: np.ndarray,
    fill: np.ndarray,
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(encoder.parameters()).device
    z = encode(encoder, frames, batch, device)
    y_hat = probe.predict(z)

    n = z.shape[0]
    k = y_hat.shape[1]

    dz = np.empty((n, *GRID))
    dy = np.empty((n, *GRID, k))

    for row, col in np.ndindex(GRID):
        z_occ = encode(encoder, occlude(frames, row, col, fill), batch, device)
        dz[:, row, col] = np.linalg.norm(z_occ - z, axis=1)
        dy[:, row, col] = np.abs(probe.predict(z_occ) - y_hat)

    return dz, dy


def summarize_maps(
    dz: np.ndarray, dy: np.ndarray, z: np.ndarray, targets: list[str]
) -> Maps:
    spread = np.sqrt(((z - z.mean(axis=0)) ** 2).sum(axis=1).mean())

    maps = {"latent": dz.mean(axis=0) / spread}
    for j, target in enumerate(targets):
        maps[target] = dy[..., j].mean(axis=0)
    return maps


def occlusion(
    buffer: EvalBuffer, algos: list[str], cfg: dict[str, Any]
) -> dict[str, Maps]:
    occ_cfg, probe_cfg = cfg["occlusion"], cfg["probes"]
    device = resolve_device(occ_cfg["device"], "occlusion.device")
    policies = load_policies(buffer, algos, device)
    y = player_state(buffer, probe_cfg["targets"])
    test = buffer["episode"] % probe_cfg["holdout_every"] == 0
    rng = np.random.default_rng(occ_cfg["seed"])
    rows = np.sort(rng.choice(np.flatnonzero(test), occ_cfg["n_frames"], False))
    frames = buffer["obs"][rows]
    fill = mean_frame(buffer["obs"])

    maps = {}
    for algo, (encoder, _) in policies.items():
        z = buffer.latents(algo)
        probe = fit_probe(Ridge(**probe_cfg["ridge"]), z, y, test)
        dz, dy = occlusion_maps(encoder, probe, frames, fill, occ_cfg["batch"])
        maps[algo] = summarize_maps(dz, dy, z, probe_cfg["targets"])
        logger.info("occluded %d cells on %d %s frames", dz[0].size, len(rows), algo)
    return maps
