import logging
from typing import Any

import numpy as np
import umap

from latent_rl.utils.eval_buffer import EvalBuffer
from latent_rl.utils.logger import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def umap_embeddings(
    buffer: EvalBuffer, algos: list[str], cfg: dict[str, Any]
) -> dict[str, np.ndarray]:
    embeddings = {}
    for algo in algos:
        reducer = umap.UMAP(**cfg["umap"])
        embeddings[algo] = np.asarray(reducer.fit_transform(buffer.latents(algo)))
        logger.info("UMAP of %s latents done", algo)
    return embeddings


def umap_colorings(buffer: EvalBuffer, panels: dict[str, Any]) -> dict[str, np.ndarray]:
    classes = list(panels["biome"]["classes"])
    if classes != buffer.meta["biome_names"]:
        raise ValueError(
            f"umap_panels.biome.classes {classes} must list the buffer's biomes "
            f"in order: {buffer.meta['biome_names']}"
        )
    return {
        "daylight": buffer["daylight"],
        "biome": buffer["biome"],
        "achievements": buffer["achievements"].sum(axis=1),
    }
