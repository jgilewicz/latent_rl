import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from latent_rl.analysis.dimensions import dimensions
from latent_rl.analysis.embedding import umap_colorings, umap_embeddings
from latent_rl.analysis.similarity import cka
from latent_rl.analysis.temporal import temporal
from latent_rl.utils import plotting
from latent_rl.utils.eval_buffer import EvalBuffer
from latent_rl.utils.logger import LOGGER_NAME, setup_logging

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "latent_comparison_config.yaml"
)
logger = logging.getLogger(LOGGER_NAME)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    return parser.parse_args()


def check_algos(buffer: EvalBuffer, algos: list[str]) -> None:
    missing = set(algos) - set(buffer.algos)
    if missing:
        raise KeyError(
            f"algos {sorted(missing)} not in the eval buffer ({buffer.algos}); "
            "add their checkpoints to eval_buffer_config.yaml and rerun it"
        )


def write_intervals(
    table: pd.DataFrame, name: str, out: Path, cfg: dict[str, Any]
) -> None:
    table.to_csv(out / f"{name}.csv", index=False)
    fig = plotting.metric_intervals(
        table, cfg["algos"], cfg[name], cfg["text"]["bootstrap_note"]
    )
    plotting.save(fig, out, name)


def main() -> None:
    cfg = load_config(parse_args().config)
    setup_logging()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    plotting.setup_style()
    buffer = EvalBuffer(Path(cfg["buffer_dir"]))
    algos = list(cfg["algos"])
    check_algos(buffer, algos)

    cka_table = cka(buffer, algos, cfg)
    cka_table.to_csv(out / "cka.csv", index=False)
    fig = plotting.cka_matrix(cka_table, cfg["algos"], cfg["text"])
    plotting.save(fig, out, "cka_matrix")

    write_intervals(dimensions(buffer, algos, cfg), "dimensions", out, cfg)
    write_intervals(temporal(buffer, algos, cfg), "temporal", out, cfg)

    fig = plotting.umap_grid(
        umap_embeddings(buffer, algos, cfg),
        umap_colorings(buffer, cfg["umap_panels"]),
        cfg["algos"],
        cfg["umap_panels"],
        cfg["text"]["umap_title"],
    )
    plotting.save(fig, out, "umap")
    logger.info("wrote latent comparison plots and tables to %s", out)


if __name__ == "__main__":
    main()
