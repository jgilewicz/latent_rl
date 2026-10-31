import argparse
import logging
from pathlib import Path

import pandas as pd

from latent_rl.scripts.latent_comparison import check_algos, load_config
from latent_rl.utils import plotting
from latent_rl.utils.eval_buffer import EvalBuffer
from latent_rl.utils.logger import LOGGER_NAME, setup_logging
from latent_rl.xai.occlusion import occlusion

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "xai_config.yaml"
logger = logging.getLogger(LOGGER_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    cfg = load_config(parse_args().config)
    setup_logging()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    plotting.setup_style()
    buffer = EvalBuffer(Path(cfg["buffer_dir"]))
    algos = list(cfg["algos"])
    check_algos(buffer, algos)

    maps = occlusion(buffer, algos, cfg)
    pd.DataFrame(
        [
            {"algo": algo, "map": key, "row": r, "col": c, "value": value}
            for algo, algo_maps in maps.items()
            for key, grid in algo_maps.items()
            for (r, c), value in pd.DataFrame(grid).stack().items()
        ]
    ).to_csv(out / "occlusion.csv", index=False)
    fig = plotting.occlusion_grid(
        maps, cfg["algos"], cfg["occlusion_maps"], cfg["text"]["occlusion_title"]
    )
    plotting.save(fig, out, "occlusion")
    logger.info("wrote occlusion maps to %s", out)


if __name__ == "__main__":
    main()
