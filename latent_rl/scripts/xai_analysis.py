import argparse
import logging
from pathlib import Path

from latent_rl.scripts.latent_comparison import (
    check_algos,
    load_config,
    write_intervals,
)
from latent_rl.utils import plotting
from latent_rl.utils.eval_buffer import EvalBuffer
from latent_rl.utils.logger import LOGGER_NAME, setup_logging
from latent_rl.xai.probe import probe

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

    table = probe(buffer, algos, cfg)
    selective = table["metric"].str.contains("_selectivity_")
    write_intervals(table[~selective], "probe_r2", out, cfg)
    write_intervals(table[selective], "probe_selectivity", out, cfg)
    logger.info("wrote XAI plots and tables to %s", out)


if __name__ == "__main__":
    main()
