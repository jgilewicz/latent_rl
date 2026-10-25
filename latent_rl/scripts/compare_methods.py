import argparse
import itertools
import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
import yaml
from rliable import library as rly
from rliable import metrics
from scipy import stats

from latent_rl.utils import plotting
from latent_rl.utils.logger import LOGGER_NAME, setup_logging

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "compare_config.yaml"
)
AGGREGATES: dict[str, Callable[[np.ndarray], float]] = {
    "Mean": metrics.aggregate_mean,
    "Median": metrics.aggregate_median,
    "IQM": metrics.aggregate_iqm,
}
logger = logging.getLogger(LOGGER_NAME)
# rliable 1.2 passes `random_state` to arch, which still honors it but warns
warnings.filterwarnings(
    "ignore", message="random_state is deprecated", category=FutureWarning
)

Runs = dict[str, list[tuple[int, pd.DataFrame]]]
Scores = dict[str, np.ndarray]


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    unknown = set(cfg["aggregate_metrics"]) - set(AGGREGATES)
    if unknown:
        raise ValueError(
            f"unknown aggregate_metrics {sorted(unknown)} in {path}; "
            f"choose from {list(AGGREGATES)}"
        )
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    return parser.parse_args()


def fetch_runs(cfg: dict[str, Any]) -> Runs:
    wb = cfg["wandb"]
    runs: Runs = {algo: [] for algo in cfg["algos"]}
    for run in wandb.Api().runs(
        f"{wb['entity']}/{wb['project']}", filters={"tags": wb["tag"]}
    ):
        if run.state != "finished":
            raise RuntimeError(
                f"run {run.name} is {run.state}; wait for it to finish "
                f"or drop the {wb['tag']!r} tag from it in W&B"
            )
        history = run.history(samples=100_000, pandas=True)
        runs[run.config["algo"]].append(
            (run.config["seed"], history.set_index("env_steps"))
        )
        logger.info("fetched %s (%d iterations)", run.name, len(history))
    missing = [algo for algo, seeds in runs.items() if not seeds]
    if missing:
        raise RuntimeError(f"no finished {wb['tag']!r} runs for {missing} in W&B")
    return {algo: sorted(seeds, key=lambda s: s[0]) for algo, seeds in runs.items()}


def window_means(history: pd.DataFrame, window: int, n_windows: int) -> pd.DataFrame:
    # the final iteration overshoots the budget by <1 batch; fold it into the
    # last window instead of creating a near-empty one
    index = np.minimum((history.index - 1) // window, n_windows - 1)
    windows = history.select_dtypes("number").groupby(index).mean()
    success = 100 * windows.filter(like="achievement/")
    windows["crafter_score"] = np.exp(np.log1p(success).mean(axis=1)) - 1
    return windows


def interval_estimates(
    scores: dict[str, Any], fn: Callable, reps: int, seed: int
) -> plotting.Estimates:
    return rly.get_interval_estimates(
        scores, fn, reps=reps, random_state=np.random.RandomState(seed)
    )


def curve_estimates(curves: Scores, boot: dict[str, int]) -> plotting.Estimates:
    def iqm_per_window(x: np.ndarray) -> np.ndarray:
        return np.array([metrics.aggregate_iqm(x[:, w]) for w in range(x.shape[1])])

    return interval_estimates(curves, iqm_per_window, boot["curve_reps"], boot["seed"])


def aggregate(
    scores: dict[str, Scores], cfg: dict[str, Any], budget: int
) -> pd.DataFrame:
    names = cfg["aggregate_metrics"]

    def all_metrics(x: np.ndarray) -> np.ndarray:
        return np.array([AGGREGATES[name](x) for name in names])

    rows = []
    for measure, per_algo in scores.items():
        points, intervals = interval_estimates(
            per_algo, all_metrics, cfg["bootstrap"]["reps"], cfg["bootstrap"]["seed"]
        )
        rows += [
            {
                "budget": budget,
                "measure": measure,
                "metric": name,
                "algo": algo,
                "point": points[algo][i],
                "ci_low": intervals[algo][0, i],
                "ci_high": intervals[algo][1, i],
            }
            for algo in per_algo
            for i, name in enumerate(names)
        ]
    return pd.DataFrame(rows)


def significance(scores: Scores, boot: dict[str, int], budget: int) -> pd.DataFrame:
    pairs = list(itertools.combinations(list(scores)[::-1], 2))
    points, intervals = interval_estimates(
        {f"{x},{y}": [scores[x], scores[y]] for x, y in pairs},
        metrics.probability_of_improvement,
        boot["reps"],
        boot["seed"],
    )
    rows = []
    for x, y in pairs:
        low, high = intervals[f"{x},{y}"].ravel()
        welch = stats.ttest_ind(scores[x].ravel(), scores[y].ravel(), equal_var=False)
        rows.append(
            {
                "budget": budget,
                "x": x,
                "y": y,
                "p_x_gt_y": points[f"{x},{y}"],
                "ci_low": low,
                "ci_high": high,
                # rliable: a difference is significant when the CI excludes 0.5
                "significant": bool(low > 0.5 or high < 0.5),
                "welch_t": welch.statistic,
                "welch_p": welch.pvalue,
            }
        )
    return pd.DataFrame(rows)


def achievement_table(windows: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    final = {
        algo: 100 * pd.concat([w.iloc[-1] for w in runs], axis=1)
        for algo, runs in windows.items()
    }
    table = pd.concat(
        {
            algo: pd.DataFrame({"mean": f.mean(axis=1), "std": f.std(axis=1)})
            for algo, f in final.items()
        },
        names=["algo", "achievement"],
    )
    table = table[
        table.index.get_level_values("achievement").str.startswith("achievement/")
    ]
    return table.rename(index=lambda k: k.removeprefix("achievement/"), level=1)


def budget_stats(
    curves: dict[str, Scores], seeds: dict[str, list[int]], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregates, tests, per_run = [], [], []
    for budget in cfg["budgets"]:
        w = budget // cfg["window"] - 1
        scores = {
            measure: {algo: c[:, [w]] for algo, c in curves[measure].items()}
            for measure in cfg["measures"]
        }
        aggregates.append(aggregate(scores, cfg, budget))
        tests.append(significance(scores["crafter_score"], cfg["bootstrap"], budget))
        per_run += [
            {"budget": budget, "algo": algo, "seed": s, "measure": m, "value": v}
            for m, per_algo in scores.items()
            for algo, c in per_algo.items()
            for s, v in zip(seeds[algo], c.ravel(), strict=True)
        ]
    return pd.DataFrame(per_run), pd.concat(aggregates), pd.concat(tests)


def make_plots(
    curves: dict[str, Scores],
    tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    cfg: dict[str, Any],
    out: Path,
) -> None:
    algos, budgets, window, text = (
        cfg["algos"],
        cfg["budgets"],
        cfg["window"],
        cfg["text"],
    )
    steps = np.arange(1, max(budgets) // window + 1) * window / 1e6
    aggregates, tests, achievements = tables
    for budget in budgets:
        tag = f"{budget // 1_000_000}M"
        fig = plotting.aggregate(
            aggregates[aggregates["budget"] == budget],
            algos,
            cfg["measures"],
            cfg["aggregate_metrics"],
        )
        plotting.save(fig, out, f"aggregate_{tag}")
        fig = plotting.probability_of_improvement(
            tests[tests["budget"] == budget], algos, text["poi"]
        )
        plotting.save(fig, out, f"probability_of_improvement_{tag}")

    estimates = curve_estimates(curves["crafter_score"], cfg["bootstrap"])
    curve_text = {**text, "ylabel": cfg["measures"]["crafter_score"]["axis"]}
    fig = plotting.sample_efficiency(steps, estimates, algos, budgets, curve_text)
    plotting.save(fig, out, "sample_efficiency")
    diag = {
        key: curve_estimates(curves[key], cfg["bootstrap"])
        for key in cfg["diagnostics"]
    }
    fig = plotting.diagnostics(steps, diag, algos, cfg["diagnostics"], text["steps"])
    plotting.save(fig, out, "latent_diagnostics")
    fig = plotting.achievements(
        achievements, algos, cfg["achievements"], text["success"]
    )
    plotting.save(fig, out, "achievements")


def main() -> None:
    cfg = load_config(parse_args().config)
    setup_logging()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    plotting.setup_style()
    n_windows = max(cfg["budgets"]) // cfg["window"]

    runs = fetch_runs(cfg)
    windows = {
        algo: [window_means(h, cfg["window"], n_windows) for _, h in seeds]
        for algo, seeds in runs.items()
    }
    curves = {
        key: {
            algo: np.stack([w[key].to_numpy() for w in ws])
            for algo, ws in windows.items()
        }
        for key in [*cfg["measures"], *cfg["diagnostics"]]
    }

    seeds = {algo: [s for s, _ in rs] for algo, rs in runs.items()}
    per_run, aggregates, tests = budget_stats(curves, seeds, cfg)
    achievements = achievement_table(windows)
    per_run.to_csv(out / "scores.csv", index=False)
    aggregates.to_csv(out / "aggregate.csv", index=False)
    tests.to_csv(out / "significance.csv", index=False)
    achievements.to_csv(out / "achievements.csv")
    make_plots(curves, (aggregates, tests, achievements), cfg, out)
    logger.info("wrote plots and tables to %s", out)


if __name__ == "__main__":
    main()
