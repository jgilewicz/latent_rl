from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

Algos = dict[str, dict[str, str]]
Estimates = tuple[dict[str, np.ndarray], dict[str, np.ndarray]]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def save(fig: Figure, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _spread_labels(values: list[float], min_gap: float) -> list[float]:
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for prev, cur in zip(order, order[1:], strict=False):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed.tolist()


def _curve_panel(
    ax: Axes, steps: np.ndarray, estimates: Estimates, algos: Algos
) -> None:
    points, intervals = estimates
    for algo, style in algos.items():
        ax.fill_between(steps, *intervals[algo], color=style["color"], alpha=0.18, lw=0)
        ax.plot(steps, points[algo], color=style["color"], label=style["label"])
    ax.set_xlim(0, steps[-1])
    ax.margins(y=0.08)
    ax.grid(axis="x", visible=False)


def _end_labels(ax: Axes, steps: np.ndarray, points: dict, algos: Algos) -> None:
    ends = [float(points[algo][-1]) for algo in algos]
    low, high = ax.get_ylim()
    for (algo, style), y in zip(
        algos.items(), _spread_labels(ends, 0.05 * (high - low)), strict=True
    ):
        ax.annotate(
            f"{style['label']}  {points[algo][-1]:.1f}",
            xy=(steps[-1], y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
        )


def sample_efficiency(
    steps: np.ndarray,
    estimates: Estimates,
    algos: Algos,
    budgets: list[int],
    text: dict[str, str],
) -> Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _curve_panel(ax, steps, estimates, algos)
    for budget in budgets[:-1]:
        ax.axvline(budget / 1e6, color=AXIS, lw=0.8, ls="--", zorder=0)
    _end_labels(ax, steps, estimates[0], algos)
    ax.set_xlabel(text["steps"])
    ax.set_ylabel(text["ylabel"])
    ax.set_title(
        text["curve_note"],
        color=INK_SECONDARY,
        size=9,
        fontweight="normal",
    )
    ax.legend(loc="upper left", ncols=len(algos))
    return fig


def diagnostics(
    steps: np.ndarray,
    estimates: dict[str, Estimates],
    algos: Algos,
    panels: dict,
    xlabel: str,
) -> Figure:
    n_cols = 2
    n_rows = -(-len(estimates) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8, 2.8 * n_rows), sharex=True, squeeze=False
    )
    for ax, (key, est) in zip(axes.flat, estimates.items(), strict=False):
        _curve_panel(ax, steps, est, algos)
        ax.set_title(panels[key]["title"])
        # early transients (e.g. probe R² << 0) would flatten the rest of the curve
        if "ylim" in panels[key]:
            ax.set_ylim(*panels[key]["ylim"])
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(algos))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _interval_panel(
    ax: Axes, rows: pd.DataFrame, algos: Algos, ref: float | None = None
) -> None:
    y = np.arange(len(rows))[::-1]
    colors = [algos[a]["color"] for a in rows["algo"]] if "algo" in rows else None
    colors = colors or [INK_SECONDARY] * len(rows)
    ax.hlines(y, rows["ci_low"], rows["ci_high"], colors=colors, lw=6, alpha=0.35)
    ax.scatter(rows["point"], y, color=colors, s=40, zorder=3, ec=SURFACE, lw=1.5)
    for yi, (_, r) in zip(y, rows.iterrows(), strict=True):
        ax.annotate(
            f"{r['point']:.2f}",
            xy=(r["ci_high"], yi),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    if ref is not None:
        ax.axvline(ref, color=INK_MUTED, lw=0.8, ls="--", zorder=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)


def aggregate(
    table: pd.DataFrame,
    algos: Algos,
    measures: dict[str, dict[str, str]],
    metrics: dict[str, str],
) -> Figure:
    panels = [(m, metric) for m in measures for metric in metrics]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(2.6 * len(panels), 0.45 * len(algos) + 1.2)
    )
    for i, (ax, (measure, metric)) in enumerate(zip(axes, panels, strict=True)):
        rows = table[(table["measure"] == measure) & (table["metric"] == metric)]
        rows = rows.set_index("algo").loc[list(algos)].reset_index()
        _interval_panel(ax, rows, algos)
        ax.set_title(f"{measures[measure]['title']} · {metrics[metric]}", size=10)
        ax.set_xlabel(measures[measure]["axis"], size=9)
        labels = [algos[a]["label"] for a in rows["algo"]][::-1] if i == 0 else []
        ax.set_yticks(range(len(rows)) if i == 0 else [], labels)
    fig.tight_layout()
    return fig


def probability_of_improvement(
    table: pd.DataFrame, algos: Algos, xlabel: str
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(table) + 1))
    rows = table.rename(columns={"p_x_gt_y": "point"}).reset_index(drop=True)
    _interval_panel(ax, rows.assign(algo=rows["x"]), algos, ref=0.5)
    ax.set_yticks(
        range(len(rows))[::-1],
        [
            f"{algos[x]['label']} > {algos[y]['label']}"
            for x, y in zip(rows["x"], rows["y"], strict=True)
        ],
    )
    ax.set_xlim(-0.05, 1.12)
    ax.set_xlabel(xlabel)
    return fig


def achievements(
    table: pd.DataFrame, algos: Algos, names: dict[str, str], xlabel: str
) -> Figure:
    means = table["mean"].unstack("algo")[list(algos)]
    means = means.loc[means.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(means) + 1))
    y = np.arange(len(means))
    ax.hlines(y, 0.1, means.max(axis=1).clip(lower=0.1), color=GRID, lw=1, zorder=0)
    offsets = np.linspace(-0.22, 0.22, len(algos))
    for offset, (algo, style) in zip(offsets, algos.items(), strict=True):
        ax.scatter(
            means[algo].clip(lower=0.1),
            y + offset,
            s=30,
            color=style["color"],
            label=style["label"],
            ec=SURFACE,
            lw=0.8,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_xticks([0.1, 1, 10, 100], ["≤0.1", "1", "10", "100"])
    ax.set_yticks(y, [names[name] for name in means.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower right", ncols=len(algos), bbox_to_anchor=(1, 1))
    return fig
