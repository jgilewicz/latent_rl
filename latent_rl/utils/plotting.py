from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
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


def _pair_matrix(table: pd.DataFrame, algos: list[str], column: str) -> np.ndarray:
    matrix = table.pivot(index="x", columns="y", values=column)
    return matrix.reindex(index=algos, columns=algos).fillna(1.0).to_numpy()


def cka_matrix(table: pd.DataFrame, algos: Algos, text: dict[str, str]) -> Figure:
    mean, low, high = (
        _pair_matrix(table, list(algos), column)
        for column in ("mean", "ci_low", "ci_high")
    )
    labels = [style["label"] for style in algos.values()]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(mean, cmap="Blues", vmin=0, vmax=1)
    for i, j in np.ndindex(mean.shape):
        # dark cells need light ink to keep the value readable
        color = SURFACE if mean[i, j] > 0.6 else INK
        ax.text(
            j,
            i - 0.1,
            f"{mean[i, j]:.2f}",
            ha="center",
            va="center",
            color=color,
            fontsize=11,
            fontweight="bold",
        )
        if i != j:
            ax.text(
                j,
                i + 0.22,
                f"[{low[i, j]:.2f}, {high[i, j]:.2f}]",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(text["cka_title"])
    ax.set_xlabel(text["bootstrap_note"], color=INK_MUTED, size=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.outline.set_visible(False)
    colorbar.set_label(text["cka_axis"])
    return fig


def metric_intervals(
    table: pd.DataFrame, algos: Algos, metrics: dict[str, dict[str, str]], note: str
) -> Figure:
    n_cols = min(len(metrics), 2)
    n_rows = -(-len(metrics) // n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, n_rows * (0.45 * len(algos) + 1.2) + 0.4),
        squeeze=False,
    )
    for i, (ax, (metric, style)) in enumerate(
        zip(axes.flat, metrics.items(), strict=True)
    ):
        rows = table[table["metric"] == metric].set_index("algo")
        rows = rows.loc[list(algos)].reset_index()
        _interval_panel(ax, rows.assign(point=rows["mean"]), algos)
        ax.set_title(style["title"], size=10)
        ax.set_xlabel(style["axis"], size=9)
        first_col = i % n_cols == 0
        labels = [algos[a]["label"] for a in rows["algo"]][::-1] if first_col else []
        ax.set_yticks(range(len(rows)) if first_col else [], labels)
    fig.supxlabel(note, color=INK_MUTED, size=8)
    fig.tight_layout()
    return fig


def _bare(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _umap_classes(
    ax: Axes, xy: np.ndarray, codes: np.ndarray, classes: dict[str, dict[str, str]]
) -> None:
    # common classes first so rare ones (cave: ~2% of frames) stay visible
    for code in np.argsort(np.bincount(codes, minlength=len(classes)))[::-1]:
        style = list(classes.values())[code]
        mask = codes == code
        ax.scatter(*xy[mask].T, color=style["color"], s=1, lw=0, rasterized=True)


def umap_grid(
    embeddings: dict[str, np.ndarray],
    colorings: dict[str, np.ndarray],
    algos: Algos,
    panels: dict[str, dict[str, Any]],
    title: str,
) -> Figure:
    fig, axes = plt.subplots(
        len(algos),
        len(panels),
        figsize=(3.2 * len(panels), 2.9 * len(algos) + 1),
        squeeze=False,
        layout="constrained",
    )
    # rows are sorted by episode; shuffle so no single episode paints on top
    n_frames = len(next(iter(colorings.values())))
    order = np.random.default_rng(0).permutation(n_frames)
    mappables: dict[str, PathCollection] = {}
    for row, (algo, style) in enumerate(algos.items()):
        for col, (key, panel) in enumerate(panels.items()):
            ax, xy = axes[row, col], embeddings[algo]
            if "classes" in panel:
                _umap_classes(ax, xy, colorings[key], panel["classes"])
            else:
                # the lightest steps vanish against the surface
                cmap = plt.get_cmap(panel["cmap"])(np.linspace(0.2, 1, 256))
                mappables[key] = ax.scatter(
                    *xy[order].T,
                    c=colorings[key][order],
                    cmap=ListedColormap(cmap),
                    s=1,
                    lw=0,
                    rasterized=True,
                )
            _bare(ax)
            if row == 0:
                ax.set_title(panel["title"])
        axes[row, 0].set_ylabel(style["label"], color=INK, size=11, weight="bold")
    for col, (key, panel) in enumerate(panels.items()):
        if "classes" in panel:
            handles = [
                Line2D([], [], ls="", marker="o", color=c["color"], label=c["label"])
                for c in panel["classes"].values()
            ]
            axes[-1, col].legend(
                handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=2
            )
        else:
            colorbar = fig.colorbar(
                mappables[key], ax=axes[-1, col], orientation="horizontal", shrink=0.8
            )
            colorbar.outline.set_visible(False)
            colorbar.set_label(panel["axis"], size=9)
    fig.suptitle(title, x=0.02, ha="left", weight="bold")
    return fig
