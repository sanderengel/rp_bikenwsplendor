import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Sequence

from splendor.plotting.style import SERIES_COLOR, clean_axes


def plot_counts(
    data: pd.DataFrame,
    column: str,
    categories: Sequence,
    title: str,
    xlabel: str,
    figsize: tuple[float, float] = (5, 3),
) -> Figure:
    """Bar chart of how many rows fall in each category of a column."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    plot_value_counts(ax, data[column], categories, title, xlabel)
    return fig


def plot_value_counts(ax: Axes, values: pd.Series, categories: Sequence, title: str, xlabel: str) -> None:
    """Bars of how many values fall in each category, including empty ones."""
    counts = values.value_counts().reindex(categories, fill_value=0)
    ax.bar(range(len(categories)), counts.to_numpy(), color=SERIES_COLOR, width=.6)
    ax.set_xticks(range(len(categories)), [str(c) for c in categories])
    ax.set(title=title, xlabel=xlabel, ylabel="Links")
    clean_axes(ax)


def plot_distribution(
    ax: Axes,
    values: pd.Series,
    title: str,
    reference: float | None = None,
    reference_label: str = "",
    integer: bool = False,
    bins: int = 30,
) -> None:
    """Histogram of values, with an optional dashed reference line. Integer values get one bin each."""
    edges = np.arange(values.min() - .5, values.max() + 1.5) if integer else bins
    ax.hist(values, bins=edges, color=SERIES_COLOR)
    if reference is not None:
        ax.axvline(reference, color="0.4", linestyle="--", linewidth=1)
        ax.text(
            reference, ax.get_ylim()[1] * .95, f"{reference_label} ",
            color="0.4", fontsize=8, va="top", ha="right", rotation=90,
        )
    ax.set(title=title, xlabel="Value", ylabel="Links")
    clean_axes(ax)


def plot_scatter(ax: Axes, x: pd.Series, y: pd.Series, title: str, xlabel: str, xscale: str = "linear") -> None:
    """Scatter of y against x."""
    ax.scatter(x, y, s=10, alpha=.5, color=SERIES_COLOR)
    ax.set(title=title, xlabel=xlabel, xscale=xscale)
    clean_axes(ax)


def plot_distributions(
    data: pd.DataFrame,
    columns: Sequence[str],
    titles: Sequence[str],
    references: Sequence[float | None] | None = None,
    reference_label: str = "",
    integer_columns: Sequence[str] = (),
    panel_size: tuple[float, float] = (4, 4),
) -> Figure:
    """One histogram per column, side by side."""
    references = references or [None] * len(columns)
    fig, axes = plt.subplots(
        1, len(columns), figsize=(panel_size[0] * len(columns), panel_size[1]),
        squeeze=False, layout="constrained",
    )
    for ax, column, title, reference in zip(axes.flat, columns, titles, references):
        plot_distribution(ax, data[column], title, reference, reference_label, column in integer_columns)
    return fig


def plot_against(
    data: pd.DataFrame,
    x_column: str,
    y_columns: Sequence[str],
    titles: Sequence[str],
    xlabel: str,
    x_scale_factor: float = 1,
    xscale: str = "linear",
    panel_size: tuple[float, float] = (4, 3),
) -> Figure:
    """One scatter panel per y column, all against the same x."""
    fig, axes = plt.subplots(
        1, len(y_columns), figsize=(panel_size[0] * len(y_columns), panel_size[1]),
        sharex=True, squeeze=False, layout="constrained",
    )
    x = data[x_column] * x_scale_factor
    for ax, column, title in zip(axes.flat, y_columns, titles):
        plot_scatter(ax, x, data[column], title, xlabel, xscale)
    axes[0, 0].set_ylabel("Value")
    return fig


def plot_scatter_matrix(data: pd.DataFrame, columns: Sequence[str], size: float = 8) -> Figure:
    """Pairwise scatter plots of the columns, with histograms on the diagonal."""
    axes = pd.plotting.scatter_matrix(
        data[columns], figsize=(size, size), alpha=.4, color=SERIES_COLOR,
        diagonal="hist", hist_kwds={"color": SERIES_COLOR},
    )
    return axes[0, 0].get_figure()
