from matplotlib.axes import Axes

SERIES_COLOR = "#4a6fa5" # Single-series color for distributions and scatter plots

def clean_axes(ax: Axes) -> None:
    """Hide the top and right spines."""
    ax.spines[["top", "right"]].set_visible(False)
