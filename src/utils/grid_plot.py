from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


plt.style.use("seaborn-v0_8-notebook")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


def create_figure_with_grid():
    """Create a matplotlib figure with a grid layout."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    return fig, ax


def plot_grid(ax, x_range, y_range=None, special_positions=None):
    """Plot a grid on the given axis."""
    if special_positions is None:
        x, y = np.meshgrid(x_range, y_range if y_range else x_range)
        ax.scatter(x, y)
    else:
        ax.scatter(special_positions[0], special_positions[1])


def plot_cuts(ax, horizontal_cuts, vertical_cuts, eps=0.3):
    """Plot horizontal and vertical cuts on the grid."""
    for cut in horizontal_cuts:
        ax.plot([cut[0] - eps, cut[2] + eps], [cut[1] + 0.5, cut[3] + 0.5], "r-")
    for cut in vertical_cuts:
        ax.plot([cut[0] + 0.5, cut[2] + 0.5], [cut[1] - eps, cut[3] + eps], "r-")


def create_and_plot_networkx_graph(edges, layout, path_length):
    """Create and plot a NetworkX graph with specified edges and layout."""
    fig, ax = create_figure_with_grid()
    if layout == "path":
        G = nx.path_graph(path_length)
    else:
        G = nx.Graph()
        G.add_edges_from(edges)
    pos = {i: (i, 0) for i in range(len(G.nodes()))} if layout == "path" else None
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        ax=ax,
        node_size=1000,
        font_size=24,
        font_color="w",
    )
    return fig


def save_figure(fig, filename):
    """Save the figure to a file."""
    fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True)


if __name__ == "__main__":
    save_path = Path("../figures/")

    fig, ax = create_figure_with_grid()
    ax.set_aspect("equal")
    plot_grid(ax, x_range=np.arange(1, 8))
    plot_cuts(
        ax,
        horizontal_cuts=[(x, y, x, y) for x in range(1, 8) for y in range(1, 7)],
        vertical_cuts=[],
    )
    save_figure(fig, save_path.joinpath("q4d_visual_1.png"))

    fig = create_and_plot_networkx_graph(edges=None, layout="path", path_length=7)
    ax.set_aspect("equal")
    save_figure(fig, "q4d_visual_2.png")

    fig, ax = create_figure_with_grid()
    plot_grid(ax, x_range=np.arange(1, 8))
    plot_cuts(
        ax,
        horizontal_cuts=[
            (1, 4, 1, 4),
            (2, 4, 2, 4),
            (2, 3, 2, 3),
            (3, 3, 3, 3),
            (4, 3, 4, 3),
            (5, 3, 5, 3),
            (6, 3, 6, 3),
            (5, 5, 5, 5),
            (6, 5, 6, 5),
            (7, 5, 7, 5),
            (7, 2, 7, 2),
            (3, 6, 3, 6),
            (4, 6, 4, 6),
            (5, 2, 5, 2),
        ],
        vertical_cuts=[
            (2, 1, 2, 1),
            (2, 2, 2, 2),
            (2, 3, 2, 3),
            (1, 4, 1, 4),
            (4, 1, 4, 1),
            (4, 2, 4, 2),
            (5, 3, 5, 3),
            (6, 3, 6, 3),
            (2, 5, 2, 5),
            (2, 6, 2, 6),
            (3, 7, 3, 7),
            (4, 4, 4, 4),
            (4, 5, 4, 5),
            (4, 6, 4, 6),
        ],
    )
    save_figure(fig, save_path.joinpath("q4d_visual_3.png"))

    fig = create_and_plot_networkx_graph(
        edges=[(0, 3), (0, 1), (1, 2), (3, 4), (4, 5), (1, 4), (0, 6), (2, 5)],
        layout="custom",
        path_length=None,
    )
    save_figure(fig, save_path.joinpath("q4d_visual_4.png"))

    fig = create_and_plot_networkx_graph(
        edges=[(0, 3), (0, 1), (1, 2), (3, 4), (4, 5), (1, 4), (1, 6), (2, 5)],
        layout="custom",
        path_length=None,
    )
    save_figure(fig, save_path.joinpath("q4d_visual_5.png"))

    fig, ax = create_figure_with_grid()
    ax.set_aspect("equal")
    plot_grid(ax, x_range=None, special_positions=(np.arange(1, 8), np.ones(7)))

    horizontal_cuts = [(i, 0, i, 0) for i in range(1, 8)] + [
        (i, 1, i, 1) for i in range(1, 8)
    ]
    vertical_cuts = [(0, 1, 0, 1), (7, 1, 7, 1)]

    plot_cuts(ax, horizontal_cuts, vertical_cuts)
    save_figure(fig, save_path.joinpath("q4d_visual_6.png"))

    fig, ax = create_figure_with_grid()
    ax.set_aspect("equal")
    plot_grid(
        ax,
        x_range=None,
        special_positions=([1, 1, 1, 2, 2, 2, 3], [1, 2, 3, 1, 2, 3, 3]),
    )

    plot_cuts(
        ax,
        horizontal_cuts=[
            (1, 0, 1, 0),
            (2, 0, 2, 0),
            (1, 3, 1, 3),
            (2, 3, 2, 3),
            (3, 3, 3, 3),
            (3, 2, 3, 2),
        ],
        vertical_cuts=[
            (0, 1, 0, 1),
            (0, 2, 0, 2),
            (0, 3, 0, 3),
            (3, 3, 3, 3),
            (2, 2, 2, 2),
            (2, 1, 2, 1),
        ],
    )
    save_figure(fig, save_path.joinpath("q4d_visual_7.png"))
