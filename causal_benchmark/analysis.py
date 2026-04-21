import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(edges: list[tuple[str, str]], title: str = "Causal Graph",
    ax=None) -> None:
    """
    Plot a causal graph from an edge list.

    Args:
        edges:  List of (cause, effect) tuples
        title:  Plot title
        ax:     Optional matplotlib Axes to draw on
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    pos = nx.spring_layout(G, seed=0)
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color="steelblue",
        node_size=1500,
        font_color="white",
        font_size=12,
        arrows=True,
        arrowsize=20,
        edge_color="gray",
    )
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def compare_graphs(
    true_edges: list[tuple[str, str]],
    pred_edges: list[tuple[str, str]],
) -> None:
    """
    Side-by-side plot of true vs predicted causal graph.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_graph(true_edges, title="Ground Truth", ax=axes[0])
    plot_graph(pred_edges, title="Predicted", ax=axes[1])
    plt.show()