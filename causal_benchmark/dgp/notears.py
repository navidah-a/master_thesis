import numpy as np
import igraph as ig
import random
from scipy.special import expit as sigmoid
from .base import dgp
import pandas as pd


class NotearsDAGP(dgp):
    """
    A flexible DGP that generates random DAGs and simulates data from them,
    based on the simulation utilities from NOTEARS (Zheng et al., NeurIPS 2018).

    Supports three graph types:
        ER  - Erdos-Renyi:  random uniform structure
        SF  - Scale-Free:   hub-like structure (Barabasi-Albert)
        BP  - Bipartite:    two-layer structure

    And six noise types for the linear SEM:
        gauss, exp, gumbel, uniform, logistic, poisson

    Reference:
        Zheng et al. (2018) - DAGs with NO TEARS (NeurIPS 2018)
        https://github.com/xunzheng/notears  (Apache-2.0 license)
    """

    GRAPH_TYPES = ["ER", "SF", "BP"]
    SEM_TYPES   = ["gauss", "exp", "gumbel", "uniform", "logistic", "poisson"]

    def __init__(
        self,
        n_nodes: int = 10,
        n_edges: int = None,
        graph_type: str = "ER",
        sem_type: str = "gauss",
        noise_scale: float = None,
        seed: int = 42,
    ):
        """
        Args:
            n_nodes:     Number of nodes (variables) in the DAG.
            n_edges:     Expected number of edges. Defaults to 2 * n_nodes.
            graph_type:  'ER', 'SF', or 'BP'.
            sem_type:    Noise distribution: 'gauss', 'exp', 'gumbel',
                         'uniform', 'logistic', or 'poisson'.
            noise_scale: Scale of the additive noise. Defaults to 1 for all nodes.
            seed:        Random seed for reproducibility.
        """
        if graph_type not in self.GRAPH_TYPES:
            raise ValueError(f"graph_type must be one of {self.GRAPH_TYPES}")
        if sem_type not in self.SEM_TYPES:
            raise ValueError(f"sem_type must be one of {self.SEM_TYPES}")

        self.n_nodes     = n_nodes
        self.n_edges     = n_edges if n_edges is not None else 2 * n_nodes
        self.graph_type  = graph_type
        self.sem_type    = sem_type
        self.noise_scale = noise_scale
        self.seed        = seed

        # Generate the DAG once — simulate() and get_ground_truth() share the same graph
        self._set_random_seed(seed)
        B_true       = self._simulate_dag(n_nodes, self.n_edges, graph_type)
        self._W_true = self._simulate_parameter(B_true)

    # ── Public API ────────────────────────────────────────────────────────────

    def simulate(self, n_samples: int) -> pd.DataFrame:
        """
        Simulate data from the stored DAG using a linear SEM.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            A DataFrame with columns X1, X2, ..., Xn.
        """
        X = self._simulate_linear_sem(self._W_true, n_samples, self.sem_type, self.noise_scale)
        columns = [f"X{i + 1}" for i in range(self.n_nodes)]
        return pd.DataFrame(X, columns=columns)

    def get_ground_truth(self) -> np.ndarray:
        """
        Return the binary adjacency matrix (1 = edge exists, 0 = no edge).
        """
        return (self._W_true != 0).astype(int)
    
    def get_ground_truth(self) -> list[tuple[str, str]]:
        rows, cols = np.where(self._W_true != 0)
        return [(f"X{r+1}", f"X{c+1}") for r, c in zip(rows, cols)]

    def get_weighted_ground_truth(self) -> np.ndarray:
        """Return the weighted adjacency matrix (includes edge weights)."""
        return self._W_true.copy()

    def name(self) -> str:
        return f"NotearsDAGP(n={self.n_nodes}, edges={self.n_edges}, {self.graph_type}, {self.sem_type})"

    # ── Graph simulation (from notears/utils.py) ──────────────────────────────

    @staticmethod
    def _set_random_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _simulate_dag(d: int, s0: int, graph_type: str) -> np.ndarray:
        """Simulate a random DAG with expected number of edges s0."""

        def _random_permutation(M):
            P = np.random.permutation(np.eye(M.shape[0]))
            return P.T @ M @ P

        def _random_acyclic_orientation(B_und):
            return np.tril(_random_permutation(B_und), k=-1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        if graph_type == "ER":
            G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == "SF":
            G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
            B = _graph_to_adjmat(G)
        elif graph_type == "BP":
            top = int(0.2 * d)
            G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
            B = _graph_to_adjmat(G)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")

        B_perm = _random_permutation(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        return B_perm

    @staticmethod
    def _simulate_parameter(
        B: np.ndarray,
        w_ranges: tuple = ((-2.0, -0.5), (0.5, 2.0)),
    ) -> np.ndarray:
        """Assign random edge weights to a binary DAG."""
        W = np.zeros(B.shape)
        S = np.random.randint(len(w_ranges), size=B.shape)
        for i, (low, high) in enumerate(w_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            W += B * (S == i) * U
        return W

    @staticmethod
    def _simulate_linear_sem(
        W: np.ndarray,
        n: int,
        sem_type: str,
        noise_scale=None,
    ) -> np.ndarray:
        """Simulate samples from a linear SEM with the given noise type."""

        def _simulate_single_equation(X, w, scale):
            if sem_type == "gauss":
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == "exp":
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == "gumbel":
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == "uniform":
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            elif sem_type == "logistic":
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            elif sem_type == "poisson":
                x = np.random.poisson(np.exp(X @ w)) * 1.0
            else:
                raise ValueError(f"Unknown sem_type: {sem_type}")
            return x

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError("noise_scale must be a scalar or have length d")
            scale_vec = noise_scale

        G = ig.Graph.Weighted_Adjacency(W.tolist())
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d

        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        return X
