import numpy as np


class Metric:
    """
    Computes evaluation metrics between an estimated and a ground truth graph.

    To add a new metric, just add a new method to this class.

    Usage:
        m = Metric(estimated, ground_truth)
        print(m.shd())
        print(m.f1())
        print(m.all())
    """

    def __init__(self, estimated: np.ndarray, ground_truth: np.ndarray):
        """
        Args:
            estimated:    The estimated adjacency matrix (n x n).
            ground_truth: The true adjacency matrix (n x n).
        """
        self.estimated = estimated
        self.ground_truth = ground_truth

    def shd(self) -> float:
        """
        Structural Hamming Distance.
        Counts edge insertions, deletions, and flips needed to reach the true graph.
        Lower is better (0 = perfect).
        """
        return float(np.sum(self.estimated != self.ground_truth))

    def precision(self) -> float:
        """
        correct edges found / total edges found
        """
        tp = np.sum((self.estimated == 1) & (self.ground_truth == 1))
        fp = np.sum((self.estimated == 1) & (self.ground_truth == 0))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def recall(self) -> float:
        """
        found edges / total edges in true graph
        """
        tp = np.sum((self.estimated == 1) & (self.ground_truth == 1))
        fn = np.sum((self.estimated == 0) & (self.ground_truth == 1))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    def f1(self) -> float:
        """ mean of precision and recall."""
        p, r = self.precision(), self.recall()
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def all(self) -> dict:
        """Return all metrics as a dictionary."""
        return {
            "SHD":       self.shd(),
            "Precision": self.precision(),
            "Recall":    self.recall(),
            "F1":        self.f1(),
        }
    
def evaluate(true_edges: list[tuple[str, str]], pred_edges: list[tuple[str, str]]) -> dict:
    """Convert edge lists to adjacency matrices and compute all metrics."""
    # Get all nodes
    nodes = sorted(set(n for e in true_edges + pred_edges for n in e))
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    true_mat = np.zeros((n, n), dtype=int)
    pred_mat = np.zeros((n, n), dtype=int)

    for src, dst in true_edges:
        true_mat[idx[src], idx[dst]] = 1
    for src, dst in pred_edges:
        pred_mat[idx[src], idx[dst]] = 1

    m = Metric(pred_mat, true_mat)  
    return {k.lower(): v for k, v in m.all().items()}

