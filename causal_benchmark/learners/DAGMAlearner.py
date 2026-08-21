import numpy as np
from dagma import utils
from dagma.linear import DagmaLinear
from structure_learner import structure_learner

class DAGMAlearner(structure_learner):
    """
    DAGMA (DAG learning via Matrix Approximation) using the dagma package.
    """

    def __init__(self, max_iter: int = 1000, learning_rate: float = 0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        X = df.to_numpy()
        model = DagmaLinear(max_iter=self.max_iter, learning_rate=self.learning_rate)
        model.fit(X)
        W_estimated = model.W.detach().numpy()
        edges = utils.get_edges_from_weight_matrix(W_estimated, df.columns)
        return edges
