import pandas as pd
from dagma import utils
from dagma.linear import DagmaLinear
from structure_learner import structure_learner

class DAGMAlearner(structure_learner):
    """
    DAGMA (DAG learning via Matrix Approximation) using the dagma package.
    """

    def __init__(self, w_threshold: float = 0.3, max_iter: int = 1000, learning_rate: float = 0.01):
        self.w_threshold = w_threshold
        self.max_iter = max_iter
        self.learning_rate = learning_rate
    
    def fit(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        
        X = df.to_numpy().astype(float)
        cols = list(df.columns)

        model = DagmaLinear(loss_type="l2")
        W_estimated = model.fit(X, w_threshold=self.w_threshold)

        # W[i, j] != 0 means i -> j (NOTEARS/DAGMA convention)
        edges = []
        for i in range(W_estimated.shape[0]):
            for j in range(W_estimated.shape[1]):
                if W_estimated[i, j] != 0:
                    edges.append((cols[i], cols[j]))
        
        return edges
