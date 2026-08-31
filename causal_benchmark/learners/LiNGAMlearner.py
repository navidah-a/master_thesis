from pgmpy.utils import get_example_model
import pandas as pd
import numpy as np
from lingam import DirectLiNGAM
from structure_learner import structure_learner

class LiNGAMlearner(structure_learner):
    """
    LiNGAM (Linear Non-Gaussian Acyclic Model) using the lingam package.
    """

    def __init__(self):
        pass

    def fit(self, df: pd.DataFrame) -> list[tuple[str, str]]:

        X = df.to_numpy()
        model = DirectLiNGAM()
        model.fit(X)
        W_estimated = model.adjacency_matrix_
        edges = []
        for i in range(W_estimated.shape[0]):
            for j in range(W_estimated.shape[1]):
                if W_estimated[i, j] != 0:
                    edges.append((df.columns[i], df.columns[j]))
        return edges