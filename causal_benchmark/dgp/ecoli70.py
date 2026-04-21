import pandas as pd
from pgmpy.utils import get_example_model
from .base import dgp


class Ecoli70(dgp):
    """
    Ecoli70 Bayesian network from pgmpy.
    Linear Gaussian BN with 70 variables.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._model = get_example_model("ecoli70")

    def simulate(self, n_samples: int) -> pd.DataFrame:
        df = self._model.simulate(n_samples=n_samples, seed=self.seed)
        return df

    def get_ground_truth(self) -> list[tuple[str, str]]:
        return list(self._model.edges())