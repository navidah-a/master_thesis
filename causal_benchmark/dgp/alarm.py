import pandas as pd
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from .base import dgp


class AlarmDGP(dgp):
    """
    Alarm Bayesian network from pgmpy.
    Discrete BN with 37 variables, commonly used in causal discovery benchmarks.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._model = get_example_model("alarm")

    def simulate(self, n_samples: int) -> pd.DataFrame:
        sampler = BayesianModelSampling(self._model)
        df = sampler.forward_sample(size=n_samples, seed=self.seed)
        return df

    def get_ground_truth(self) -> list[tuple[str, str]]:
        return list(self._model.edges())