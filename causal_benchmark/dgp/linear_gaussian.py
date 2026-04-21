import numpy as np
import pandas as pd
from .base import dgp

class linear_gaussian(dgp):
    """
    Simple linear Gaussian DGP.
    Ground truth graph: X -> Y -> Z, X -> Z
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def simulate(self, n_samples: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)

        X = rng.normal(0, 1, n_samples)
        Y = 0.8 * X + rng.normal(0, 0.5, n_samples)
        Z = 0.5 * X + 0.6 * Y + rng.normal(0, 0.5, n_samples)

        return pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def get_ground_truth(self) -> list[tuple[str, str]]:
        return [("X", "Y"), ("X", "Z"), ("Y", "Z")]