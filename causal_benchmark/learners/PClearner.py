from pgmpy.estimators import PC
from structure_learner import structure_learner
import pandas as pd

class PClearner(structure_learner):
    """
    PC algorithm for causal structure learning.
    """
    def __init__(self, alpha: float = 0.05, ci_test: str = "pearsonr"):
        """
        Args:
            alpha: Significance level for conditional independence tests.
            ci_test: Type of conditional independence test to use (e.g. "pearsonr", "spearmanr", "kendalltau").
        """
        self.alpha = alpha
        self.ci_test = ci_test
    
    def fit(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        est = PC(data=df)
        model = est.estimate(
            significance_level=self.alpha,
            ci_test=self.ci_test,
            return_type="cpdag",
        )
        return list(model.edges())