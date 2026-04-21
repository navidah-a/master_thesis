from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BIC
from structure_learner import structure_learner
import pandas as pd


class HCSLearner(structure_learner):
    """
    Hill Climb Search structure learner with BIC score.
    """

    def __init__(self, max_indegree: int = None, epsilon: float = 1e-4):
        self.max_indegree = max_indegree
        self.epsilon = epsilon

    def fit(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        hc = HillClimbSearch(df)
        model = hc.estimate(
            scoring_method=BIC(df),
            max_indegree=self.max_indegree,
            epsilon=self.epsilon,
        )
        return list(model.edges())