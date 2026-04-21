from pgmpy.estimators import BIC, GES
from structure_learner import structure_learner
import pandas as pd


class GESLearner(structure_learner):
    """
    GES (Greedy Equivalence Search) using pgmpy's GES estimator with BIC score.
    """

    def __init__(self):
        pass

    def fit(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        #scoring = BIC(data=df)
        est = GES(df)
        model = est.estimate(scoring_method=BIC(data=df))
        return list(model.edges())


