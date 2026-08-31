import pandas as pd
from dodiscover import make_context
from dodiscover.toporder import DAS
from structure_learner import structure_learner

class DASlearner(structure_learner):
    """
    DAS (Discovering Ancestral Structures) using the dodiscover package.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def fit(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        context = make_context(df)
        das = DAS(alpha=self.alpha)
        das.fit(context)
        edges = das.get_edges()
        return edges