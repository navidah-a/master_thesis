from abc import ABC, abstractmethod
import pandas as pd

class structure_learner(ABC):
    """
    Abstract base class for all structure learning algorithms.

    To add a new structure learner:
    1. Create a new file in causal_benchmark/learners/
    2. Subclass BaseStructureLearner
    3. Implement learn_structure()
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        pass

    def name(self) -> str:
        """Name for this structure learner. Defaults to class name."""
        return self.__class__.__name__
    
