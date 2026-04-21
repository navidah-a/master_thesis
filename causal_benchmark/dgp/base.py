from abc import ABC, abstractmethod
import pandas as pd


class dgp(ABC):
    """
    Abstract base class for all Data Generating Processes.

    To add a new DGP:
    1. Create a new file in causal_benchmark/dgp/
    2. Subclass BaseDGP
    3. Implement simulate() and get_ground_truth()
    """

    @abstractmethod
    def simulate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate a dataset from this DGP.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            A pandas DataFrame with one column per variable.
        """ 


    @abstractmethod
    def get_ground_truth(self) -> list[tuple[str, str]]:
        """
        Return the true causal graph for this DGP.

        Returns:
            List of (cause, effect) string tuples.
        """

    def name(self) -> str:
        """Defaults to class name."""
        return self.__class__.__name__
    

