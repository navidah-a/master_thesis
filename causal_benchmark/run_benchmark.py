import pandas as pd
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

from dgp import linear_gaussian
from dgp.ecoli70 import Ecoli70
from dgp.alarm import AlarmDGP     
from learners import PClearner, GESLearner, HCSLearner
import metrics, analysis

def run_benchmark(
    dgps: list,
    learners: list,
    sample_sizes: list[int],
    n_runs: int = 5,
) -> pd.DataFrame:

    results = []

    for dgp in dgps:
        true_edges = dgp.get_ground_truth()

        for learner in learners:
            for n_samples in sample_sizes:
                for run in range(n_runs):

                    dgp.seed = run
                    df = dgp.simulate(n_samples=n_samples)

                    try:
                        pred_edges = learner.fit(df)
                        scores = metrics.evaluate(true_edges, pred_edges)
                    except Exception as e:
                        print(
                            f"Failed: {dgp.name()} / {learner.name()} / "
                            f"n={n_samples} / run={run}: {e}"
                        )
                        scores = {
                            "shd": None,
                            "precision": None,
                            "recall": None,
                            "f1": None,
                        }

                    results.append({
                        "dgp": dgp.name(),
                        "learner": learner.name(),
                        "n_samples": n_samples,
                        "run": run,
                        **scores,
                    })

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean std per dgp x learner x sample size."""
    return (
        results
        .groupby(["dgp", "learner", "n_samples"])
        [["shd", "precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .round(3)
    )


if __name__ == "__main__":
    dgps = [
        #linear_gaussian(),
        #Ecoli70(),
        AlarmDGP(),
    ]
    learners = [
        #PClearner(alpha=0.05),
        #GESLearner(),
        HCSLearner(max_indegree=3, epsilon=1e-4),
    ]
    sample_sizes = [100] # tune here 500, 1000, 5000

    results = run_benchmark(dgps, learners, sample_sizes, n_runs=5)
    print(results)

    summary = summarize(results)
    print(summary)

    # Visualize ground truth graphs
    for dgp in dgps:
        analysis.plot_graph(dgp.get_ground_truth(), title=f"Ground truth: {dgp.name()}")

    
