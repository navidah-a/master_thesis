import os
import pandas as pd
import numpy as np
from datetime import datetime

import config
import metrics
import analysis

from dgp.notears import NotearsDAGP
from dgp.alarm import AlarmDGP
from learners.GESlearner import GESLearner
from learners.PClearner import PClearner


# ── DGP and Learner factories ─────────────────────────────────────────────────

def build_dgps() -> list:
    """Instantiate DGPs from config."""
    dgps = []
    for d in config.DGPS:
        if d["type"] == "NotearsDAGP":
            dgps.append(NotearsDAGP(
                n_nodes=d["n_nodes"],
                n_edges=d["n_edges"],
                graph_type=d["graph_type"],
                sem_type=d["sem_type"],
                seed=config.BASE_SEED,
            ))
        elif d["type"] == "AlarmDGP":
            dgps.append(AlarmDGP(seed=config.BASE_SEED))
    return dgps


def build_learners() -> list:
    """Instantiate learners from config."""
    learners = []
    for l in config.LEARNERS:
        if l["type"] == "GESLearner":
            learners.append(GESLearner())
        elif l["type"] == "PClearner":
            learners.append(PClearner(alpha=l.get("alpha", 0.05)))
        elif l["type"] == "HCSLearner":
            from learners.HCSlearner import HCSLearner
            learners.append(HCSLearner(
                max_indegree=l.get("max_indegree", 3),
                epsilon=l.get("epsilon", 1e-4),
            ))
    return learners


# ── Seeding ───────────────────────────────────────────────────────────────────

def get_run_seed(run: int) -> int:
    """
    Derive a unique seed for each run from the base seed.
    This ensures reproducibility while giving different data per run.
    e.g. BASE_SEED=42, run=0 -> 420, run=1 -> 421, ...
    """
    return config.BASE_SEED * 10 + run


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(dgps: list, learners: list, sample_sizes: list, n_runs: int) -> pd.DataFrame:
    results = []

    for dgp in dgps:
        true_edges = dgp.get_ground_truth()

        for learner in learners:
            for n_samples in sample_sizes:
                for run in range(n_runs):

                    seed = get_run_seed(run)
                    np.random.seed(seed)

                    df = dgp.simulate(n_samples=n_samples)

                    try:
                        pred_edges = learner.fit(df)
                        scores = metrics.evaluate(true_edges, pred_edges)
                    except Exception as e:
                        print(
                            f"Failed: {dgp.name()} / {learner.name()} / "
                            f"n={n_samples} / run={run}: {e}"
                        )
                        scores = {"shd": None, "precision": None, "recall": None, "f1": None}

                    results.append({
                        "dgp":       dgp.name(),
                        "learner":   learner.name(),
                        "n_samples": n_samples,
                        "run":       run,
                        "seed":      seed,
                        **scores,
                    })

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std per dgp x learner x sample size."""
    return (
        results
        .groupby(["dgp", "learner", "n_samples"])
        [["shd", "precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .round(3)
    )


# ── Storage ───────────────────────────────────────────────────────────────────

def save_results(results: pd.DataFrame, summary: pd.DataFrame):
    """Save parameters, results and summary together in one file, formatted like terminal output."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    params = pd.DataFrame({
        "parameter": ["base_seed", "n_runs", "sample_sizes", "dgps", "learners"],
        "value": [
            config.BASE_SEED,
            config.N_RUNS,
            str(config.SAMPLE_SIZES),
            str([d["type"] for d in config.DGPS]),
            str([l["type"] for l in config.LEARNERS]),
        ]
    })

    path = os.path.join(config.RESULTS_DIR, f"experiment_{timestamp}.txt")

    with open(path, "w") as f:
        f.write("=== Parameters ===\n")
        f.write(params.to_string(index=False))

        f.write("\n\n=== Results ===\n")
        f.write(results.to_string(index=False))

        f.write("\n\n=== Summary ===\n")
        f.write(summary.to_string())

    print(f"\nSaved experiment to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dgps     = build_dgps()
    learners = build_learners()

    results = run_benchmark(dgps, learners, config.SAMPLE_SIZES, n_runs=config.N_RUNS)
    print(results)

    summary = summarize(results)
    print(summary)

    save_results(results, summary)

    for dgp in dgps:
        analysis.plot_graph(dgp.get_ground_truth(), title=f"Ground truth: {dgp.name()}")