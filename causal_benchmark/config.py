"""
config.py — Central configuration for all benchmark experiments.
This file defines all the settings for the benchmark, including:
- Reproducibility settings (base seed, number of runs)
- Experiment settings (sample sizes, DGPs, learners)
- Storage settings (results directory)
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
BASE_SEED = 42        # master seed — all run seeds are derived from this
N_RUNS = 5         # number of repeated runs per (dgp, learner, n_samples)

# ── Experiment settings ───────────────────────────────────────────────────────
SAMPLE_SIZES = [100]

DGPS = [
    # Uncomment to add more DGPs
    {"type": "NotearsDAGP", "n_nodes": 10, "n_edges": 20, "graph_type": "ER", "sem_type": "gauss"},
    # {"type": "NotearsDAGP", "n_nodes": 10, "n_edges": 20, "graph_type": "SF", "sem_type": "gauss"},
    # {"type": "AlarmDGP"},
]

LEARNERS = [
    # Uncomment to add more learners
    {"type": "GESLearner"},
    #{"type": "PClearner", "alpha": 0.05},
    # {"type": "HCSLearner", "max_indegree": 3, "epsilon": 1e-4},
]

# ── Storage ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"   # folder where CSVs are saved