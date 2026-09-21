A benchmarking framework for causal structure learning algorithms, developed as part of a master's thesis. The goal is to make it easy to compare how well different algorithms recover the true causal graph from data, across a variety of datasets and evaluation metrics.


# Setup

## R Dependency (for `cdt.metrics.SID`)

The `SID` metric (via the `cdt` package) requires a system-level R installation
with several R packages — this is separate from the Python virtual environment
and must be installed once per machine.

### Requirements
- R (installed via Homebrew, not Anaconda — see note below)
- `rpy2` (Python package, in `requirements.txt`)
- R packages: `SID`, `pcalg`, `ggm`, `graph`, `RBGL`

### Setup (macOS)

1. Install R via Homebrew: