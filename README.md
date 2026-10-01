# Bidder-equivariant combinatorial auctions

PyTorch implementations of an MLP RegretNet-style baseline, bidder-permutation augmentation, and a DeepSet mechanism for small combinatorial auctions. The models share the same allocation decoder and revenue/regret training objective.

## Setup

Use Python 3.11 and the pinned CPU dependencies:

```bash
git clone https://github.com/Z-LI0403/bidder-equivariant-combinatorial-auctions.git
cd bidder-equivariant-combinatorial-auctions
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Run

Check the complete software pipeline with a short training run:

```bash
python -m src.reproduce --mode smoke --output reproductions/smoke
```

Train one configured model:

```bash
python -m src.train --config configs/p1_pilot_deepset.json --seed 91
```

Regenerate the benchmark:

```bash
python -m src.reproduce --mode full --output reproductions/full --workers 6
```

Each reproduction requires a new output directory. See [reproduction instructions](docs/REPRODUCIBILITY.md) for outputs and checks.

## Model and measurement

Inputs have shape `[batch, bidders, 2**items - 1]`, with nonempty bundles ordered by bitmask. Valuations contain nonnegative item coefficients and pairwise complementarity coefficients. Only bidder permutations are considered; item order is fixed.

The shared decoder represents a lottery over all `(bidders + 1)**items` feasible deterministic item assignments, including unallocated items. This enforces allocation feasibility but scales exponentially. Payments are a sigmoid fraction of reported expected allocated value; truthful individual rationality uses risk-neutral, quasi-linear expected utility.

Regret is estimated by bounded, finite-budget unilateral misreport searches. These values are lower bounds on worst-case regret and do not certify incentive compatibility. Permutation error is measured independently for allocations and payments.

## Benchmark files

[Per-seed metrics](results/per_seed.csv), [aggregate metrics](results/aggregate.csv), and [held-out evaluations](results/post_evaluation_per_seed.csv) retain all benchmark settings, including adverse outcomes. Aggregates use five seeds (11, 22, 33, 44, 55) and sample standard deviations. The configuration manifest maps 205 training jobs across 41 model/settings combinations; some experiment groups share jobs.

Plots are in [figures/study_v1](figures/study_v1). They describe these finite experiments; they do not establish universal revenue or generalization advantages.

## Code

- `src/models.py`, `feasibility.py`: model backbones and shared decoder.
- `src/valuations.py`, `regret.py`, `evaluation.py`: data, deviation search, and metrics.
- `src/train.py`, `study.py`, `post_evaluate.py`: training and benchmark execution.
- `src/audit.py`, `equivariance.py`, `validate_study.py`, `reproduce_final.py`: independent numerical checks.
- `src/analyze_study.py`: tables, paired descriptive statistics, and plots.
- `tests/`: manually solvable cases, feasibility, equivariance, attacks, and configuration checks.
