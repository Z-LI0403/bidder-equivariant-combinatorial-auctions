# Reproducing the benchmark

Install Python 3.11 and `requirements.lock.txt` as shown in the root README. The runner uses CPU execution, one PyTorch thread per training process, fixed random seeds, and deterministic algorithms. Matching software and hardware gives the strongest reproducibility; bitwise equality across platforms is not assumed.

## Quick software check

```bash
python -m src.reproduce --mode smoke --output reproductions/smoke
```

This copies the runnable code, tests, and configurations to a new directory, runs the unit tests and structural permutation checks, then trains an MLP for five updates on a tiny synthetic dataset. Its metrics verify execution only; five updates need not satisfy the low-regret thresholds.

## Full benchmark

```bash
python -m src.reproduce --mode full --output reproductions/full --workers 6
```

The runner first performs four fixed pilot runs and independent measurement audits. Failed prerequisites stop the run before scaling. It then executes the configuration manifest, evaluates distribution and bidder-count shifts without retraining, calculates tables and figures, checks saved datasets and metrics, and repeats two seed-11 runs to compare tensors and numerical trajectories.

The full command includes 205 primary jobs, four pilots, and two repeat runs; it is substantially slower than the smoke check. Reduce `--workers` to fit available memory. Repeated runs are excluded from the five-seed benchmark statistics. Evaluation uses the final iterate, with fixed attack budgets and common data seeds across model comparisons.

Outputs under the new directory:

- `results/runs/`: checkpoints, generated data, per-profile measurements, configurations, and trajectories.
- `results/study_v1/analysis/`: per-seed and aggregate CSVs, descriptive comparisons, and held-out evaluations.
- `results/verification/`: consistency, repeatability, and test outputs.
- `figures/study_v1/`: PNG and PDF plots.

Fresh runtime records and large tensor files are ignored by Git. The published CSVs in the root `results/` directory are compact benchmark measurements.

## Reading the numbers

`regret_mean_bidder` is the mean positive unilateral gain per bidder found by the configured attack. `regret_mean_total` sums bidder gains before averaging profiles. Tail metrics summarize the same finite searches. Allocation/payment permutation errors are normalized errors under bidder relabeling.

The aggregate CSV reports mean and sample standard deviation across five seeds. Paired comparisons use seed differences and descriptive 95% t intervals with four degrees of freedom, without multiple-comparison adjustment. Revenue comparisons require attention to the accompanying regret. Neither finite searches nor small numerical permutation errors replace mathematical proofs.

The shared feasible lottery enumerates deterministic assignments and becomes expensive as bidder/item counts grow. The included benchmark covers two or three bidders and three or four items.
