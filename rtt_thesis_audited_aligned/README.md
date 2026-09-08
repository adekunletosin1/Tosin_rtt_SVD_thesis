# RTT-SVD audited implementation aligned with thesis Chapters 1--5

This package consolidates the RTT-SVD implementation against the current
thesis draft using Chapters 1--5 only. The old Chapters 6 and 7 are not used as
implementation specifications, and none of their old experiment results are
carried into the new experiment suite.

## Thesis-to-code map

| Thesis item | Code |
|---|---|
| TT representation and storage | `TensorTrain` |
| Algorithm 2, deterministic TT-SVD | `tt_svd` |
| Algorithm 3, TT-rounding as written in the draft | `tt_round` |
| Algorithm 4, Gaussian range finder | `randomized_range_finder` |
| Algorithm 5, power range finder | `randomized_range_finder_power` |
| Algorithm 6, randomized matrix SVD | `randomized_svd` |
| Algorithm 7, Huber randomized TT-SVD | `rtt_svd` |
| Equal-final-rank post-processing for experiments | `rtt_svd_rounded` |
| Proposition 5.5 coordinate-sparse access model | `rtt_svd_sparse` |

## Algorithm 7

`rtt_svd` is the right-to-left Huber--Schneider--Wolf construction described
in Chapter 5. It is not a left-to-right TT-SVD with a randomized matrix SVD at
each unfolding.

For `j=d,d-1,...,2`, the implementation draws a fresh standard Gaussian test
tensor in matricized form, contracts it with the current working tensor,
computes the economic RQ factorization, stores the row-orthogonal factor as
core `W_j`, and contracts that core into the next working tensor. No SVD
truncation occurs inside the randomized sweep.

The Algorithm-7 output has effective representation ranks `s=r+p`, clipped
only by unavoidable unfolding dimensions:

```python
tt_s = rtt_svd(X, ranks=r, oversampling=p, random_state=42)
```

For comparisons in which TT-SVD and RTT-SVD must have the same final target
rank, use:

```python
tt_r = rtt_svd_rounded(X, ranks=r, oversampling=p, random_state=42)
```

This post-processing operates only on TT cores. `tt_round` remains the separate
implementation of Algorithm 3 exactly as it is written in the current draft.

## Ten-experiment suite

The experiment driver imports the core algorithms above and does not define a
second RTT-SVD implementation.

1. Nearly low-rank tensors with Gaussian perturbations, following the numerical question in Huber Sec. 5.1.
2. Oversampling study on nearly low-rank and quadratic-decay tensors, following Huber Sec. 5.2.
3. Approximation quality versus tensor order for both tensor families, following Huber Sec. 5.3.
4. Coordinate-sparse RTT-SVD runtime versus dense deterministic TT-SVD, following Huber Sec. 5.4.
5. Error concentration over independent Gaussian sketches.
6. Gaussian single-step range-finder bound validation using only `p>=4`, as required by Theorem 5.1 in the current draft.
7. Accuracy versus target TT-rank for exponential and polynomial decay.
8. Mode-size scalability and exact-recovery check on exact TT tensors.
9. Singular-value decay across all principal TT matricizations.
10. Storage-accuracy trade-off.

The ALS/random-low-rank experiment from Huber Sec. 5.5 is deliberately
excluded.

Each experiment produces exactly one figure design in vector PDF and 360-dpi
PNG form. Aggregated JSON and raw CSV results are written to `results/data`.
Combined panels are used where an experiment has several related views.

## Validation

Run the algorithm tests first:

```powershell
python -m rtt_thesis_audited_aligned.self_test
```

Expected output:

```text
All Chapters 1-5 alignment self-tests passed.
```

Run the reduced experiment profile before the final numerical run:

```powershell
$env:RTT_RUN_PROFILE="smoke"
python -m rtt_thesis_audited_aligned.run_experiments
```

The smoke profile is for software validation only. Do not use its numerical
values in the thesis.

## Final thesis run

```powershell
$env:RTT_RUN_PROFILE="thesis"
python -m rtt_thesis_audited_aligned.run_experiments
```

To choose another output folder:

```powershell
$env:RTT_RESULTS_DIR="thesis_results"
python -m rtt_thesis_audited_aligned.run_experiments
```

To rerun selected experiments only:

```powershell
$env:RTT_EXPERIMENTS="2,5,8"
python -m rtt_thesis_audited_aligned.run_experiments
```

The `paper` profile raises the Huber-style repeated studies to 256 trials per
point and is substantially more expensive.
