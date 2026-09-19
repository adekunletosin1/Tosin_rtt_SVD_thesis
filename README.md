# Randomized TT-SVD thesis implementation

This repository contains the Python implementation and numerical experiments for the master's thesis:

**Randomized Tensor Train Singular Value Decomposition for Higher-Order Tensors: Algorithm Implementation and Analysis**

The implementation follows the algorithms developed and discussed in Chapters 3--5 of the thesis. Chapter 6 contains the numerical implementation, experiments, and analysis produced with this code.

The main Python package is:

```text
rtt_SVD_thesis/

The final Chapter 6 results are stored in:

```text
results_chapter6_final/
```

## Repository structure

The final repository has the following structure:

```text
Tosin_rtt_SVD_thesis/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── rtt_SVD_thesis/
│   ├── __init__.py
│   ├── alignment_manifest.json
│   ├── chapter6_experiments.py
│   ├── experiments.py
│   ├── self_test.py
│   ├── sparse_rtt_svd.py
│   └── tt_core.py
│
└── results_chapter6_final/
    ├── data/
    └── figures/

## Thesis-to-code map

| Thesis item | Python implementation |
| --- | --- |
| Tensor Train representation and storage | `TensorTrain` |
| Algorithm 2: deterministic TT-SVD | `tt_svd` |
| Algorithm 3: TT-rounding | `tt_round` |
| Algorithm 4: Gaussian randomized range finder | `randomized_range_finder` |
| Algorithm 5: randomized range finder with power iteration | `randomized_range_finder_power` |
| Algorithm 6: randomized matrix SVD | `randomized_svd` |
| Algorithm 7: randomized TT-SVD | `rtt_svd` |
| Equal-final-rank randomized TT-SVD used in comparison experiments | `rtt_svd_rounded` |
| Coordinate-sparse randomized TT-SVD | `rtt_svd_sparse` |

The main dense tensor routines are implemented in:

```text
rtt_SVD_thesis/tt_core.py
```

The sparse randomized TT-SVD implementation is contained in:

```text
rtt_SVD_thesis/sparse_rtt_svd.py
```
The Chapter 6 numerical experiments are implemented in:

```text
rtt_SVD_thesis/chapter6_experiments.py
```
A direct randomized TT-SVD approximation can be computed with:

```python
tt_s = rtt_svd(
    X,
    ranks=r,
    oversampling=p,
    random_state=42,
)
```
For experiments requiring deterministic TT-SVD and randomized TT-SVD to have the same final target TT rank, the randomized result is subsequently rounded:

```python
tt_r = rtt_svd_rounded(
    X,
    ranks=r,
    oversampling=p,
    random_state=42,
)
```

The experiment driver imports the tensor algorithms from the implementation modules. It does not define a separate randomized TT-SVD algorithm.

The complete experiment suite can be started from the repository root with:

```powershell
python -m rtt_SVD_thesis.chapter6_experiments
```

    ## Numerical output

The experiment results are stored in:

```text
results_chapter6_final/
```

The directory is divided into:

```text
results_chapter6_final/data/
results_chapter6_final/figures/
```
## Final experiment output names

The final experiment output stems are:

```text
exp01_accuracy_vs_bond_dimension
exp02_oversampling_parameter_study
exp03_runtime_vs_tensor_order
exp04_sparse_rtt_svd_complexity
exp05_scalability_vs_mode_size
exp06_robustness_to_additive_noise
exp07_method_comparison
exp08_singular_value_decay
exp09_error_concentration
exp10_storage_accuracy_tradeoff
exp11_gaussian_rangefinder_bound
```

This ordering should be kept consistent with Chapter 6 of the thesis.

## Reproducibility

The numerical experiments use explicit random seeds.

The base experiment seed is:

```text
42
```
## Main reference

The randomized TT-SVD implementation follows the construction developed by:

Benjamin Huber, Reinhold Schneider, and Sebastian Wolf,

**A Randomized Tensor Train Singular Value Decomposition**

The thesis uses this work as the main reference for the randomized TT-SVD construction and its error analysis.

The implementation in this repository is written for the thesis and is organized to correspond to the notation, algorithms, and numerical study presented there.

## Repository

GitHub repository:

```text
https://github.com/adekunletosin1/Tosin_rtt_SVD_thesis
