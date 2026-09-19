#!/usr/bin/env python3
"""
Chapter 6 experiments are implementations of Chapters 1--5 algorithms.

The randomized TT method is not reimplemented here. All RTT-SVD calls go through
``tt_core.rtt_svd_rounded``: Chapter 5 Algorithm 7 (right-to-left, independent
Gaussian test tensor at every bond, RQ factorization, no SVD truncation inside
the randomized sweep) followed by core-only rank reduction for equal-final-rank
comparisons. Deterministic TT-SVD uses Chapter 3 Algorithm 2. The matrix range
finder and randomized matrix SVD use Chapter 4 Algorithms 4 and 6.

Experiments follow the order, figure titles, axis labels, and legend wording of
Chapter 6 of the current thesis draft:
1. Accuracy versus Bond Dimension
2. Oversampling Parameter Study
3. Runtime versus Tensor Order
4. Sparse RTT-SVD Complexity versus Tensor Order
5. Scalability versus Mode Size
6. Robustness to Additive Noise
7. Comparison with TT-SVD and Randomized Tucker
8. Singular Value Decay
9. Error Concentration
10. Storage-Accuracy Trade-off
11. Gaussian Range-Finder Bound
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import scipy
from scipy.linalg import qr, svd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .tt_core import (
    TensorTrain,
    tt_svd as _chapter_tt_svd,
    rtt_svd as _chapter_rtt_svd,
    rtt_svd_rounded as _chapter_rtt_svd_rounded,
    randomized_range_finder as _chapter_range_finder,
    randomized_svd as _chapter_randomized_svd,
)
from .sparse_rtt_svd import (
    SparseTensor,
    random_sparse_tensor,
    rtt_svd_sparse as _chapter_sparse_rtt_svd,
    rtt_svd_sparse_rounded as _chapter_sparse_rtt_svd_rounded,
)

# ============================================================
# Configuration
# ============================================================

BASE_SEED = 42
RESULTS_DIR = Path(os.getenv("RTT_RESULTS_DIR", "results"))
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"
for _directory in (RESULTS_DIR, FIGURES_DIR, DATA_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

RUN_PROFILE = os.getenv("RTT_RUN_PROFILE", "thesis").strip().lower()
if RUN_PROFILE not in {"smoke", "thesis", "paper"}:
    raise ValueError("RTT_RUN_PROFILE must be one of: smoke, thesis, paper")

PROFILE = {
    "smoke": {
        "huber_trials": 1,
        "general_trials": 2,
        "concentration_trials": 20,
        "bound_trials": 20,
        "timing_trials": 1,
        "timing_repeats": 1,
    },
    "thesis": {
        "huber_trials": 16,
        "general_trials": 10,
        "concentration_trials": 500,
        "bound_trials": 200,
        "timing_trials": 5,
        "timing_repeats": 3,
    },
    "paper": {
        "huber_trials": 256,
        "general_trials": 15,
        "concentration_trials": 1000,
        "bound_trials": 500,
        "timing_trials": 8,
        "timing_repeats": 3,
    },
}[RUN_PROFILE]

HUBER_TRIALS = PROFILE["huber_trials"]
GENERAL_TRIALS = PROFILE["general_trials"]
CONCENTRATION_TRIALS = PROFILE["concentration_trials"]
BOUND_TRIALS = PROFILE["bound_trials"]
TIMING_TRIALS = PROFILE["timing_trials"]
TIMING_REPEATS = PROFILE["timing_repeats"]

if RUN_PROFILE == "smoke":
    HUBER_QUALITY_ORDERS = [4, 6, 8]
    SPARSE_RUNTIME_ORDERS = [4, 8, 12, 16]
else:
    HUBER_QUALITY_ORDERS = list(range(4, 11))
    SPARSE_RUNTIME_ORDERS = list(range(4, 61, 4))

MAX_DENSE_REFERENCE_ENTRIES = int(
    os.getenv("RTT_MAX_DENSE_ENTRIES", str(2**20))
)

_requested = os.getenv("RTT_EXPERIMENTS", "").strip()
if _requested:
    EXPERIMENTS_TO_RUN = sorted(
        {int(x.strip()) for x in _requested.split(",") if x.strip()}
    )
    if any(x < 1 or x > 11 for x in EXPERIMENTS_TO_RUN):
        raise ValueError("RTT_EXPERIMENTS may contain only integers 1,...,11")
else:
    EXPERIMENTS_TO_RUN = list(range(1, 12))

# ============================================================
# Thesis figure style
# ============================================================

# Colourblind-friendly palette, based on the Okabe-Ito family.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#202124",
    "grey": "#6B7280",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.titlesize": 11.0,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9.2,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "axes.linewidth": 0.85,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 130,
    "savefig.dpi": 360,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})



# ============================================================
# Rank helpers
# ============================================================

def feasible_tt_ranks(shape: Sequence[int], rank: int | Sequence[int]) -> Tuple[int, ...]:
    """Clamp requested TT ranks to the dimensions of each matricization."""
    shape = tuple(int(n) for n in shape)
    d = len(shape)

    if isinstance(rank, (int, np.integer)):
        requested = [int(rank)] * (d - 1)
    else:
        requested = [int(r) for r in rank]
        if len(requested) != d - 1:
            raise ValueError("A TT rank tuple must have length d-1.")

    ranks = []
    left = 1
    total = int(np.prod(shape, dtype=object))
    for k in range(d - 1):
        left *= shape[k]
        right = total // left
        ranks.append(max(1, min(requested[k], left, right)))
    return tuple(ranks)


def normalize_dense(A: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(A.ravel())
    return A / nrm if nrm > 0 else A.copy()


def relative_error_dense(A: np.ndarray, B: np.ndarray) -> float:
    denom = np.linalg.norm(A.ravel())
    if denom == 0:
        return float(np.linalg.norm(B.ravel()))
    return float(np.linalg.norm((A - B).ravel()) / denom)


def relative_error_tt(tt: TensorTrain, A: np.ndarray) -> float:
    return relative_error_dense(A, tt.to_full())




# ============================================================
# Thin adapters to the authoritative Chapters 1--5 algorithms
# ============================================================

def tt_svd(A: np.ndarray, target_ranks: int | Sequence[int]) -> TensorTrain:
    ranks = feasible_tt_ranks(A.shape, target_ranks)
    return _chapter_tt_svd(A, ranks)


def rtt_svd(
    A: np.ndarray,
    target_ranks: int | Sequence[int],
    p: int = 10,
    seed: int | None = None,
) -> TensorTrain:
    """Algorithm 7 followed by Chapter-3 core-only TT rounding to target r."""
    ranks = feasible_tt_ranks(A.shape, target_ranks)
    return _chapter_rtt_svd_rounded(
        A,
        ranks,
        oversampling=p,
        random_state=seed,
    )


def rtt_svd_unrounded(
    A: np.ndarray,
    target_ranks: int | Sequence[int],
    p: int = 10,
    seed: int | None = None,
) -> TensorTrain:
    """Unrounded Chapter-5 Algorithm 7 output with representation ranks r+p."""
    ranks = feasible_tt_ranks(A.shape, target_ranks)
    return _chapter_rtt_svd(
        A,
        ranks,
        oversampling=p,
        random_state=seed,
    )


def randomized_range_finder(M: np.ndarray, r: int, p: int, seed: int) -> np.ndarray:
    return _chapter_range_finder(
        M,
        rank=r,
        oversampling=p,
        random_state=seed,
    )


def generate_sparse_tensor(
    shape: Sequence[int],
    n_entries: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample N Gaussian coordinate contributions, allowing repeated positions."""
    rng = np.random.default_rng(seed)
    shape = tuple(int(n) for n in shape)
    coords = np.column_stack([
        rng.integers(0, n, size=int(n_entries)) for n in shape
    ]).astype(np.int64)
    values = rng.standard_normal(int(n_entries)).astype(float)
    return coords, values


def sparse_to_dense(
    shape: Sequence[int],
    coords: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    A = np.zeros(tuple(shape), dtype=float)
    np.add.at(A, tuple(coords[:, k] for k in range(coords.shape[1])), values)
    return A


def sparse_rtt_svd(
    shape: Sequence[int],
    coords: np.ndarray,
    values: np.ndarray,
    target_ranks: int | Sequence[int],
    p: int = 10,
    seed: int | None = None,
) -> TensorTrain:
    S = SparseTensor(shape, coords, values)
    ranks = feasible_tt_ranks(shape, target_ranks)
    return _chapter_sparse_rtt_svd_rounded(
        S,
        ranks,
        oversampling=p,
        random_state=seed,
    )

# ============================================================
# Tensor generators
def generate_random_tt_tensor(
    shape: Sequence[int],
    rank: int | Sequence[int],
    seed: int,
    normalize: bool = True,
) -> np.ndarray:
    """Generate a dense tensor from independent Gaussian TT cores."""
    shape = tuple(int(n) for n in shape)
    ranks = (1,) + feasible_tt_ranks(shape, rank) + (1,)
    rng = np.random.RandomState(seed)

    cores = []
    for k, n_k in enumerate(shape):
        core = rng.standard_normal((ranks[k], n_k, ranks[k + 1]))
        # Mild scaling controls norm growth without changing generic rank.
        core /= math.sqrt(max(1, n_k * ranks[k]))
        cores.append(core)

    A = TensorTrain(cores).to_full()
    return normalize_dense(A) if normalize else A


def generate_nearly_low_rank_tensor(
    shape: Sequence[int],
    exact_rank: int,
    tau: float,
    seed: int,
) -> np.ndarray:
    """Huber-style x = x_exact/||x_exact|| + tau*n/||n||."""
    x_exact = generate_random_tt_tensor(shape, exact_rank, seed=seed, normalize=True)
    rng = np.random.RandomState(seed + 7919)
    noise = normalize_dense(rng.standard_normal(tuple(shape)))
    return x_exact + float(tau) * noise


def generate_spectral_tt_tensor(
    shape: Sequence[int],
    source_rank: int = 20,
    decay: str = "quadratic",
    alpha: float = 0.55,
    seed: int = BASE_SEED,
) -> np.ndarray:
    """
    Generate a TT tensor with an imposed bond-spectrum decay.

    This follows the construction described by Huber et al.: start from random
    Gaussian TT cores, contract adjacent cores, compute an SVD, replace the
    singular values by a prescribed decay, and separate the pair again. Later
    steps can perturb spectra imposed earlier, as in the reference paper.
    """
    shape = tuple(int(n) for n in shape)
    ranks = (1,) + feasible_tt_ranks(shape, source_rank) + (1,)
    rng = np.random.RandomState(seed)

    cores = [
        rng.standard_normal((ranks[k], shape[k], ranks[k + 1]))
        for k in range(len(shape))
    ]

    for k in range(len(shape) - 1):
        left = cores[k]
        right = cores[k + 1]
        r0, n0, rbond = left.shape
        _, n1, r2 = right.shape

        pair = np.tensordot(left, right, axes=([2], [0]))
        mat = pair.reshape(r0 * n0, n1 * r2)
        U, _, Vt = svd(mat, full_matrices=False, check_finite=False)

        keep = min(rbond, U.shape[1], Vt.shape[0])
        idx = np.arange(1, keep + 1, dtype=float)

        if decay == "quadratic":
            sigma = 1.0 / idx**2
        elif decay == "polynomial":
            sigma = 1.0 / idx**1.5
        elif decay == "exponential":
            sigma = np.exp(-alpha * (idx - 1.0))
        else:
            raise ValueError("decay must be quadratic, polynomial, or exponential")

        U = U[:, :keep]
        Vt = Vt[:keep, :]
        cores[k] = U.reshape(r0, n0, keep)
        cores[k + 1] = (sigma[:, None] * Vt).reshape(keep, n1, r2)

    return normalize_dense(TensorTrain(cores).to_full())


def generate_cp_decay_tensor(
    shape: Sequence[int],
    decay: str,
    alpha: float = 0.5,
    beta: float = 1.5,
    seed: int = BASE_SEED,
) -> np.ndarray:
    """Current-thesis synthetic tensor with controlled CP spectral weights."""
    shape = tuple(int(n) for n in shape)
    q = min(shape)
    rng = np.random.RandomState(seed)

    if decay == "exp":
        sigma = np.exp(-alpha * np.arange(q, dtype=float))
    elif decay == "poly":
        sigma = np.arange(1, q + 1, dtype=float) ** (-beta)
    else:
        raise ValueError("decay must be 'exp' or 'poly'")

    factors = []
    for n_k in shape:
        U = rng.standard_normal((n_k, q))
        U, _ = qr(U, mode="economic", check_finite=False)
        factors.append(U[:, :q])

    # Generic einsum for d <= 20, using letters for physical indices.
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(shape) > len(letters):
        raise ValueError("generate_cp_decay_tensor supports at most 52 modes")
    subs = [f"{letters[k]}j" for k in range(len(shape))]
    out = "".join(letters[:len(shape)])
    operands = [factors[0] * sigma[None, :]] + factors[1:]
    A = np.einsum(",".join(subs) + "->" + out, *operands, optimize=True)
    return normalize_dense(A)



# ============================================================
# Theoretical bound helper
# ============================================================

def amplification_factor(r: int, p: int, t: float = 1.0, u: float = 1.0) -> float:
    if p <= 0:
        raise ValueError("The amplification factor requires p > 0.")
    return float(
        1.0
        + t * np.sqrt(12.0 * r / p)
        + u * t * np.e * np.sqrt(r + p) / (p + 1.0)
    )



# ============================================================
# I/O helpers
# ============================================================

def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_json(stem: str, data: dict) -> Path:
    path = DATA_DIR / f"{stem}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(data), f, indent=2)
    print(f"  data: {path}")
    return path


def save_csv(stem: str, rows: List[dict]) -> Path | None:
    if not rows:
        return None
    path = DATA_DIR / f"{stem}.csv"
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  raw : {path}")
    return path


def save_figure(fig: plt.Figure, stem: str) -> None:
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  fig : {png}")
    print(f"  fig : {pdf}")


def clean_previous_outputs() -> None:
    """Remove earlier exp01...exp11 files to prevent duplicate figure/data sets."""
    for directory in (FIGURES_DIR, DATA_DIR):
        for path in directory.iterdir():
            if path.is_file() and path.name.lower().startswith("exp"):
                if path.suffix.lower() in {".png", ".pdf", ".json", ".csv"}:
                    path.unlink()


def style_axis(ax: plt.Axes, *, log_grid: bool = False) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_axisbelow(True)
    if log_grid:
        ax.grid(True, which="major", color="#CBD5E1", linewidth=0.65, alpha=0.70)
        ax.grid(True, which="minor", color="#E2E8F0", linewidth=0.45, alpha=0.55)
    else:
        ax.grid(True, which="major", color="#D7DEE8", linewidth=0.65, alpha=0.72)
    for spine in ax.spines.values():
        spine.set_color("#4B5563")
        spine.set_linewidth(0.8)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02, 0.97, label,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["black"],
    )


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def benchmark(callable_obj, repeats: int) -> float:
    """Median wall-clock time over repeated calls."""
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        callable_obj()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def safe_histogram_edges(values: Sequence[float], max_bins: int = 28) -> np.ndarray | int:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 10

    lo = float(np.min(x))
    hi = float(np.max(x))
    span = hi - lo
    ulp = max(abs(float(np.spacing(lo))), abs(float(np.spacing(hi))), np.finfo(float).tiny)
    min_width = 4.0 * ulp

    if span <= min_width:
        center = float(np.mean(x))
        half = max(16.0 * ulp, abs(center) * 1e-14, 1e-15)
        return np.linspace(center - half, center + half, 6)

    desired = min(max_bins, max(5, int(np.sqrt(x.size))))
    representable = max(1, int(np.floor(span / min_width)))
    bins = max(1, min(desired, representable))
    edges = np.unique(np.linspace(lo, hi, bins + 1))
    if edges.size < 2:
        center = float(np.mean(x))
        half = max(16.0 * ulp, abs(center) * 1e-14, 1e-15)
        edges = np.array([center - half, center + half])
    return edges



# ============================================================
# Experiment 1
# ============================================================

# ============================================================
# Chapter 6 comparison helpers
# ============================================================

def randomized_tucker(
    A: np.ndarray,
    rank: int,
    p: int = 5,
    seed: int = BASE_SEED,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Randomized Tucker comparison using Chapter 4 Algorithm 6 per mode.

    Algorithm 6 itself uses the Gaussian range finder of Algorithm 4. Each mode
    receives an independent random seed. The returned multilinear rank is the
    nominal rank ``rank`` rather than ``rank+p`` so the comparison reports the
    intended Tucker storage.
    """
    A = np.asarray(A, dtype=float)
    factors: list[np.ndarray] = []

    for mode, n_mode in enumerate(A.shape):
        M = np.moveaxis(A, mode, 0).reshape(n_mode, -1)
        r_eff = min(int(rank), min(M.shape))
        if r_eff < 1:
            raise ValueError("Tucker rank must be positive.")
        p_eff = min(int(p), min(M.shape) - r_eff)
        U, _, _ = _chapter_randomized_svd(
            M,
            rank=r_eff,
            oversampling=p_eff,
            random_state=int(seed + 1009 * (mode + 1)),
        )
        factors.append(U)

    core = A.copy()
    for mode, U in enumerate(factors):
        core = np.tensordot(U.T, core, axes=(1, mode))
        core = np.moveaxis(core, 0, mode)

    reconstruction = core
    for mode, U in enumerate(factors):
        reconstruction = np.tensordot(U, reconstruction, axes=(1, mode))
        reconstruction = np.moveaxis(reconstruction, 0, mode)

    return reconstruction, factors, core


def tucker_storage(factors: Sequence[np.ndarray], core: np.ndarray) -> int:
    return int(core.size + sum(U.size for U in factors))


def _profile_count(thesis: int, smoke: int, paper: int | None = None) -> int:
    if RUN_PROFILE == "smoke":
        return int(smoke)
    if RUN_PROFILE == "paper":
        return int(paper if paper is not None else thesis)
    return int(thesis)


def _positive_band(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tiny = np.finfo(float).tiny
    return np.maximum(mean - std, tiny), np.maximum(mean + std, tiny)


def _shape_power(n: int, d: int) -> str:
    return rf"${n}^{d}$"


# ============================================================
# Experiment 1: Accuracy versus Bond Dimension
# ============================================================

def experiment_01_accuracy_vs_bond_dimension() -> dict:
    print("\n[EXP 1] Accuracy versus Bond Dimension")

    shape = (16, 16, 16, 16)
    ranks = list(range(1, 13))
    p = 10
    n_trials = _profile_count(10, 2, 20)
    models = [
        ("exponential", "exp", {"alpha": 0.8},r"Exponential decay ($\alpha=0.8$)"),
    ("polynomial", "poly", {"beta": 2.0},r"Polynomial decay ($\beta=2.0$)"),
    ]

    raw: List[dict] = []
    summary: dict[str, dict] = {}

    for model_name, decay, kwargs, _ in models:
        tt_by_rank = {r: [] for r in ranks}
        rtt_by_rank = {r: [] for r in ranks}

        for trial in range(n_trials):
            seed_tensor = BASE_SEED + 10_000 * trial + (0 if decay == "exp" else 500_000)
            A = generate_cp_decay_tensor(shape, decay, seed=seed_tensor, **kwargs)

            for r in ranks:
                tt = tt_svd(A, r)
                seed_rtt = BASE_SEED + 1_000_000 + 10_000 * trial + 101 * r + (0 if decay == "exp" else 700_000)
                rtt = rtt_svd(A, r, p=p, seed=seed_rtt)
                e_tt = relative_error_tt(tt, A)
                e_rtt = relative_error_tt(rtt, A)
                tt_by_rank[r].append(e_tt)
                rtt_by_rank[r].append(e_rtt)
                raw.append({
                    "model": model_name,
                    "trial": trial,
                    "rank": r,
                    "p": p,
                    "tt_svd_error": e_tt,
                    "rtt_svd_error": e_rtt,
                    "tensor_seed": seed_tensor,
                    "rtt_seed": seed_rtt,
                })

        summary[model_name] = {
            "rank": ranks,
            "tt_mean": [mean_std(tt_by_rank[r])[0] for r in ranks],
            "tt_std": [mean_std(tt_by_rank[r])[1] for r in ranks],
            "rtt_mean": [mean_std(rtt_by_rank[r])[0] for r in ranks],
            "rtt_std": [mean_std(rtt_by_rank[r])[1] for r in ranks],
        }

    result = {
        "title": "Experiment 1: Accuracy versus Bond Dimension",
        "shape": shape,
        "p": p,
        "n_trials": n_trials,
        "ranks": ranks,
        "models": {"exponential_alpha": 0.8, "polynomial_beta": 2.0},
        "summary": summary,
        "algorithm": "Chapter 3 Algorithm 2 versus Chapter 5 Algorithm 7 followed by core-only rank reduction",
    }
    save_json("exp01_accuracy_vs_bond_dimension", result)
    save_csv("exp01_accuracy_vs_bond_dimension_raw", raw)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), sharey=False)
    for ax, (model_name, _, _, panel_title) in zip(axes, models):
        s = summary[model_name]
        x = np.asarray(ranks)
        tt_m, tt_s = np.asarray(s["tt_mean"]), np.asarray(s["tt_std"])
        rt_m, rt_s = np.asarray(s["rtt_mean"]), np.asarray(s["rtt_std"])
        ax.semilogy(x, tt_m, "o-", color=COLORS["orange"], label="TT-SVD (deterministic)")
        ax.semilogy(x, rt_m, "o--", color=COLORS["blue"], label=f"RTT-SVD (p = {p})")
        lo, hi = _positive_band(tt_m, tt_s)
        ax.fill_between(x, lo, hi, color=COLORS["orange"], alpha=0.15)
        lo, hi = _positive_band(rt_m, rt_s)
        ax.fill_between(x, lo, hi, color=COLORS["blue"], alpha=0.15)
        ax.set_title(panel_title)
        ax.set_xlabel(r"Target TT rank $r$")
        ax.set_ylabel(r"Relative error $\varepsilon$")
        ax.set_xticks(ranks)
        style_axis(ax, log_grid=True)
        ax.legend(frameon=False)
    # fig.suptitle(
    #     f"Approximation Accuracy vs Bond Dimension\n"
    #     f"(tensor shape $16^4$, averaged over {n_trials} trials)",
    #     y=1.02,
    # )
    fig.tight_layout()
    save_figure(fig, "exp01_accuracy_vs_bond_dimension")
    return result


# ============================================================
# Experiment 2: Oversampling Parameter Study
# ============================================================

def experiment_02_oversampling_parameter_study() -> dict:
    print("\n[EXP 2] Oversampling Parameter Study")

    shape = (16, 16, 16, 16)
    alpha = 0.6
    r = 6
    p_values = [0, 2, 5, 10, 15, 20, 30]
    n_trials = _profile_count(15, 3, 30)
    A = generate_cp_decay_tensor(shape, "exp", alpha=alpha, seed=BASE_SEED + 200)
    tt_ref = tt_svd(A, r)
    tt_error = relative_error_tt(tt_ref, A)

    errors = {p: [] for p in p_values}
    times = {p: [] for p in p_values}
    raw: List[dict] = []

    for p in p_values:
        for trial in range(n_trials):
            seed = BASE_SEED + 2_000_000 + 1000 * trial + p
            t0 = time.perf_counter()
            approx = rtt_svd(A, r, p=p, seed=seed)
            elapsed = time.perf_counter() - t0
            err = relative_error_tt(approx, A)
            errors[p].append(err)
            times[p].append(elapsed)
            raw.append({"p": p, "trial": trial, "rtt_svd_error": err, "wall_clock_s": elapsed, "seed": seed})

    error_mean = [mean_std(errors[p])[0] for p in p_values]
    error_std = [mean_std(errors[p])[1] for p in p_values]
    time_mean = [mean_std(times[p])[0] for p in p_values]
    time_std = [mean_std(times[p])[1] for p in p_values]

    result = {
        "title": "Experiment 2: Oversampling Parameter Study",
        "shape": shape,
        "alpha": alpha,
        "rank": r,
        "p_values": p_values,
        "n_trials": n_trials,
        "tt_svd_reference_error": tt_error,
        "error_mean": error_mean,
        "error_std": error_std,
        "time_mean_s": time_mean,
        "time_std_s": time_std,
    }
    save_json("exp02_oversampling_parameter_study", result)
    save_csv("exp02_oversampling_parameter_study_raw", raw)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25))
    x = np.asarray(p_values)
    em, es = np.asarray(error_mean), np.asarray(error_std)
    tm, ts = np.asarray(time_mean), np.asarray(time_std)

    axes[0].semilogy(x, em, "o-", color=COLORS["blue"], label="RTT-SVD")
    lo, hi = _positive_band(em, es)
    axes[0].fill_between(x, lo, hi, color=COLORS["blue"], alpha=0.18)
    axes[0].axhline(tt_error, linestyle="--", color=COLORS["grey"], label="TT-SVD reference")
    axes[0].set_title(r"Relative error as a function of $p$")
    axes[0].set_xlabel(r"Oversampling $p$")
    axes[0].set_ylabel(r"Relative error $\varepsilon$")
    axes[0].set_xticks(p_values)
    style_axis(axes[0], log_grid=True)
    axes[0].legend(frameon=False)

    axes[1].plot(x, tm, "o-", color=COLORS["blue"], label="RTT-SVD")
    axes[1].fill_between(x, np.maximum(tm - ts, 0.0), tm + ts, color=COLORS["blue"], alpha=0.18)
    axes[1].set_title(r"Runtime as a function of $p$")
    axes[1].set_xlabel(r"Oversampling $p$")
    axes[1].set_ylabel(r"Wall-Clock Time (s)")
    axes[1].set_xticks(p_values)
    style_axis(axes[1])
    axes[1].legend(frameon=False)

    # fig.suptitle(f"Effect of Oversampling Parameter p on RTT-SVD (bond dim r = {r}, shape $16^4$)", y=1.01)
    fig.tight_layout()
    save_figure(fig, "exp02_oversampling_parameter_study")
    return result


# ============================================================
# Experiment 3: Runtime versus Tensor Order
# ============================================================

def experiment_03_runtime_vs_tensor_order() -> dict:
    print("\n[EXP 3] Runtime versus Tensor Order")

    n = 8
    r = 4
    p = 10
    orders = [4, 5, 6, 7]
    n_tensors = _profile_count(5, 1, 8)
    repeats = _profile_count(3, 1, 5)

    tt_times_by_d = {d: [] for d in orders}
    rtt_times_by_d = {d: [] for d in orders}
    speedups_by_d = {d: [] for d in orders}
    raw: List[dict] = []

    for d in orders:
        shape = (n,) * d
        for trial in range(n_tensors):
            seed_tensor = BASE_SEED + 3_000_000 + 10_000 * d + trial
            A = generate_random_tt_tensor(shape, r, seed=seed_tensor, normalize=True)

            tt_runs = []
            rtt_runs = []
            for rep in range(repeats):
                t0 = time.perf_counter()
                tt_svd(A, r)
                tt_runs.append(time.perf_counter() - t0)

                seed_rtt = BASE_SEED + 3_500_000 + 100_000 * d + 1000 * trial + rep
                t0 = time.perf_counter()
                rtt_svd(A, r, p=p, seed=seed_rtt)
                rtt_runs.append(time.perf_counter() - t0)

            tt_time = float(np.mean(tt_runs))
            rtt_time = float(np.mean(rtt_runs))
            speedup = tt_time / rtt_time if rtt_time > 0 else float("inf")
            tt_times_by_d[d].append(tt_time)
            rtt_times_by_d[d].append(rtt_time)
            speedups_by_d[d].append(speedup)
            raw.append({
                "order": d,
                "trial": trial,
                "timing_repeats": repeats,
                "tt_svd_time_s": tt_time,
                "rtt_svd_time_s": rtt_time,
                "speedup_tt_over_rtt": speedup,
                "tensor_seed": seed_tensor,
            })

    tt_mean = [mean_std(tt_times_by_d[d])[0] for d in orders]
    tt_std = [mean_std(tt_times_by_d[d])[1] for d in orders]
    rtt_mean = [mean_std(rtt_times_by_d[d])[0] for d in orders]
    rtt_std = [mean_std(rtt_times_by_d[d])[1] for d in orders]
    sp_mean = [mean_std(speedups_by_d[d])[0] for d in orders]
    sp_std = [mean_std(speedups_by_d[d])[1] for d in orders]

    result = {
        "title": "Experiment 3: Runtime versus Tensor Order",
        "mode_size": n,
        "rank": r,
        "p": p,
        "orders": orders,
        "n_tensors_per_order": n_tensors,
        "timing_repeats": repeats,
        "tt_svd_time_mean_s": tt_mean,
        "tt_svd_time_std_s": tt_std,
        "rtt_svd_time_mean_s": rtt_mean,
        "rtt_svd_time_std_s": rtt_std,
        "speedup_mean": sp_mean,
        "speedup_std": sp_std,
        "timing_scope": "dense reference implementations; this experiment does not test the sparse complexity proposition",
    }
    save_json("exp03_runtime_vs_tensor_order", result)
    save_csv("exp03_runtime_vs_tensor_order_raw", raw)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25))
    x = np.asarray(orders)
    tt_m, tt_s = np.asarray(tt_mean), np.asarray(tt_std)
    rt_m, rt_s = np.asarray(rtt_mean), np.asarray(rtt_std)
    sp_m, sp_s = np.asarray(sp_mean), np.asarray(sp_std)

    axes[0].semilogy(x, tt_m, "o-", color=COLORS["orange"], label="TT-SVD")
    axes[0].semilogy(x, rt_m, "o--", color=COLORS["blue"], label="RTT-SVD")
    lo, hi = _positive_band(tt_m, tt_s)
    axes[0].fill_between(x, lo, hi, color=COLORS["orange"], alpha=0.15)
    lo, hi = _positive_band(rt_m, rt_s)
    axes[0].fill_between(x, lo, hi, color=COLORS["blue"], alpha=0.15)
    axes[0].set_title(r"Runtime as a function of tensor order $d$")
    axes[0].set_xlabel(r"Tensor order $d$")
    axes[0].set_ylabel(r"Wall-Clock Time (s)")
    axes[0].set_xticks(orders)
    style_axis(axes[0], log_grid=True)
    axes[0].legend(frameon=False)

    axes[1].plot(x, sp_m, "o-", color=COLORS["blue"])
    axes[1].fill_between(x, np.maximum(sp_m - sp_s, 0.0), sp_m + sp_s, color=COLORS["blue"], alpha=0.18)
    axes[1].axhline(1.0, linestyle="--", color=COLORS["grey"], linewidth=1.0, label=r"$S=1$")
    axes[1].set_title(r"Speedup as a function of $d$")
    axes[1].set_xlabel(r"Tensor order $d$")
    axes[1].set_ylabel(r"Speedup $S$")
    axes[1].set_xticks(orders)
    style_axis(axes[1])

    # fig.suptitle(f"Computational Speedup vs Tensor Order (n = {n}, r = {r}, p = {p})", y=1.01)
    fig.tight_layout()
    save_figure(fig, "exp03_runtime_vs_tensor_order")
    return result

# ============================================================
# Experiment 4: Sparse RTT-SVD Complexity versus Tensor Order
# ============================================================

def experiment_04_sparse_rtt_svd_complexity() -> dict:
    """
    Empirical sparse-scaling experiment for Proposition 5.5.

    The experiment keeps mode size n, number of nonzeros N,
    target rank r, and oversampling p fixed while varying tensor
    order d.

    The timed routine is the coordinate-sparse implementation of
    Chapter 5, Algorithm 7. No dense tensor is materialized and
    TT-rounding is excluded from the timed region.
    """

    print("\n[EXP 4] Sparse RTT-SVD Complexity versus Tensor Order")

    # Fixed parameters in
    # O(d * (s^2 N + s^3 n)), s = r + p.
    n = 16
    nnz = 20_000
    r = 4
    p = 5
    s = r + p

    # 16^14 = 2^56, so the largest ambient size remains inside
    # the integer range used by the current COO generator.
    if RUN_PROFILE == "smoke":
        orders = [5, 6, 8]
        n_tensors = 1
        repeats = 1
    elif RUN_PROFILE == "paper":
        orders = [5, 6, 8, 10, 12, 14]
        n_tensors = 8
        repeats = 5
    else:
        orders = [5, 6, 8, 10, 12, 14]
        n_tensors = 5
        repeats = 3

    times_by_order = {d: [] for d in orders}
    raw: List[dict] = []

    for d in orders:
        shape = (n,) * d
        ranks = (r,) * (d - 1)

        ambient_entries = n ** d
        density = nnz / ambient_entries

        for trial in range(n_tensors):

            tensor_seed = (
                BASE_SEED
                + 11_000_000
                + 10_000 * d
                + trial
            )

            A_sparse = random_sparse_tensor(
                shape,
                nnz=nnz,
                random_state=tensor_seed,
            )

            # One untimed warm-up run reduces first-call effects.
            warmup_seed = (
                BASE_SEED
                + 11_500_000
                + 10_000 * d
                + trial
            )

            _chapter_sparse_rtt_svd(
                A_sparse,
                ranks,
                oversampling=p,
                random_state=warmup_seed,
            )

            run_times = []
            output_ranks = None

            for rep in range(repeats):

                sketch_seed = (
                    BASE_SEED
                    + 12_000_000
                    + 100_000 * d
                    + 1000 * trial
                    + rep
                )

                t0 = time.perf_counter()

                tt_sparse = _chapter_sparse_rtt_svd(
                    A_sparse,
                    ranks,
                    oversampling=p,
                    random_state=sketch_seed,
                )

                elapsed = time.perf_counter() - t0
                run_times.append(elapsed)

                output_ranks = tuple(tt_sparse.ranks)

            trial_time = float(np.mean(run_times))
            times_by_order[d].append(trial_time)

            raw.append({
                "order": d,
                "mode_size": n,
                "nnz": A_sparse.nnz,
                "ambient_entries": ambient_entries,
                "density": density,
                "target_rank": r,
                "oversampling": p,
                "sketch_rank_s": s,
                "trial": trial,
                "timing_repeats": repeats,
                "runtime_s": trial_time,
                "output_ranks": str(output_ranks),
                "tensor_seed": tensor_seed,
            })

    mean_times = [
        mean_std(times_by_order[d])[0]
        for d in orders
    ]

    std_times = [
        mean_std(times_by_order[d])[1]
        for d in orders
    ]

    # --------------------------------------------------------
    # Linear fit T(d) = a d + b
    # --------------------------------------------------------

    x = np.asarray(orders, dtype=float)
    y = np.asarray(mean_times, dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept

    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    if ss_tot > 0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = 1.0

    time_per_mode = y / x
    std_per_mode = np.asarray(std_times) / x

    # The theoretical operation-count expression.
    theoretical_work = [
        int(d * (s**2 * nnz + s**3 * n))
        for d in orders
    ]

    result = {
        "title":
            "Experiment 4: Sparse RTT-SVD Complexity versus Tensor Order",

        "algorithm":
            "Chapter 5 Algorithm 7, coordinate-sparse implementation",

        "theoretical_complexity":
            "O(d * (s^2 N + s^3 n))",

        "mode_size_n": n,
        "nnz_N": nnz,
        "target_rank_r": r,
        "oversampling_p": p,
        "sketch_rank_s": s,

        "orders": orders,
        "n_tensors_per_order": n_tensors,
        "timing_repeats": repeats,

        "runtime_mean_s": mean_times,
        "runtime_std_s": std_times,

        "time_per_mode_s": time_per_mode.tolist(),

        "theoretical_work_units": theoretical_work,

        "linear_fit_slope_s_per_mode": float(slope),
        "linear_fit_intercept_s": float(intercept),
        "linear_fit_r_squared": float(r_squared),

        "timing_scope":
            "Coordinate-sparse Algorithm 7 only; no dense "
            "materialization and no TT-rounding in timed region.",
    }

    save_json(
        "exp04_sparse_rtt_svd_complexity",
        result,
    )

    save_csv(
        "exp04_sparse_rtt_svd_complexity_raw",
        raw,
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.25),
    )

    # Runtime versus order
    axes[0].errorbar(
        x,
        y,
        yerr=np.asarray(std_times),
        fmt="o-",
        capsize=3,
        color=COLORS["blue"],
        label="Sparse RTT-SVD",
    )

    axes[0].plot(
        x,
        fitted,
        "--",
        color=COLORS["orange"],
        label=fr"Linear fit ($R^2={r_squared:.3f}$)",
    )

    axes[0].set_title(
        r"Sparse Runtime per Tensor Order $d$"
    )
    axes[0].set_xlabel(r"Tensor order $d$")
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_xticks(orders)
    style_axis(axes[0])
    axes[0].legend(frameon=False)

    # Runtime divided by order
    axes[1].errorbar(
        x,
        time_per_mode,
        yerr=std_per_mode,
        fmt="o-",
        capsize=3,
        color=COLORS["blue"],
        label="Sparse RTT-SVD",
    )

    axes[1].set_title(
        "Runtime per Tensor Mode"
    )
    axes[1].set_xlabel(r"Tensor order $d$")
    axes[1].set_ylabel(r"Wall-clock time per mode (s)")
    axes[1].set_xticks(orders)
    style_axis(axes[1])
    axes[1].legend(frameon=False)

    # fig.suptitle(
    #     f"Sparse RTT-SVD Scaling "
    #     f"(n = {n}, N = {nnz:,}, r = {r}, p = {p})",
    #     y=1.01,
    # )

    fig.tight_layout()

    save_figure(
        fig,
        "exp04_sparse_rtt_svd_complexity",
    )
    return result

# ============================================================
# Experiment 5: Scalability versus Mode Size
# ============================================================

def experiment_05_scalability_vs_mode_size() -> dict:
    print("\n[EXP 5] Scalability versus Mode Size")

    d = 5
    r = 4
    p = 10
    alpha = 0.5
    mode_sizes = [6, 8, 10, 12, 16]
    n_trials = _profile_count(5, 1, 8)
    repeats = _profile_count(2, 1, 3)

    raw: List[dict] = []
    by_n: dict[int, dict[str, list[float]]] = {
        n: {"tt_time": [], "rtt_time": [], "speedup": [], "tt_error": [], "rtt_error": []}
        for n in mode_sizes
    }

    for n in mode_sizes:
        shape = (n,) * d
        for trial in range(n_trials):
            seed_tensor = BASE_SEED + 4_000_000 + 10_000 * n + trial
            A = generate_cp_decay_tensor(shape, "exp", alpha=alpha, seed=seed_tensor)
            tt_runs, rtt_runs = [], []
            tt_last = rtt_last = None

            for rep in range(repeats):
                t0 = time.perf_counter()
                tt_last = tt_svd(A, r)
                tt_runs.append(time.perf_counter() - t0)

                seed_rtt = BASE_SEED + 4_500_000 + 100_000 * n + 1000 * trial + rep
                t0 = time.perf_counter()
                rtt_last = rtt_svd(A, r, p=p, seed=seed_rtt)
                rtt_runs.append(time.perf_counter() - t0)

            assert tt_last is not None and rtt_last is not None
            tt_time = float(np.mean(tt_runs))
            rtt_time = float(np.mean(rtt_runs))
            e_tt = relative_error_tt(tt_last, A)
            e_rtt = relative_error_tt(rtt_last, A)
            speedup = tt_time / rtt_time if rtt_time > 0 else float("inf")

            by_n[n]["tt_time"].append(tt_time)
            by_n[n]["rtt_time"].append(rtt_time)
            by_n[n]["speedup"].append(speedup)
            by_n[n]["tt_error"].append(e_tt)
            by_n[n]["rtt_error"].append(e_rtt)
            raw.append({
                "mode_size": n,
                "trial": trial,
                "tt_svd_time_s": tt_time,
                "rtt_svd_time_s": rtt_time,
                "speedup": speedup,
                "tt_svd_error": e_tt,
                "rtt_svd_error": e_rtt,
                "tensor_seed": seed_tensor,
            })

    def stat(field: str, which: int) -> list[float]:
        return [mean_std(by_n[n][field])[which] for n in mode_sizes]

    result = {
        "title": "Experiment 5: Scalability versus Mode Size",
        "order": d,
        "rank": r,
        "p": p,
        "alpha": alpha,
        "mode_sizes": mode_sizes,
        "n_trials": n_trials,
        "timing_repeats": repeats,
        "tt_time_mean_s": stat("tt_time", 0),
        "tt_time_std_s": stat("tt_time", 1),
        "rtt_time_mean_s": stat("rtt_time", 0),
        "rtt_time_std_s": stat("rtt_time", 1),
        "speedup_mean": stat("speedup", 0),
        "speedup_std": stat("speedup", 1),
        "tt_error_mean": stat("tt_error", 0),
        "tt_error_std": stat("tt_error", 1),
        "rtt_error_mean": stat("rtt_error", 0),
        "rtt_error_std": stat("rtt_error", 1),
    }
    save_json("exp05_scalability_vs_mode_size", result)
    save_csv("exp05_scalability_vs_mode_size_raw", raw)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.15))
    x = np.asarray(mode_sizes)
    tt_tm = np.asarray(result["tt_time_mean_s"])
    rt_tm = np.asarray(result["rtt_time_mean_s"])
    tt_em = np.asarray(result["tt_error_mean"])
    rt_em = np.asarray(result["rtt_error_mean"])
    sp_m = np.asarray(result["speedup_mean"])

    axes[0].semilogy(x, tt_tm, "o-", color=COLORS["orange"], label="TT-SVD")
    axes[0].semilogy(x, rt_tm, "o--", color=COLORS["blue"], label="RTT-SVD")
    axes[0].set_title(r"Runtime per Mode Size $n$")
    axes[0].set_xlabel(r"Mode size $n$")
    axes[0].set_ylabel(r"Wall-clock time (s)")
    style_axis(axes[0], log_grid=True)
    axes[0].legend(frameon=False)

    axes[1].semilogy(x, tt_em, "o-", color=COLORS["orange"], label="TT-SVD")
    axes[1].semilogy(x, rt_em, "o--", color=COLORS["blue"], label="RTT-SVD")
    axes[1].set_title(r"Relative Error per Mode Size $n$")
    axes[1].set_xlabel(r"Mode size $n$")
    axes[1].set_ylabel(r"Relative error $\varepsilon$")
    style_axis(axes[1], log_grid=True)
    axes[1].legend(frameon=False)

    axes[2].plot(x, sp_m, "o-", color=COLORS["blue"])
    axes[2].axhline(1.0, linestyle="--", color=COLORS["grey"], linewidth=1.0, label=r"$S=1$")
    axes[2].set_title(r"Speedup per Mode Size $n$")
    axes[2].set_xlabel(r"Mode size $n$")
    axes[2].set_ylabel(r"Speedup $S$")
    style_axis(axes[2])
    axes[2].legend(frameon=False)

    # fig.suptitle(f"Scalability vs Mode Size (d = {d}, r = {r}, p = {p})", y=1.01)
    fig.tight_layout()
    save_figure(fig, "exp05_scalability_vs_mode_size")
    return result


# ============================================================
# Experiment 6: Robustness to Additive Noise
# ============================================================

def experiment_06_robustness_to_additive_noise() -> dict:
    print("\n[EXP 6] Robustness to Additive Noise")

    shape = (12, 12, 12, 12)
    alpha = 0.7
    snr_values = [0, 5, 10, 15, 20]
    r = 4
    p = 10
    n_trials = _profile_count(10, 2, 20)
    clean = generate_cp_decay_tensor(shape, "exp", alpha=alpha, seed=BASE_SEED + 500)

    by_snr = {snr: {"tt": [], "rtt": []} for snr in snr_values}
    raw: List[dict] = []

    for snr in snr_values:
        for trial in range(n_trials):
            noise_seed = BASE_SEED + 5_000_000 + 10_000 * snr + trial
            rng = np.random.default_rng(noise_seed)
            z = rng.standard_normal(shape)
            z_norm = np.linalg.norm(z.ravel())
            desired_noise_norm = np.linalg.norm(clean.ravel()) / (10.0 ** (snr / 20.0))
            noisy = clean + z * (desired_noise_norm / z_norm)

            tt = tt_svd(noisy, r)
            rtt_seed = BASE_SEED + 5_500_000 + 10_000 * snr + trial
            rtt = rtt_svd(noisy, r, p=p, seed=rtt_seed)
            e_tt = relative_error_dense(clean, tt.to_full())
            e_rtt = relative_error_dense(clean, rtt.to_full())
            by_snr[snr]["tt"].append(e_tt)
            by_snr[snr]["rtt"].append(e_rtt)
            raw.append({
                "snr_db": snr,
                "trial": trial,
                "tt_svd_error_wrt_clean": e_tt,
                "rtt_svd_error_wrt_clean": e_rtt,
                "noise_seed": noise_seed,
                "rtt_seed": rtt_seed,
            })

    tt_mean = [mean_std(by_snr[s]["tt"])[0] for s in snr_values]
    tt_std = [mean_std(by_snr[s]["tt"])[1] for s in snr_values]
    rt_mean = [mean_std(by_snr[s]["rtt"])[0] for s in snr_values]
    rt_std = [mean_std(by_snr[s]["rtt"])[1] for s in snr_values]

    result = {
        "title": "Experiment 6: Robustness to Additive Noise",
        "shape": shape,
        "alpha": alpha,
        "rank": r,
        "p": p,
        "snr_db": snr_values,
        "n_trials": n_trials,
        "tt_mean": tt_mean,
        "tt_std": tt_std,
        "rtt_mean": rt_mean,
        "rtt_std": rt_std,
        "error_reference": "clean tensor",
    }
    save_json("exp06_robustness_to_additive_noise", result)
    save_csv("exp06_robustness_to_additive_noise_raw", raw)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.asarray(snr_values)
    tt_m, tt_s = np.asarray(tt_mean), np.asarray(tt_std)
    rt_m, rt_s = np.asarray(rt_mean), np.asarray(rt_std)
    ax.semilogy(x, tt_m, "o-", color=COLORS["orange"], label="TT-SVD")
    ax.semilogy(x, rt_m, "o--", color=COLORS["blue"], label=f"RTT-SVD (p = {p})")
    lo, hi = _positive_band(tt_m, tt_s)
    ax.fill_between(x, lo, hi, color=COLORS["orange"], alpha=0.16)
    lo, hi = _positive_band(rt_m, rt_s)
    ax.fill_between(x, lo, hi, color=COLORS["blue"], alpha=0.16)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"Relative error $\varepsilon$")
    ax.set_xticks(snr_values)
    
    style_axis(ax, log_grid=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "exp06_robustness_to_additive_noise")
    return result


# ============================================================
# Experiment 7: Comparison with TT-SVD and Randomized Tucker
# ============================================================

def experiment_07_method_comparison() -> dict:
    print("\n[EXP 7] Comparison with TT-SVD and Randomized Tucker")

    shape = (12, 12, 12, 12)
    source_rank = 10
    target_rank = 4
    p = 5
    n_trials = _profile_count(6, 2, 12)
    A = generate_random_tt_tensor(shape, source_rank, seed=BASE_SEED + 600, normalize=True)

    methods = ["RTT-SVD", "TT-SVD", "Randomized Tucker"]
    errors = {m: [] for m in methods}
    times = {m: [] for m in methods}
    storage: dict[str, int] = {}
    raw: List[dict] = []

    for trial in range(n_trials):
        seed_rtt = BASE_SEED + 6_000_000 + trial
        t0 = time.perf_counter()
        rtt = rtt_svd(A, target_rank, p=p, seed=seed_rtt)
        t_rtt = time.perf_counter() - t0
        e_rtt = relative_error_tt(rtt, A)

        t0 = time.perf_counter()
        tt = tt_svd(A, target_rank)
        t_tt = time.perf_counter() - t0
        e_tt = relative_error_tt(tt, A)

        seed_tucker = BASE_SEED + 6_500_000 + trial
        t0 = time.perf_counter()
        Tucker_rec, factors, core = randomized_tucker(A, target_rank, p=p, seed=seed_tucker)
        t_tucker = time.perf_counter() - t0
        e_tucker = relative_error_dense(A, Tucker_rec)

        trial_values = {
            "RTT-SVD": (e_rtt, t_rtt),
            "TT-SVD": (e_tt, t_tt),
            "Randomized Tucker": (e_tucker, t_tucker),
        }
        storage["RTT-SVD"] = rtt.storage()
        storage["TT-SVD"] = tt.storage()
        storage["Randomized Tucker"] = tucker_storage(factors, core)

        for method, (err, elapsed) in trial_values.items():
            errors[method].append(err)
            times[method].append(elapsed)
            raw.append({
                "trial": trial,
                "method": method,
                "relative_frobenius_error": err,
                "wall_clock_s": elapsed,
                "storage": storage[method],
                "compression_ratio": A.size / storage[method],
                "rtt_seed": seed_rtt if method == "RTT-SVD" else "",
                "tucker_seed": seed_tucker if method == "Randomized Tucker" else "",
            })

    error_mean = [mean_std(errors[m])[0] for m in methods]
    error_std = [mean_std(errors[m])[1] for m in methods]
    time_mean = [mean_std(times[m])[0] for m in methods]
    time_std = [mean_std(times[m])[1] for m in methods]
    ratios = [A.size / storage[m] for m in methods]

    result = {
        "title": "Experiment 7: Comparison with TT-SVD and Randomized Tucker",
        "shape": shape,
        "source_tt_rank": source_rank,
        "target_rank": target_rank,
        "p": p,
        "n_trials": n_trials,
        "methods": methods,
        "error_mean": error_mean,
        "error_std": error_std,
        "time_mean_s": time_mean,
        "time_std_s": time_std,
        "storage": [storage[m] for m in methods],
        "compression_ratio": ratios,
        "tucker_algorithm": "Chapter 4 Algorithm 6 independently on each mode unfolding, followed by Tucker projection",
    }
    save_json("exp07_method_comparison", result)
    save_csv("exp07_method_comparison_raw", raw)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.15))
    x = np.arange(len(methods))
    bar_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    axes[0].bar(x, error_mean, yerr=error_std, capsize=3, color=bar_colors, alpha=0.85)
    axes[0].set_xticks(x, methods, rotation=12)
    axes[0].set_ylabel(r"Relative error $\varepsilon$")
    axes[0].set_title("Approximation Error")
    style_axis(axes[0])

    axes[1].bar(x, time_mean, yerr=time_std, capsize=3, color=bar_colors, alpha=0.85)
    axes[1].set_xticks(x, methods, rotation=12)
    axes[1].set_ylabel(r"Wall-clock time (s)")
    axes[1].set_title("Runtime")
    style_axis(axes[1])

    axes[2].bar(x, ratios, color=bar_colors, alpha=0.85)
    axes[2].set_xticks(x, methods, rotation=12)
    axes[2].set_ylabel(r"Compression ratio $\rho$")
    axes[2].set_title("Compression Ratio")
    style_axis(axes[2])

    # fig.suptitle(
    #     f"Method Comparison on a TT-Structured Tensor\n"
    #     f"(shape $12^4$, source TT rank {source_rank}, target rank {target_rank}, p = {p}, {n_trials} trials)",
    #     y=1.03,
    # )
    fig.tight_layout()
    save_figure(fig, "exp07_method_comparison")
    return result


# ============================================================
# Experiment 8: Singular Value Decay
# ============================================================

def experiment_08_singular_value_decay() -> dict:
    print("\n[EXP 8] Singular Value Decay")

    shape = (32, 32, 32, 32)
    specifications = [
    (r"Exponential decay ($\alpha = 0.5$)", "exp", {"alpha": 0.5}),
    (r"Exponential decay ($\alpha = 1.0$)", "exp", {"alpha": 1.0}),
    (r"Polynomial decay ($\beta = 1.5$)", "poly", {"beta": 1.5}),
    (r"Polynomial decay ($\beta = 3.0$)", "poly", {"beta": 3.0}),
]
    threshold = 1e-3
    spectra: dict[str, list[float]] = {}
    raw: List[dict] = []

    for idx, (label, decay, kwargs) in enumerate(specifications):
        A = generate_cp_decay_tensor(shape, decay, seed=BASE_SEED + 700 + idx, **kwargs)
        M = A.reshape(shape[0], -1)
        svals = svd(M, compute_uv=False, check_finite=False)
        normalized = svals / svals[0]
        spectra[label] = normalized.tolist()
        for j, value in enumerate(normalized, start=1):
            raw.append({"model": label, "singular_value_index": j, "normalized_singular_value": float(value)})

    result = {
        "title": "Experiment 8: Singular Value Decay",
        "shape": shape,
        "unfolding": "mode-1",
        "threshold": threshold,
        "spectra": spectra,
    }
    save_json("exp08_singular_value_decay", result)
    save_csv("exp08_singular_value_decay_raw", raw)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    for ax, (label, _, _), (_, vals) in zip(axes.ravel(), specifications, spectra.items()):
        vals_arr = np.asarray(vals)
        j = np.arange(1, vals_arr.size + 1)
        ax.semilogy(j, vals_arr, "-", color=COLORS["blue"])
        ax.axhline(threshold, linestyle="--", color=COLORS["grey"], label=r"$10^{-3}$")
        ax.set_title(label)
        ax.set_xlabel(r"Singular Value Index $j$")
        ax.set_ylabel(r"$\sigma_j/\sigma_1$")
        style_axis(ax, log_grid=True)
        ax.legend(frameon=False)
    
    fig.tight_layout()
    save_figure(fig, "exp08_singular_value_decay")
    return result


# ============================================================
# Experiment 9: Error Concentration
# ============================================================

def experiment_09_error_concentration() -> dict:
    print("\n[EXP 9] Error Concentration")

    shape = (12, 12, 12, 12)
    alpha = 0.6
    r = 5
    # User-selected oversampling values. The current draft previously used 5,10,20.
    p_values = [5, 10, 15]
    n_trials = _profile_count(500, 30, 1000)
    A = generate_cp_decay_tensor(shape, "exp", alpha=alpha, seed=BASE_SEED + 800)
    tt = tt_svd(A, r)
    tt_error = relative_error_tt(tt, A)

    distributions: dict[int, list[float]] = {p: [] for p in p_values}
    raw: List[dict] = []
    summaries: dict[int, dict] = {}

    for p in p_values:
        total_errors = []
        for trial in range(n_trials):
            seed = BASE_SEED + 8_000_000 + 100_000 * p + trial
            approx = rtt_svd(A, r, p=p, seed=seed)
            err = relative_error_tt(approx, A)
            excess = err - tt_error
            distributions[p].append(1e6 * excess)
            total_errors.append(err)
            raw.append({
                "p": p,
                "trial": trial,
                "rtt_svd_error": err,
                "tt_svd_reference_error": tt_error,
                "excess_error": excess,
                "scaled_excess_error_1e6": 1e6 * excess,
                "seed": seed,
            })
        arr = np.asarray(total_errors)
        summaries[p] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "q0.95": float(np.quantile(arr, 0.95)),
            "q0.99": float(np.quantile(arr, 0.99)),
            "mean_excess": float(np.mean(arr - tt_error)),
        }

    result = {
        "title": "Experiment 9: Error Concentration",
        "shape": shape,
        "alpha": alpha,
        "rank": r,
        "p_values": p_values,
        "n_trials": n_trials,
        "tt_svd_reference_error": tt_error,
        "summaries": summaries,
    }
    save_json("exp09_error_concentration", result)
    save_csv("exp09_error_concentration_raw", raw)

    fig, ax = plt.subplots(figsize=(7.7, 4.9))
    data = [distributions[p] for p in p_values]
    bp = ax.boxplot(data, tick_labels=[str(p) for p in p_values], showfliers=True, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor(COLORS["sky"])
        box.set_alpha(0.45)
    ax.axhline(
    0.0,
    linestyle="--",
    color=COLORS["orange"],
    label=r"$\varepsilon_{\mathrm{RTT}}=\varepsilon_{\mathrm{TT}}$"
)
    ax.set_xlabel(r"Oversampling $p$")
    ax.set_ylabel(
    r"$10^6(\varepsilon_{\mathrm{RTT}}-\varepsilon_{\mathrm{TT}})$"
)
    
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "exp09_error_concentration")
    return result


# ============================================================
# Experiment 10: Storage-Accuracy Trade-off
# ============================================================

def experiment_10_storage_accuracy_tradeoff() -> dict:
    print("\n[EXP 10] Storage-Accuracy Trade-off")

    shape = (16, 16, 16, 16)
    alpha = 0.6
    ranks = list(range(1, 14))
    p = 10
    n_trials = _profile_count(8, 2, 16)
    A = generate_cp_decay_tensor(shape, "exp", alpha=alpha, seed=BASE_SEED + 900)

    storage_fraction = []
    tt_error = []
    rtt_mean, rtt_std = [], []
    raw: List[dict] = []

    for r in ranks:
        tt = tt_svd(A, r)
        e_tt = relative_error_tt(tt, A)
        frac = tt.storage() / A.size
        vals = []
        for trial in range(n_trials):
            seed = BASE_SEED + 9_000_000 + 1000 * r + trial
            rtt = rtt_svd(A, r, p=p, seed=seed)
            e_rtt = relative_error_tt(rtt, A)
            vals.append(e_rtt)
            raw.append({
                "rank": r,
                "trial": trial,
                "storage_fraction": frac,
                "tt_svd_error": e_tt,
                "rtt_svd_error": e_rtt,
                "seed": seed,
            })
        storage_fraction.append(frac)
        tt_error.append(e_tt)
        rtt_mean.append(mean_std(vals)[0])
        rtt_std.append(mean_std(vals)[1])

    result = {
        "title": "Experiment 10: Storage-Accuracy Trade-off",
        "shape": shape,
        "alpha": alpha,
        "ranks": ranks,
        "p": p,
        "n_trials": n_trials,
        "storage_fraction": storage_fraction,
        "tt_svd_error": tt_error,
        "rtt_svd_error_mean": rtt_mean,
        "rtt_svd_error_std": rtt_std,
    }
    save_json("exp10_storage_accuracy_tradeoff", result)
    save_csv("exp10_storage_accuracy_tradeoff_raw", raw)

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    x = np.asarray(storage_fraction)
    ax.semilogy(x, tt_error, "o-", color=COLORS["orange"], label="TT-SVD")
    ax.semilogy(x, rtt_mean, "o--", color=COLORS["blue"], label=rf"RTT-SVD (p = {p})")
    for r in [1, 3, 5, 7, 9, 11, 13]:
        idx = ranks.index(r)
        ax.annotate(rf"r = {r}", (x[idx], rtt_mean[idx]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"TT storage fraction $\rho_{\mathrm{store}}$")
    ax.set_ylabel(r"Relative Error $\varepsilon$")
    
    style_axis(ax, log_grid=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "exp10_storage_accuracy_tradeoff")
    return result


# ============================================================
# Experiment 11: Gaussian Range-Finder Bound
# ============================================================

def experiment_11_gaussian_rangefinder_bound() -> dict:
    print("\n[EXP 11] Gaussian Range-Finder Bound")

    shape = (32, 16, 16, 16)
    alpha = 0.35
    r = 4
    p_values = [2, 4, 6, 8, 10]
    n_trials = _profile_count(200, 20, 500)
    A = generate_cp_decay_tensor(shape, "exp", alpha=alpha, seed=BASE_SEED + 1000)
    M = A.reshape(shape[0], -1)
    M_norm = float(np.linalg.norm(M, ord="fro"))
    svals = svd(M, compute_uv=False, check_finite=False)
    rank_r_tail = float(np.sqrt(np.sum(svals[r:] ** 2)) / M_norm)

    means, stds, q95s, expected_bounds = [], [], [], []
    raw: List[dict] = []

    for p in p_values:
        expected_bound = math.sqrt(1.0 + r / (p - 1.0)) * rank_r_tail
        expected_bounds.append(expected_bound)
        vals = []
        for trial in range(n_trials):
            seed = BASE_SEED + 10_000_000 + 100_000 * p + trial
            Q = randomized_range_finder(M, r, p, seed)
            residual = float(np.linalg.norm(M - Q @ (Q.T @ M), ord="fro") / M_norm)
            vals.append(residual)
            raw.append({
                "p": p,
                "trial": trial,
                "relative_frobenius_residual": residual,
                "expected_frobenius_bound": expected_bound,
                "optimal_rank_r_tail": rank_r_tail,
                "seed": seed,
            })
        arr = np.asarray(vals)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)))
        q95s.append(float(np.quantile(arr, 0.95)))

    result = {
        "title": "Experiment 11: Gaussian Range-Finder Bound",
        "tensor_shape": shape,
        "unfolding_shape": M.shape,
        "alpha": alpha,
        "rank": r,
        "p_values": p_values,
        "n_trials": n_trials,
        "empirical_mean": means,
        "empirical_std": stds,
        "q0.95": q95s,
        "expected_frobenius_bound": expected_bounds,
        "optimal_rank_r_tail": rank_r_tail,
        "bound_formula": "sqrt(1 + r/(p-1)) * (sum_{j>r} sigma_j^2)^(1/2) / ||A||_F",
        "scope": "isolated Chapter 4 Gaussian matrix range finder, not a global RTT-SVD error bound",
    }
    save_json("exp11_gaussian_rangefinder_bound", result)
    save_csv("exp11_gaussian_rangefinder_bound_raw", raw)

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    x = np.asarray(p_values)
    mean_arr = np.asarray(means)
    std_arr = np.asarray(stds)
    ax.semilogy(x, expected_bounds, "s--", color=COLORS["orange"], label="Expected Bound")
    ax.axhline(rank_r_tail, linestyle="--", color=COLORS["green"], label="Rank-r Tail")
    ax.errorbar(x, mean_arr, yerr=std_arr, fmt="o-", color=COLORS["blue"], capsize=3, label="Empirical Mean")
    ax.set_xlabel("Oversampling $p$")
    ax.set_ylabel( r"$\|M-QQ^\top M\|_F/\|M\|_F$")
    ax.set_xticks(p_values)
    
    style_axis(ax, log_grid=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "exp11_gaussian_rangefinder_bound")
    return result

# ============================================================
# Registry and main driver
# ============================================================

EXPERIMENT_FUNCTIONS = {
    1: experiment_01_accuracy_vs_bond_dimension,
    2: experiment_02_oversampling_parameter_study,
    3: experiment_03_runtime_vs_tensor_order,
    4: experiment_04_sparse_rtt_svd_complexity,
    5: experiment_05_scalability_vs_mode_size,
    6: experiment_06_robustness_to_additive_noise,
    7: experiment_07_method_comparison,
    8: experiment_08_singular_value_decay,
    9: experiment_09_error_concentration,
    10: experiment_10_storage_accuracy_tradeoff,
    11: experiment_11_gaussian_rangefinder_bound,
}

EXPERIMENT_NAMES = {
    1: "Accuracy versus Bond Dimension",
    2: "Oversampling Parameter Study",
    3: "Runtime versus Tensor Order",
    4: "Sparse RTT-SVD Complexity versus Tensor Order",
    5: "Scalability versus Mode Size",
    6: "Robustness to Additive Noise",
    7: "Comparison with TT-SVD and Randomized Tucker",
    8: "Singular Value Decay",
    9: "Error Concentration",
    10: "Storage-Accuracy Trade-off",
    11: "Gaussian Range-Finder Bound",
}

FIGURE_STEMS = [
    "exp01_accuracy_vs_bond_dimension",
    "exp02_oversampling_parameter_study",
    "exp03_runtime_vs_tensor_order",
    "exp04_sparse_rtt_svd_complexity",
    "exp05_scalability_vs_mode_size",
    "exp06_robustness_to_additive_noise",
    "exp07_method_comparison",
    "exp08_singular_value_decay",
    "exp09_error_concentration",
    "exp10_storage_accuracy_tradeoff",
    "exp11_gaussian_rangefinder_bound",
]


def main() -> None:
    print("=" * 78)
    print("Randomized TT-SVD: Chapter 6 Eleven-Experiment Suite")
    print("=" * 78)
    print(f"Run profile       : {RUN_PROFILE}")
    print(f"Experiments       : {EXPERIMENTS_TO_RUN}")
    print(f"Results directory : {RESULTS_DIR.resolve()}")
    print(f"Base seed         : {BASE_SEED}")
    print("RTT implementation: Chapter 5 Algorithm 7 + core-only equal-rank reduction")

    if EXPERIMENTS_TO_RUN == list(range(1, 12)):
        clean_previous_outputs()

    metadata = {
        "title": "Chapter 6 experiment suite aligned with Chapters 1-5 algorithms",
        "run_profile": RUN_PROFILE,
        "base_seed": BASE_SEED,
        "experiments_requested": EXPERIMENTS_TO_RUN,
        "experiment_names": EXPERIMENT_NAMES,
        "algorithm_implementation": {
            "tt_svd": "Chapter 3 Algorithm 2",
            "range_finder": "Chapter 4 Algorithm 4",
            "randomized_matrix_svd": "Chapter 4 Algorithm 6",
            "rtt_svd": "Chapter 5 Algorithm 7: right-to-left independent Gaussian sketches and RQ; then core-only rank reduction for same-rank comparisons",
            "randomized_tucker": "Algorithm 6 applied independently to mode unfoldings, then Tucker projection",
        },
        "error_concentration_p_values": [5, 10, 15],
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": {},
    }

    suite_start = time.perf_counter()
    for number in EXPERIMENTS_TO_RUN:
        func = EXPERIMENT_FUNCTIONS[number]
        t0 = time.perf_counter()
        try:
            func()
            elapsed = time.perf_counter() - t0
            metadata["status"][str(number)] = {
                "name": EXPERIMENT_NAMES[number],
                "status": "completed",
                "elapsed_s": elapsed,
            }
            print(f"  completed EXP {number} in {elapsed:.2f} s")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            metadata["status"][str(number)] = {
                "name": EXPERIMENT_NAMES[number],
                "status": "failed",
                "elapsed_s": elapsed,
                "error": repr(exc),
            }
            metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata["suite_elapsed_s"] = time.perf_counter() - suite_start
            save_json("run_metadata", metadata)
            raise

    metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["suite_elapsed_s"] = time.perf_counter() - suite_start
    save_json("run_metadata", metadata)
    save_json("experiment_manifest", {
        "experiment_order": EXPERIMENT_NAMES,
        "figure_stems": FIGURE_STEMS,
        "formats": ["png", "pdf"],
        "data_formats": ["json", "csv"],
    })

    print("\n" + "=" * 78)
    print("Run complete")
    print(f"Data    : {DATA_DIR.resolve()}")
    print(f"Figures : {FIGURES_DIR.resolve()}")
    print(f"Elapsed : {metadata['suite_elapsed_s']:.2f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()