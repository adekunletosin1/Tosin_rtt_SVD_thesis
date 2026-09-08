#!/usr/bin/env python3
"""
Ten-experiment RTT-SVD thesis suite.

This driver imports the single authoritative Chapter 1--5 implementation from
``tt_core.py`` and ``sparse_rtt_svd.py``. It does not define a second tensor
algorithm. Chapters 6 and 7 of the old draft are not used as implementation
specifications.

Experiments
-----------
1. Nearly low-rank tensors with Gaussian perturbations (Huber Sec. 5.1)
2. Oversampling parameter study, two tensor families (Huber Sec. 5.2)
3. Approximation quality versus tensor order, two tensor families (Huber Sec. 5.3)
4. Sparse runtime versus tensor order (Huber Sec. 5.4)
5. Error concentration over independent Gaussian sketches
6. Gaussian single-step range-finder bound validation
7. Accuracy versus target TT-rank
8. Mode-size scalability and exact-recovery check
9. Singular-value decay across principal TT matricizations
10. Storage-accuracy trade-off

The ALS/random-low-rank experiment in Huber Sec. 5.5 is intentionally excluded.
Every experiment produces one final figure design, saved as vector PDF and
360-dpi PNG, plus JSON and CSV numerical data.
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
)
from .sparse_rtt_svd import (
    SparseTensor,
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
    if any(x < 1 or x > 10 for x in EXPERIMENTS_TO_RUN):
        raise ValueError("RTT_EXPERIMENTS may contain only integers 1,...,10")
else:
    EXPERIMENTS_TO_RUN = list(range(1, 11))

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
    """Remove earlier exp01...exp10 files to prevent duplicate figure/data sets."""
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

def experiment_01_nearly_low_rank_noise() -> dict:
    print("\n[EXP 1] Nearly low-rank tensors with Gaussian perturbations")

    shape = (4,) * 10
    r_star = 10
    r = 10
    p = 5
    tau_values = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    target = feasible_tt_ranks(shape, r)

    raw: List[dict] = []
    det_by_tau = {tau: [] for tau in tau_values}
    rnd_by_tau = {tau: [] for tau in tau_values}

    for trial in range(HUBER_TRIALS):
        seed = BASE_SEED + 10_000 * trial
        exact = generate_random_tt_tensor(shape, r_star, seed=seed, normalize=True)
        rng = np.random.RandomState(seed + 7919)
        noise = normalize_dense(rng.standard_normal(shape))

        for tau in tau_values:
            A = exact + tau * noise
            det = tt_svd(A, target)
            rnd = rtt_svd(A, target, p=p, seed=seed + 1_000_003 + int(1000 * tau))
            e_det = relative_error_tt(det, A)
            e_rnd = relative_error_tt(rnd, A)

            det_by_tau[tau].append(e_det)
            rnd_by_tau[tau].append(e_rnd)
            raw.append({
                "trial": trial,
                "tau": tau,
                "det_error": e_det,
                "rnd_error": e_rnd,
                "error_ratio": e_rnd / max(e_det, 1e-30),
            })

    det_mean, det_std, rnd_mean, rnd_std = [], [], [], []
    for tau in tau_values:
        m, s = mean_std(det_by_tau[tau]); det_mean.append(m); det_std.append(s)
        m, s = mean_std(rnd_by_tau[tau]); rnd_mean.append(m); rnd_std.append(s)

    result = {
        "description": "Huber Sec. 5.1 aligned nearly-low-rank noise study",
        "shape": shape,
        "target_rank_nominal": r,
        "target_ranks_feasible": target,
        "exact_rank_nominal": r_star,
        "p": p,
        "tau_values": tau_values,
        "n_trials": HUBER_TRIALS,
        "det_mean": det_mean,
        "det_std": det_std,
        "rnd_mean": rnd_mean,
        "rnd_std": rnd_std,
    }
    save_json("exp01_nearly_low_rank_noise", result)
    save_csv("exp01_nearly_low_rank_noise_raw", raw)

    x = np.asarray(tau_values)
    dm, ds = np.asarray(det_mean), np.asarray(det_std)
    rm, rs = np.asarray(rnd_mean), np.asarray(rnd_std)

    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    ax.plot(x, dm, marker="o", color=COLORS["blue"], label="TT-SVD")
    ax.fill_between(x, np.maximum(dm - ds, 0), dm + ds, color=COLORS["blue"], alpha=0.16)
    ax.plot(x, rm, marker="s", color=COLORS["vermillion"], label=f"RTT-SVD ($p={p}$)")
    ax.fill_between(x, np.maximum(rm - rs, 0), rm + rs, color=COLORS["vermillion"], alpha=0.16)
    ax.set_xlabel(r"Noise level $\tau$")
    ax.set_ylabel(r"Relative Frobenius error")
    ax.set_xlim(min(x), max(x))
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, "exp01_nearly_low_rank_noise")
    return result


# ============================================================
# Experiment 2
# ============================================================

def experiment_02_oversampling() -> dict:
    print("\n[EXP 2] Approximation quality versus oversampling")

    shape = (4,) * 10
    r = 10
    r_star = 10
    tau = 0.05
    p_values = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25]
    target = feasible_tt_ranks(shape, r)

    families = {
        "nearly_low_rank": {},
        "quadratic_decay": {},
    }
    raw: List[dict] = []

    for family in families:
        ratios_by_p = {p: [] for p in p_values}

        for trial in range(HUBER_TRIALS):
            seed = BASE_SEED + 20_000 * trial + (0 if family == "nearly_low_rank" else 5000)

            if family == "nearly_low_rank":
                A = generate_nearly_low_rank_tensor(shape, r_star, tau, seed)
            else:
                A = generate_spectral_tt_tensor(
                    shape, source_rank=20, decay="quadratic", seed=seed
                )

            det = tt_svd(A, target)
            e_det = relative_error_tt(det, A)

            for p in p_values:
                rnd = rtt_svd(A, target, p=p, seed=seed + 1_100_000 + 101 * p)
                e_rnd = relative_error_tt(rnd, A)
                ratio = e_rnd / max(e_det, 1e-30)
                ratios_by_p[p].append(ratio)
                raw.append({
                    "family": family,
                    "trial": trial,
                    "p": p,
                    "det_error": e_det,
                    "rnd_error": e_rnd,
                    "error_ratio": ratio,
                })

        means, stds = [], []
        for p in p_values:
            m, s = mean_std(ratios_by_p[p])
            means.append(m); stds.append(s)
        families[family] = {"ratio_mean": means, "ratio_std": stds}

    result = {
        "description": "Huber Sec. 5.2 aligned oversampling study",
        "shape": shape,
        "target_rank_nominal": r,
        "target_ranks_feasible": target,
        "tau_nearly_low_rank": tau,
        "p_values": p_values,
        "n_trials": HUBER_TRIALS,
        "families": families,
    }
    save_json("exp02_oversampling", result)
    save_csv("exp02_oversampling_raw", raw)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.65), constrained_layout=True, sharey=True)
    specs = [
        ("nearly_low_rank", "Nearly low-rank + noise", COLORS["blue"]),
        ("quadratic_decay", "Quadratic spectral decay", COLORS["green"]),
    ]

    for idx, (family, title, colour) in enumerate(specs):
        ax = axes[idx]
        mean = np.asarray(families[family]["ratio_mean"])
        std = np.asarray(families[family]["ratio_std"])
        p_arr = np.asarray(p_values)
        ax.plot(p_arr, mean, marker="o", color=colour, label=r"Mean $\varepsilon_{rnd}/\varepsilon_{det}$")
        ax.fill_between(p_arr, np.maximum(mean - std, 0), mean + std, color=colour, alpha=0.18, label=r"$\pm1$ std.")
        ax.axhline(1.0, color=COLORS["black"], linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel(r"Oversampling $p$")
        ax.set_title(title)
        ax.set_xticks(p_values)
        style_axis(ax)
        panel_label(ax, f"({chr(97+idx)})")
        ax.legend(frameon=False, loc="upper right")

    axes[0].set_ylabel(r"Error ratio $\varepsilon_{rnd}/\varepsilon_{det}$")
    save_figure(fig, "exp02_oversampling_combined")
    return result


# ============================================================
# Experiment 3
# ============================================================

def experiment_03_quality_vs_order() -> dict:
    print("\n[EXP 3] Approximation quality versus tensor order")

    n = 4
    r = 10
    r_star = 10
    p = 5
    tau = 0.05
    orders = HUBER_QUALITY_ORDERS
    raw: List[dict] = []
    families = {"nearly_low_rank": {}, "quadratic_decay": {}}

    for family in families:
        means, stds = [], []

        for d in orders:
            shape = (n,) * d
            target = feasible_tt_ranks(shape, r)
            ratios = []

            for trial in range(HUBER_TRIALS):
                seed = BASE_SEED + 30_000 * trial + 100 * d + (0 if family == "nearly_low_rank" else 7000)

                if family == "nearly_low_rank":
                    A = generate_nearly_low_rank_tensor(shape, r_star, tau, seed)
                else:
                    A = generate_spectral_tt_tensor(shape, source_rank=20, decay="quadratic", seed=seed)

                det = tt_svd(A, target)
                rnd = rtt_svd(A, target, p=p, seed=seed + 1_200_000)
                e_det = relative_error_tt(det, A)
                e_rnd = relative_error_tt(rnd, A)
                ratio = e_rnd / max(e_det, 1e-30)
                ratios.append(ratio)

                raw.append({
                    "family": family,
                    "trial": trial,
                    "order": d,
                    "p": p,
                    "det_error": e_det,
                    "rnd_error": e_rnd,
                    "error_ratio": ratio,
                })

            m, s = mean_std(ratios)
            means.append(m); stds.append(s)

        families[family] = {"ratio_mean": means, "ratio_std": stds}

    result = {
        "description": "Huber Sec. 5.3 aligned quality-versus-order study",
        "mode_size": n,
        "orders": orders,
        "target_rank_nominal": r,
        "exact_rank_nominal": r_star,
        "p": p,
        "tau_nearly_low_rank": tau,
        "n_trials": HUBER_TRIALS,
        "families": families,
        "note": "Dense Python reproduction uses d=4,...,10 for computational tractability.",
    }
    save_json("exp03_quality_vs_order", result)
    save_csv("exp03_quality_vs_order_raw", raw)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.65), constrained_layout=True, sharey=True)
    specs = [
        ("nearly_low_rank", "Nearly low-rank + noise", COLORS["purple"]),
        ("quadratic_decay", "Quadratic spectral decay", COLORS["orange"]),
    ]

    for idx, (family, title, colour) in enumerate(specs):
        ax = axes[idx]
        mean = np.asarray(families[family]["ratio_mean"])
        std = np.asarray(families[family]["ratio_std"])
        x = np.asarray(orders)
        ax.plot(x, mean, marker="o", color=colour)
        ax.fill_between(x, np.maximum(mean - std, 0), mean + std, color=colour, alpha=0.18)
        ax.axhline(1.0, color=COLORS["black"], linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel(r"Tensor order $d$")
        ax.set_title(title)
        ax.set_xticks(orders)
        style_axis(ax)
        panel_label(ax, f"({chr(97+idx)})")

    axes[0].set_ylabel(r"Error ratio $\varepsilon_{rnd}/\varepsilon_{det}$")
    save_figure(fig, "exp03_quality_vs_order_combined")
    return result


# ============================================================
# Experiment 4
# ============================================================

def experiment_04_sparse_runtime() -> dict:
    print("\n[EXP 4] Sparse runtime versus tensor order")

    n = 2
    N = 500
    r = 10
    p = 10
    orders = SPARSE_RUNTIME_ORDERS
    raw: List[dict] = []

    rnd_means, rnd_stds = [], []
    det_means, det_stds = [], []
    speedups = []

    for d in orders:
        shape = (n,) * d
        target = feasible_tt_ranks(shape, r)
        rnd_times = []
        det_times = []
        dense_feasible = int(np.prod(shape, dtype=object)) <= MAX_DENSE_REFERENCE_ENTRIES

        for trial in range(TIMING_TRIALS):
            seed = BASE_SEED + 40_000 * trial + d
            coords, values = generate_sparse_tensor(shape, N, seed)

            tr = benchmark(
                lambda: sparse_rtt_svd(shape, coords, values, target, p=p, seed=seed + 1_300_000),
                TIMING_REPEATS,
            )
            rnd_times.append(tr)

            td = np.nan
            if dense_feasible:
                def _det_call():
                    A_dense = sparse_to_dense(shape, coords, values)
                    _ = tt_svd(A_dense, target)
                td = benchmark(_det_call, TIMING_REPEATS)
                det_times.append(td)

            raw.append({
                "trial": trial,
                "order": d,
                "mode_size": n,
                "N_nonzero_samples": N,
                "rnd_time_s": tr,
                "det_time_s": td,
                "dense_reference_attempted": bool(dense_feasible),
            })

        rm, rs = mean_std(rnd_times)
        rnd_means.append(rm); rnd_stds.append(rs)

        if det_times:
            dm, ds = mean_std(det_times)
            det_means.append(dm); det_stds.append(ds)
            speedups.append(dm / rm if rm > 0 else np.nan)
        else:
            det_means.append(np.nan); det_stds.append(np.nan); speedups.append(np.nan)

        print(f"  d={d:2d}: sparse RTT={rm:.4e}s" + (f", TT={det_means[-1]:.4e}s" if np.isfinite(det_means[-1]) else ", TT=skipped"))

    result = {
        "description": "Huber Sec. 5.4 sparse runtime study",
        "orders": orders,
        "mode_size": n,
        "N_nonzero_samples": N,
        "target_rank_nominal": r,
        "p": p,
        "timing_trials": TIMING_TRIALS,
        "timing_repeats": TIMING_REPEATS,
        "max_dense_reference_entries": MAX_DENSE_REFERENCE_ENTRIES,
        "rnd_mean_s": rnd_means,
        "rnd_std_s": rnd_stds,
        "det_mean_s": det_means,
        "det_std_s": det_stds,
        "speedup": speedups,
    }
    save_json("exp04_sparse_runtime", result)
    save_csv("exp04_sparse_runtime_raw", raw)

    x = np.asarray(orders)
    rm = np.asarray(rnd_means)
    rs = np.asarray(rnd_stds)
    dm = np.asarray(det_means)
    ds = np.asarray(det_stds)
    sp = np.asarray(speedups)

    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.7), constrained_layout=True)

    ax = axes[0]
    ax.plot(x, rm, marker="o", color=COLORS["green"], label="Sparse RTT-SVD")
    ax.fill_between(x, np.maximum(rm - rs, 1e-12), rm + rs, color=COLORS["green"], alpha=0.17)
    mask = np.isfinite(dm)
    if np.any(mask):
        ax.plot(x[mask], dm[mask], marker="s", color=COLORS["vermillion"], label="Dense TT-SVD")
        ax.fill_between(x[mask], np.maximum(dm[mask] - ds[mask], 1e-12), dm[mask] + ds[mask], color=COLORS["vermillion"], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel(r"Tensor order $d$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(r"Runtime, $N=500$, $n=2$")
    style_axis(ax, log_grid=True)
    panel_label(ax, "(a)")
    ax.legend(frameon=False)

    ax = axes[1]
    if np.any(mask):
        ax.plot(x[mask], sp[mask], marker="D", color=COLORS["blue"])
        ax.axhline(1.0, color=COLORS["black"], linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"Tensor order $d$")
    ax.set_ylabel(r"Speedup $T_{TT}/T_{RTT}$")
    ax.set_title("Measured speedup where dense TT-SVD is feasible")
    style_axis(ax)
    panel_label(ax, "(b)")

    save_figure(fig, "exp04_sparse_runtime_combined")
    return result


# ============================================================
# Experiment 5
# ============================================================

def experiment_05_error_concentration() -> dict:
    print("\n[EXP 5] Error concentration")

    # Choose r+p strictly below the edge unfolding dimension for every p.
    # This avoids a saturated full-range sketch, which would remove the
    # randomness and make a concentration study uninformative.
    shape = (16, 16, 16, 16)
    r = 4
    p_values = [5, 10, 15]
    A = generate_cp_decay_tensor(shape, "exp", alpha=0.6, seed=BASE_SEED)
    target = feasible_tt_ranks(shape, r)
    det_error = relative_error_tt(tt_svd(A, target), A)

    errors_by_p: Dict[int, List[float]] = {p: [] for p in p_values}
    raw: List[dict] = []

    for p_value in p_values:
        for trial in range(CONCENTRATION_TRIALS):
            seed = BASE_SEED + 50_000 * trial + p_value
            e = relative_error_tt(
                rtt_svd(A, target, p=p_value, seed=seed),
                A,
            )
            errors_by_p[p_value].append(e)
            raw.append({
                "trial": trial,
                "p": p_value,
                "rnd_error": e,
                "det_error": det_error,
                "error_ratio": e / det_error,
                "excess_percent": 100.0 * (e / det_error - 1.0),
            })

    stats = {}
    for p_value in p_values:
        x = np.asarray(errors_by_p[p_value])
        stats[str(p_value)] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(np.median(x)),
            "q95": float(np.percentile(x, 95)),
            "q99": float(np.percentile(x, 99)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean_error_ratio": float(np.mean(x / det_error)),
        }

    result = {
        "description": "Distribution of RTT-SVD error over independent Gaussian sketches",
        "shape": shape,
        "target_rank": r,
        "p_values": p_values,
        "n_trials": CONCENTRATION_TRIALS,
        "det_error": det_error,
        "statistics": stats,
        "design_note": "All r+p values are below the edge unfolding dimension, so no tested sketch is saturated.",
    }
    save_json("exp05_error_concentration", result)
    save_csv("exp05_error_concentration_raw", raw)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, 4.45),
        constrained_layout=True,
        sharey=False,
    )
    hist_colours = [COLORS["blue"], COLORS["green"], COLORS["purple"]]

    for idx, (ax, p_value, colour) in enumerate(zip(axes, p_values, hist_colours)):
        vals = np.asarray(errors_by_p[p_value])
        excess_pct = 100.0 * (vals / det_error - 1.0)
        edges = safe_histogram_edges(excess_pct)
        ax.hist(
            excess_pct,
            bins=edges,
            density=True,
            color=colour,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.axvline(
            float(np.mean(excess_pct)),
            color=COLORS["black"],
            linestyle="--",
            linewidth=1.6,
            label="RTT-SVD mean",
        )
        ax.axvline(
            0.0,
            color=COLORS["vermillion"],
            linestyle=":",
            linewidth=1.6,
            label="TT-SVD reference",
        )
        ax.set_xlabel(r"Excess error over TT-SVD (\%)")
        ax.set_title(fr"$p={p_value}$")
        style_axis(ax)
        panel_label(ax, f"({chr(97+idx)})")

    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
    )
    save_figure(fig, "exp05_error_concentration_combined")
    return result


# ============================================================
# Experiment 6
# ============================================================

def experiment_06_rangefinder_bound() -> dict:
    print("\n[EXP 6] Gaussian range-finder bound validation")

    shape = (32, 16, 16, 16)
    r = 4
    p_values = [4, 6, 8, 10, 15, 20]
    A = generate_cp_decay_tensor(shape, "exp", alpha=0.35, seed=BASE_SEED)
    M = A.reshape(shape[0], -1)
    svals = svd(M, compute_uv=False, check_finite=False)
    tail = float(np.sqrt(np.sum(svals[r:] ** 2)))

    means, stds, q95, bounds = [], [], [], []
    raw: List[dict] = []

    for p in p_values:
        residuals = []
        bound = amplification_factor(r, p) * tail
        bounds.append(bound)

        for trial in range(BOUND_TRIALS):
            seed = BASE_SEED + 60_000 * trial + p
            Q = randomized_range_finder(M, r, p, seed)
            residual = float(np.linalg.norm(M - Q @ (Q.T @ M), ord="fro"))
            residuals.append(residual)
            raw.append({
                "trial": trial,
                "p": p,
                "residual": residual,
                "theoretical_bound": bound,
            })

        x = np.asarray(residuals)
        means.append(float(np.mean(x)))
        stds.append(float(np.std(x)))
        q95.append(float(np.percentile(x, 95)))

    result = {
        "description": "Single-step Gaussian range-finder bound validation",
        "matrix_shape": M.shape,
        "target_rank": r,
        "p_values": p_values,
        "n_trials": BOUND_TRIALS,
        "tail_energy": tail,
        "empirical_mean": means,
        "empirical_std": stds,
        "q95": q95,
        "theoretical_bound": bounds,
    }
    save_json("exp06_rangefinder_bound", result)
    save_csv("exp06_rangefinder_bound_raw", raw)

    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    p_arr = np.asarray(p_values)
    m = np.asarray(means); s = np.asarray(stds)
    ax.errorbar(p_arr, m, yerr=s, marker="o", capsize=3, color=COLORS["blue"], label=r"Empirical mean $\pm$ std.")
    ax.plot(p_arr, q95, marker="s", color=COLORS["purple"], label="95th percentile")
    ax.plot(p_arr, bounds, marker="^", color=COLORS["vermillion"], label="Theoretical bound")
    ax.set_yscale("log")
    ax.set_xlabel(r"Oversampling $p$")
    ax.set_ylabel(r"$\|M-QQ^{\mathsf{T}}M\|_{\mathrm{F}}$")
    ax.set_xticks(p_values)
    style_axis(ax, log_grid=True)
    ax.legend(frameon=False)
    save_figure(fig, "exp06_rangefinder_bound")
    return result


# ============================================================
# Experiment 7
# ============================================================

def experiment_07_accuracy_vs_rank() -> dict:
    print("\n[EXP 7] Accuracy versus target TT-rank")

    shape = (16, 16, 16, 16)
    ranks = list(range(1, 13))
    p = 10
    raw: List[dict] = []
    families = {}

    specs = {
        "exponential": ("exp", {"alpha": 0.5}),
        "polynomial": ("poly", {"beta": 1.5}),
    }

    for label, (decay, params) in specs.items():
        det_mean, det_std, rnd_mean, rnd_std = [], [], [], []

        for r in ranks:
            de, re = [], []
            for trial in range(GENERAL_TRIALS):
                seed = BASE_SEED + 70_000 * trial + 100 * r + (0 if label == "exponential" else 9000)
                A = generate_cp_decay_tensor(shape, decay, seed=seed, **params)
                target = feasible_tt_ranks(shape, r)
                e_det = relative_error_tt(tt_svd(A, target), A)
                e_rnd = relative_error_tt(rtt_svd(A, target, p=p, seed=seed + 1_400_000), A)
                de.append(e_det); re.append(e_rnd)
                raw.append({
                    "family": label,
                    "trial": trial,
                    "rank": r,
                    "det_error": e_det,
                    "rnd_error": e_rnd,
                })

            m, s = mean_std(de); det_mean.append(m); det_std.append(s)
            m, s = mean_std(re); rnd_mean.append(m); rnd_std.append(s)

        families[label] = {
            "det_mean": det_mean, "det_std": det_std,
            "rnd_mean": rnd_mean, "rnd_std": rnd_std,
        }

    result = {
        "description": "Current-thesis accuracy versus target TT-rank",
        "shape": shape,
        "ranks": ranks,
        "p": p,
        "n_trials": GENERAL_TRIALS,
        "families": families,
    }
    save_json("exp07_accuracy_vs_rank", result)
    save_csv("exp07_accuracy_vs_rank_raw", raw)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True, sharey=True)
    titles = ["Exponential decay", "Polynomial decay"]

    for idx, (ax, family, title) in enumerate(zip(axes, ["exponential", "polynomial"], titles)):
        block = families[family]
        x = np.asarray(ranks)
        dm, ds = np.asarray(block["det_mean"]), np.asarray(block["det_std"])
        rm, rs = np.asarray(block["rnd_mean"]), np.asarray(block["rnd_std"])

        ax.plot(x, dm, marker="o", color=COLORS["blue"], label="TT-SVD")
        ax.fill_between(x, np.maximum(dm-ds, 1e-16), dm+ds, color=COLORS["blue"], alpha=0.14)
        ax.plot(x, rm, marker="s", color=COLORS["vermillion"], label=f"RTT-SVD ($p={p}$)")
        ax.fill_between(x, np.maximum(rm-rs, 1e-16), rm+rs, color=COLORS["vermillion"], alpha=0.14)
        ax.set_yscale("log")
        ax.set_xlabel(r"Target TT-rank $r$")
        ax.set_title(title)
        ax.set_xticks(ranks)
        style_axis(ax, log_grid=True)
        panel_label(ax, f"({chr(97+idx)})")
        ax.legend(frameon=False)

    axes[0].set_ylabel("Relative Frobenius error")
    save_figure(fig, "exp07_accuracy_vs_rank_combined")
    return result


# ============================================================
# Experiment 8
# ============================================================

def experiment_08_mode_size_exact_recovery() -> dict:
    print("\n[EXP 8] Mode-size scalability and exact recovery")

    d = 4
    mode_sizes = [8, 12, 16, 20]
    r = 4
    p = 10
    raw: List[dict] = []

    det_time_mean, det_time_std = [], []
    rnd_time_mean, rnd_time_std = [], []
    det_error_mean, rnd_error_mean = [], []

    for n in mode_sizes:
        shape = (n,) * d
        target = feasible_tt_ranks(shape, r)
        dt, rt, de, re = [], [], [], []

        for trial in range(max(3, min(GENERAL_TRIALS, 5))):
            seed = BASE_SEED + 80_000 * trial + n
            A = generate_random_tt_tensor(shape, r, seed=seed, normalize=True)

            t_det = benchmark(lambda: tt_svd(A, target), max(1, min(TIMING_REPEATS, 2)))
            t_rnd = benchmark(lambda: rtt_svd(A, target, p=p, seed=seed + 1_500_000), max(1, min(TIMING_REPEATS, 2)))

            det_tt = tt_svd(A, target)
            rnd_tt = rtt_svd(A, target, p=p, seed=seed + 1_500_000)
            e_det = relative_error_tt(det_tt, A)
            e_rnd = relative_error_tt(rnd_tt, A)

            dt.append(t_det); rt.append(t_rnd); de.append(e_det); re.append(e_rnd)
            raw.append({
                "trial": trial,
                "mode_size": n,
                "det_time_s": t_det,
                "rnd_time_s": t_rnd,
                "det_error": e_det,
                "rnd_error": e_rnd,
            })

        m, s = mean_std(dt); det_time_mean.append(m); det_time_std.append(s)
        m, s = mean_std(rt); rnd_time_mean.append(m); rnd_time_std.append(s)
        det_error_mean.append(float(np.mean(de)))
        rnd_error_mean.append(float(np.mean(re)))

    result = {
        "description": "Dense mode-size scalability on exact TT-rank-r tensors",
        "order": d,
        "mode_sizes": mode_sizes,
        "exact_rank": r,
        "target_rank": r,
        "p": p,
        "det_time_mean_s": det_time_mean,
        "det_time_std_s": det_time_std,
        "rnd_time_mean_s": rnd_time_mean,
        "rnd_time_std_s": rnd_time_std,
        "det_error_mean": det_error_mean,
        "rnd_error_mean": rnd_error_mean,
    }
    save_json("exp08_mode_size_exact_recovery", result)
    save_csv("exp08_mode_size_exact_recovery_raw", raw)

    x = np.asarray(mode_sizes)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(x, det_time_mean, yerr=det_time_std, marker="o", capsize=3, color=COLORS["blue"], label="TT-SVD")
    ax.errorbar(x, rnd_time_mean, yerr=rnd_time_std, marker="s", capsize=3, color=COLORS["vermillion"], label=f"RTT-SVD ($p={p}$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"Mode size $n$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(r"Scalability for order $d=4$")
    ax.set_xticks(mode_sizes)
    style_axis(ax, log_grid=True)
    panel_label(ax, "(a)")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(x, det_error_mean, marker="o", color=COLORS["blue"], label="TT-SVD")
    ax.plot(x, rnd_error_mean, marker="s", color=COLORS["vermillion"], label="RTT-SVD")
    ax.set_yscale("log")
    ax.set_xlabel(r"Mode size $n$")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Exact-recovery check")
    ax.set_xticks(mode_sizes)
    style_axis(ax, log_grid=True)
    panel_label(ax, "(b)")
    ax.legend(frameon=False)

    save_figure(fig, "exp08_mode_size_exact_recovery_combined")
    return result


# ============================================================
# Experiment 9
# ============================================================

def experiment_09_singular_value_decay() -> dict:
    print("\n[EXP 9] Singular-value decay across TT matricizations")

    shape = (16, 16, 16, 16)
    tensors = {
        "exponential": generate_spectral_tt_tensor(shape, source_rank=16, decay="exponential", alpha=0.7, seed=BASE_SEED),
        "quadratic": generate_spectral_tt_tensor(shape, source_rank=16, decay="quadratic", seed=BASE_SEED + 1),
    }

    result = {
        "description": "Singular values of all d-1 TT matricizations",
        "shape": shape,
        "families": {},
    }
    raw: List[dict] = []

    for family, A in tensors.items():
        sv_by_split = []
        for split in range(1, len(shape)):
            M = A.reshape(int(np.prod(shape[:split])), -1)
            svals = svd(M, compute_uv=False, check_finite=False)
            svals = svals / max(svals[0], 1e-300)
            sv_by_split.append(svals.tolist())
            for idx, val in enumerate(svals, start=1):
                raw.append({
                    "family": family,
                    "split": split,
                    "index": idx,
                    "normalized_singular_value": float(val),
                })
        result["families"][family] = {"normalized_singular_values": sv_by_split}

    save_json("exp09_singular_value_decay", result)
    save_csv("exp09_singular_value_decay_raw", raw)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True, sharey=True)
    family_specs = [
        ("exponential", "Exponential spectral decay", [COLORS["blue"], COLORS["sky"], COLORS["purple"]]),
        ("quadratic", "Quadratic spectral decay", [COLORS["green"], COLORS["orange"], COLORS["vermillion"]]),
    ]

    for panel, (family, title, cols) in enumerate(family_specs):
        ax = axes[panel]
        spectra = result["families"][family]["normalized_singular_values"]
        for split, (vals, colour) in enumerate(zip(spectra, cols), start=1):
            vals = np.asarray(vals)
            # Plot the leading spectrum only.  Values beyond the imposed TT
            # rank are numerical round-off and obscure the informative decay.
            n_show = min(20, len(vals))
            vals_plot = vals[:n_show]
            ax.semilogy(np.arange(1, n_show + 1), vals_plot, marker="o", markersize=4.5, color=colour, label=fr"split $k={split}$")
        ax.set_xlabel("Singular-value index")
        ax.set_title(title)
        style_axis(ax, log_grid=True)
        panel_label(ax, f"({chr(97+panel)})")
        ax.legend(frameon=False)

    axes[0].set_ylabel(r"Normalized singular value $\sigma_j/\sigma_1$")
    save_figure(fig, "exp09_singular_value_decay_combined")
    return result


# ============================================================
# Experiment 10
# ============================================================

def experiment_10_storage_accuracy() -> dict:
    print("\n[EXP 10] Storage-accuracy trade-off")

    shape = (16, 16, 16, 16)
    ranks = list(range(1, 14))
    p = 10
    dense_storage = int(np.prod(shape))
    raw: List[dict] = []

    det_mean, rnd_mean, storage_fraction, compression_ratio = [], [], [], []

    for r in ranks:
        det_errors, rnd_errors, fractions = [], [], []

        for trial in range(GENERAL_TRIALS):
            seed = BASE_SEED + 100_000 * trial + r
            A = generate_cp_decay_tensor(shape, "exp", alpha=0.5, seed=seed)
            target = feasible_tt_ranks(shape, r)
            det = tt_svd(A, target)
            rnd = rtt_svd(A, target, p=p, seed=seed + 1_600_000)

            e_det = relative_error_tt(det, A)
            e_rnd = relative_error_tt(rnd, A)
            frac = det.storage() / dense_storage

            det_errors.append(e_det); rnd_errors.append(e_rnd); fractions.append(frac)
            raw.append({
                "trial": trial,
                "rank": r,
                "stored_parameters": det.storage(),
                "dense_parameters": dense_storage,
                "storage_fraction": frac,
                "det_error": e_det,
                "rnd_error": e_rnd,
                "error_ratio": e_rnd / max(e_det, 1e-30),
            })

        det_mean.append(float(np.mean(det_errors)))
        rnd_mean.append(float(np.mean(rnd_errors)))
        sf = float(np.mean(fractions))
        storage_fraction.append(sf)
        compression_ratio.append(1.0 / sf)

    result = {
        "description": "Storage-accuracy trade-off for a 16^4 exponentially decaying tensor",
        "shape": shape,
        "ranks": ranks,
        "p": p,
        "n_trials": GENERAL_TRIALS,
        "dense_storage": dense_storage,
        "storage_fraction": storage_fraction,
        "compression_ratio": compression_ratio,
        "det_error_mean": det_mean,
        "rnd_error_mean": rnd_mean,
    }
    save_json("exp10_storage_accuracy", result)
    save_csv("exp10_storage_accuracy_raw", raw)

    sf_pct = 100 * np.asarray(storage_fraction)
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.7), constrained_layout=True)

    ax = axes[0]
    ax.plot(sf_pct, det_mean, marker="o", color=COLORS["blue"], label="TT-SVD")
    ax.plot(sf_pct, rnd_mean, marker="s", color=COLORS["vermillion"], label=f"RTT-SVD ($p={p}$)")
    ax.set_yscale("log")
    ax.set_xlabel("TT storage (% of dense tensor)")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Accuracy at a fixed storage budget")
    style_axis(ax, log_grid=True)
    panel_label(ax, "(a)")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(ranks, compression_ratio, marker="D", color=COLORS["green"])
    ax.set_yscale("log")
    ax.set_xlabel(r"Target TT-rank $r$")
    ax.set_ylabel("Compression ratio (dense / TT)")
    ax.set_title("Compression versus target rank")
    ax.set_xticks(ranks)
    style_axis(ax, log_grid=True)
    panel_label(ax, "(b)")

    save_figure(fig, "exp10_storage_accuracy_combined")
    return result


# ============================================================
# Metadata and main
# ============================================================

EXPERIMENT_FUNCTIONS = {
    1: experiment_01_nearly_low_rank_noise,
    2: experiment_02_oversampling,
    3: experiment_03_quality_vs_order,
    4: experiment_04_sparse_runtime,
    5: experiment_05_error_concentration,
    6: experiment_06_rangefinder_bound,
    7: experiment_07_accuracy_vs_rank,
    8: experiment_08_mode_size_exact_recovery,
    9: experiment_09_singular_value_decay,
    10: experiment_10_storage_accuracy,
}

EXPERIMENT_NAMES = {
    1: "Nearly low-rank tensors with Gaussian perturbations",
    2: "Oversampling parameter study",
    3: "Approximation quality versus tensor order",
    4: "Sparse runtime versus tensor order",
    5: "Error concentration",
    6: "Gaussian range-finder bound validation",
    7: "Accuracy versus target TT-rank",
    8: "Mode-size scalability and exact recovery",
    9: "Singular-value decay across TT matricizations",
    10: "Storage-accuracy trade-off",
}


def main() -> None:
    print("=" * 78)
    print("Randomized TT-SVD: Final Ten-Experiment Thesis Suite")
    print("=" * 78)
    print(f"Run profile       : {RUN_PROFILE}")
    print(f"Experiments       : {EXPERIMENTS_TO_RUN}")
    print(f"Results directory : {RESULTS_DIR.resolve()}")
    print(f"Base seed         : {BASE_SEED}")

    # For a full final run, clear earlier experiment outputs so the directories
    # contain one coherent experiment set. For subset reruns, keep other files.
    if EXPERIMENTS_TO_RUN == list(range(1, 11)):
        clean_previous_outputs()

    metadata = {
        "title": "Randomized TT-SVD Ten-Experiment Suite, Chapters 1-5 aligned",
        "run_profile": RUN_PROFILE,
        "base_seed": BASE_SEED,
        "experiments_requested": EXPERIMENTS_TO_RUN,
        "experiment_names": EXPERIMENT_NAMES,
        "profile_settings": PROFILE,
        "algorithm_implementation": "Chapter 5 Algorithm 7, right-to-left independent Gaussian sketches, followed by Chapter 3 core-only TT rounding for equal-rank comparisons",
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": {},
    }

    all_results = {}
    suite_start = time.perf_counter()

    for number in EXPERIMENTS_TO_RUN:
        func = EXPERIMENT_FUNCTIONS[number]
        t0 = time.perf_counter()
        try:
            result = func()
            elapsed = time.perf_counter() - t0
            all_results[f"exp{number:02d}"] = result
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
    metadata["completed_experiments"] = len(all_results)
    save_json("run_metadata", metadata)

    manifest = {
        "figure_designs": [
            "exp01_nearly_low_rank_noise",
            "exp02_oversampling_combined",
            "exp03_quality_vs_order_combined",
            "exp04_sparse_runtime_combined",
            "exp05_error_concentration_combined",
            "exp06_rangefinder_bound",
            "exp07_accuracy_vs_rank_combined",
            "exp08_mode_size_exact_recovery_combined",
            "exp09_singular_value_decay_combined",
            "exp10_storage_accuracy_combined",
        ],
        "note": "Each design is saved once as PNG and once as PDF.",
    }
    save_json("experiment_manifest", manifest)

    print("\n" + "=" * 78)
    print("Run complete")
    print(f"Data    : {DATA_DIR.resolve()}")
    print(f"Figures : {FIGURES_DIR.resolve()}")
    print(f"Elapsed : {metadata['suite_elapsed_s']:.2f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()
