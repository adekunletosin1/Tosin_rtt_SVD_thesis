"""
tt_core.py
==========

Authoritative dense Tensor Train implementation aligned with Chapters 1--5
of the current thesis draft.

Implemented thesis algorithms
-----------------------------
- Algorithm 2: deterministic TT-SVD.
- Algorithm 3: prescribed-rank TT-rounding.
- Algorithm 4: Gaussian randomized range finder.
- Algorithm 5: Gaussian randomized range finder with power iteration.
- Algorithm 6: randomized matrix SVD.
- Algorithm 7: Huber--Schneider--Wolf randomized TT-SVD.

The randomized TT-SVD in this file is the right-to-left construction from
Chapter 5. It does not replace each deterministic TT-SVD step by a randomized
matrix SVD. At every bond it draws a fresh independent Gaussian test tensor
(in matricized form), forms the contracted sketch, computes the RQ
factorization, and extracts a right-orthogonal TT core.

The function ``rtt_svd`` returns the Algorithm-7 representation with effective
ranks s_k = r_k + p, clipped only by unavoidable unfolding-dimension bounds.
The convenience function ``rtt_svd_rounded`` subsequently applies core-only
TT-rounding to the prescribed target ranks. No dense reconstruction is used in
that final rank reduction.

All arrays are real-valued, consistently with the scope of the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.linalg as la


Array = np.ndarray


def _rng(random_state=None) -> np.random.Generator:
    """Return a NumPy Generator without altering an existing Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_shape(shape: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(int(n) for n in shape)
    if len(shape) < 2:
        raise ValueError("a tensor train requires order d >= 2")
    if any(n < 1 for n in shape):
        raise ValueError("all mode sizes must be positive")
    return shape


def _validate_target_ranks(
    shape: Sequence[int],
    ranks: Sequence[int],
) -> tuple[int, ...]:
    shape = _validate_shape(shape)
    ranks = tuple(int(r) for r in ranks)
    if len(ranks) != len(shape) - 1:
        raise ValueError(
            f"expected d-1 = {len(shape)-1} interior ranks, got {len(ranks)}"
        )
    if any(r < 1 for r in ranks):
        raise ValueError("all target TT-ranks must be positive")
    return ranks


def _feasible_bond_rank(shape: Sequence[int], bond: int) -> int:
    """Maximum rank of the principal unfolding after mode ``bond``."""
    left = int(np.prod(shape[: bond + 1], dtype=np.int64))
    right = int(np.prod(shape[bond + 1 :], dtype=np.int64))
    return min(left, right)


@dataclass(frozen=True)
class TensorTrain:
    """Tensor Train stored as three-dimensional TT cores.

    Core ``k`` has shape ``(r_{k-1}, n_k, r_k)`` with boundary ranks
    ``r_0 = r_d = 1``.
    """

    cores: tuple[Array, ...]

    def __init__(self, cores: Iterable[Array]):
        copied = tuple(np.asarray(g, dtype=float).copy() for g in cores)
        self._check(copied)
        object.__setattr__(self, "cores", copied)

    @staticmethod
    def _check(cores: tuple[Array, ...]) -> None:
        if len(cores) < 2:
            raise ValueError("a TT needs at least two cores")
        if any(g.ndim != 3 for g in cores):
            raise ValueError("every TT core must be a three-dimensional array")
        if cores[0].shape[0] != 1 or cores[-1].shape[2] != 1:
            raise ValueError("boundary TT-ranks r_0 and r_d must equal 1")
        for k in range(len(cores) - 1):
            if cores[k].shape[2] != cores[k + 1].shape[0]:
                raise ValueError(
                    f"rank mismatch between cores {k} and {k+1}: "
                    f"{cores[k].shape[2]} != {cores[k+1].shape[0]}"
                )

    @property
    def order(self) -> int:
        return len(self.cores)

    @property
    def mode_sizes(self) -> tuple[int, ...]:
        return tuple(int(g.shape[1]) for g in self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(int(g.shape[2]) for g in self.cores[:-1])

    def storage(self) -> int:
        """Number of scalar parameters stored in the TT cores."""
        return int(sum(g.size for g in self.cores))

    def norm(self) -> float:
        """Frobenius norm by sequential TT-network contraction."""
        gram = np.ones((1, 1), dtype=float)
        for g in self.cores:
            gram = np.einsum("ab,aic,bid->cd", gram, g, g, optimize=True)
        value = float(gram[0, 0])
        return float(np.sqrt(max(value, 0.0)))

    def to_full(self) -> Array:
        """Contract the TT cores to the full dense tensor."""
        g0 = self.cores[0]
        full = g0.reshape(g0.shape[1], g0.shape[2])
        for g in self.cores[1:]:
            r_left, n_k, r_right = g.shape
            full = full.reshape(-1, r_left) @ g.reshape(
                r_left, n_k * r_right
            )
            full = full.reshape(-1, r_right)
        return full.reshape(self.mode_sizes)

    def copy(self) -> "TensorTrain":
        return TensorTrain(self.cores)


def relative_error(X: Array, tt: TensorTrain) -> float:
    """Relative Frobenius error ||X - TT||_F / ||X||_F."""
    X = np.asarray(X, dtype=float)
    denom = np.linalg.norm(X)
    if denom == 0:
        return 0.0 if np.linalg.norm(tt.to_full()) == 0 else np.inf
    return float(np.linalg.norm(X - tt.to_full()) / denom)


# ---------------------------------------------------------------------------
# Algorithm 2: deterministic TT-SVD
# ---------------------------------------------------------------------------

def tt_svd(X: Array, ranks: Sequence[int]) -> TensorTrain:
    """Deterministic left-to-right TT-SVD with prescribed TT-ranks."""
    X = np.asarray(X, dtype=float)
    shape = _validate_shape(X.shape)
    ranks = _validate_target_ranks(shape, ranks)

    cores: list[Array] = []
    working = X.copy()
    r_prev = 1

    for k in range(len(shape) - 1):
        n_k = shape[k]
        matrix = working.reshape(r_prev * n_k, -1)
        U, singular_values, Vt = la.svd(
            matrix, full_matrices=False, check_finite=False
        )

        r_k = min(ranks[k], singular_values.size)
        U = U[:, :r_k]
        singular_values = singular_values[:r_k]
        Vt = Vt[:r_k, :]

        cores.append(U.reshape(r_prev, n_k, r_k))
        working = singular_values[:, None] * Vt
        r_prev = r_k

    cores.append(working.reshape(r_prev, shape[-1], 1))
    return TensorTrain(cores)


# ---------------------------------------------------------------------------
# Algorithm 3: TT-rounding, prescribed-rank branch
# ---------------------------------------------------------------------------

def tt_round(tt: TensorTrain, target_ranks: Sequence[int]) -> TensorTrain:
    """Round a TT directly on its cores to prescribed interior ranks.

    This implements the prescribed-rank branch of Algorithm 3 in the thesis:

    1. left-to-right QR orthogonalization;
    2. right-to-left SVD truncation.

    The dense tensor is never formed.
    """
    target_ranks = _validate_target_ranks(tt.mode_sizes, target_ranks)
    cores = [g.copy() for g in tt.cores]
    d = len(cores)

    # Phase 1: left-to-right orthogonalization.
    for k in range(d - 1):
        r_left, n_k, r_right = cores[k].shape
        matrix = cores[k].reshape(r_left * n_k, r_right)

        Q, R = la.qr(
            matrix,
            mode="economic",
            check_finite=False,
        )
        q_rank = Q.shape[1]

        cores[k] = Q.reshape(r_left, n_k, q_rank)
        cores[k + 1] = np.tensordot(
            R,
            cores[k + 1],
            axes=([1], [0]),
        )

    # Phase 2: right-to-left SVD truncation.
    for k in range(d - 1, 0, -1):
        r_left, n_k, r_right = cores[k].shape
        matrix = cores[k].reshape(r_left, n_k * r_right)

        U, singular_values, Vt = la.svd(
            matrix,
            full_matrices=False,
            check_finite=False,
        )

        r_new = min(target_ranks[k - 1], singular_values.size)
        U_r = U[:, :r_new]
        s_r = singular_values[:r_new]
        Vt_r = Vt[:r_new, :]

        cores[k] = (s_r[:, None] * Vt_r).reshape(
            r_new, n_k, r_right
        )

        cores[k - 1] = np.tensordot(
            cores[k - 1],
            U_r,
            axes=([2], [0]),
        )

    return TensorTrain(cores)


# ---------------------------------------------------------------------------
# Core-only rounding for a right-orthogonal TT representation
# ---------------------------------------------------------------------------

def tt_round_right_orthogonal(
    tt: TensorTrain,
    target_ranks: Sequence[int],
) -> TensorTrain:
    """Round a TT by right orthogonalization and left-to-right truncation.

    This form is especially natural for the output of Algorithm 7, whose cores
    W_2,...,W_d are already right-orthogonal. It is a core-only post-processing
    routine for equal-rank numerical comparisons. It is kept separate from
    ``tt_round``, which follows Algorithm 3 exactly as written in Chapter 3 of
    the current draft.
    """
    target_ranks = _validate_target_ranks(tt.mode_sizes, target_ranks)
    cores = [g.copy() for g in tt.cores]
    d = len(cores)

    # Right-to-left orthogonalization.
    for k in range(d - 1, 0, -1):
        r_left, n_k, r_right = cores[k].shape
        matrix = cores[k].reshape(r_left, n_k * r_right)
        Q, R = la.qr(
            matrix.T,
            mode="economic",
            check_finite=False,
        )
        new_left = Q.shape[1]
        cores[k] = Q.T.reshape(new_left, n_k, r_right)
        cores[k - 1] = np.tensordot(
            cores[k - 1],
            R.T,
            axes=([2], [0]),
        )

    # Left-to-right SVD truncation.
    for k in range(d - 1):
        r_left, n_k, r_right = cores[k].shape
        matrix = cores[k].reshape(r_left * n_k, r_right)
        U, singular_values, Vt = la.svd(
            matrix,
            full_matrices=False,
            check_finite=False,
        )
        r_new = min(target_ranks[k], singular_values.size)
        U = U[:, :r_new]
        singular_values = singular_values[:r_new]
        Vt = Vt[:r_new, :]

        cores[k] = U.reshape(r_left, n_k, r_new)
        transfer = singular_values[:, None] * Vt
        cores[k + 1] = np.tensordot(
            transfer,
            cores[k + 1],
            axes=([1], [0]),
        )

    return TensorTrain(cores)


# ---------------------------------------------------------------------------
# Algorithm 4: randomized range finder
# ---------------------------------------------------------------------------

def randomized_range_finder(
    A: Array,
    rank: int,
    oversampling: int = 10,
    random_state=None,
) -> Array:
    """Gaussian randomized range finder from Chapter 4, Algorithm 4."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a matrix")

    m, n = A.shape
    rank = int(rank)
    p = int(oversampling)
    if rank < 1 or p < 0:
        raise ValueError("rank must be positive and oversampling nonnegative")

    s = rank + p
    if s > min(m, n):
        raise ValueError(
            f"Algorithm 4 requires r+p <= min(m,n); got {s} > {min(m,n)}"
        )

    rng = _rng(random_state)
    G = rng.standard_normal((n, s))
    Y = A @ G
    Q, _ = la.qr(Y, mode="economic", check_finite=False)
    return Q


# ---------------------------------------------------------------------------
# Algorithm 5: randomized range finder with power iteration
# ---------------------------------------------------------------------------

def randomized_range_finder_power(
    A: Array,
    rank: int,
    oversampling: int = 10,
    power: int = 1,
    random_state=None,
) -> Array:
    """Power-iteration range finder from Chapter 4, Algorithm 5."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a matrix")

    m, n = A.shape
    rank = int(rank)
    p = int(oversampling)
    q = int(power)

    if rank < 1 or p < 0 or q < 1:
        raise ValueError("require rank >= 1, p >= 0, and power q >= 1")

    s = rank + p
    if s > min(m, n):
        raise ValueError(
            f"Algorithm 5 requires r+p <= min(m,n); got {s} > {min(m,n)}"
        )

    rng = _rng(random_state)
    G = rng.standard_normal((n, s))
    Y = A @ G

    for _ in range(q):
        Q1, _ = la.qr(Y, mode="economic", check_finite=False)
        Y_right = A.T @ Q1
        Q2, _ = la.qr(
            Y_right,
            mode="economic",
            check_finite=False,
        )
        Y = A @ Q2

    Q, _ = la.qr(Y, mode="economic", check_finite=False)
    return Q


# ---------------------------------------------------------------------------
# Algorithm 6: randomized matrix SVD
# ---------------------------------------------------------------------------

def randomized_svd(
    A: Array,
    rank: int,
    oversampling: int = 10,
    random_state=None,
) -> tuple[Array, Array, Array]:
    """Randomized matrix SVD from Chapter 4, Algorithm 6."""
    A = np.asarray(A, dtype=float)
    Q = randomized_range_finder(
        A,
        rank=rank,
        oversampling=oversampling,
        random_state=random_state,
    )
    B = Q.T @ A
    U_small, singular_values, Vt = la.svd(
        B,
        full_matrices=False,
        check_finite=False,
    )

    r = min(int(rank), singular_values.size)
    U = Q @ U_small[:, :r]
    return U, singular_values[:r], Vt[:r, :]


# ---------------------------------------------------------------------------
# Algorithm 7: randomized TT-SVD of Huber, Schneider and Wolf
# ---------------------------------------------------------------------------

def rtt_svd(
    X: Array,
    ranks: Sequence[int],
    oversampling: int = 10,
    random_state=None,
) -> TensorTrain:
    """Right-to-left randomized TT-SVD from Chapter 5, Algorithm 7.

    ``ranks`` are the comparison target ranks r. Algorithm 7 constructs
    representation ranks s_k = r_k + p. At a bond where this exceeds the
    maximal possible unfolding rank, the effective rank is clipped by the
    corresponding matrix dimensions.

    A fresh independent standard Gaussian test tensor is drawn at every bond.
    Its matricization is generated directly as a Gaussian matrix.

    The thesis states Algorithm 7 using an RQ factorization. This function uses
    SciPy's economic RQ directly, so the stored core is the row-orthogonal
    factor specified by that algorithm.

    No SVD truncation occurs inside this randomized sweep.
    """
    X = np.asarray(X, dtype=float)
    shape = _validate_shape(X.shape)
    ranks = _validate_target_ranks(shape, ranks)

    p = int(oversampling)
    if p < 0:
        raise ValueError("oversampling p must be nonnegative")

    rng = _rng(random_state)
    d = len(shape)

    working = X.reshape(-1, 1)
    carried = 1
    cores: list[Array | None] = [None] * d

    for j in range(d, 1, -1):
        k = j - 1
        n_j = shape[k]

        lead = working.shape[0] // n_j
        block = working.reshape(lead, n_j, carried)

        # M has rows indexed by (mode j, carried rank) and columns by modes
        # 1,...,j-1. This is the transpose of the relevant working unfolding.
        M = block.transpose(1, 2, 0).reshape(n_j * carried, lead)

        desired = ranks[k - 1] + p
        s_eff = min(desired, M.shape[0], M.shape[1])
        if s_eff < 1:
            raise RuntimeError("effective sketch rank became zero")

        # Matricization of the independent Gaussian test tensor g_j.
        G_j = rng.standard_normal((s_eff, lead))

        # A_j in Algorithm 7. It has shape s_{j-1} x (n_j s_j).
        A_j = G_j @ M.T

        # Economic RQ: A_j = R_j Q_j, with Q_j Q_j^T = I.
        _, Q_j = la.rq(
            A_j,
            mode="economic",
            check_finite=False,
        )

        r_new = Q_j.shape[0]
        cores[k] = Q_j.reshape(r_new, n_j, carried)

        # Contract b_{j+1} with W_j to obtain the next working tensor b_j.
        working = (Q_j @ M).T
        carried = r_new

    cores[0] = working.reshape(1, shape[0], carried)

    return TensorTrain(cores)  # type: ignore[arg-type]


def rtt_svd_rounded(
    X: Array,
    ranks: Sequence[int],
    oversampling: int = 10,
    random_state=None,
) -> TensorTrain:
    """Algorithm 7 followed by right-orthogonal core-only rank reduction.

    This post-processing is used for equal-final-rank numerical comparisons.
    The separate ``tt_round`` function remains Algorithm 3 exactly as written
    in Chapter 3 of the current draft.
    """
    full_rank_tt = rtt_svd(
        X,
        ranks=ranks,
        oversampling=oversampling,
        random_state=random_state,
    )
    return tt_round_right_orthogonal(full_rank_tt, ranks)


def right_orthogonality_residual(tt: TensorTrain) -> float:
    """Maximum ||W_k W_k^T - I||_F for cores k=2,...,d.

    Here W_k denotes the mode-1 matricization of the kth TT core. Algorithm 7
    constructs these cores with orthonormal rows.
    """
    residual = 0.0
    for g in tt.cores[1:]:
        mat = g.reshape(g.shape[0], -1)
        eye = np.eye(mat.shape[0])
        residual = max(
            residual,
            float(np.linalg.norm(mat @ mat.T - eye)),
        )
    return residual
