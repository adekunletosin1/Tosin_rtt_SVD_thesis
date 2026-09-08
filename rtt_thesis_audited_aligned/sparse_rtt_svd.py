"""
sparse_rtt_svd.py
=================

Coordinate-sparse counterpart of Chapter 5, Algorithm 7, consistent with the
sparse access model in Proposition 5.5.

The input is stored in COO form. The ambient dense tensor and the full Gaussian
test tensors are never materialized. Gaussian entries are generated only for
leading coordinates that are actually touched by nonzero data. The working
state remains sparse in the unprocessed leading modes.

The function ``rtt_svd_sparse`` returns the Algorithm-7 rank-(r+p) TT.
``rtt_svd_sparse_rounded`` applies the same core-only TT-rounding used by the
dense implementation.

This module is an implementation of the access pattern described by the
theoretical sparse cost argument. It does not claim that Python wall-clock
constants realize the asymptotic bound on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg as la

from .tt_core import TensorTrain, _rng, _validate_shape, _validate_target_ranks, tt_round


@dataclass(frozen=True)
class SparseTensor:
    """Real order-d tensor in coalesced COO form."""

    shape: tuple[int, ...]
    idx: np.ndarray
    val: np.ndarray

    def __init__(self, shape, idx, val):
        shape = _validate_shape(shape)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, len(shape))
        val = np.asarray(val, dtype=float).reshape(-1)

        if idx.shape[0] != val.size:
            raise ValueError("idx and val must contain the same number of entries")

        if idx.size:
            for mode, n in enumerate(shape):
                if np.any(idx[:, mode] < 0) or np.any(idx[:, mode] >= n):
                    raise ValueError(f"COO index out of range in mode {mode}")

            linear = np.ravel_multi_index(idx.T, shape)
            unique, inverse = np.unique(linear, return_inverse=True)
            summed = np.zeros(unique.size, dtype=float)
            np.add.at(summed, inverse, val)

            keep = summed != 0.0
            unique = unique[keep]
            summed = summed[keep]
            idx = np.array(np.unravel_index(unique, shape)).T
            val = summed

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "val", val)

    @property
    def order(self) -> int:
        return len(self.shape)

    @property
    def nnz(self) -> int:
        return int(self.val.size)

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=float)
        if self.nnz:
            dense[tuple(self.idx.T)] = self.val
        return dense


def random_sparse_tensor(shape, nnz, random_state=None) -> SparseTensor:
    """Generate a random COO tensor with distinct uniformly sampled positions."""
    shape = _validate_shape(shape)
    total = int(np.prod(shape, dtype=np.int64))
    nnz = min(int(nnz), total)
    if nnz < 0:
        raise ValueError("nnz must be nonnegative")

    rng = _rng(random_state)
    linear = rng.choice(total, size=nnz, replace=False)
    idx = np.array(np.unravel_index(linear, shape)).T
    val = rng.standard_normal(nnz)
    return SparseTensor(shape, idx, val)


def _leading_linear_indices(A: SparseTensor) -> tuple[np.ndarray, tuple[int, ...]]:
    lead_shape = A.shape[:-1]
    if not lead_shape:
        raise ValueError("sparse RTT-SVD requires tensor order >= 2")
    if A.nnz == 0:
        return np.empty(0, dtype=np.int64), lead_shape
    lead_lin = np.ravel_multi_index(A.idx[:, :-1].T, lead_shape)
    return lead_lin, lead_shape


def rtt_svd_sparse(
    A: SparseTensor,
    ranks: Sequence[int],
    oversampling: int = 10,
    random_state=None,
) -> TensorTrain:
    """Coordinate-sparse right-to-left randomized TT-SVD.

    The returned representation corresponds to Algorithm 7 before target-rank
    rounding.
    """
    shape = A.shape
    d = A.order
    ranks = _validate_target_ranks(shape, ranks)

    p = int(oversampling)
    if p < 0:
        raise ValueError("oversampling p must be nonnegative")

    if A.nnz == 0:
        # The zero tensor can be represented exactly with unit ranks.
        zero_cores = []
        for k, n_k in enumerate(shape):
            r_left = 1
            r_right = 1
            zero_cores.append(np.zeros((r_left, n_k, r_right)))
        return TensorTrain(zero_cores)

    rng = _rng(random_state)
    cores: list[np.ndarray | None] = [None] * d

    # First right-to-left step, j=d, directly from COO coordinates.
    lead_lin, lead_shape = _leading_linear_indices(A)
    unique_lead, inverse_lead = np.unique(lead_lin, return_inverse=True)
    n_d = shape[-1]

    desired = ranks[-1] + p
    s_eff = min(desired, n_d, unique_lead.size)

    # Only Gaussian columns corresponding to touched leading indices are drawn.
    G_touched = rng.standard_normal((s_eff, unique_lead.size))

    # A_d = g_d contracted with A, shape s_{d-1} x n_d.
    A_d = np.zeros((s_eff, n_d), dtype=float)
    for row in range(s_eff):
        contrib = A.val * G_touched[row, inverse_lead]
        np.add.at(A_d[row], A.idx[:, -1], contrib)

    _, Q = la.rq(A_d, mode="economic", check_finite=False)
    carried = Q.shape[0]
    cores[-1] = Q.reshape(carried, n_d, 1)

    # b_d restricted to touched leading coordinates.
    B = np.zeros((unique_lead.size, carried), dtype=float)
    contributions = A.val[:, None] * Q[:, A.idx[:, -1]].T
    np.add.at(B, inverse_lead, contributions)

    remaining_shape = lead_shape
    remaining_lin = unique_lead

    # Continue j=d-1,...,2 while keeping the unprocessed leading block sparse.
    m = len(remaining_shape)
    while m >= 2:
        n_j = remaining_shape[-1]
        inner_shape = remaining_shape[:-1]

        mode_j = remaining_lin % n_j
        inner_lin = remaining_lin // n_j
        unique_inner, inverse_inner = np.unique(
            inner_lin,
            return_inverse=True,
        )

        # Restricted working unfolding M:
        # rows=(mode j, carried), columns=touched inner leading indices.
        M = np.zeros((n_j, carried, unique_inner.size), dtype=float)
        for c in range(carried):
            np.add.at(
                M[:, c, :],
                (mode_j, inverse_inner),
                B[:, c],
            )
        M2 = M.reshape(n_j * carried, unique_inner.size)

        global_mode_index = m - 1  # zero-based current mode within full tensor
        desired = ranks[global_mode_index - 1] + p
        s_eff = min(desired, M2.shape[0], M2.shape[1])

        G_j = rng.standard_normal((s_eff, M2.shape[1]))
        A_j = G_j @ M2.T
        _, Q_j = la.rq(A_j, mode="economic", check_finite=False)

        r_new = Q_j.shape[0]
        cores[global_mode_index] = Q_j.reshape(
            r_new, n_j, carried
        )

        B = (Q_j @ M2).T
        remaining_shape = inner_shape
        remaining_lin = unique_inner
        carried = r_new
        m -= 1

    # First core W_1 = b_2.
    n_1 = remaining_shape[0]
    first = np.zeros((n_1, carried), dtype=float)
    first[remaining_lin] = B
    cores[0] = first.reshape(1, n_1, carried)

    return TensorTrain(cores)  # type: ignore[arg-type]


def rtt_svd_sparse_rounded(
    A: SparseTensor,
    ranks: Sequence[int],
    oversampling: int = 10,
    random_state=None,
) -> TensorTrain:
    """Sparse Algorithm 7 followed by core-only TT-rounding."""
    tt = rtt_svd_sparse(
        A,
        ranks=ranks,
        oversampling=oversampling,
        random_state=random_state,
    )
    return tt_round(tt, ranks)
