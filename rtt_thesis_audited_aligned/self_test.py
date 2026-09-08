"""Numerical correctness checks for the Chapters 1--5 aligned implementation."""

from __future__ import annotations

import numpy as np

from .tt_core import (
    TensorTrain,
    randomized_range_finder,
    randomized_range_finder_power,
    randomized_svd,
    relative_error,
    right_orthogonality_residual,
    rtt_svd,
    rtt_svd_rounded,
    tt_round,
    tt_svd,
)
from .sparse_rtt_svd import SparseTensor, rtt_svd_sparse, rtt_svd_sparse_rounded


def make_exact_tt(shape, ranks, seed=0):
    rng = np.random.default_rng(seed)
    full_ranks = (1, *ranks, 1)
    cores = []
    for k, n_k in enumerate(shape):
        g = rng.standard_normal((full_ranks[k], n_k, full_ranks[k + 1]))
        cores.append(g)
    return TensorTrain(cores)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run():
    # TensorTrain reconstruction and norm.
    exact = make_exact_tt((5, 4, 3, 4), (3, 3, 2), seed=1)
    X = exact.to_full()
    check(
        abs(exact.norm() - np.linalg.norm(X)) <= 1e-11 * max(1.0, np.linalg.norm(X)),
        "TensorTrain.norm does not match dense Frobenius norm",
    )

    # Algorithm 2: exact recovery when prescribed ranks contain the true ranks.
    det = tt_svd(X, (3, 3, 2))
    check(relative_error(X, det) < 1e-12, "TT-SVD exact recovery failed")

    # Algorithm 3: same-rank rounding preserves the represented tensor.
    rounded_same = tt_round(exact, (3, 3, 2))
    check(
        np.linalg.norm(rounded_same.to_full() - X) / np.linalg.norm(X) < 1e-12,
        "same-rank TT-rounding changed the tensor",
    )

    # Algorithm 3: lower target ranks are obeyed without dense reconstruction.
    rounded_low = tt_round(exact, (2, 2, 2))
    check(
        all(a <= b for a, b in zip(rounded_low.ranks, (2, 2, 2))),
        "TT-rounding violated target ranks",
    )

    # Algorithms 4 and 5: orthonormal range bases.
    A = np.random.default_rng(2).standard_normal((40, 25))
    Q = randomized_range_finder(A, rank=6, oversampling=4, random_state=3)
    check(
        np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])) < 1e-12,
        "randomized range finder returned a non-orthonormal basis",
    )
    Qp = randomized_range_finder_power(
        A, rank=6, oversampling=4, power=2, random_state=3
    )
    check(
        np.linalg.norm(Qp.T @ Qp - np.eye(Qp.shape[1])) < 1e-12,
        "power range finder returned a non-orthonormal basis",
    )

    # Algorithm 6: dimensions and finite approximation.
    U, s, Vt = randomized_svd(A, rank=6, oversampling=4, random_state=4)
    check(U.shape == (40, 6), "randomized SVD U shape incorrect")
    check(s.shape == (6,), "randomized SVD singular-value shape incorrect")
    check(Vt.shape == (6, 25), "randomized SVD Vt shape incorrect")
    check(np.all(np.isfinite(s)), "randomized SVD returned nonfinite values")

    # Algorithm 7: exact recovery of a tensor whose TT-rank is at most r.
    true_ranks = (2, 2, 2)
    exact2 = make_exact_tt((5, 5, 5, 5), true_ranks, seed=5)
    X2 = exact2.to_full()
    rand_full = rtt_svd(
        X2,
        true_ranks,
        oversampling=2,
        random_state=6,
    )
    check(
        relative_error(X2, rand_full) < 1e-11,
        "Algorithm 7 exact-recovery test failed",
    )
    check(
        right_orthogonality_residual(rand_full) < 1e-11,
        "Algorithm 7 right-orthogonality test failed",
    )

    # Core-only rounding of Algorithm-7 output to the target ranks.
    rand_round = rtt_svd_rounded(
        X2,
        true_ranks,
        oversampling=2,
        random_state=6,
    )
    check(
        relative_error(X2, rand_round) < 1e-11,
        "rounded Algorithm-7 exact-recovery test failed",
    )
    check(
        all(a <= b for a, b in zip(rand_round.ranks, true_ranks)),
        "rounded Algorithm-7 output violates target ranks",
    )

    # Sparse Algorithm 7: use maximal feasible ranks on a small sparse tensor.
    shape = (3, 3, 3)
    idx = np.array(
        [
            [0, 0, 0],
            [0, 2, 1],
            [1, 1, 2],
            [2, 0, 2],
            [2, 2, 0],
        ],
        dtype=int,
    )
    val = np.array([1.0, -2.0, 0.5, 1.25, -0.75])
    S = SparseTensor(shape, idx, val)
    dense = S.to_dense()

    sparse_tt = rtt_svd_sparse(
        S,
        ranks=(3, 3),
        oversampling=0,
        random_state=7,
    )
    sparse_err = np.linalg.norm(dense - sparse_tt.to_full()) / np.linalg.norm(dense)
    check(sparse_err < 1e-11, "sparse Algorithm-7 full-rank recovery failed")

    sparse_round = rtt_svd_sparse_rounded(
        S,
        ranks=(3, 3),
        oversampling=0,
        random_state=7,
    )
    sparse_round_err = np.linalg.norm(dense - sparse_round.to_full()) / np.linalg.norm(dense)
    check(sparse_round_err < 1e-11, "sparse rounded recovery failed")

    print("All Chapters 1-5 alignment self-tests passed.")


if __name__ == "__main__":
    run()
