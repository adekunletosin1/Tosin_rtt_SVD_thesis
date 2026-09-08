"""RTT-SVD thesis implementation aligned with Chapters 1--5."""

from .tt_core import (
    TensorTrain,
    relative_error,
    tt_svd,
    tt_round,
    tt_round_right_orthogonal,
    randomized_range_finder,
    randomized_range_finder_power,
    randomized_svd,
    rtt_svd,
    rtt_svd_rounded,
    right_orthogonality_residual,
)
from .sparse_rtt_svd import (
    SparseTensor,
    random_sparse_tensor,
    rtt_svd_sparse,
    rtt_svd_sparse_rounded,
)

__all__ = [
    "TensorTrain",
    "relative_error",
    "tt_svd",
    "tt_round",
    "tt_round_right_orthogonal",
    "randomized_range_finder",
    "randomized_range_finder_power",
    "randomized_svd",
    "rtt_svd",
    "rtt_svd_rounded",
    "right_orthogonality_residual",
    "SparseTensor",
    "random_sparse_tensor",
    "rtt_svd_sparse",
    "rtt_svd_sparse_rounded",
]
