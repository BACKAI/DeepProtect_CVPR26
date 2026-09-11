"""Order-aware LDA used for text-guided attribute directions."""

import numpy as np


class OrderAwareLDA:
    """Solve the paper's regularized generalized Rayleigh quotient.

    The published notation writes ``(z_i-z_j)`` in Eq. (8); as a scatter
    regularizer this is implemented as its corresponding outer product.  The
    direction is solved with least squares (LSQR when SciPy is available).
    """

    def __init__(self, lambda_reg: float = 1.0, ridge: float = 1e-5):
        self.lambda_reg = float(lambda_reg)
        self.ridge = float(ridge)
        self.direction = None
        self.means_ = None
        self.selected_indices_ = None
        self.selected_ranks_ = None

    def fit(self, features: np.ndarray, scores: np.ndarray, m: int = 30):
        features = np.asarray(features, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or features.shape[0] != scores.shape[0]:
            raise ValueError("features must be [N, D] and scores must be [N].")
        if features.shape[0] < 2 * m:
            raise ValueError(f"Need at least 2*m={2*m} bank entries, got {features.shape[0]}.")

        ranking = np.argsort(scores)[::-1]
        top = ranking[:m]
        bottom = ranking[-m:]
        selected = np.concatenate([top, bottom])
        selected_features = features[selected]
        selected_scores = scores[selected]
        full_ranks = np.empty_like(ranking)
        full_ranks[ranking] = np.arange(ranking.shape[0])
        ranks = full_ranks[selected].astype(np.float64)

        groups = (np.arange(2 * m) >= m).astype(np.int64)
        means = np.stack([selected_features[groups == label].mean(axis=0) for label in (0, 1)])
        self.means_ = means
        self.selected_indices_ = selected
        self.selected_ranks_ = ranks

        centered = selected_features - means[groups]
        sw = centered.T @ centered / max(1, selected_features.shape[0] - 2)
        difference = means[0] - means[1]
        sb = np.outer(difference, difference)

        order_regularizer = np.zeros_like(sw)
        for i in range(selected_features.shape[0]):
            for j in range(i + 1, selected_features.shape[0]):
                dz = selected_features[i] - selected_features[j]
                order_regularizer += abs(ranks[i] - ranks[j]) * np.outer(dz, dz)
        scale = max(1.0, np.trace(sw) / max(1, sw.shape[0]))
        system = sw + self.lambda_reg * order_regularizer / max(1, selected_features.shape[0])
        system = system + self.ridge * scale * np.eye(system.shape[0])

        # The rank-one between-class scatter makes the solution equivalent to
        # the regularized LDA direction A^{-1}(mu_top-mu_bottom).
        try:
            from scipy.sparse.linalg import lsqr

            direction = lsqr(system, difference, atol=1e-7, btol=1e-7, iter_lim=2000)[0]
        except ImportError:
            direction = np.linalg.lstsq(system, difference, rcond=1e-6)[0]
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        self.direction = direction.astype(np.float32)
        return self

