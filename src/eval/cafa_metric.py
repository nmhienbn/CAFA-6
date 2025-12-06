import numpy as np
from scipy.sparse import csr_matrix


def fmax_score(y_true: csr_matrix, y_score: np.ndarray, thresholds=None):
    """
    Args:
        y_true: csr [N, M] {0,1}
        y_score: dense [N, M] float
        thresholds: list of threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    y_true = y_true.tocoo()
    N, M = y_true.shape
    best_f = 0.0
    best_t = 0.5

    # precompute positives per protein
    true_counts_per_protein = np.asarray(y_true.sum(axis=1)).reshape(-1)

    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int8)  # [N, M]
        # per-protein
        tp = (y_pred * y_true.toarray()).sum(axis=1)
        pred_counts = y_pred.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(pred_counts == 0, 1.0, tp / np.maximum(pred_counts, 1))
            recall = np.where(
                true_counts_per_protein == 0,
                1.0,
                tp / np.maximum(true_counts_per_protein, 1),
            )

        p = precision.mean()
        r = recall.mean()
        if p + r == 0:
            f = 0.0
        else:
            f = 2 * p * r / (p + r)
        if f > best_f:
            best_f = f
            best_t = t

    return best_f, best_t
