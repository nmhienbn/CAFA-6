import numpy as np
from scipy.sparse import csr_matrix

from .cafa_metric import fmax_score


def search_threshold(y_true: csr_matrix, y_score: np.ndarray):
    fmax, t_opt = fmax_score(y_true, y_score)
    return {"Fmax": fmax, "threshold": t_opt}
