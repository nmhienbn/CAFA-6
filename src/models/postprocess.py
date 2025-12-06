import numpy as np
from typing import Dict, List


def enforce_parent_ge_child(preds: np.ndarray, parents: Dict[int, List[int]], iters: int = 2):
    """
    preds: [N, M]
    parents: term_idx -> list[parent_idx]
    Đảm bảo parent_prob >= max(child_prob)
    """
    N, M = preds.shape
    out = preds.copy()
    for _ in range(iters):
        for child, ps in parents.items():
            if not ps:
                continue
            child_col = out[:, child]
            max_child = child_col
            for p in ps:
                out[:, p] = np.maximum(out[:, p], max_child)
    return out


def condprob_mod(preds: np.ndarray, parents: Dict[int, List[int]], threshold_parent: float = 0.0):
    """
    Ý tưởng CondProbMod đơn giản:
    - Nếu tất cả parent của term < threshold_parent -> đè prob term đó về nhỏ hơn
    Cái này là approximation, bạn có thể chỉnh lại theo paper ProtBoost.
    """
    out = preds.copy()
    N, M = preds.shape
    for t, ps in parents.items():
        if not ps:
            continue
        parent_probs = out[:, ps]
        max_parent = parent_probs.max(axis=1)
        # nếu parent quá thấp thì suppress child
        mask_low = max_parent < threshold_parent
        out[mask_low, t] *= max_parent[mask_low]
    return out
