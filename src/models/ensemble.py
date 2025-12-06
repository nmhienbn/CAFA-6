import numpy as np
import json
from pathlib import Path


def load_model_preds(base_names, folds, base_output_dir="outputs"):
    """
    Returns:
      train_preds: list of [N, M]
      test_preds: list of [N_test, M]
    """
    train_list = []
    test_list = []
    for name in base_names:
        base_dir = Path(base_output_dir) / name
        train_scores = np.load(base_dir / "train_scores_merged.npy")
        test_scores = np.load(base_dir / "test_scores_mean.npy")
        train_list.append(train_scores)
        test_list.append(test_scores)
    return train_list, test_list


def weighted_average_ensemble(train_list, test_list, weights=None):
    K = len(train_list)
    if weights is None:
        weights = np.ones(K) / K
    weights = np.asarray(weights).reshape(K, 1, 1)

    train_stack = np.stack(train_list, axis=0)  # [K, N, M]
    test_stack = np.stack(test_list, axis=0)    # [K, N_test, M]

    train_ens = (weights * train_stack).sum(axis=0)
    test_ens = (weights * test_stack).sum(axis=0)
    return train_ens, test_ens
