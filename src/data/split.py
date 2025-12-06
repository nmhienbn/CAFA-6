import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from sklearn.model_selection import StratifiedKFold


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, required=True)
    ap.add_argument("--ontology", type=str, default="BP")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    processed = Path(args.processed)
    mapping_dir = processed / "mapping"

    # load label
    Y = load_npz(mapping_dir / f"Y_{args.ontology}.npz")  # [N, M]

    # đơn giản: stratify theo số lượng term mỗi protein
    y_strat = np.asarray(Y.sum(axis=1)).reshape(-1)  # [N]
    y_strat = np.clip(y_strat, 0, 10)  # tránh quá nhiều class

    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.random_state
    )

    fold_ids = np.zeros(Y.shape[0], dtype=np.int64)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros_like(y_strat), y_strat)):
        fold_ids[val_idx] = fold

    splits_dir = processed / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    np.save(splits_dir / f"train_folds_{args.ontology}.npy", fold_ids)

    print("Saved splits to", splits_dir)


if __name__ == "__main__":
    main()
