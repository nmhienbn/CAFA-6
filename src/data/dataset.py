import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from scipy.sparse import load_npz
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    processed_dir: str
    ontology: str
    feature_paths: List[str]
    folds_path: str = ""
    fold_id: int = 0
    split: str = "train"  # "train" | "val" | "trainval" | "test"


def load_features(feature_paths: List[str]) -> np.ndarray:
    mats = []
    for p in feature_paths:
        x = np.load(p)  # [N, d]
        mats.append(x)
    X = np.concatenate(mats, axis=1)
    return X


class ProteinTrainDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        processed = Path(cfg.processed_dir)
        mapping_dir = processed / "mapping"

        self.X = load_features(cfg.feature_paths)  # [N, D]
        self.Y = load_npz(mapping_dir / f"Y_{cfg.ontology}.npz")  # csr [N, M]

        if cfg.folds_path:
            fold_ids = np.load(cfg.folds_path)  # [N]
            if cfg.split == "train":
                mask = fold_ids != cfg.fold_id
            elif cfg.split == "val":
                mask = fold_ids == cfg.fold_id
            elif cfg.split == "trainval":
                mask = np.ones_like(fold_ids, dtype=bool)
            else:
                raise ValueError("Invalid split")
            self.idx = np.where(mask)[0]
        else:
            self.idx = np.arange(self.X.shape[0])

        self.X = self.X[self.idx]
        self.Y = self.Y[self.idx]

        self.X = self.X.astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y.getrow(i).toarray().astype(np.float32).squeeze(0)
        return torch.from_numpy(x), torch.from_numpy(y)


class ProteinTestDataset(Dataset):
    def __init__(self, feature_paths: List[str]):
        self.X = load_features(feature_paths).astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        return torch.from_numpy(x)
