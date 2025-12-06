import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from scipy.sparse import load_npz

from src.data.dataset import ProteinTestDataset
from src.models.seq_mlp_bce import SeqMLPBCE
from src.models.seq_mlp_softf1 import SeqMLPSoftF1
from src.eval.threshold_search import search_threshold


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--output_dir", type=str, default="outputs")
    return ap.parse_args()


def build_model(cfg_model, input_dim, output_dim):
    from train import build_model as _build

    return _build(cfg_model, input_dim, output_dim)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    with open(cfg["include_data"], "r") as f:
        data_cfg = yaml.safe_load(f)["data"]

    processed_dir = Path(data_cfg["processed"])
    ontology = data_cfg["ontology"]
    mapping_dir = processed_dir / "mapping"

    Y = load_npz(mapping_dir / f"Y_{ontology}.npz")
    output_dim = Y.shape[1]

    tmp_feat = np.load(cfg["features"]["train"][0])
    input_dim = tmp_feat.shape[1] * len(cfg["features"]["train"])

    model = build_model(cfg["model"], input_dim, output_dim)

    # load checkpoint
    ckpt_dir = Path(args.output_dir) / cfg["name"] / f"fold{args.fold}"
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # load full train features (để có full preds cho stacking nếu cần)
    X_train = np.concatenate(
        [np.load(p) for p in cfg["features"]["train"]], axis=1
    ).astype(np.float32)
    train_dataset = ProteinTestDataset(cfg["features"]["train"])
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)

    train_scores = []
    for xb in train_loader:
        xb = xb.to(device)
        with torch.no_grad():
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
        train_scores.append(probs)
    train_scores = np.concatenate(train_scores, axis=0)  # [N_train, M]

    np.save(ckpt_dir / "train_scores.npy", train_scores)

    # test preds
    test_dataset = ProteinTestDataset(cfg["features"]["test"])
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    test_scores = []
    for xb in test_loader:
        xb = xb.to(device)
        with torch.no_grad():
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
        test_scores.append(probs)
    test_scores = np.concatenate(test_scores, axis=0)
    np.save(ckpt_dir / "test_scores.npy", test_scores)

    # threshold từ full train
    res = search_threshold(Y, train_scores)
    with open(ckpt_dir / "train_threshold.json", "w") as f:
        json.dump(res, f)

    print("Saved train/test scores and threshold to", ckpt_dir)


if __name__ == "__main__":
    main()
