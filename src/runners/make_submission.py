import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--submission_path", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(cfg["include_data"], "r") as f:
        data_cfg = yaml.safe_load(f)["data"]

    processed_dir = Path(data_cfg["processed"])
    ontology = data_cfg["ontology"]
    mapping_dir = processed_dir / "mapping"

    idx2protein = np.load(mapping_dir / "idx2protein.npy", allow_pickle=True)
    with open(mapping_dir / f"go2idx_{ontology}.json", "r") as f:
        go2idx = json.load(f)

    # đọc test_scores + threshold
    ckpt_dir = Path(args.output_dir) / cfg["name"] / f"fold{args.fold}"
    test_scores = np.load(ckpt_dir / "test_scores.npy")
    with open(ckpt_dir / "train_threshold.json", "r") as f:
        thr_info = json.load(f)
    t_opt = thr_info["threshold"]

    # cần map test index -> protein_id; giả sử test_features align với list protein_id trong 1 file
    # easiest: lưu file test_proteins.npy cùng lúc build mapping
    test_proteins = np.load(mapping_dir / "test_proteins.npy", allow_pickle=True)

    idx2go = {v: k for k, v in go2idx.items()}

    # threshold
    pred_bin = (test_scores >= t_opt)

    rows = []
    for i, pid in enumerate(test_proteins):
        cols = np.where(pred_bin[i])[0]
        if len(cols) == 0:
            continue
        for c in cols:
            go_id = idx2go[c]
            score = float(test_scores[i, c])
            rows.append((pid, go_id, score))

    df = pd.DataFrame(rows, columns=["protein_id", "go_id", "score"])
    df.to_csv(args.submission_path, index=False)
    print("Saved submission to", args.submission_path)


if __name__ == "__main__":
    main()
