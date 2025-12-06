import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.sparse import load_npz

from src.data.dataset import DataConfig, ProteinTrainDataset
from src.eval.threshold_search import search_threshold
from src.models.seq_mlp_bce import SeqMLPBCE
from src.models.seq_mlp_softf1 import SeqMLPSoftF1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default="outputs")
    return ap.parse_args()


def build_model(cfg_model, input_dim, output_dim):
    mtype = cfg_model["type"]
    hidden_dims = cfg_model.get("hidden_dims", [2048, 2048])
    dropout = cfg_model.get("dropout", 0.3)

    if mtype == "seq_mlp_bce":
        return SeqMLPBCE(input_dim, output_dim, hidden_dims, dropout)
    elif mtype == "seq_mlp_softf1":
        bce_weight = cfg_model.get("bce_weight", 0.5)
        return SeqMLPSoftF1(input_dim, output_dim, hidden_dims, dropout, bce_weight)
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # load data config
    with open(cfg["include_data"], "r") as f:
        data_cfg = yaml.safe_load(f)["data"]

    processed_dir = Path(data_cfg["processed"])
    ontology = data_cfg["ontology"]
    mapping_dir = processed_dir / "mapping"

    # load label to get output_dim
    Y = load_npz(mapping_dir / f"Y_{ontology}.npz")
    output_dim = Y.shape[1]

    # load one feature file to get input_dim
    import numpy as np

    tmp_feat = np.load(cfg["features"]["train"][0])
    input_dim = tmp_feat.shape[1] * len(cfg["features"]["train"])  # vì concat

    folds_path = processed_dir / "splits" / f"train_folds_{ontology}.npy"

    train_data_cfg = DataConfig(
        processed_dir=str(processed_dir),
        ontology=ontology,
        feature_paths=cfg["features"]["train"],
        folds_path=str(folds_path),
        fold_id=args.fold,
        split="train",
    )

    val_data_cfg = DataConfig(
        processed_dir=str(processed_dir),
        ontology=ontology,
        feature_paths=cfg["features"]["train"],
        folds_path=str(folds_path),
        fold_id=args.fold,
        split="val",
    )

    train_ds = ProteinTrainDataset(train_data_cfg)
    val_ds = ProteinTrainDataset(val_data_cfg)

    training_cfg = cfg["training"]
    batch_size = training_cfg["batch_size"]
    num_workers = training_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_model(cfg["model"], input_dim, output_dim)

    devices = training_cfg.get("devices", [0])
    device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    lr = training_cfg["lr"]
    weight_decay = training_cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_epochs = training_cfg["max_epochs"]
    early_patience = training_cfg.get("early_stopping_patience", 5)
    precision = training_cfg.get("precision", "fp32")

    if isinstance(model, SeqMLPSoftF1):
        loss_fn = None
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    best_fmax = -1
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            if precision == "fp16":
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    if isinstance(model, SeqMLPSoftF1):
                        loss = model.compute_loss(logits, yb)
                    else:
                        loss = loss_fn(logits, yb)
            else:
                logits = model(xb)
                if isinstance(model, SeqMLPSoftF1):
                    loss = model.compute_loss(logits, yb)
                else:
                    loss = loss_fn(logits, yb)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch} train loss: {avg_loss:.4f}")

        # eval Fmax
        model.eval()
        val_scores = []
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
            val_scores.append(probs)

        val_scores = np.concatenate(val_scores, axis=0)  # [N_val, M]

        fold_ids = np.load(folds_path)
        val_mask = fold_ids == args.fold
        Y_val = Y[val_mask]

        from src.eval.threshold_search import search_threshold

        res = search_threshold(Y_val, val_scores)
        fmax = res["Fmax"]
        t_opt = res["threshold"]
        print(f"Epoch {epoch} Val Fmax={fmax:.4f} @ t={t_opt:.3f}")

        if fmax > best_fmax:
            best_fmax = fmax
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_patience:
            print("Early stopping")
            break

    # save best model + val preds
    output_dir = Path(args.output_dir) / cfg["name"] / f"fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "model.pt")

    # save OOF preds for this fold
    # recompute val preds with best model
    model.eval()
    val_scores = []
    for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
        val_scores.append(probs)
    val_scores = np.concatenate(val_scores, axis=0)

    np.save(output_dir / "val_scores.npy", val_scores)
    np.save(output_dir / "val_idx.npy", np.where(fold_ids == args.fold)[0])

    with open(output_dir / "val_metrics.json", "w") as f:
        json.dump({"Fmax": float(best_fmax)}, f)

    print("Saved model and val preds to", output_dir)


if __name__ == "__main__":
    main()
