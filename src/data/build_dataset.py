import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_cafa", type=str, required=True)
    ap.add_argument("--processed", type=str, required=True)
    ap.add_argument("--ontology", type=str, default="BP", choices=["BP", "MF", "CC", "ALL"])
    ap.add_argument("--min_term_freq", type=int, default=1)
    return ap.parse_args()


def load_train_terms(path: Path, ontology: str):
    """
    Giả sử train_terms.tsv có cột:
    protein_id, go_id, ontology
    """
    df = pd.read_csv(path, sep="\t")
    if ontology != "ALL":
        df = df[df["ontology"] == ontology]
    return df


def build_mappings(df_terms: pd.DataFrame, processed_dir: Path, ontology: str, min_term_freq: int):
    proteins = sorted(df_terms["protein_id"].unique().tolist())
    protein2idx = {p: i for i, p in enumerate(proteins)}
    idx2protein = np.array(proteins, dtype=object)

    term_counts = df_terms["go_id"].value_counts()
    keep_terms = term_counts[term_counts >= min_term_freq].index.tolist()
    df_terms = df_terms[df_terms["go_id"].isin(keep_terms)]

    terms = sorted(keep_terms)
    go2idx = {g: i for i, g in enumerate(terms)}

    print(f"N proteins: {len(proteins)}, N terms ({ontology}): {len(terms)}")

    mapping_dir = processed_dir / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_dir / "protein2idx.json", "w") as f:
        json.dump(protein2idx, f)

    np.save(mapping_dir / "idx2protein.npy", idx2protein)

    with open(mapping_dir / f"go2idx_{ontology}.json", "w") as f:
        json.dump(go2idx, f)

    return protein2idx, go2idx


def build_label_matrix(df_terms: pd.DataFrame, protein2idx, go2idx, processed_dir: Path, ontology: str):
    rows, cols = [], []
    for _, row in df_terms.iterrows():
        p = row["protein_id"]
        g = row["go_id"]
        if p not in protein2idx or g not in go2idx:
            continue
        rows.append(protein2idx[p])
        cols.append(go2idx[g])

    data = np.ones(len(rows), dtype=np.int8)
    N = len(protein2idx)
    M = len(go2idx)

    Y = coo_matrix((data, (rows, cols)), shape=(N, M), dtype=np.int8).tocsr()
    mapping_dir = processed_dir / "mapping"
    save_npz(mapping_dir / f"Y_{ontology}.npz", Y)

    print(f"Saved Y_{ontology}.npz with shape {Y.shape} and nnz={Y.nnz}")


def main():
    args = parse_args()
    raw_cafa = Path(args.raw_cafa)
    processed = Path(args.processed)

    train_terms_path = raw_cafa / "Train" / "train_terms.tsv"
    df_terms = load_train_terms(train_terms_path, args.ontology)

    protein2idx, go2idx = build_mappings(
        df_terms, processed, args.ontology, args.min_term_freq
    )
    build_label_matrix(df_terms, protein2idx, go2idx, processed, args.ontology)


if __name__ == "__main__":
    main()
