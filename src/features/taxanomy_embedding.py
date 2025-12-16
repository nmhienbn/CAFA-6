"""
Tạo species feature cho CAFA6 và lưu ra .npy

Input:
- Train/train_taxonomy.tsv
    + Format: EntryID<TAB>taxonomyID

- Test/testsuperset.fasta
    + Header: >UNIPROT_ID TAXON_ID
      Ví dụ: >A0A0C5B5G6 9606

Output (ở OUT_DIR):
- train_species_onehot.npy   : [N_train, num_taxa]
- test_species_onehot.npy    : [N_test,  num_taxa]
- train_species_idx.npy      : [N_train]  (taxon_idx)
- test_species_idx.npy       : [N_test]
- train_species_proteins.tsv : protein <TAB> taxon <TAB> taxon_idx
- test_species_proteins.tsv  : ...
- species_vocab.json         : mapping taxon_id <-> index
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import json
import numpy as np
import pandas as pd

def load_train_taxonomy(taxonomy_path: str | Path) -> pd.DataFrame:
    csv_path = Path(taxonomy_path)

    df = pd.read_csv(csv_path, sep="\t")

    col0, col1 = df.columns[:2]
    df = df.rename(columns={col0: "protein", col1: "taxon"})

    df["protein"] = df["protein"].astype(str)
    df["taxon"] = df["taxon"].astype(int)
    return df[["protein", "taxon"]]


def load_test_taxonomy_from_fasta(fasta_path: str | Path) -> pd.DataFrame:
    fasta_path = Path(fasta_path)

    proteins = []
    taxa = []

    with fasta_path.open("r") as f:
        for line in f:
            if not line.startswith(">"):
                continue

            header = line[1:].strip()
            parts = header.split()
            if len(parts) < 2:
                raise ValueError(f"Header not in format 'ID TAXONID': {header}")

            prot_id = parts[0]
            tax_id = int(parts[1])

            proteins.append(prot_id)
            taxa.append(tax_id)

    df = pd.DataFrame({"protein": proteins, "taxon": taxa})
    df["protein"] = df["protein"].astype(str)
    df["taxon"] = df["taxon"].astype(int)
    return df

@dataclass
class SpeciesIndexer:
    """
    Mapping taxon_id -> index dùng cho embedding / one-hot.

    taxon2idx : Dict[int, int]
        Map taxon (NCBI taxid) ==> index (0..K-1 or 0..K-2 if rare_idx).
    rare_idx : Optional[int]
        Index cho "rare/unknown" taxon (optional).
    num_taxa : int
        Số lượng index (kể cả rare_idx nếu có).
    """

    taxon2idx: Dict[int, int]
    rare_idx: Optional[int]
    num_taxa: int

    @classmethod
    def build(
        cls,
        train_tax: pd.DataFrame,
        test_tax: Optional[pd.DataFrame] = None,
        min_count: int = 1,
        use_test_for_vocab: bool = True,
    ) -> "SpeciesIndexer":
        """
        Build index từ train (+ optional test).

        - min_count:
            taxon xuất hiện < min_count sẽ gộp vào rare_idx (nếu >1).
        - use_test_for_vocab:
            True  => đếm cả taxon trong test (đảm bảo test taxid có index riêng)
            False => chỉ đếm train; taxon chỉ có ở test -> rare_idx.
        """
        if "taxon" not in train_tax.columns:
            raise ValueError("train_tax phải có cột 'taxon'.")

        if test_tax is not None and "taxon" not in test_tax.columns:
            raise ValueError("test_tax phải có cột 'taxon'.")

        if use_test_for_vocab and test_tax is not None:
            all_tax = pd.concat(
                [train_tax[["taxon"]], test_tax[["taxon"]]],
                ignore_index=True,
            )
        else:
            all_tax = train_tax[["taxon"]]

        all_tax["taxon"] = all_tax["taxon"].astype(int)
        counts = all_tax["taxon"].value_counts()

        kept_taxa = counts[counts >= min_count].index.tolist()

        taxon2idx: Dict[int, int] = {}
        idx = 0
        for t in kept_taxa:
            taxon2idx[int(t)] = idx
            idx += 1

        # Bucket cho rare / unknown
        rare_idx: Optional[int] = None
        if (counts < min_count).sum() > 0:
            rare_idx = idx
            idx += 1

        num_taxa = idx

        return cls(taxon2idx=taxon2idx, rare_idx=rare_idx, num_taxa=num_taxa)

    # ----------------------- map hàm đơn lẻ -----------------------

    def _map_taxon(self, t: int) -> int:
        t = int(t)
        if t in self.taxon2idx:
            return self.taxon2idx[t]
        if self.rare_idx is None:
            raise KeyError(f"Unknown taxon {t} và rare_idx=None.")
        return self.rare_idx

    # ----------------------- public API -----------------------

    def map_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm cột 'taxon_idx' vào DataFrame (cần có cột 'taxon').
        """
        if "taxon" not in df.columns:
            raise ValueError("df phải có cột 'taxon' để map.")
        out = df.copy()
        out["taxon_idx"] = out["taxon"].map(self._map_taxon).astype(int)
        return out

def one_hot_from_index(taxon_idx: np.ndarray, num_taxa: int) -> np.ndarray:
    """
    taxon_idx: [N] (int)
    return: one-hot [N, num_taxa]
    """
    N = taxon_idx.shape[0]
    one_hot = np.zeros((N, num_taxa), dtype=np.float32)
    one_hot[np.arange(N), taxon_idx] = 1.0
    return one_hot


def save_species_features(
    train_tax: pd.DataFrame,
    test_tax: pd.DataFrame,
    indexer: SpeciesIndexer,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "taxon_idx" not in train_tax.columns:
        train_tax = indexer.map_df(train_tax)
    if "taxon_idx" not in test_tax.columns:
        test_tax = indexer.map_df(test_tax)

    train_idx = train_tax["taxon_idx"].to_numpy(dtype=np.int64)
    test_idx = test_tax["taxon_idx"].to_numpy(dtype=np.int64)

    train_onehot = one_hot_from_index(train_idx, indexer.num_taxa)
    test_onehot = one_hot_from_index(test_idx, indexer.num_taxa)

    np.save(out_dir / "train_species_onehot.npy", train_onehot)
    np.save(out_dir / "test_species_onehot.npy", test_onehot)
    np.save(out_dir / "train_species_idx.npy", train_idx)
    np.save(out_dir / "test_species_idx.npy", test_idx)

    train_tax[["protein", "taxon", "taxon_idx"]].to_csv(
        out_dir / "train_species_proteins.tsv", sep="\t", index=False
    )
    test_tax[["protein", "taxon", "taxon_idx"]].to_csv(
        out_dir / "test_species_proteins.tsv", sep="\t", index=False
    )

    vocab = {
        "taxon2idx": {str(k): int(v) for k, v in indexer.taxon2idx.items()},
        "rare_idx": indexer.rare_idx,
        "num_taxa": indexer.num_taxa,
    }
    with (out_dir / "species_vocab.json").open("w") as f:
        json.dump(vocab, f, indent=2)

    print(f"[OK] Saved species features to {out_dir}")


def main():
    # -------------------------------------------------------
    TRAIN_TAX_PATH = "data/raw/cafa6/Train/train_taxonomy.tsv"
    TEST_FASTA_PATH = "data/raw/cafa6/Test/testsuperset.fasta"
    OUT_DIR = "data/processed/cafa6/taxanomy"
    # -------------------------------------------------------

    train_tax = load_train_taxonomy(TRAIN_TAX_PATH)
    test_tax = load_test_taxonomy_from_fasta(TEST_FASTA_PATH)

    indexer = SpeciesIndexer.build(
        train_tax=train_tax,
        test_tax=test_tax,
        min_count=1,
        use_test_for_vocab=True,
    )

    train_tax = indexer.map_df(train_tax)
    test_tax = indexer.map_df(test_tax)

    save_species_features(train_tax, test_tax, indexer, OUT_DIR)


if __name__ == "__main__":
    main()
