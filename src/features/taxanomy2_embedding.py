from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import json
import numpy as np
import pandas as pd

def normalize_taxonkit_output(
    raw_path: str | Path,
    out_path: str | Path,
) -> None:
    raw_path = Path(raw_path)
    out_path = Path(out_path)

    df = pd.read_csv(raw_path, sep="\t", header=None, dtype=str)

    if df.shape[1] < 8:
        raise ValueError(
            f"Expect at least 8 columns (taxid + 7 ranks), got {df.shape[1]}"
        )

    df = df.iloc[:, :8]
    df.columns = [
        "taxon",
        "superkingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    df["taxon"] = df["taxon"].astype(int)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Saved normalized lineage to {out_path}")

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


def load_taxon_lineage(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["taxon"] = df["taxon"].astype(int)

    expected_cols = [
        "taxon",
        "superkingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df[expected_cols]


def attach_lineage(df_tax: pd.DataFrame, lineage: pd.DataFrame) -> pd.DataFrame:
    return df_tax.merge(lineage, on="taxon", how="left")

@dataclass
class RankIndexer:
    value2idx: Dict[str, int]
    num_values: int
    unknown_idx: int

    @classmethod
    def from_series(cls, s: pd.Series, add_unknown: bool = True) -> "RankIndexer":
        """
        s: Series các string (genus / family / order), có thể NaN.
        """
        s = s.fillna("unknown").astype(str)
        uniq = sorted(s.unique())

        value2idx: Dict[str, int] = {}
        idx = 0
        if add_unknown:
            value2idx["unknown"] = idx
            idx += 1

        for v in uniq:
            if add_unknown and v == "unknown":
                continue
            value2idx[v] = idx
            idx += 1

        num_values = idx
        unknown_idx = value2idx["unknown"] if add_unknown else -1
        return cls(value2idx=value2idx, num_values=num_values, unknown_idx=unknown_idx)

    def map_series(self, s: pd.Series) -> pd.Series:
        s = s.fillna("unknown").astype(str)
        return s.map(lambda v: self.value2idx.get(v, self.unknown_idx)).astype(int)


@dataclass
class TaxonomyIndexers:
    genus: RankIndexer
    family: RankIndexer
    order: RankIndexer

    @classmethod
    def build_from_lineage(cls, lineage: pd.DataFrame) -> "TaxonomyIndexers":
        genus_indexer = RankIndexer.from_series(lineage["genus"])
        family_indexer = RankIndexer.from_series(lineage["family"])
        order_indexer = RankIndexer.from_series(lineage["order"])
        return cls(
            genus=genus_indexer,
            family=family_indexer,
            order=order_indexer,
        )


def add_rank_indices(
    df: pd.DataFrame,
    indexers: TaxonomyIndexers,
) -> pd.DataFrame:
    df = df.copy()
    df["genus_idx"] = indexers.genus.map_series(df["genus"])
    df["family_idx"] = indexers.family.map_series(df["family"])
    df["order_idx"] = indexers.order.map_series(df["order"])
    return df

def one_hot_from_index(idx: np.ndarray, num_values: int) -> np.ndarray:
    """
    idx: [N] (int)\\
    returns: [N, num_values] one-hot
    """
    N = idx.shape[0]
    one_hot = np.zeros((N, num_values), dtype=np.float32)
    one_hot[np.arange(N), idx] = 1.0
    return one_hot


def save_taxonomy_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    indexers: TaxonomyIndexers,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, indexer in [
        ("genus", indexers.genus),
        ("family", indexers.family),
        ("order", indexers.order),
    ]:
        dim = indexer.num_values

        train_idx = train_df[f"{name}_idx"].to_numpy(dtype=np.int64)
        test_idx = test_df[f"{name}_idx"].to_numpy(dtype=np.int64)

        train_oh = one_hot_from_index(train_idx, dim)
        test_oh = one_hot_from_index(test_idx, dim)

        np.save(out_dir / f"train_{name}_idx.npy", train_idx)
        np.save(out_dir / f"test_{name}_idx.npy", test_idx)
        np.save(out_dir / f"train_{name}_onehot.npy", train_oh)
        np.save(out_dir / f"test_{name}_onehot.npy", test_oh)

    cols = [
        "protein",
        "taxon",
        "order",
        "family",
        "genus",
        "order_idx",
        "family_idx",
        "genus_idx",
    ]
    train_df[cols].to_csv(
        out_dir / "train_taxonomy_highlevel.tsv", sep="\t", index=False
    )
    test_df[cols].to_csv(
        out_dir / "test_taxonomy_highlevel.tsv", sep="\t", index=False
    )

    vocab = {
        "genus": {
            "value2idx": indexers.genus.value2idx,
            "unknown_idx": indexers.genus.unknown_idx,
            "num_values": indexers.genus.num_values,
        },
        "family": {
            "value2idx": indexers.family.value2idx,
            "unknown_idx": indexers.family.unknown_idx,
            "num_values": indexers.family.num_values,
        },
        "order": {
            "value2idx": indexers.order.value2idx,
            "unknown_idx": indexers.order.unknown_idx,
            "num_values": indexers.order.num_values,
        },
    }
    with (out_dir / "taxonomy_highlevel_vocab.json").open("w") as f:
        json.dump(vocab, f, indent=2)

    print(f"[OK] Saved high-level taxonomy features to {out_dir}")


def main():
    TRAIN_TAX_PATH = "data/raw/cafa6/Train/train_taxonomy.tsv"
    TEST_FASTA_PATH = "data/raw/cafa6/Test/testsuperset.fasta"
    RAW_LINEAGE_PATH = "data/raw/cafa6/taxon_lineage_raw.tsv"   # from taxonkit
    LINEAGE_PATH = "data/raw/cafa6/taxon_lineage.tsv"
    OUT_DIR = "features/taxonomy_highlevel"
    # ----------------------------------------------

    # B1: chuẩn hóa raw -> lineage.tsv
    normalize_taxonkit_output(RAW_LINEAGE_PATH, LINEAGE_PATH)

    # B2: load train/test taxon
    train_tax = load_train_taxonomy(TRAIN_TAX_PATH)
    test_tax = load_test_taxonomy_from_fasta(TEST_FASTA_PATH)

    # B3: load lineage + build indexers
    lineage = load_taxon_lineage(LINEAGE_PATH)
    indexers = TaxonomyIndexers.build_from_lineage(lineage)

    # B4: join lineage và map sang idx
    train_full = attach_lineage(train_tax, lineage)
    test_full = attach_lineage(test_tax, lineage)

    train_full = add_rank_indices(train_full, indexers)
    test_full = add_rank_indices(test_full, indexers)

    # B5: save .npy + mapping
    save_taxonomy_features(train_full, test_full, indexers, OUT_DIR)


if __name__ == "__main__":
    main()
