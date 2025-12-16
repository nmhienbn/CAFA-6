from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import json
import numpy as np
import pandas as pd

# ... (Giữ nguyên các hàm load_train_taxonomy và load_test_taxonomy_from_fasta) ...
def load_train_taxonomy(taxonomy_path: str | Path) -> pd.DataFrame:
    csv_path = Path(taxonomy_path)
    df = pd.read_csv(csv_path, sep="\t", header=None)
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
            proteins.append(parts[0])
            taxa.append(int(parts[1]))
    df = pd.DataFrame({"protein": proteins, "taxon": taxa})
    df["protein"] = df["protein"].astype(str)
    df["taxon"] = df["taxon"].astype(int)
    return df

@dataclass
class SpeciesIndexer:
    """
    taxon2idx : Dict[int, int]
    rare_idx : Optional[int] (Chỉ dùng nếu muốn gom nhóm rare vào 1 cột riêng)
    num_taxa : int
    """

    taxon2idx: Dict[int, int]
    rare_idx: Optional[int]
    num_taxa: int

    @classmethod
    def build(
        cls,
        train_tax: pd.DataFrame,
        test_tax: Optional[pd.DataFrame] = None,
        top_k: int = 500,
        include_rare: bool = False, 
        use_test_for_vocab: bool = True,
    ) -> "SpeciesIndexer":
        # 1. Gộp data để đếm tần suất
        if use_test_for_vocab and test_tax is not None:
            all_tax = pd.concat([train_tax[["taxon"]], test_tax[["taxon"]]], ignore_index=True)
        else:
            all_tax = train_tax[["taxon"]]

        # 2. Đếm và lấy Top K
        counts = all_tax["taxon"].astype(int).value_counts(sort=True, ascending=False)
        kept_taxa = counts.index[:top_k].tolist()

        # 3. Tạo mapping
        taxon2idx: Dict[int, int] = {}
        idx = 0
        for t in kept_taxa:
            taxon2idx[int(t)] = idx
            idx += 1

        # 4. Xử lý rare
        rare_idx: Optional[int] = None
        
        # Nếu include_rare=True: Tạo thêm 1 cột riêng cho loài hiếm
        if include_rare and (len(counts) > top_k):
            rare_idx = idx
            idx += 1
        
        # Nếu include_rare=False: Không tạo cột mới. 
        # Những loài lạ sẽ map về -1 và khi one-hot sẽ thành vector 0.

        return cls(taxon2idx=taxon2idx, rare_idx=rare_idx, num_taxa=idx)

    def _map_taxon(self, t: int) -> int:
        t = int(t)
        if t in self.taxon2idx:
            return self.taxon2idx[t]
        
        # Nếu có rare bucket thì trả về rare bucket index
        if self.rare_idx is not None:
            return self.rare_idx
            
        # Nếu không có rare bucket (include_rare=False), trả về -1
        # -1 sẽ được xử lý là "all-zeros" ở bước one-hot
        return -1

    def map_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map taxon -> taxon_idx.
        KHÔNG DROP dòng nào cả. Loài lạ sẽ có taxon_idx = -1.
        """
        if "taxon" not in df.columns:
            raise ValueError("df phải có cột 'taxon' để map.")
        
        out = df.copy()
        mapped = out["taxon"].map(self._map_taxon)
        out["taxon_idx"] = mapped.fillna(-1).astype(int) # Đảm bảo fillna an toàn

        # Đã xóa đoạn code drop row
        return out.reset_index(drop=True)

# -----------------------------------------------------------
# HÀM ONE-HOT ĐÃ SỬA ĐỔI
# -----------------------------------------------------------
def one_hot_from_index(taxon_idx: np.ndarray, num_taxa: int) -> np.ndarray:
    """
    Tạo one-hot encoding.
    Nếu taxon_idx là -1 (loài hiếm), vector sẽ là toàn số 0.
    """
    N = taxon_idx.shape[0]
    one_hot = np.zeros((N, num_taxa), dtype=np.float32)
    
    # Tạo mask: chỉ lấy những index hợp lệ (>= 0)
    valid_mask = taxon_idx >= 0
    
    # Chỉ gán 1.0 cho những dòng có index hợp lệ
    # Những dòng có index -1 sẽ bị bỏ qua -> giữ nguyên giá trị 0.0 ban đầu
    one_hot[valid_mask, taxon_idx[valid_mask]] = 1.0
    
    return one_hot

def save_species_features(
    train_tax: pd.DataFrame,
    test_tax: pd.DataFrame,
    indexer: SpeciesIndexer,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Mapping Train Data...")
    if "taxon_idx" not in train_tax.columns:
        train_tax = indexer.map_df(train_tax)
    
    print("Mapping Test Data...")
    if "taxon_idx" not in test_tax.columns:
        test_tax = indexer.map_df(test_tax)

    # Convert cột taxon_idx sang numpy array
    train_idx = train_tax["taxon_idx"].to_numpy(dtype=np.int64)
    test_idx = test_tax["taxon_idx"].to_numpy(dtype=np.int64)
    
    # Tạo one-hot (đã hỗ trợ vector 0 cho index -1)
    train_onehot = one_hot_from_index(train_idx, indexer.num_taxa)
    test_onehot = one_hot_from_index(test_idx, indexer.num_taxa)

    # Save NPY
    np.save(out_dir / "train_species_onehot.npy", train_onehot)
    np.save(out_dir / "test_species_onehot.npy", test_onehot)
    np.save(out_dir / "train_species_idx.npy", train_idx)
    np.save(out_dir / "test_species_idx.npy", test_idx)

    # Save Metadata TSV (Lưu lại toàn bộ, kể cả loài hiếm)
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
    print(f"Final Train size: {len(train_tax)}, Test size: {len(test_tax)}")


def main():
    # -------------------------------------------------------
    TRAIN_TAX_PATH = "data/raw/cafa6/Train/train_taxonomy.tsv"
    TEST_FASTA_PATH = "data/raw/cafa6/Test/testsuperset.fasta"
    OUT_DIR = "features/taxonomy_top500" 
    # -------------------------------------------------------

    train_tax = load_train_taxonomy(TRAIN_TAX_PATH)
    test_tax = load_test_taxonomy_from_fasta(TEST_FASTA_PATH)
    
    print(f"Original Train: {len(train_tax)}, Original Test: {len(test_tax)}")

    # BUILD INDEXER
    indexer = SpeciesIndexer.build(
        train_tax=train_tax,
        test_tax=test_tax,
        top_k=500,              
        include_rare=False,     # False: Không tạo cột riêng cho rare
        use_test_for_vocab=True,
    )

    save_species_features(train_tax, test_tax, indexer, OUT_DIR)

if __name__ == "__main__":
    main()