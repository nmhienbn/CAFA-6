import os
import pandas as pd
from Bio import SeqIO
import argparse

def parse_fasta_headers(fasta_paths):
    """
    Parse FASTA headers để lấy Protein ID và Taxon ID.
    CAFA header format thường là: >DBID|PROTEIN_ID|TAXON_ID ...
    Hoặc check file train_taxonomy.tsv nếu có.
    """
    data = []
    ids = []
    
    for path in fasta_paths:
        print(f"Processing {path}...")
        for record in SeqIO.parse(path, "fasta"):
            # Ví dụ header: sp|P12345|10090 ...
            # Tùy format năm nay, code dưới đây giả định format chuẩn UniProt/CAFA
            # Nếu header chỉ có ID, ta cần file taxonomy riêng.
            
            # Giả sử format CAFA 6 năm nay đơn giản, ta lấy ID từ record.id
            pid = record.id
            ids.append(pid)
            
            # Taxon extraction logic (cần điều chỉnh theo dữ liệu thật)
            # Ở đây giả định bạn có file train_taxonomy.tsv thì tốt hơn
            # Code này chỉ lấy list protein ID
            
    return sorted(list(set(ids)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fasta", required=True)
    parser.add_argument("--test_fasta", required=True)
    parser.add_argument("--train_terms", required=True)
    parser.add_argument("--train_taxa", required=True) # File train_taxonomy.tsv từ Kaggle
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Tạo all_proteins.txt
    print("Reading FASTA files...")
    # Logic đơn giản: Đọc ID từ fasta
    # Lưu ý: Bạn nên dùng Biopython để parse chính xác
    train_ids = [r.id for r in SeqIO.parse(args.train_fasta, "fasta")]
    test_ids = [r.id for r in SeqIO.parse(args.test_fasta, "fasta")]
    all_ids = sorted(list(set(train_ids + test_ids)))
    
    with open(os.path.join(args.out_dir, "all_proteins.txt"), "w") as f:
        f.write("\n".join(all_ids))
    print(f"Saved {len(all_ids)} proteins to all_proteins.txt")

    # 2. Tạo protein_taxon.tsv
    # File train_taxonomy.tsv của Kaggle thường có dạng: ProteinID \t TaxonID
    print("Processing taxonomy...")
    df_tax = pd.read_csv(args.train_taxa, sep="\t", names=["protein_id", "taxon_id"])
    
    # Với tập Test, thường không có taxon public ngay. 
    # Nếu không có, ta tạm thời chỉ dùng train để map PPI.
    # Hoặc parse từ header FASTA test (nếu có info).
    df_tax.to_csv(os.path.join(args.out_dir, "protein_taxon.tsv"), sep="\t", index=False)
    print("Saved protein_taxon.tsv")

    # 3. Chuẩn hóa train_go_annotations.tsv
    print("Processing annotations...")
    # Kaggle: ProteinID, TermID, Aspect (BPO/CCO/MFO)
    df_terms = pd.read_csv(args.train_terms, sep="\t") 
    # Cần format: protein_id, go_id, label (mặc định positive là 1)
    df_terms = df_terms.rename(columns={"term": "go_id", "EntryID": "protein_id"}) # Check column names!
    if "label" not in df_terms.columns:
        df_terms["label"] = 1
    
    # Chọn cột cần thiết
    out_cols = ["protein_id", "go_id", "label"]
    # Map column names chính xác theo file Kaggle
    # Ví dụ CAFA 5: EntryID	term	aspect
    if "EntryID" in df_terms.columns:
        df_terms = df_terms.rename(columns={"EntryID": "protein_id", "term": "go_id"})
        
    df_terms = df_terms[["protein_id", "go_id", "label"]]
    df_terms.to_csv(os.path.join(args.out_dir, "train_go_annotations.tsv"), sep="\t", index=False)
    print("Saved train_go_annotations.tsv")

if __name__ == "__main__":
    main()