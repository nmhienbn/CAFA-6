import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--protein_taxon_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--ann_protein_col", default="protein_id")
    parser.add_argument("--ann_go_col", default="go_id")
    # Các tham số thừa để tương thích lệnh gọi của bạn
    parser.add_argument("--ann_label_col", default="label") 
    parser.add_argument("--ann_label_threshold", type=float, default=0.5)

    args = parser.parse_args()

    print("Loading data...")
    df_ann = pd.read_csv(args.annotation_path, sep="\t")
    df_tax = pd.read_csv(args.protein_taxon_path, sep="\t")
    
    # Merge Annotation với Taxon
    # df_ann: protein_id, go_id
    # df_tax: protein_id, taxon_id
    merged = pd.merge(df_ann, df_tax, left_on=args.ann_protein_col, right_on="protein_id", how="inner")
    
    print("Computing stats...")
    # Đếm số lượng protein unique có annotation cho mỗi taxon
    stats = merged.groupby("taxon_id")[args.ann_protein_col].nunique().reset_index()
    stats.columns = ["taxon_id", "n_annotated_proteins"]
    
    # Đếm tổng số lượng annotations (cặp protein-go)
    count_anns = merged.groupby("taxon_id")[args.ann_go_col].count().reset_index()
    count_anns.columns = ["taxon_id", "n_annotations"]
    
    final_stats = pd.merge(stats, count_anns, on="taxon_id")
    
    # Sort giảm dần
    final_stats = final_stats.sort_values("n_annotated_proteins", ascending=False)
    
    print(f"Top 5 species:\n{final_stats.head(5)}")
    
    final_stats.to_csv(args.out_path, sep="\t", index=False)
    print(f"Saved stats to {args.out_path}")

if __name__ == "__main__":
    main()