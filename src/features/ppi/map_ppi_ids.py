import pandas as pd
import argparse
import os
import gzip

def load_mapping(mapping_path):
    """
    Load file mapping: STRING_ID -> CAFA_ID
    File mapping này bạn phải tự tạo từ STRING Aliases.
    Format: string_id \t protein_id
    """
    print(f"Loading mapping from {mapping_path}")
    return pd.read_csv(mapping_path, sep="\t")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppi_in", required=True)
    parser.add_argument("--mapping_path", required=True) # File string_to_protein_id.tsv
    parser.add_argument("--ppi_out", required=True)
    # Các tham số cột (giữ mặc định như yêu cầu của bạn)
    parser.add_argument("--ppi_protein1_col", default="protein1")
    parser.add_argument("--ppi_protein2_col", default="protein2")
    parser.add_argument("--ppi_weight_col", default="weight")
    parser.add_argument("--map_src_col", default="string_id")
    parser.add_argument("--map_tgt_col", default="protein_id")
    parser.add_argument("--drop_self", action="store_true")
    parser.add_argument("--agg_method", default="max")
    
    args = parser.parse_args()

    # 1. Load PPI (STRING raw)
    print("Loading PPI raw...")
    df_ppi = pd.read_csv(args.ppi_in, sep="\t")
    
    # 2. Load Mapping
    # Giả sử bạn đã có file này. Nếu chưa, xem hướng dẫn bên dưới code.
    df_map = pd.read_csv(args.mapping_path, sep="\t")
    # Tạo dict cho nhanh: string_id -> protein_id
    # Lưu ý: STRING ID thường có dạng "9606.ENSP000..."
    mapper = dict(zip(df_map[args.map_src_col], df_map[args.map_tgt_col]))

    # 3. Map IDs
    print("Mapping Protein 1...")
    df_ppi['p1_mapped'] = df_ppi[args.ppi_protein1_col].map(mapper)
    print("Mapping Protein 2...")
    df_ppi['p2_mapped'] = df_ppi[args.ppi_protein2_col].map(mapper)

    # 4. Filter missing
    initial_len = len(df_ppi)
    df_ppi = df_ppi.dropna(subset=['p1_mapped', 'p2_mapped'])
    print(f"Dropped {initial_len - len(df_ppi)} edges due to missing mapping.")

    # 5. Drop self-loops
    if args.drop_self:
        df_ppi = df_ppi[df_ppi['p1_mapped'] != df_ppi['p2_mapped']]

    # 6. Chuẩn hóa format đầu ra
    # Sắp xếp p1, p2 để coi (A,B) như (B,A) -> giảm duplicate undirected
    # (Tùy chọn, STRING thường đã đối xứng, nhưng remap có thể làm lệch)
    # Ở đây ta giữ nguyên hướng hoặc sort
    
    out_df = df_ppi[['p1_mapped', 'p2_mapped', args.ppi_weight_col]].copy()
    out_df.columns = ['protein1', 'protein2', 'weight']

    # 7. Aggregate duplicates (nếu nhiều string id map về cùng 1 cafa id)
    print(f"Aggregating duplicates using {args.agg_method}...")
    if args.agg_method == 'max':
        out_df = out_df.groupby(['protein1', 'protein2'], as_index=False)['weight'].max()
    
    print(f"Saving {len(out_df)} edges to {args.ppi_out}")
    out_df.to_csv(args.ppi_out, sep="\t", index=False)

if __name__ == "__main__":
    main()