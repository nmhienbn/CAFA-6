import pandas as pd
import numpy as np
import networkx as nx
import argparse
import json
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    # Input Data
    parser.add_argument("--ppi_path", required=True)
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--protein_taxon_path", required=True)
    parser.add_argument("--species_annot_path", required=True)
    parser.add_argument("--target_ids_path", required=True) # Danh sách protein cần predict (Test set)
    
    # Output
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--meta_path", required=True)
    
    # Columns
    parser.add_argument("--protein1_col", default="protein1")
    parser.add_argument("--protein2_col", default="protein2")
    parser.add_argument("--weight_col", default="weight")
    parser.add_argument("--ann_protein_col", default="protein_id")
    parser.add_argument("--ann_go_col", default="go_id")
    parser.add_argument("--ann_label_col", default="label") 
    
    # Parameters
    parser.add_argument("--topk_species", type=int, default=15)
    parser.add_argument("--k_neighbors", type=int, default=50) # Top K neighbor mạnh nhất
    parser.add_argument("--min_edge_weight", type=float, default=0.0)
    parser.add_argument("--normalize_scores", action="store_true")
    # Dummy
    parser.add_argument("--ann_label_threshold", type=float, default=0.5)

    args = parser.parse_args()

    # 1. Load Species Info & Filter Top 15
    print("Loading Species Stats...")
    df_stats = pd.read_csv(args.species_annot_path, sep="\t")
    top_species = df_stats.nlargest(args.topk_species, 'n_annotated_proteins')['taxon_id'].tolist()
    print(f"Top {args.topk_species} species IDs: {top_species}")

    # 2. Load Protein Taxon map
    print("Loading Taxon Map...")
    df_tax = pd.read_csv(args.protein_taxon_path, sep="\t") # protein_id, taxon_id
    # Tạo set các protein thuộc top species (Training source tốt)
    valid_train_proteins = set(df_tax[df_tax['taxon_id'].isin(top_species)]['protein_id'])
    
    # 3. Load Annotations (Train)
    print("Loading Annotations...")
    df_ann = pd.read_csv(args.annotation_path, sep="\t")
    # Filter: Chỉ giữ annotation của các loài tốt để lan truyền
    df_ann = df_ann[df_ann[args.ann_protein_col].isin(valid_train_proteins)]
    
    # Tạo dictionary: Protein -> Set of GO terms
    # Để tối ưu, ta gom nhóm
    train_go_map = df_ann.groupby(args.ann_protein_col)[args.ann_go_col].apply(set).to_dict()

    # 4. Load PPI Graph
    print("Loading PPI...")
    df_ppi = pd.read_csv(args.ppi_path, sep="\t")
    # Filter edge weight
    df_ppi = df_ppi[df_ppi[args.weight_col] > args.min_edge_weight]
    
    # Build Graph (NetworkX hoặc Adjacency Dict)
    # Dùng Dict of Dicts cho nhanh: G[u][v] = w
    G = {}
    print("Building adjacency list...")
    for _, row in tqdm(df_ppi.iterrows(), total=len(df_ppi)):
        p1, p2, w = row[args.protein1_col], row[args.protein2_col], row[args.weight_col]
        if p1 not in G: G[p1] = {}
        if p2 not in G: G[p2] = {}
        G[p1][p2] = w
        G[p2][p1] = w # Undirected

    # 5. Load Targets (Proteins cần dự đoán)
    with open(args.target_ids_path, 'r') as f:
        target_ids = [line.strip() for line in f if line.strip()]

    # 6. Predict (Net-KNN)
    print(f"Predicting for {len(target_ids)} proteins...")
    results = []
    
    for target in tqdm(target_ids):
        if target not in G:
            continue
            
        # Lấy neighbors của target
        neighbors = G[target] # {neighbor_id: weight}
        
        # Sort neighbors theo weight giảm dần và lấy Top K
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:args.k_neighbors]
        
        # Tính điểm cho từng GO term
        # Score(GO) = Sum(Weight_neighbor) nếu neighbor có GO đó
        go_scores = {}
        total_weight = 0.0
        
        for n_id, w in sorted_neighbors:
            if n_id in train_go_map: # Neighbor này có annotation và thuộc Top Species
                total_weight += w
                for go_id in train_go_map[n_id]:
                    go_scores[go_id] = go_scores.get(go_id, 0.0) + w
        
        # Normalize
        if args.normalize_scores and total_weight > 0:
            for go_id in go_scores:
                go_scores[go_id] /= total_weight
        
        # Lưu kết quả
        # Format output mong đợi: protein_id, go_id, score
        # Để tiết kiệm, ta chỉ lưu Top GO terms hoặc threshold
        for go_id, score in go_scores.items():
            if score > 0.01: # Threshold nhẹ để giảm dung lượng file
                results.append(f"{target}\t{go_id}\t{score:.4f}")

    # 7. Write Output
    print("Writing results...")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        f.write("protein_id\tgo_id\tscore\n")
        f.write("\n".join(results))
    
    # Meta
    os.makedirs(os.path.dirname(args.meta_path), exist_ok=True)
    with open(args.meta_path, 'w') as f:
        json.dump({"n_targets_predicted": len(target_ids), "method": "NetKNN_Top15"}, f)

if __name__ == "__main__":
    main()