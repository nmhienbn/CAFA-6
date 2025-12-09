import pandas as pd
import numpy as np
import networkx as nx
import argparse
import json
from scipy import sparse
# Cần cài đặt node2vec: 
# Hoặc dùng implementation đơn giản nếu không muốn cài thêm lib nặng

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppi_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--meta_path", required=True)
    
    # Cấu hình
    parser.add_argument("--compute_closeness", action="store_true")
    parser.add_argument("--compute_betweenness", action="store_true")
    parser.add_argument("--betweenness_k", type=int, default=100) # Approx k nodes
    parser.add_argument("--compute_node2vec", action="store_true")
    parser.add_argument("--node2vec_dim", type=int, default=64)
    
    # Dummy args để khớp với lệnh gọi của bạn (chưa dùng trong logic đơn giản này)
    parser.add_argument("--protein1_col", default="protein1")
    parser.add_argument("--protein2_col", default="protein2")
    parser.add_argument("--weight_col", default="weight")
    parser.add_argument("--target_ids_path", default="")
    parser.add_argument("--protein_taxon_path", default="")
    parser.add_argument("--species_annot_path", default="")
    parser.add_argument("--topk_species", type=int, default=15)
    
    args = parser.parse_args()

    print("Loading PPI Graph...")
    df = pd.read_csv(args.ppi_path, sep="\t")
    G = nx.from_pandas_edgelist(df, args.protein1_col, args.protein2_col, edge_attr=args.weight_col)
    
    features = {}
    nodes = list(G.nodes())
    features['protein_id'] = nodes
    
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 1. Degree Centrality (Cơ bản, luôn tính)
    print("Computing Degree Centrality...")
    deg = nx.degree_centrality(G)
    features['degree'] = [deg[n] for n in nodes]

    # 2. Closeness (Rất chậm với đồ thị lớn > 50k node)
    if args.compute_closeness:
        print("Computing Closeness Centrality (Warning: Slow)...")
        # Chỉ nên chạy nếu đồ thị nhỏ hoặc sample
        clo = nx.closeness_centrality(G)
        features['closeness'] = [clo[n] for n in nodes]

    # 3. Betweenness (Approximate)
    if args.compute_betweenness:
        print(f"Computing Betweenness (Approx k={args.betweenness_k})...")
        bet = nx.betweenness_centrality(G, k=args.betweenness_k)
        features['betweenness'] = [bet[n] for n in nodes]

    # 4. Node2Vec
    if args.compute_node2vec:
        print("Computing Node2Vec...")
        try:
            from pecanpy import node2vec
            # PecanPy tự động tối ưu hóa đa luồng rất tốt
            g = node2vec.SparseOTF(p=1, q=1, workers=args.workers, verbose=True)
            g.read_edg(args.ppi_path, weighted=True, implicit=False)
            vectors = g.embed(dim=args.node2vec_dim, num_walks=10, walk_length=10, window_size=5)
        except ImportError:
            print("Module 'node2vec' not found. Skipping.")
            vectors = np.zeros((len(nodes), args.node2vec_dim))

    # Save outputs
    print(f"Saving to {args.out_path}...")
    
    # Convert list features to numpy arrays
    save_dict = {k: np.array(v) for k, v in features.items()}
    if args.compute_node2vec and 'vectors' in locals():
        save_dict['node2vec'] = vectors
        
    np.savez(args.out_path, **save_dict)
    
    # Save meta
    with open(args.meta_path, 'w') as f:
        json.dump({"nodes": len(nodes), "features": list(features.keys())}, f)

if __name__ == "__main__":
    main()