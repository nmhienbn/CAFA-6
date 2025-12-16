import argparse
import json
import numpy as np
import pandas as pd
import time
import sys
import os
import warnings

# Tắt warning
warnings.filterwarnings("ignore")

# Kiểm tra RAPIDS
try:
    import cudf
    import cugraph
except ImportError:
    print("❌ LỖI: Không tìm thấy thư viện RAPIDS (cudf, cugraph).")
    sys.exit(1)

# Kiểm tra Gensim (Cần cho Node2Vec embedding)
try:
    from gensim.models import Word2Vec
except ImportError:
    print("⚠️ CẢNH BÁO: Không tìm thấy thư viện 'gensim'.")
    print("   Cài đặt bằng: pip install gensim")
    # Chúng ta sẽ xử lý lỗi này trong hàm main nếu người dùng bật node2vec

def main():
    parser = argparse.ArgumentParser(description="Build PPI features using GPU (cuGraph)")
    parser.add_argument("--ppi_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--meta_path", required=True)
    
    # Column names
    parser.add_argument("--protein1_col", default="protein1")
    parser.add_argument("--protein2_col", default="protein2")
    parser.add_argument("--weight_col", default="weight")
    
    # Feature flags
    parser.add_argument("--compute_betweenness", action="store_true")
    parser.add_argument("--betweenness_k", type=int, default=100)
    
    parser.add_argument("--compute_node2vec", action="store_true")
    parser.add_argument("--node2vec_dim", type=int, default=64)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--num_walks", type=int, default=10)
    
    parser.add_argument("--compute_pagerank", action="store_true")

    args = parser.parse_args()
    
    start_global = time.time()
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print(f"Loading PPI from {args.ppi_path} to GPU...")
    gdf = cudf.read_csv(args.ppi_path, sep="\t")
    
    # Đổi tên cột chuẩn
    gdf = gdf.rename(columns={
        args.protein1_col: "source", 
        args.protein2_col: "destination", 
        args.weight_col: "weight"
    })
    
    # Tạo Graph (cugraph tự động renumber string sang int nếu cần)
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source='source', destination='destination', edge_attr='weight')
    
    print(f"Graph loaded. Nodes: {G.number_of_vertices()}, Edges: {G.number_of_edges()}")

    # Master Index
    nodes_series = G.nodes()
    master_df = cudf.DataFrame({'vertex': nodes_series})
    master_df = master_df.sort_values('vertex').reset_index(drop=True)
    
    features_meta = ['protein_id']
    
    # ---------------------------------------------------------
    # 2. Degree Centrality
    # ---------------------------------------------------------
    print("Computing Degree Centrality...")
    deg_df = cugraph.degree_centrality(G)
    master_df = master_df.merge(deg_df, on='vertex', how='left').fillna(0)
    master_df = master_df.rename(columns={'degree_centrality': 'degree'})
    features_meta.append('degree')

    # ---------------------------------------------------------
    # 3. Betweenness Centrality
    # ---------------------------------------------------------
    if args.compute_betweenness:
        print(f"Computing Betweenness (k={args.betweenness_k})...")
        try:
            bet_df = cugraph.betweenness_centrality(G, k=args.betweenness_k)
            master_df = master_df.merge(bet_df, on='vertex', how='left').fillna(0)
            master_df = master_df.rename(columns={'betweenness_centrality': 'betweenness'})
            features_meta.append('betweenness')
        except Exception as e:
            print(f"⚠️ Betweenness failed: {e}")

    # ---------------------------------------------------------
    # 4. PageRank
    # ---------------------------------------------------------
    if args.compute_pagerank:
        print("Computing PageRank...")
        pr_df = cugraph.pagerank(G)
        master_df = master_df.merge(pr_df, on='vertex', how='left').fillna(0)
        master_df = master_df.rename(columns={'pagerank': 'pagerank'})
        features_meta.append('pagerank')

    # ---------------------------------------------------------
    # 5. Node2Vec (FIXED)
    # ---------------------------------------------------------
    n2v_vectors = None
    if args.compute_node2vec:
        print(f"Computing Node2Vec (dim={args.node2vec_dim}, walks={args.num_walks})...")
        t0 = time.time()
        
        # Kiểm tra Gensim lần nữa
        if 'Word2Vec' not in sys.modules and 'gensim.models' not in sys.modules:
             try:
                 from gensim.models import Word2Vec
             except ImportError:
                 print("❌ ERROR: Cần cài đặt 'gensim' để chạy Node2Vec.")
                 sys.exit(1)

        try:
            # BƯỚC 1: Tạo Random Walks trên GPU bằng cugraph
            # Để có 'num_walks' cho mỗi node, ta phải lặp lại danh sách start_nodes bấy nhiêu lần
            start_nodes = G.nodes()
            # Lặp lại series start_nodes (concat)
            start_nodes_expanded = cudf.concat([start_nodes] * args.num_walks)
            
            # Gọi cugraph.node2vec (trả về paths, weights, lengths)
            # Hàm này trả về dataframe với các cột là các bước đi
            paths_df, _, _ = cugraph.node2vec(
                G, 
                start_vertices=start_nodes_expanded, 
                max_depth=args.walk_length,
                p=1.0, q=1.0 # p, q = 1.0 tương đương DeepWalk/Uniform, chỉnh nếu cần
            )
            
            # BƯỚC 2: Chuẩn bị dữ liệu cho Gensim (CPU)
            # Chuyển paths từ GPU DataFrame sang Pandas -> Numpy -> List of Strings
            # Gensim Word2Vec yêu cầu list các câu (sentences), mỗi câu là list các từ (words/node_ids) dạng String
            
            # Lấy các cột chứa node id trong path (bỏ cột vertex id gốc hoặc weight nếu có)
            # paths_df thường chỉ chứa các cột integer đại diện cho vertex ở mỗi bước
            
            # Chuyển về CPU numpy array
            paths_np = paths_df.to_pandas().values
            
            # Chuyển sang list of lists of strings
            # Lưu ý: paths_np là ma trận [num_total_walks, walk_length]
            walks = []
            for row in paths_np:
                # Filter các giá trị padding (nếu graph nhỏ hoặc cụt đường, cugraph có thể fill -1)
                # Chuyển int -> str
                walk = [str(int(node_id)) for node_id in row if node_id >= 0]
                walks.append(walk)
            
            print(f"  Generated {len(walks)} walks. Training Word2Vec...")
            
            # BƯỚC 3: Train Word2Vec
            model = Word2Vec(
                sentences=walks, 
                vector_size=args.node2vec_dim, 
                window=5, 
                min_count=0, 
                sg=1, # Skip-gram
                workers=4,
                epochs=5
            )
            
            # BƯỚC 4: Trích xuất Vector và Map vào DataFrame
            # Tạo dictionary {node_id_str: vector}
            # Cần đảm bảo thứ tự khớp với master_df
            
            # Lấy danh sách vertex từ master_df (đang là int hoặc str tùy input gốc)
            # Chuyển sang chuỗi để lookup trong model gensim
            target_nodes = master_df['vertex'].to_pandas().astype(str).values
            
            vectors = []
            zeros = np.zeros(args.node2vec_dim)
            
            for node_str in target_nodes:
                if node_str in model.wv:
                    vectors.append(model.wv[node_str])
                else:
                    vectors.append(zeros)
            
            n2v_vectors = np.array(vectors) # Shape: (num_nodes, dim)
            
            print(f"Done Node2Vec in {time.time()-t0:.2f}s. Shape: {n2v_vectors.shape}")

        except Exception as e:
            print(f"⚠️ Node2Vec Error: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping Node2Vec.")

    # ---------------------------------------------------------
    # 6. Save Outputs
    # ---------------------------------------------------------
    print("Exporting data to CPU & Saving...")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    final_df = master_df.to_pandas()
    save_dict = {}
    
    # Lưu protein_id (đảm bảo str)
    save_dict['protein_id'] = final_df['vertex'].astype(str).values
    
    for feat in features_meta:
        if feat == 'protein_id': continue
        if feat in final_df.columns:
            save_dict[feat] = final_df[feat].values
            
    if n2v_vectors is not None:
        save_dict['node2vec'] = n2v_vectors

    np.savez(args.out_path, **save_dict)
    
    meta = {
        "nodes": len(final_df),
        "features": list(save_dict.keys()),
        "engine": "cugraph (GPU) + gensim (CPU)"
    }
    with open(args.meta_path, 'w') as f:
        json.dump(meta, f)
        
    print(f"✅ Completed in {time.time() - start_global:.2f}s. Saved to {args.out_path}")

if __name__ == "__main__":
    main()