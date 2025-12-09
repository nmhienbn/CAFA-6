import argparse
import json
import numpy as np
import pandas as pd
import time
import sys
import os
import warnings

# Tắt warning không cần thiết
warnings.filterwarnings("ignore")

# Kiểm tra import cugraph/cudf
try:
    import cudf
    import cugraph
except ImportError:
    print("❌ LỖI: Không tìm thấy thư viện RAPIDS (cudf, cugraph).")
    print("Vui lòng cài đặt môi trường GPU: pip install cugraph-cu11 --extra-index-url https://pypi.nvidia.com")
    sys.exit(1)

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
    
    # 1. Load Data vào GPU DataFrame (cuDF)
    print(f"Loading PPI from {args.ppi_path} to GPU...")
    # Đọc tsv bằng cudf
    gdf = cudf.read_csv(args.ppi_path, sep="\t")
    
    # Đổi tên cột chuẩn
    gdf = gdf.rename(columns={
        args.protein1_col: "source", 
        args.protein2_col: "destination", 
        args.weight_col: "weight"
    })
    
    # Tạo Graph
    # Lưu ý: cugraph cần source/destination là int32/int64 hoặc dùng string nhưng cần renumbring tự động
    # Các bản mới tự động handle string IDs nhưng tốt nhất cứ để nó tự xử
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source='source', destination='destination', edge_attr='weight')
    
    print(f"Graph loaded on GPU. Nodes: {G.number_of_vertices()}, Edges: {G.number_of_edges()}")

    # Lấy danh sách tất cả các Nodes để làm Master Index
    # cugraph trả về nodes dưới dạng Series cudf
    nodes_series = G.nodes()
    
    # Tạo master DataFrame để join các feature
    master_df = cudf.DataFrame({'vertex': nodes_series})
    master_df = master_df.sort_values('vertex').reset_index(drop=True)
    
    features_meta = ['protein_id']
    
    # ---------------------------------------------------------
    # 2. Degree Centrality
    # ---------------------------------------------------------
    print("Computing Degree Centrality...")
    t0 = time.time()
    # cugraph.degree_centrality trả về [vertex, degree_centrality]
    deg_df = cugraph.degree_centrality(G)
    
    # Merge
    master_df = master_df.merge(deg_df, on='vertex', how='left').fillna(0)
    master_df = master_df.rename(columns={'degree_centrality': 'degree'})
    features_meta.append('degree')
    print(f"Done in {time.time()-t0:.2f}s")

    # ---------------------------------------------------------
    # 3. Betweenness Centrality (Approx)
    # ---------------------------------------------------------
    if args.compute_betweenness:
        print(f"Computing Betweenness (k={args.betweenness_k})...")
        t0 = time.time()
        try:
            bet_df = cugraph.betweenness_centrality(G, k=args.betweenness_k)
            master_df = master_df.merge(bet_df, on='vertex', how='left').fillna(0)
            master_df = master_df.rename(columns={'betweenness_centrality': 'betweenness'})
            features_meta.append('betweenness')
            print(f"Done in {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"⚠️ Betweenness failed: {e}. Skipping.")

    # ---------------------------------------------------------
    # 4. PageRank
    # ---------------------------------------------------------
    if args.compute_pagerank:
        print("Computing PageRank...")
        t0 = time.time()
        pr_df = cugraph.pagerank(G)
        master_df = master_df.merge(pr_df, on='vertex', how='left').fillna(0)
        master_df = master_df.rename(columns={'pagerank': 'pagerank'})
        features_meta.append('pagerank')
        print(f"Done in {time.time()-t0:.2f}s")

    # ---------------------------------------------------------
    # 5. Node2Vec
    # ---------------------------------------------------------
    n2v_vectors = None
    if args.compute_node2vec:
        print(f"Computing Node2Vec (dim={args.node2vec_dim})...")
        t0 = time.time()
        
        try:
            # SỬA LỖI: Gọi trực tiếp cugraph.node2vec thay vì import submodule
            # Check xem function nằm ở đâu (thay đổi theo version)
            if hasattr(cugraph, 'node2vec'):
                n2v_func = cugraph.node2vec
            else:
                # Fallback cho các bản cũ hơn hoặc cấu trúc khác
                raise ImportError("Function cugraph.node2vec not found")

            # Gọi hàm
            # Kết quả n2v_df sẽ có cột 'vertex' và các cột embedding (0, 1, 2...)
            n2v_df = n2v_func(
                G, 
                # embedding_dim=args.node2vec_dim, 
                walk_length=args.walk_length, 
                num_walks=args.num_walks,
                p=1.0, q=1.0
            )
            
            # Merge để đảm bảo thứ tự node khớp với master_df
            # Lưu ý: Node2Vec có thể drop node cô lập, nên cần merge left từ master
            merged_n2v = master_df[['vertex']].merge(n2v_df, on='vertex', how='left').fillna(0.0)
            
            # Sort lại theo vertex để khớp với protein_id list
            merged_n2v = merged_n2v.sort_values('vertex')
            
            # Lấy các cột embedding (loại bỏ cột vertex)
            # Tên cột thường là 0, 1, 2 (int) hoặc "0", "1" (str)
            # Ta lọc các cột không phải là 'vertex'
            embedding_cols = [c for c in merged_n2v.columns if c != 'vertex']
            
            # Chuyển sang numpy array (CPU memory)
            # to_pandas() chuyển cudf -> pandas, sau đó .values lấy numpy
            n2v_vectors = merged_n2v[embedding_cols].to_pandas().values
            
            print(f"Done in {time.time()-t0:.2f}s. Shape: {n2v_vectors.shape}")
            
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
    
    # Chuyển master_df về Pandas (CPU)
    final_df = master_df.to_pandas()
    
    save_dict = {}
    
    # Lưu list protein_id (đảm bảo convert sang str nếu cần)
    save_dict['protein_id'] = final_df['vertex'].astype(str).values
    
    # Lưu các features scalar
    for feat in features_meta:
        if feat == 'protein_id': continue
        if feat in final_df.columns:
            save_dict[feat] = final_df[feat].values
        
    # Lưu node2vec embeddings
    if n2v_vectors is not None:
        save_dict['node2vec'] = n2v_vectors

    # Save .npz
    np.savez(args.out_path, **save_dict)
    
    # Save meta
    meta = {
        "nodes": len(final_df),
        "features": list(save_dict.keys()),
        "engine": "cugraph (GPU)"
    }
    os.makedirs(os.path.dirname(args.meta_path), exist_ok=True)
    with open(args.meta_path, 'w') as f:
        json.dump(meta, f)
        
    print(f"✅ Completed in {time.time() - start_global:.2f}s. Saved to {args.out_path}")

if __name__ == "__main__":
    main()