import pandas as pd
import numpy as np
import networkx as nx
import argparse
import json
import multiprocessing
from functools import partial
import time

def chunks(l, n):
    """Chia list thành các chunk đều nhau"""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def _betweenness_worker(G, nodes_subset):
    """Hàm worker để tính betweenness cho một tập node con"""
    # Sử dụng betweenness_centrality_subset tính trên tập source cụ thể
    return nx.betweenness_centrality_subset(G, sources=nodes_subset, weight=None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppi_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--meta_path", required=True)
    
    # Cấu hình tính năng
    parser.add_argument("--compute_closeness", action="store_true", help="Rất chậm, cân nhắc kỹ!")
    parser.add_argument("--compute_betweenness", action="store_true")
    parser.add_argument("--betweenness_k", type=int, default=100)
    parser.add_argument("--compute_node2vec", action="store_true")
    parser.add_argument("--node2vec_dim", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4, help="Số luồng CPU sử dụng")
    
    # Dummy args
    parser.add_argument("--protein1_col", default="protein1")
    parser.add_argument("--protein2_col", default="protein2")
    parser.add_argument("--weight_col", default="weight")
    
    args = parser.parse_args()
    
    # Auto-detect max workers
    if args.workers == -1:
        args.workers = multiprocessing.cpu_count()
    print(f"🚀 Running with {args.workers} CPU workers")

    print(f"Loading PPI Graph from {args.ppi_path}...")
    df = pd.read_csv(args.ppi_path, sep="\t")
    # Graph vô hướng
    G = nx.from_pandas_edgelist(df, args.protein1_col, args.protein2_col, edge_attr=args.weight_col)
    nodes = list(G.nodes())
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    features = {'protein_id': nodes}

    # 1. Degree (Nhanh, chạy luôn)
    print("Computing Degree Centrality...")
    deg = nx.degree_centrality(G)
    features['degree'] = [deg[n] for n in nodes]

    # 2. Parallel Betweenness (Tối ưu Multi-process)
    if args.compute_betweenness:
        print(f"Computing Betweenness (Parallel, k={args.betweenness_k})...")
        start_time = time.time()
        
        # Chọn ngẫu nhiên k nodes làm source
        import random
        k = min(args.betweenness_k, len(nodes))
        sources = random.sample(nodes, k)
        
        # Chia source cho các workers
        source_chunks = list(chunks(sources, max(1, k // args.workers)))
        
        # Chạy song song
        # Lưu ý: Pass G qua pool có thể tốn RAM, nhưng với graph trung bình thì ổn
        with multiprocessing.Pool(processes=args.workers) as pool:
            func = partial(_betweenness_worker, G)
            results = pool.map(func, source_chunks)
            
        # Gộp kết quả (Cộng dồn score)
        bet_total = {n: 0.0 for n in nodes}
        for res in results:
            for n, score in res.items():
                bet_total[n] += score
        
        # Normalize (Optional: chia cho số lượng chunk hoặc để raw)
        features['betweenness'] = [bet_total[n] for n in nodes]
        print(f"Betweenness done in {time.time() - start_time:.2f}s")

    # 3. Closeness (Vẫn chậm, không khuyến khích parallel vì overhead lớn)
    if args.compute_closeness:
        print("⚠️ Computing Closeness (Single-thread, very slow)...")
        clo = nx.closeness_centrality(G)
        features['closeness'] = [clo[n] for n in nodes]

    # 4. Parallel Node2Vec
    if args.compute_node2vec:
        print("Computing Node2Vec...")
        try:
            from node2vec import Node2Vec
            # Thêm parameter workers=args.workers
            n2v = Node2Vec(G, dimensions=args.node2vec_dim, walk_length=10, num_walks=10, 
                           workers=args.workers, quiet=False) 
            model = n2v.fit(window=5, min_count=1, batch_words=4)
            vectors = np.array([model.wv[str(n)] for n in nodes])
        except ImportError:
            print("❌ Module 'node2vec' not found.")
            vectors = np.zeros((len(nodes), args.node2vec_dim))

    # Save
    print(f"Saving to {args.out_path}...")
    save_dict = {k: np.array(v) for k, v in features.items()}
    if args.compute_node2vec and 'vectors' in locals():
        save_dict['node2vec'] = vectors
        
    np.savez(args.out_path, **save_dict)
    
    with open(args.meta_path, 'w') as f:
        json.dump({"nodes": len(nodes), "features": list(features.keys())}, f)

if __name__ == "__main__":
    main()