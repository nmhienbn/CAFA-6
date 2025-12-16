# Tóm tắt

1. Lấy PPI từ STRING → chuẩn hóa thành `ppi_edges.tsv`.
2. Từ `ppi_edges.tsv` → sinh **PPI features** (`ppi_features.npz`).
3. Từ PPI + GO train + species info → **Net-KNN + top 15 species** (`netknn_top15.tsv`).

# 0. File sử dụng

```text
CAFA-6/
  data/
    raw/
      cafa6/                    # Kaggle
      external/
        string/
          v12/
            protein.links.full/ # toàn bộ file *.protein.links.full.v12.0.txt(.gz)
    processed/
      mapping/
        string_to_protein_id.tsv   # STRING id -> CAFA protein_id
        protein_taxon.tsv          # protein_id, taxon_id
        train_go_annotations.tsv   # protein_id, go_id, label
        all_proteins.txt           # 1 protein_id / dòng (train + test)
        species_annot_stats.tsv    # sẽ build bằng script
      ppi/
        ppi_edges_string_all.tsv   # gộp 2000+ species (STRING id)
        ppi_edges_cafa.tsv         # đã remap sang protein_id CAFA
  features/
    ppi/
      ppi_features.npz
      ppi_features_meta.json
  outputs/
    preds/
      netknn_top15.tsv
      netknn_top15_meta.json
  src/
    features/
      ppi/
        prepare_string_ppi_all.py
        map_ppi_ids.py
        build_species_annot_stats.py
        build_ppi_features.py
        net_knn_top15.py

```

---


## Sinh file phụ trợ
Tạo ra all_proteins.txt, protein_taxon.tsv, train_go_annotations.tsv từ dữ liệu Kaggle.
```
python src/features/ppi/build_cafa_mappings.py \
  --train_fasta data/raw/cafa6/Train/train_sequences.fasta \
  --test_fasta data/raw/cafa6/Test/testsupertest.fasta \
  --train_terms data/raw/cafa6/Train/train_terms.tsv \
  --train_taxa data/raw/cafa6/Train/train_taxonomy.tsv \
  --out_dir data/processed/mapping
```

## 1. Download protein alias
Bạn cần tải file protein.aliases.v12.0.txt.gz của STRING. Sau đó chạy lệnh grep hoặc zgrep để lọc ra các dòng chứa ID UniProt (thường là source UniProt_AC hoặc BLAST_UniProt_AC).

```bash
wget https://stringdb-downloads.org/download/protein.aliases.v12.0.txt.gz
zgrep "UniProt_AC" protein.aliases.v12.0.txt.gz | cut -f1,2 > data/processed/mapping/string_to_protein_id.tsv
```


Hoặc nếu máy không nhiều bộ nhớ:
```bash
python src/features/ppi/fetch_string_mapping.py \
  --input_proteins data/processed/mapping/all_proteins.txt \
  --out_path data/processed/mapping/string_to_protein_id.tsv

# Merge các file
python src/features/ppi/prepare_string_ppi_all.py \
  --input_dir data/raw/external/string/v12/protein.links.full.cafa \
  --pattern "*.protein.links.full.v12.0.txt.gz" \
  --score_col combined_score \
  --min_score 400 \
  --out_path data/processed/ppi/ppi_edges_string_all.tsv
```

## 2. Remap STRING id -> protein_id (CAFA)


```bash
python src/features/ppi/map_ppi_ids.py \
  --ppi_in data/processed/ppi/ppi_edges_string_all.tsv \
  --mapping_path data/processed/mapping/string_to_protein_id.tsv \
  --ppi_out data/processed/ppi/ppi_edges_cafa.tsv \
  --ppi_protein1_col protein1 \
  --ppi_protein2_col protein2 \
  --ppi_weight_col weight \
  --map_src_col string_id \
  --map_tgt_col protein_id \
  --drop_self \
  --agg_method max
```

# 3. Build species_annot_stats
Đếm số lượng GO terms được gán cho mỗi loài, giúp xác định loài nào có dữ liệu huấn luyện tốt nhất.
```bash
python src/features/ppi/build_species_annot_stats.py \
  --annotation_path data/processed/mapping/train_go_annotations.tsv \
  --protein_taxon_path data/processed/mapping/protein_taxon.tsv \
  --out_path data/processed/mapping/species_annot_stats.tsv \
  --ann_protein_col protein_id \
  --ann_go_col go_id \
  --ann_label_col label
```

# 4. PPI features
Tính toán đặc trưng đồ thị, node2vec...
```bash
pip install node2vec
python src/features/ppi/build_ppi_features.py \
  --ppi_path data/processed/ppi/ppi_edges_cafa.tsv \
  --out_path features/ppi/ppi_features.npz \
  --meta_path features/ppi/ppi_features_meta.json \
  --protein1_col protein1 \
  --protein2_col protein2 \
  --weight_col weight \
  --compute_closeness \
  --compute_betweenness --betweenness_k 100 \
  --compute_node2vec --node2vec_dim 64

or 
pip install node2vec
python src/features/ppi/build_ppi_features_fast.py \
  --ppi_path data/processed/ppi/ppi_edges_cafa.tsv \
  --out_path features/ppi/ppi_features.npz \
  --meta_path features/ppi/ppi_features_meta.json \
  --weight_col weight \
  --compute_betweenness --betweenness_k 200 \
  --compute_node2vec --node2vec_dim 64 \
  --workers 12

or 
pip install pecanpy
python src/features/ppi/build_ppi_features_pecanpy.py \
  --ppi_path data/processed/ppi/ppi_edges_cafa.tsv \
  --out_path features/ppi/ppi_features.npz \
  --meta_path features/ppi/ppi_features_meta.json \
  --protein1_col protein1 \
  --protein2_col protein2 \
  --weight_col weight \
  --compute_closeness \
  --compute_betweenness --betweenness_k 100 \
  --compute_node2vec --node2vec_dim 64

or
conda create -n rapids_env -c conda-forge python=3.10 cudatoolkit=11.8 -y
conda activate rapids_env
python -m pip install cugraph-cu11==23.04.* cudf-cu11==23.04.* --extra-index-url https://pypi.nvidia.com
python -m pip install gensim pandas scipy

python src/features/ppi/build_ppi_features_gpu.py \
  --ppi_path data/processed/ppi/ppi_edges_cafa.tsv \
  --out_path features/ppi/ppi_features.npz \
  --meta_path features/ppi/ppi_features_meta.json \
  --weight_col weight \
  --compute_betweenness --betweenness_k 100 \
  --compute_node2vec --node2vec_dim 64 \
  --compute_pagerank
```

# 5. Net-KNN + top-15 species
Thuật toán lan truyền nhãn (Label Propagation) đơn giản dựa trên hàng xóm có trọng số (Weighted KNN).
```bash
python src/features/ppi/net_knn_top15.py \
  --ppi_path data/processed/ppi/ppi_edges_cafa.tsv \
  --annotation_path data/processed/mapping/train_go_annotations.tsv \
  --protein_taxon_path data/processed/mapping/protein_taxon.tsv \
  --species_annot_path data/processed/mapping/species_annot_stats.tsv \
  --target_ids_path data/processed/mapping/all_proteins.txt \
  --out_path features/ppi/netknn_top15.tsv \
  --meta_path features/ppi/netknn_top15_meta.json \
  --topk_species 15 \
  --k_neighbors 50 \
  --min_edge_weight 400 \
  --normalize_scores
```
