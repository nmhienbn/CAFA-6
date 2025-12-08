# Tóm tắt

1. Lấy PPI từ STRING → chuẩn hóa thành `ppi_edges.tsv`.
2. Từ `ppi_edges.tsv` → sinh **PPI features** (`ppi_features.npz`).
3. Từ PPI + GO train + species info → **Net-KNN + top 15 species** (`netknn_top15.tsv`).

Tôi cho bạn **3 script chính + 1 script phụ build species_annot**:

* `prepare_ppi_from_string.py`
* `build_ppi_features.py`
* `build_species_annot_stats.py` (để ra `species_annot_stats.tsv`)
* `net_knn_top15.py`

---

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

Giả định: **ID trong PPI đã cùng namespace với `protein_id` trong CAFA** (UniProt, hoặc gì đó bạn chọn). Nếu bạn dùng STRING `9606.ENSP...` thì tự remap sang CAFA id trước.

---

## 1. Download - TODO: Remove 0B files.
```bash
cd cafa6

mkdir -p data/raw/external/string/v12/protein.links.full.cafa

python src/features/ppi/download_string_ppi_cafa_species.py \
  --protein_taxon_path features/taxonomy/protein_taxon.tsv \
  --top_k_taxa 50 \
  --out_dir data/raw/external/string/v12/protein.links.full.cafa
```

## 2. Merge
```
python src/features/ppi/prepare_string_ppi_all.py \
  --input_dir data/raw/external/string/v12/protein.links.full.cafa \
  --pattern "*.protein.links.full.v12.0.txt.gz" \
  --score_col combined_score \
  --min_score 400 \
  --out_path data/processed/ppi/ppi_edges_string_all.tsv
```

## 2 Remap STRING id -> protein_id (CAFA)
```
python src/features/ppi/map_ppi_ids.py \
  --ppi_in data/processed/ppi/ppi_edges_string_all.tsv \
  --ppi_protein1_col protein1 \
  --ppi_protein2_col protein2 \
  --ppi_weight_col weight \
  --mapping_path data/processed/mapping/string_to_protein_id.tsv \
  --map_src_col string_id \
  --map_tgt_col protein_id \
  --drop_self \
  --agg_method max \
  --ppi_out data/processed/ppi/ppi_edges_cafa.tsv
```

# 2.2 Build species_annot_stats
python src/features/ppi/build_species_annot_stats.py \
  --annotation_path data/processed/mapping/train_go_annotations.tsv \
  --protein_taxon_path data/processed/mapping/protein_taxon.tsv \
  --ann_protein_col protein_id \
  --ann_go_col go_id \
  --ann_label_col label \
  --ann_label_threshold 0.5 \
  --out_path data/processed/mapping/species_annot_stats.tsv

# 2.3 PPI features
python src/features/ppi/build_ppi_features.py \
  --ppi_path data/processed/ppi/ppi_edges_cafa.tsv \
  --protein1_col protein1 \
  --protein2_col protein2 \
  --weight_col weight \
  --target_ids_path data/processed/mapping/all_proteins.txt \
  --protein_taxon_path data/processed/mapping/protein_taxon.tsv \
  --species_annot_path data/processed/mapping/species_annot_stats.tsv \
  --topk_species 15 \
  --compute_closeness \
  --compute_betweenness --betweenness_k 256 \
  --compute_node2vec \
  --node2vec_dim 128 \
  --out_path features/ppi/ppi_features.npz \
  --meta_path features/ppi/ppi_features_meta.json

# 2.4 Net-KNN + top-15 species
python src/features/ppi/net_knn_top15.py \
  --ppi_path data/processed/ppi/ppi_edges_cafa.tsv \
  --protein1_col protein1 \
  --protein2_col protein2 \
  --weight_col weight \
  --annotation_path data/processed/mapping/train_go_annotations.tsv \
  --ann_protein_col protein_id \
  --ann_go_col go_id \
  --ann_label_col label \
  --ann_label_threshold 0.5 \
  --target_ids_path data/processed/mapping/all_proteins.txt \
  --protein_taxon_path data/processed/mapping/protein_taxon.tsv \
  --species_annot_path data/processed/mapping/species_annot_stats.tsv \
  --topk_species 15 \
  --k_neighbors 50 \
  --min_edge_weight 0.0 \
  --normalize_scores \
  --out_path outputs/preds/netknn_top15.tsv \
  --meta_path outputs/preds/netknn_top15_meta.json
