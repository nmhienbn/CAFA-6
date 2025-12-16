Repository pipeline cho CAFA 6 (Critical Assessment of Functional Annotation)

## 1. Cấu trúc Repository
Tổng quan về các thư mục chính trong dự án:

```
CAFA-6
├── configs/                     : Các file cấu hình .yaml (model params, data paths)
├── features/                    : Nơi chứa output features sau khi extract (embeddings .npy, ids)
├── notebooks/                   : Các notebook test model cơ bản
├── U900/                        : Model U900, Chi tiết chạy full Pipeline.ipynb
├── src/                         : Mã nguồn chính của dự án
│   ├── data/                    : Xử lý dữ liệu thô, Dataset class, split data
│   ├── eval/                    : Metric đánh giá (CAFA F-score) và tìm threshold tối ưu
│   ├── models/                  : Định nghĩa kiến trúc model (MLP, Ensemble, Post-process)
│   ├── runners/                 : Script huấn luyện (train), suy luận (infer), tạo submission
│   └── features/                : Các script trích xuất đặc trưng (Features Extraction)
│       ├── ppi/                 : Protein-Protein Interaction (STRING, kNN)
│       ├── structures/          : Xử lý cấu trúc 3D (FoldSeek, AlphaFold, BLAST)
│       ├── embedding.sh         : Chạy embedding trên GPU
│       ├── extract_single_model.py : Core script chạy infer embedding
│       └── taxonomy*_embedding.py  : Taxonomy Embedding
└── requirements.txt             : Các thư viện cần thiết
```

---

## 2. Hướng dẫn chạy Embedding pLM
Đây là phần feature quan trọng nhất trong dự án.

Chạy để chạy song song.

```bash
bash src/features/embedding.sh
```

Yêu cầu máy chạy:
```
- GPU: 8 x A100 80GB
- CPU: AMD 2TB, 256 cores
- Disk Storage: 2TB (At least 300GB for this project)
```
Kết quả chạy: 
https://drive.google.com/drive/u/0/folders/1cEiFOeUMki87HMmWcs5JGW761xF-bmbJ


## 3. Chạy model
- Chạy [Notebook code.ipynb](code.ipynb) trên môi trường Google Colab, với các embedding tương ứng như trong link google drive.
- Chạy [Notebook refine](refine.ipynb) ở local, chú ý cài đặt các môi trường.
- Chạy [Notebook remove negative](goa-negative-propagation.ipynb) ở local, chú ý cài đặt các môi trường.

## Phụ lục
- [CAFA 6](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
- [CAFA 5](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction)
- [GOA](https://geneontology.org/docs/guide-go-evidence-codes/)