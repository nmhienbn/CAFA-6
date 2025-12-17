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

Yêu cầu máy chạy:
```
- GPU: 8 x A100 80GB
- CPU: AMD 2TB, 256 cores
- Disk Storage: 2TB (At least 300GB for this project)
```

Yêu cầu môi trường
```
conda create -n lastdance python=3.9 -y
conda activate lastdance
conda install -c conda-forge taxonkit -y
pip install -r requirements.txt
```

Chạy để chạy song song.

```bash
bash src/features/embedding.sh
```
Kết quả chạy: 
https://drive.google.com/drive/u/0/folders/1cEiFOeUMki87HMmWcs5JGW761xF-bmbJ

Note chia sẻ: Nếu muốn chạy FoldSeek trên các phiên bản cuda cũ, ví dụ như CUDA 11.7, sử dụng clone project [https://github.com/steineggerlab/foldseek](https://github.com/steineggerlab/foldseek). Cài cudatoolkit=11.7 và nvcc trong channel nvidia của conda. Thêm flag ` -DCMAKE_CUDA_ARCHITECTURES=native` khi chạy cmake

## 3. Chạy model
- Chạy [Notebook code.ipynb](code.ipynb) trên môi trường Google Colab, với các embedding tương ứng như trong link google drive.
- Chạy [Notebook refine](refine.ipynb) ở local, chú ý cài đặt các môi trường.
- Chạy [Notebook remove negative](goa-negative-propagation.ipynb) ở local, chú ý cài đặt các môi trường - Nó là file goa từ version 227 từ UniProt. Nếu cần lặp lại kết quả thì lấy file gaf từ [Notebook refine](refine.ipynb) và lọc thông tin cần thiết.

## 4. Kết quả
- [Báo cáo PDF](report/CAFA6_ML.pdf)

## Phụ lục
- [CAFA 6](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
- [CAFA 5](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction)
- [GOA](https://geneontology.org/docs/guide-go-evidence-codes/)
