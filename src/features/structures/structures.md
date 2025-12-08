# 1. Chuẩn bị thư mục
Giả sử layout:
```text
cafa6/
  data/
    Train/train_sequences.fasta
    Test/testsuperset.fasta
  afdb/
    db/         # AFDB FoldSeek database
    prostt5/    # ProstT5 weights
    tmp/        # tmp folder cho FoldSeek
    out/        # m8 output
  struct_feats/
    # sẽ chứa .npy features
  scripts/
```

Tạo nhanh:

```
mkdir -p data/{afdb/{db,prostt5,tmp,out},struct_feats,scripts}
```

# 2. Cài FoldSeek (GPU build)

```text
cd data

# GPU build (Ampere trở lên)
wget https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz
tar xvfz foldseek-linux-gpu.tar.gz
export PATH="$(pwd)/data/structures/foldseek/bin:$PATH"
source ~/.bashrc

# Test
foldseek -h
```

FoldSeek hỗ trợ:
- Alphafold/UniProt – full 214M entries (~700 GB download, ~950 GB RAM DB).
- Alphafold/UniProt50 – cluster 50% identity, AFDB50 (~54M entries, ~151 GB RAM nếu giữ Cα).
- Alphafold/Swiss-Prot – 550k entries, nhỏ, curated.

### Tải AFDB50
```
foldseek databases Alphafold/Swiss-Prot afdb/db/afdb50 afdb/tmp --threads 32
```

### Tải ProstT5 weights (để search từ FASTA)

FoldSeek hỗ trợ structure-based sequence search bằng ProstT5, nhanh hơn ColabFold 400–4000x.
```
mkdir -p data/structures/afdb/prostt5
mkdir -p data/structures/afdb/tmp/prostt5_dl

foldseek databases ProstT5 \
  data/structures/afdb/prostt5/weights \
  data/structures/afdb/tmp/prostt5_dl

ls -lh data/structures/afdb/prostt5/weights
```

# 3. Chạy FoldSeek easy-search cho CAFA6 (FASTA → AFDB)
## Train vs AFDB

```bash
cd cafa6

foldseek createdb \
  data/raw/cafa6/Train/train_sequences.fasta \
  data/structures/afdb/db/train_prostt5 \
  --prostt5-model data/structures/afdb/prostt5/weights \
  --threads 32
  # chỉ thêm --gpu 1 nếu bạn CHẮC đang chạy trên node có GPU hỗ trợ ProstT5

ls -lh data/structures/afdb/db/train_prostt5*

foldseek easy-search \
  data/structures/afdb/db/train_prostt5 \
  data/structures/afdb/db/afdb50 \
  data/structures/afdb/out/train_vs_afdb50.m8 \
  data/structures/afdb/tmp/train_vs_afdb50 \
  -e 1e-3 \
  --max-seqs 200 \
  --threads 32 \
  --gpu 1

```

## Test vs AFDB
```bash
foldseek createdb \
  data/raw/cafa6/Test/testsuperset.fasta \
  data/structures/afdb/db/test_prostt5 \
  --prostt5-model data/structures/afdb/prostt5/weights \
  --threads 32
  
foldseek easy-search \
  data/structures/afdb/db/test_prostt5 \
  data/structures/afdb/db/afdb50 \
  data/structures/afdb/out/testsuperset_vs_afdb50.m8 \
  data/structures/afdb/tmp/testsuperset_vs_afdb50 \
  -e 1e-3 \
  --max-seqs 200 \
  --threads 32 \
  --gpu 2
```

# Run scripts
```bash
python src/features/extract_foldseek_features.py \
  --foldseek_tsv data/structures/afdb/out/train_vs_afdb50.m8 \
  --fasta_path   data/raw/cafa6/Train/train_sequences.fasta \
  --save_dir     features/foldseek \
  --short_name   afdb50 \
  --top_k "1,3,5,10" \
  --max_hits_per_query 200

python src/features/extract_foldseek_features.py \
  --foldseek_tsv data/structures/afdb/out/testsuperset_vs_afdb50.m8 \
  --fasta_path   data/raw/cafa6/Test/testsuperset.fasta \
  --save_dir     features/foldseek \
  --short_name   afdb50 \
  --top_k "1,3,5,10" \
  --max_hits_per_query 200
```


# Install FoldSeek
```text
# 0. Tạo env build riêng
conda create -n foldseek-gpu-build -c conda-forge cmake make gcc gxx -y
conda activate foldseek-gpu-build
conda install -c conda-forge rust -y
conda install -c nvidia cuda-nvcc=11.7 cuda-toolkit=11.7 -y


# 1. Load CUDA 11.7 do cluster cung cấp (nếu dùng modules)
module load cuda/11.7   # hoặc tên module tương đương
# Sau đó thường sẽ có biến $CUDA_HOME hoặc /usr/local/cuda-11.7

# 2. Lấy source
git clone https://github.com/steineggerlab/foldseek.git
cd foldseek
mkdir build && cd build

# 3. CMake với CUDA 11.7
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=. \
      -DENABLE_CUDA=1 \
      -DCUDAToolkit_ROOT=/usr/local/cuda-11.7 \
      -DCMAKE_CUDA_ARCHITECTURES="80-real" \
      -DMMSEQS_USE_DPX=OFF ..
      
# 4. Build & install
make -j16
make install

# 5. Ưu tiên binary mới build
export PATH=$(pwd)/bin:$PATH
which foldseek    # phải ra .../foldseek/build/bin/foldseek
```