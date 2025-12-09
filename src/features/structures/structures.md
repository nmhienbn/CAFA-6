# 1. Chuẩn bị thư mục
Giả sử layout:
```text
cafa6/
  data/
    Train/train_sequences.fasta
    Test/testsuperset.fasta
    afdb/
      db/
      prostt5/
      tmp/
      out/
  features/
    foldseek/
  struct_feats/
  scripts/

```

Tạo nhanh:

```
mkdir -p data/{afdb/{db,prostt5,tmp,out},}
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
foldseek databases Alphafold/Swiss-Prot \
  data/afdb/db/afdb_sp \
  data/afdb/tmp \
  --threads 32 \
  --remove-tmp-files 1

foldseek databases Alphafold/Proteome \
  data/afdb/db/afdb_proteome \
  data/afdb/tmp \
  --threads 32 \
  --remove-tmp-files 1

```

### Tải ProstT5 weights (để search từ FASTA)

FoldSeek hỗ trợ structure-based sequence search bằng ProstT5, nhanh hơn ColabFold 400–4000x.
```
mkdir -p data/afdb/prostt5
mkdir -p data/afdb/tmp/prostt5_dl

foldseek databases ProstT5 \
  data/afdb/prostt5/weights \
  data/afdb/tmp/prostt5_dl

ls -lh data/afdb/prostt5/weights
```

# 3. Chạy FoldSeek easy-search cho CAFA6 (FASTA → AFDB)
## Train vs AFDB

```bash
CUDA_VISIBLE_DEVICES=2 \  # nếu build GPU; nếu build CPU-only thì bỏ dòng này
foldseek easy-search \
  data/raw/cafa6/Train/train_sequences.fasta \
  data/afdb/db/afdb_sp \
  data/afdb/out/train_vs_afdb_sp.m8 \
  data/afdb/tmp/train_vs_afdb_sp \
  --prostt5-model data/afdb/prostt5/weights \
  -e 1e-3 \
  --max-seqs 200 \
  --threads 32 \
  --gpu 1   # chỉ giữ nếu bạn build được CUDA; nếu build CPU-only thì bỏ


```

## Test vs AFDB
```bash
CUDA_VISIBLE_DEVICES=3 \  # nếu build GPU; nếu build CPU-only thì bỏ dòng này
foldseek easy-search \
  data/raw/cafa6/Test/testsuperset.fasta \
  data/afdb/db/afdb_sp \
  data/afdb/out/test_vs_afdb_sp.m8 \
  data/afdb/tmp/test_vs_afdb_sp \
  --prostt5-model data/afdb/prostt5/weights \
  -e 1e-3 \
  --max-seqs 200 \
  --threads 32
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
cd /data/hien
git clone https://github.com/Daniel-Liu-c0deb0t/block-aligner.git
cd block-aligner/c
cargo build --release --features simd_avx2

export CONDA_PREFIX=/data/hien/.conda/envs/cafa
cp target/release/libblock_aligner_c.a $CONDA_PREFIX/lib/
cd $CONDA_PREFIX/lib
ln -sf libblock_aligner_c.a libblock-aligner-c.a

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
cd /data/hien/foldseek
rm -rf build
mkdir build && cd build



export CUDA_ROOT=/data/hien/.conda/envs/cafa
export CUDA_PATH=$CUDA_ROOT
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT

export LD_LIBRARY_PATH=$CUDA_ROOT/lib:$CUDA_ROOT/lib64:$LD_LIBRARY_PATH


cmake  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_POLICY_DEFAULT_CMP0079=NEW \
  -DCMAKE_POLICY_DEFAULT_CMP0074=NEW \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT \
  -DCMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib \
  -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc \
  -DRust_CUDA_ROOT=$CUDA_ROOT \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=1 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-11.7 ..




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