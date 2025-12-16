#!/bin/bash
PYTHON_EXEC="/data/hien/.conda/envs/cafa/bin/python"
INPUT_FASTA="data/raw/cafa6/Train/train_sequences.fasta"
OUTPUT_DIR="data/processed/pdb/train_structures_pdb"
ESMFOLD_DIR="src/features/structures/run_esmfold.py"
PRELOAD_DIR="src/features/structures/preload_model.py"
SPLIT_DIR="data/processed/splitted_fasta"
# INPUT_FASTA="data/raw/cafa6/Test/testsuperset.fasta"
# OUTPUT_DIR="data/processed/pdb/test_structures_pdb"
NUM_GPUS=8

mkdir -p $OUTPUT_DIR
mkdir -p splitted_fasta

# 1. Tính số dòng để chia đều
TOTAL_LINES=$(wc -l < $INPUT_FASTA)
LINES_PER_FILE=$(( ($TOTAL_LINES + $NUM_GPUS - 1) / $NUM_GPUS ))

# 2. Chia file fasta (đảm bảo không cắt giữa chừng sequence - dùng split theo line không an toàn lắm, nhưng nếu fasta chuẩn 2 dòng/seq thì ok. Tốt nhất dùng tool chia chuyên dụng, ở đây giả sử fasta đơn giản)
# Cách an toàn hơn dùng awk để chia theo record `>`:
awk -v n=$NUM_GPUS -v dir=$SPLIT_DIR '/^>/{f=dir"/part_"++i%n".fasta"} {print > f}' $INPUT_FASTA

# 3. Chạy Loop song song
echo "Starting inference on $NUM_GPUS GPUs..."
$PYTHON_EXEC $PRELOAD_DIR

for i in $(seq 0 $(($NUM_GPUS - 1))); do
    PART_FILE="${SPLIT_DIR}/part_${i}.fasta"
    echo "Launching GPU $i for file $PART_FILE"
    
    # Chạy background (&)
    $PYTHON_EXEC $ESMFOLD_DIR --fasta $PART_FILE --out $OUTPUT_DIR --gpu $i > $OUTPUT_DIR/log_gpu_$i.txt 2>&1 &
done

wait
echo "All predictions completed!"