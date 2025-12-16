#!/bin/bash

# --- CẤU HÌNH ĐƯỜNG DẪN (SỬA LẠI NẾU CẦN) ---
PYTHON_EXEC="/data/hien/.conda/envs/cafa/bin/python"
SRC_PATH="src/features/extract_single_model.py"
TRAIN_PATH="data/raw/cafa6/Train/train_sequences.fasta"
TEST_PATH="data/raw/cafa6/Test/testsuperset.fasta"
SAVE_DIR="features"

# Lưu ý chung: ĐÃ BỎ --use_fp16 để tránh lỗi NaN
# Batch size để 16 cho an toàn với bộ nhớ (FP32 tốn VRAM hơn)

echo "Bắt đầu chạy lại các model bị lỗi NaN..."

# # ==============================================================================
# # GPU 0: Ankh Large - TEST (220k seqs)
# # ==============================================================================
# CUDA_VISIBLE_DEVICES=0 python $SRC_PATH \
#   --model_name ElnaggarLab/ankh-large \
#   --short_name ankh_large \
#   --fasta_path $TEST_PATH \
#   --save_dir $SAVE_DIR \
#   --device cuda:0 \
#   --batch_size 16 \
#   --is_ankh &

# # ==============================================================================
# # GPU 1: Ankh Large - TRAIN (80k seqs)
# # ==============================================================================
# CUDA_VISIBLE_DEVICES=1 python $SRC_PATH \
#   --model_name ElnaggarLab/ankh-large \
#   --short_name ankh_large \
#   --fasta_path $TRAIN_PATH \
#   --save_dir $SAVE_DIR \
#   --device cuda:0 \
#   --batch_size 16 \
#   --is_ankh &

# ==============================================================================
# GPU 2: Ankh3 Large - TEST (220k seqs)
# ==============================================================================
CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC $SRC_PATH \
  --model_name ElnaggarLab/ankh3-large \
  --short_name ankh3_large \
  --fasta_path $TEST_PATH \
  --save_dir $SAVE_DIR \
  --device cuda:0 \
  --batch_size 16 \
  --is_ankh &

# ==============================================================================
# GPU 3: Ankh3 Large - TRAIN (80k seqs)
# ==============================================================================
CUDA_VISIBLE_DEVICES=1 $PYTHON_EXEC $SRC_PATH \
  --model_name ElnaggarLab/ankh3-large \
  --short_name ankh3_large \
  --fasta_path $TRAIN_PATH \
  --save_dir $SAVE_DIR \
  --device cuda:0 \
  --batch_size 16 \
  --is_ankh &

# ==============================================================================
# GPU 4: ProtT5 XL - TEST (220k seqs)
# ==============================================================================
CUDA_VISIBLE_DEVICES=2 $PYTHON_EXEC $SRC_PATH \
  --model_name Rostlab/prot_t5_xl_uniref50 \
  --short_name protT5_xl \
  --fasta_path $TEST_PATH \
  --save_dir $SAVE_DIR \
  --device cuda:0 \
  --batch_size 16 \
  --is_prott5 &

# ==============================================================================
# GPU 5: ProtT5 XL - TRAIN (80k seqs)
# ==============================================================================
CUDA_VISIBLE_DEVICES=4 $PYTHON_EXEC $SRC_PATH \
  --model_name Rostlab/prot_t5_xl_uniref50 \
  --short_name protT5_xl \
  --fasta_path $TRAIN_PATH \
  --save_dir $SAVE_DIR \
  --device cuda:0 \
  --batch_size 16 \
  --is_prott5 &

# GPU 6 & 7: Rảnh (Hoặc dùng nếu bạn muốn chạy lại ProtBERT cho chắc chắn)

wait
echo "DONE FIXING ERROR MODELS"