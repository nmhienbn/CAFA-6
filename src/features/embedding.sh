#!/bin/bash

PYTHON_EXEC="/data/hien/.conda/envs/cafa/bin/python"
SRC_PATH="src/features/extract_single_model.py"
TRAIN_PATH="data/raw/cafa6/Train/train_sequences.fasta"
TEST_PATH="data/raw/cafa6/Test/testsuperset.fasta"
SAVE_DIR="features"

echo "Bắt đầu chạy Embeddings trên 8 GPU A100..."

# ==============================================================================
# GPU 0: ESM2 15B (TEST) - Nặng nhất
# ==============================================================================
CUDA_VISIBLE_DEVICES=0 python $SRC_PATH \
  --model_name facebook/esm2_t48_15B_UR50D \
  --short_name esm2_15B \
  --fasta_path $TEST_PATH \
  --save_dir $SAVE_DIR \
  --device cuda:0 \
  --batch_size 4 \
  --use_fp16 \
  --max_len 1024 &

# ==============================================================================
# GPU 1: ESM2 15B (TRAIN) - Nặng nhì
# ==============================================================================
CUDA_VISIBLE_DEVICES=1 python $SRC_PATH \
  --model_name facebook/esm2_t48_15B_UR50D \
  --short_name esm2_15B \
  --fasta_path $TRAIN_PATH \
  --save_dir $SAVE_DIR \
  --device cuda:0 \
  --batch_size 4 \
  --use_fp16 \
  --max_len 1024 &

# ==============================================================================
# GPU 2: ProtT5 XL (TEST) - KHÔNG DÙNG FP16 (Fix NaN)
# ==============================================================================
CUDA_VISIBLE_DEVICES=2 python $SRC_PATH \
  --model_name Rostlab/prot_t5_xl_uniref50 \
  --short_name protT5_xl \
  --fasta_path $TEST_PATH \
  --save_dir $SAVE_DIR \
  --device cuda:0 \
  --batch_size 16 \
  --is_prott5 & 
# Lưu ý: Batch size giảm vì chạy FP32 tốn VRAM.

# ==============================================================================
# GPU 3: Ankh Large (TEST) - KHÔNG DÙNG FP16 (Fix NaN)
# ==============================================================================
CUDA_VISIBLE_DEVICES=3 python $SRC_PATH \
  --model_name ElnaggarLab/ankh-large \
  --short_name ankh_large \
  --fasta_path $TEST_PATH \
  --save_dir $SAVE_DIR \
  --device cuda:0 \
  --batch_size 16 \
  --is_ankh &

# ==============================================================================
# GPU 4: Ankh3 Large (TEST) - KHÔNG DÙNG FP16 (Fix NaN)
# ==============================================================================
CUDA_VISIBLE_DEVICES=4 python $SRC_PATH \
  --model_name ElnaggarLab/ankh3-large \
  --short_name ankh3_large \
  --fasta_path $TEST_PATH \
  --save_dir $SAVE_DIR \
  --device cuda:0 \
  --batch_size 16 \
  --is_ankh &

# ==============================================================================
# GPU 5: GOM CÁC FILE TRAIN CỦA T5/ANKH (Chạy tuần tự - KHÔNG FP16)
# ==============================================================================
CUDA_VISIBLE_DEVICES=5 bash -c "
python $SRC_PATH --model_name Rostlab/prot_t5_xl_uniref50 --short_name protT5_xl --fasta_path $TRAIN_PATH --save_dir $SAVE_DIR --device cuda:0 --batch_size 16 --is_prott5 ;
python $SRC_PATH --model_name ElnaggarLab/ankh-large      --short_name ankh_large --fasta_path $TRAIN_PATH --save_dir $SAVE_DIR --device cuda:0 --batch_size 16 --is_ankh ;
python $SRC_PATH --model_name ElnaggarLab/ankh3-large     --short_name ankh3_large --fasta_path $TRAIN_PATH --save_dir $SAVE_DIR --device cuda:0 --batch_size 16 --is_ankh
" &

# ==============================================================================
# GPU 6: ESM2 3B + ESM1b 650M (Cả Train & Test) - DÙNG FP16
# ==============================================================================
CUDA_VISIBLE_DEVICES=6 bash -c "
python $SRC_PATH --model_name facebook/esm2_t36_3B_UR50D    --short_name esm2_3B    --fasta_path $TRAIN_PATH --save_dir $SAVE_DIR --device cuda:0 --batch_size 64 --use_fp16 ;
python $SRC_PATH --model_name facebook/esm2_t36_3B_UR50D    --short_name esm2_3B    --fasta_path $TEST_PATH  --save_dir $SAVE_DIR --device cuda:0 --batch_size 64 --use_fp16 ;
python $SRC_PATH --model_name facebook/esm1b_t33_650M_UR50S --short_name esm1b_650M --fasta_path $TRAIN_PATH --save_dir $SAVE_DIR --device cuda:0 --batch_size 64 --use_fp16 ;
python $SRC_PATH --model_name facebook/esm1b_t33_650M_UR50S --short_name esm1b_650M --fasta_path $TEST_PATH  --save_dir $SAVE_DIR --device cuda:0 --batch_size 64 --use_fp16
" &

# ==============================================================================
# GPU 7: ESM2 650M + ProtBERT (Cả Train & Test) - DÙNG FP16
# ==============================================================================
CUDA_VISIBLE_DEVICES=7 bash -c "
python $SRC_PATH --model_name facebook/esm2_t33_650M_UR50D --short_name esm2_650M --fasta_path $TRAIN_PATH --save_dir $SAVE_DIR --device cuda:0 --batch_size 64 --use_fp16 ;
python $SRC_PATH --model_name facebook/esm2_t33_650M_UR50D --short_name esm2_650M --fasta_path $TEST_PATH  --save_dir $SAVE_DIR --device cuda:0 --batch_size 64 --use_fp16 ;
python $SRC_PATH --model_name Rostlab/prot_bert_bfd        --short_name protBERT   --fasta_path $TRAIN_PATH --save_dir $SAVE_DIR --device cuda:0 --batch_size 64 --use_fp16 --is_protbert ;
python $SRC_PATH --model_name Rostlab/prot_bert_bfd        --short_name protBERT   --fasta_path $TEST_PATH  --save_dir $SAVE_DIR --device cuda:0 --batch_size 64 --use_fp16 --is_protbert
" &

wait
echo "HOÀN TẤT TOÀN BỘ EMBEDDING!"