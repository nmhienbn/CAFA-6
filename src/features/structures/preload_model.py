from transformers import AutoTokenizer, EsmForProteinFolding
import torch

print("Downloading model and tokenizer to HuggingFace cache...")

# 1. Tải Tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

# 2. Tải Model
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

print("Done! Model cached successfully.")