#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')

# !pip install -q transformers biopython kaggle
# print("✅ Đã cài đặt xong thư viện!")


# In[ ]:


# import os
# from google.colab import files

# # Upload file kaggle.json
# print("Vui lòng upload file kaggle.json của bạn:")
# files.upload()

# # Cấu hình Kaggle API
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# # Tải dữ liệu cuộc thi (Sẽ mất khoảng 1-2 phút)
# print("⏳ Đang tải dữ liệu CAFA 6...")
# !kaggle competitions download -c cafa-6-protein-function-prediction
# !unzip -q cafa-6-protein-function-prediction.zip -d /content/cafa6_data
# print("✅ Đã tải và giải nén dữ liệu tại /content/cafa6_data")


# In[ ]:


import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import gc
import os
import sys
from tqdm.auto import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[ ]:


with open("configs/base_model.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

if CONFIG.get("DEVICE", "auto") == "auto":
    CONFIG["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)
print(f"🚀 CAFA 6 - DUAL MODEL (ANKH + ESM) | Device: {CONFIG['DEVICE']}")

torch.manual_seed(CONFIG['SEED'])
np.random.seed(CONFIG['SEED'])


# # 1. MEMORY-SAFE DATASET
# CHÌA KHÓA ĐỂ KHÔNG TRÀN RAM

# In[ ]:


class MultiSourceDataset(Dataset):
    def __init__(self, embedding_paths_dict, y_tensor=None, indices=None):
        self.mmaps = {}
        self.keys = list(embedding_paths_dict.keys())

        # Load mmap
        for name, path in embedding_paths_dict.items():
            self.mmaps[name] = np.load(path, mmap_mode='r')

        # Base length
        first_key = self.keys[0]
        self.total_len = len(self.mmaps[first_key])

        self.indices = indices if indices is not None else np.arange(self.total_len)
        self.y = y_tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        inputs = []
        # Load từng vector -> List of Tensors
        for key in self.keys:
            vec = torch.from_numpy(self.mmaps[key][real_idx].copy()).float()
            inputs.append(vec)

        if self.y is not None:
            return inputs, self.y[real_idx]
        return (inputs,)


# # 2. LOAD DATA & PROCESS LABELS

# In[ ]:


print("\n[1/5] Checking Files...")
train_ids = np.load(CONFIG['TRAIN_ID_PATH'])
print(f"   ✓ Train IDs: {len(train_ids)}")


# In[ ]:


print("\n[2/5] Processing Labels (IA Strategy)...")
# 1. Load Terms & IA
df_terms = pd.read_csv(CONFIG['TRAIN_TERMS'], sep='\t', header=0,
                       names=['EntryID', 'term', 'aspect'])
df_ia = pd.read_csv(CONFIG['IA_FILE'], sep='\t', names=['term', 'ia'])
ia_dict = dict(zip(df_ia['term'], df_ia['ia']))

# 2. Tính Score
# map ontology -> aspect char
ONTOLOGY2ASPECT = {'MF': 'F', 'BP': 'P', 'CC': 'C'}
aspect_char = ONTOLOGY2ASPECT[CONFIG['ONTOLOGY']]

# chỉ giữ terms thuộc ontology này
df_terms_aspect = df_terms[df_terms['aspect'] == aspect_char]

# load top_terms cho ontology này
top_terms = np.load(CONFIG['TOP_TERMS_NPY'], allow_pickle=True).tolist()

# lọc lại theo top_terms
df_filtered = df_terms_aspect[df_terms_aspect['term'].isin(top_terms)]
id_to_terms = df_filtered.groupby('EntryID')['term'].apply(list).to_dict()

# Dọn dẹp
del df_terms, df_terms_aspect, df_filtered
gc.collect()


# # 3. PREPARE LABELS (RAM OPTIMIZED)

# In[ ]:


# Hàm parse đơn giản không cần thư viện ngoài
def parse_obo_parents(obo_path):
    # Trả về dict: term -> set of parents (bao gồm is_a và part_of)
    go_parents = {}
    current_term = None
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_term = None
            elif line.startswith("id: "):
                current_term = line[4:]
                if current_term not in go_parents:
                    go_parents[current_term] = set()
            elif line.startswith("is_a: ") and current_term:
                parent = line[6:].split(' ! ')[0]
                go_parents[current_term].add(parent)
            elif line.startswith("relationship: part_of ") and current_term:
                parent = line[22:].split(' ! ')[0]
                go_parents[current_term].add(parent)
    return go_parents


# In[ ]:


print("\n[3/5] Preparing Labels (Sparse Mode)...")

# 1. Sparse MLB
mlb = MultiLabelBinarizer(classes=top_terms, sparse_output=True)
mlb.fit([top_terms])

# 2. Build DAG Parent Index Map ---
print("Building DAG parent map...")
term_to_idx = {term: i for i, term in enumerate(top_terms)}
parents_idx = {} # Map: child_idx -> list of parent_indices

go_parents_dict = parse_obo_parents(CONFIG['OBO_PATH'])

# Chỉ giữ lại các quan hệ nằm trong top_terms (10,000 nhãn của model)
for child_term, idx in term_to_idx.items():
    if child_term in go_parents_dict:
        raw_parents = go_parents_dict[child_term]
        valid_parents_indices = []
        for p in raw_parents:
            if p in term_to_idx: # Chỉ lấy parent cũng nằm trong top_terms
                valid_parents_indices.append(term_to_idx[p])

        if valid_parents_indices:
            parents_idx[idx] = valid_parents_indices

print(f"DAG built. Found parents for {len(parents_idx)}/{len(top_terms)} terms.")

# 3. Transform -> Sparse Matrix
y_labels_list = [id_to_terms.get(pid, []) for pid in train_ids]
y_train_sparse = mlb.transform(y_labels_list)
del y_labels_list, train_ids # Xóa ID list không dùng nữa
gc.collect()

# 4. Weights (Optional)
# Nếu dùng IA weight thì giữ, nếu muốn model tự học thì comment dòng này và bỏ pos_weight trong Loss
weights_list = [ia_dict.get(t, 0.0) for t in mlb.classes_]
pos_weight_tensor = torch.tensor(weights_list, dtype=torch.float32).to(CONFIG['DEVICE'])

# 5. Convert to Dense Float32 & Label Smoothing
print("   ⏳ Converting Labels to Tensor...")
# Convert từng phần nhỏ hoặc convert hết nếu RAM > 12GB (với 80k row x 10k col float32 ~ 3.2GB -> Ổn)
y_train_binary = y_train_sparse.astype(np.float32).toarray()
y_train_tensor = torch.from_numpy(y_train_binary)

if CONFIG['LABEL_SMOOTHING'] > 0:
    y_train_tensor.mul_(1 - CONFIG['LABEL_SMOOTHING']).add_(CONFIG['LABEL_SMOOTHING'] / len(top_terms))

del y_train_sparse, y_train_binary
gc.collect()


# # 4. CREATE DATA LOADERS

# In[ ]:


print("\n[4/5] Creating DataLoaders...")

train_paths = {k: v['train'] for k, v in CONFIG['EMBEDDINGS'].items()}

indices = np.arange(len(y_train_tensor))
train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=CONFIG['SEED'])

train_dataset = MultiSourceDataset(train_paths, y_train_tensor, indices=train_idx)
val_dataset   = MultiSourceDataset(train_paths, y_train_tensor, indices=val_idx)

# --- AUTO DETECT INPUT DIMS ---
print("   ⏳ Measuring input shapes from dataset...")
sample_inputs, _ = train_dataset[0]
INPUT_DIMS_LIST = [x.shape[0] for x in sample_inputs]
print(f"   ✓ Detected Input Dims: {INPUT_DIMS_LIST}")

train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=8)
val_loader   = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=8)


# # 5. MODEL & TRAINING

# In[ ]:


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, layers_config, dropout):
        super().__init__()
        layers = []

        # 1. LAYER NORM ĐẦU VÀO (BẮT BUỘC)
        # Để cân bằng 'âm lượng' giữa ESM (hét to) và Ankh (nói nhỏ)
        layers.append(nn.LayerNorm(in_dim))

        prev_dim = in_dim
        for i, dim in enumerate(layers_config):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim), # Ổn định training
                nn.GELU(),           # Hiện đại hơn ReLU
                nn.Dropout(dropout)
            ])
            prev_dim = dim

        self.net = nn.Sequential(*layers)
        self.out_dim = prev_dim

    def forward(self, x):
        return self.net(x)


# In[ ]:


# Module lọc nhiễu (Attention)
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block để lọc nhiễu sau khi gộp"""
    def __init__(self, in_dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.Sigmoid() # Tạo ra mask từ 0 đến 1
        )

    def forward(self, x):
        # x shape: [Batch, Dim]
        # Attention weight: [Batch, Dim]
        w = self.fc(x)
        # Nhân trọng số vào x: Cái nào quan trọng thì giữ, rác thì nhân với 0
        return x * w


# In[ ]:


class MultiModalNet(nn.Module):
    def __init__(self, input_dims_list, encoder_layers, dropout, num_classes):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.fusion_input_dim = 0

        print("\n🏗️ Building Advanced Architecture:")

        # 1. Xây dựng các nhánh Encoder
        for i, in_dim in enumerate(input_dims_list):
            print(f"   ➤ Branch {i+1}: Input {in_dim} -> Output {encoder_layers[-1]}")
            enc = EncoderBlock(in_dim, encoder_layers, dropout)
            self.encoders.append(enc)
            self.fusion_input_dim += enc.out_dim

        print(f"   ➤ Fusion Dim: {self.fusion_input_dim}")

        # 2. SE-Block (Bộ lọc thông minh)
        self.attention_filter = SEBlock(self.fusion_input_dim)

        # 3. Layer tổng hợp cuối cùng
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.fusion_input_dim),
            nn.Linear(self.fusion_input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),

            # Thêm một lớp nữa để tăng khả năng học
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes)
        )

    def forward(self, inputs_list):
        features = []
        # Đi qua từng nhánh
        for i, encoder in enumerate(self.encoders):
            feat = encoder(inputs_list[i])
            features.append(feat)

        # Gộp lại (Concatenate)
        combined = torch.cat(features, dim=1)

        # LỌC NHIỄU (Điểm khác biệt lớn nhất)
        # Mạng sẽ tự học cách "tắt tiếng" các đặc trưng rác từ ProtT5 nếu nó thấy không cần thiết
        refined = self.attention_filter(combined)

        return self.head(refined)


# In[ ]:


print("\n[5/5] Building Model...")
model = MultiModalNet(
    input_dims_list = INPUT_DIMS_LIST,
    encoder_layers  = CONFIG['ENCODER_LAYERS'],
    dropout         = CONFIG['DROPOUT_RATE'],
    num_classes     = len(top_terms) # Dùng số lượng thực tế sau lọc
).to(CONFIG['DEVICE'])

# Loss & Optimizer
# Có thể thử bỏ pos_weight nếu thấy loss dao động quá mạnh
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(f"\n🚀 START TRAINING ({CONFIG['EPOCHS']} Epochs)...")
best_val_loss = float('inf')
best_model_path = f"{CONFIG['SAVE_DIR']}/prott5_esm2_ankh_best.pth"

for epoch in range(CONFIG['EPOCHS']):
    model.train()
    train_loss = 0
    for X_b, y_b in train_loader:
        X_b, y_b = [x.to(CONFIG['DEVICE']) for x in X_b], y_b.to(CONFIG['DEVICE'])
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = [x.to(CONFIG['DEVICE']) for x in X_b], y_b.to(CONFIG['DEVICE'])
            logits = model(X_b)
            loss = criterion(logits, y_b)
            val_loss += loss.item()

    avg_val = val_loss / len(val_loader)
    scheduler.step(avg_val)

    print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), best_model_path)
        print("  ⭐ New Best Model!")


# # 6. INFERENCE (STREAMING)

# In[ ]:


def propagate_dag(scores, parents_idx):
    # scores: numpy array (num_classes,)
    # parents_idx: dict {child_idx: [parent_indices]}
    # Lặp để lan truyền điểm từ con lên cha
    # (Cách đơn giản: lặp vài lần để hội tụ hoặc đi từ lá lên gốc)

    # Để tối ưu tốc độ trong vòng lặp Inference, ta chạy cố định khoảng 3-4 pass
    # thay vì while changed (tránh infinite loop và nhanh hơn)
    for _ in range(18): 
        for child_idx, parent_indices in parents_idx.items():
            child_score = scores[child_idx]
            for parent_idx in parent_indices:
                # Rule: Score cha không được nhỏ hơn score con
                if scores[parent_idx] < child_score:
                    scores[parent_idx] = child_score
    return scores


# In[ ]:


print("\n🔮 PREDICTING...")

# Dọn dẹp Training
del train_loader, val_loader, train_dataset, val_dataset, y_train_tensor, optimizer
gc.collect()
torch.cuda.empty_cache()

model.load_state_dict(torch.load(best_model_path))
model.eval()

# Load Test IDs & Paths
test_paths = {k: v['test'] for k, v in CONFIG['EMBEDDINGS'].items()}
test_ids = np.load(CONFIG['TEST_ID_PATH'])

# Test Dataset (Dual Memmap)
test_dataset = MultiSourceDataset(test_paths)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE']*2, shuffle=False, num_workers=2)

submission_path = f"{CONFIG['SAVE_DIR']}/submission.tsv"
n_predictions = 0

print("Starting Inference with DAG Propagation...")
with open(submission_path, 'w') as f:
    current_idx = 0
    with torch.no_grad():
        for (X_b,) in tqdm(test_loader, desc="Inference"):
            X_b = [x.to(CONFIG['DEVICE']) for x in X_b]
            logits = model(X_b)
            probs_batch = torch.sigmoid(logits).cpu().numpy()

            ids_batch = test_ids[current_idx : current_idx + len(probs_batch)]
            current_idx += len(probs_batch)

            for i, pid in enumerate(ids_batch):
                probs = probs_batch[i]
                probs = propagate_dag(probs, parents_idx)

                # TOP-K CỨNG (Chiến thuật 0.27 điểm)
                top_k = CONFIG['MAX_PREDS_PER_PROTEIN']
                ind = np.argpartition(probs, -top_k)[-top_k:]
                ind = ind[np.argsort(probs[ind])][::-1]

                for idx in ind:
                    score = probs[idx]
                    if score > CONFIG['MIN_CONFIDENCE']:
                        f.write(f"{pid}\t{top_terms[idx]}\t{score:.3f}\n")
                        n_predictions += 1

            del probs_batch, X_b, logits

print(f"\n✅ DONE! File: {submission_path}")
print(f"\n✅ DONE! Predictions: {n_predictions:,}")


# In[ ]:


# # Chạy lệnh này trong một cell mới
# !kaggle competitions submit \
#     -c cafa-6-protein-function-prediction \
#     -f output/base_model/submission.tsv \
#     -m "concat 3 models"


# In[ ]:




