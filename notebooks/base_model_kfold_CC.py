#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')

# !pip install -q transformers biopython kaggle
# print("✅ Đã cài đặt xong thư viện!")


# In[2]:


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


# In[3]:


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


# In[4]:


with open("configs/base_model_kfold.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

if CONFIG.get("DEVICE", "auto") == "auto":
    CONFIG["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG['ONTOLOGY'] = "CC"
CONFIG['TOP_TERMS_NPY'] = "features/top_terms_by_aspect/top_terms_CC.npy"

os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)
print(f"🚀 CAFA 6 - DUAL MODEL (ANKH + ESM) | Device: {CONFIG['DEVICE']}")

torch.manual_seed(CONFIG['SEED'])
np.random.seed(CONFIG['SEED'])


# # 1. MEMORY-SAFE DATASET
# CHÌA KHÓA ĐỂ KHÔNG TRÀN RAM

# In[5]:


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

# In[6]:


print("\n[1/5] Checking Files...")
train_ids = np.load(CONFIG['TRAIN_ID_PATH'])
print(f"   ✓ Train IDs: {len(train_ids)}")


# In[7]:


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


# # 3. PREPARE LABELS
# (RAM OPTIMIZED)

# In[8]:


print("\n[3/5] Preparing Labels (Sparse Mode)...")

# 1. Sparse MLB
mlb = MultiLabelBinarizer(classes=top_terms, sparse_output=True)
mlb.fit([top_terms])

# 2. Transform -> Sparse Matrix
y_labels_list = [id_to_terms.get(pid, []) for pid in train_ids]
y_train_sparse = mlb.transform(y_labels_list)
del y_labels_list, train_ids # Xóa ID list không dùng nữa
gc.collect()

# 3. Weights (Optional)
# Nếu dùng IA weight thì giữ, nếu muốn model tự học thì comment dòng này và bỏ pos_weight trong Loss
weights_list = [ia_dict.get(t, 0.0) for t in mlb.classes_]
pos_weight_tensor = torch.tensor(weights_list, dtype=torch.float32).to(CONFIG['DEVICE'])

# 4. Convert to Dense Float32 & Label Smoothing
print("   ⏳ Converting Labels to Tensor...")
# Convert từng phần nhỏ hoặc convert hết nếu RAM > 12GB (với 80k row x 10k col float32 ~ 3.2GB -> Ổn)
# true labels 0/1 (dùng cho tuning F-max)
y_train_binary = y_train_sparse.astype(np.float32).toarray()
y_true_np = y_train_binary.copy()   # lưu lại cho threshold tuning
y_train_tensor = torch.from_numpy(y_train_binary)

if CONFIG['LABEL_SMOOTHING'] > 0:
    y_train_tensor.mul_(1 - CONFIG['LABEL_SMOOTHING']).add_(CONFIG['LABEL_SMOOTHING'] / len(top_terms))

del y_train_sparse
gc.collect()

y_true_np.shape


# In[9]:


# ----- chọn nhãn phổ biến nhất để stratification -----
label_freq = y_true_np.sum(axis=0)              # (n_labels,)
order = np.argsort(-label_freq)                 # sort desc
top_strat_labels = order[:CONFIG['N_LABELS_STRAT']]

y_for_strat = y_true_np[:, top_strat_labels]    # (n_samples, N_LABELS_STRAT)
print('y_for_strat shape:', y_for_strat.shape)


# # 4. CREATE DATA LOADERS

# In[10]:


print("\n[4/5] Creating DataLoaders...")

train_paths = {k: v['train'] for k, v in CONFIG['EMBEDDINGS'].items()}

# tạo dataset tạm để lấy dimensionality
tmp_dataset = MultiSourceDataset(train_paths)
sample_inputs, = tmp_dataset[0]   # vì y=None -> __getitem__ trả (inputs,)
INPUT_DIMS_LIST = [x.shape[0] for x in sample_inputs]
del tmp_dataset

n_samples, n_labels = y_true_np.shape
indices = np.arange(n_samples)

mskf = MultilabelStratifiedKFold(
    n_splits=CONFIG['N_FOLDS'],
    shuffle=True,
    random_state=CONFIG['SEED']
)

# OOF predictions
oof_pred = np.zeros((n_samples, n_labels), dtype=np.float32)


# # 5. MODEL & TRAINING

# In[11]:


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


# In[12]:


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


# In[13]:


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

# Loss & Optimizer
# Có thể thử bỏ pos_weight nếu thấy loss dao động quá mạnh
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

for fold, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(n_samples), y_for_strat)):
    print(f'=== Fold {fold} ===')
    train_dataset = MultiSourceDataset(train_paths, y_train_tensor, indices=train_idx)
    val_dataset   = MultiSourceDataset(train_paths, y_train_tensor, indices=val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False, num_workers=2
    )
    model = MultiModalNet(
        input_dims_list = INPUT_DIMS_LIST,
        encoder_layers  = CONFIG['ENCODER_LAYERS'],
        dropout         = CONFIG['DROPOUT_RATE'],
        num_classes     = len(top_terms) # Dùng số lượng thực tế sau lọc
    ).to(CONFIG['DEVICE'])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"\n🚀 START TRAINING ({CONFIG['EPOCHS']} Epochs)...")
    best_val_loss = float('inf')
    best_model_path = f"{CONFIG['SAVE_DIR']}/best_fold{fold}_{CONFIG['ONTOLOGY']}.pth"

    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b = [x.to(CONFIG['DEVICE']) for x in X_b]
            y_b = y_b.to(CONFIG['DEVICE'])

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
                X_b = [x.to(CONFIG['DEVICE']) for x in X_b]
                y_b = y_b.to(CONFIG['DEVICE'])
                logits = model(X_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(f"Fold {fold} | Epoch {epoch+1}/{CONFIG['EPOCHS']} "
              f"| train {avg_train:.4f} | val {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            print("  ⭐ New Best Model!")

    # ===== OOF prediction cho fold này =====
    print(f'Infer OOF for fold {fold}')
    model.load_state_dict(torch.load(best_model_path, map_location=CONFIG['DEVICE']))
    model.eval()

    fold_val_pred = np.zeros((len(val_idx), n_labels), dtype=np.float32)
    pos = 0
    with torch.no_grad():
        for X_b, y_b in DataLoader(
            val_dataset, batch_size=CONFIG['BATCH_SIZE'],
            shuffle=False, num_workers=2
        ):
            X_b = [x.to(CONFIG['DEVICE']) for x in X_b]
            logits = model(X_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            bs = probs.shape[0]
            fold_val_pred[pos:pos+bs] = probs
            pos += bs

    oof_pred[val_idx] = fold_val_pred

    del train_loader, val_loader, train_dataset, val_dataset, model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

# lưu lại để debug / dùng ngoài
np.save(f"{CONFIG['SAVE_DIR']}/oof_pred_{CONFIG['ONTOLOGY']}.npy", oof_pred)
np.save(f"{CONFIG['SAVE_DIR']}/y_true_{CONFIG['ONTOLOGY']}.npy", y_true_np)


# # 6. INFERENCE (STREAMING)

# In[ ]:


def tune_fmax_threshold(y_true, y_score, num_thresholds=99):
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_t, best_f = 0.3, 0.0

    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int8)

        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()

        if tp == 0:
            continue

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2 * p * r / (p + r)

        if f > best_f:
            best_f = f
            best_t = t

    return best_t, best_f

thr, fmax = tune_fmax_threshold(y_true_np, oof_pred)
print(f"[{CONFIG['ONTOLOGY']}] Best threshold = {thr:.4f}, Fmax = {fmax:.4f}")

thr_path = f"{CONFIG['SAVE_DIR']}/thr_{CONFIG['ONTOLOGY']}.txt"
with open(thr_path, 'w') as f:
    f.write(f"{thr:.6f}\n")


# In[ ]:


# === Inference cho ontology hiện tại ===

test_paths = {k: v['test'] for k, v in CONFIG['EMBEDDINGS'].items()}
test_ids = np.load(CONFIG['TEST_ID_PATH'])

test_dataset = MultiSourceDataset(test_paths)
test_loader = DataLoader(
    test_dataset, batch_size=CONFIG['BATCH_SIZE'] * 2,
    shuffle=False, num_workers=2
)

# load các model từng fold
models = []
for fold in range(CONFIG['N_FOLDS']):
    model_path = f"{CONFIG['SAVE_DIR']}/best_fold{fold}_{CONFIG['ONTOLOGY']}.pth"
    m = MultiModalNet(
        input_dims_list=INPUT_DIMS_LIST,
        encoder_layers=CONFIG['ENCODER_LAYERS'],
        dropout=CONFIG['DROPOUT_RATE'],
        num_classes=len(top_terms)
    ).to(CONFIG['DEVICE'])
    m.load_state_dict(torch.load(model_path, map_location=CONFIG['DEVICE']))
    m.eval()
    models.append(m)

# load threshold đã tune
with open(f"{CONFIG['SAVE_DIR']}/thr_{CONFIG['ONTOLOGY']}.txt") as f:
    THR = float(f.read().strip())

submission_path = f"{CONFIG['SAVE_DIR']}/submission_{CONFIG['ONTOLOGY']}.tsv"

from tqdm.auto import tqdm
n_predictions = 0

with open(submission_path, 'w') as f:
    current_idx = 0
    with torch.no_grad():
        for (X_b,) in tqdm(test_loader, desc=f"Inference {CONFIG['ONTOLOGY']}"):
            X_b = [x.to(CONFIG['DEVICE']) for x in X_b]

            # ensemble trung bình logits của 5 fold
            logits_sum = None
            for m in models:
                logits = m(X_b)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            logits_avg = logits_sum / len(models)

            probs_batch = torch.sigmoid(logits_avg).cpu().numpy()

            ids_batch = test_ids[current_idx: current_idx + len(probs_batch)]
            current_idx += len(probs_batch)

            for i, pid in enumerate(ids_batch):
                probs = probs_batch[i]

                top_k = CONFIG['MAX_PREDS_PER_PROTEIN']
                ind = np.argpartition(probs, -top_k)[-top_k:]
                ind = ind[np.argsort(probs[ind])][::-1]

                for idx in ind:
                    score = probs[idx]
                    if score >= THR:     # dùng threshold F-max
                        f.write(f"{pid}\t{top_terms[idx]}\t{score:.3f}\n")
                        n_predictions += 1

print(f"{CONFIG['ONTOLOGY']} done, {n_predictions} predictions")


# In[ ]:


# # Chạy lệnh này trong một cell mới
# !kaggle competitions submit \
#     -c cafa-6-protein-function-prediction \
#     -f /content/drive/MyDrive/CAFA6_Results/prott5_esm2_ankh_Run/submission.tsv \
#     -m "esm2 ankh prot t5 new model"


# In[ ]:




