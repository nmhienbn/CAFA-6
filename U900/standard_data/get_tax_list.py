import pandas as pd
import os
import sys

# --- CẤU HÌNH ĐƯỜNG DẪN (Bạn hãy chỉnh lại cho đúng thư mục của bạn) ---
BASE_PATH = "/data/hien/CAFA-6/U900" 
# Đường dẫn đến thư mục chứa file feather (thường là helpers/fasta)
FASTA_HELPERS_PATH = os.path.join(BASE_PATH, "helpers/fasta")
TRAIN_FEATHER = os.path.join(FASTA_HELPERS_PATH, "train_seq.feather")
TEST_FEATHER = os.path.join(FASTA_HELPERS_PATH, "test_seq.feather")

# Số lượng loài muốn lấy (giải pháp gốc lấy khoảng 30)
TOP_K = 30 

print(f"=== BẮT ĐẦU TẠO TAXONOMY LIST ===")
print(f"Thư mục dữ liệu: {FASTA_HELPERS_PATH}")

# ---------------------------------------------------------
# 1. ĐỌC DỮ LIỆU
# ---------------------------------------------------------
print("\n[1] Đang đọc file dữ liệu feather...")
try:
    if not os.path.exists(TRAIN_FEATHER):
        print(f"❌ LỖI: Không tìm thấy file {TRAIN_FEATHER}")
        print("   -> Bạn đã chạy script 'parse_fasta.py' chưa?")
        sys.exit(1)
        
    if not os.path.exists(TEST_FEATHER):
        print(f"❌ LỖI: Không tìm thấy file {TEST_FEATHER}")
        sys.exit(1)

    # Chỉ đọc cột taxonomyID cho nhẹ
    train_df = pd.read_feather(TRAIN_FEATHER, columns=['taxonomyID'])
    print(f"   ✅ Train: {len(train_df):,} protein.")
    
    test_df = pd.read_feather(TEST_FEATHER, columns=['taxonomyID'])
    print(f"   ✅ Test: {len(test_df):,} protein.")

except Exception as e:
    print(f"❌ LỖI ĐỌC FILE: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 2. THỐNG KÊ VÀ TÌM TOP K
# ---------------------------------------------------------
print(f"\n[2] Đang thống kê tần suất Taxonomy ID trên cả Train & Test...")
try:
    # Gộp 2 tập dữ liệu
    all_tax = pd.concat([train_df['taxonomyID'], test_df['taxonomyID']], axis=0)
    
    # Đếm số lần xuất hiện
    counts = all_tax.value_counts()
    print(f"   - Tổng số loài (unique Taxonomy ID): {len(counts):,}")
    
    # Lấy Top K
    top_taxons_series = counts.head(TOP_K)
    top_ids = top_taxons_series.index.tolist()
    
    # Chuyển về kiểu int (đề phòng dữ liệu bị dính float/str)
    top_ids = [int(x) for x in top_ids]
    
except Exception as e:
    print(f"❌ LỖI XỬ LÝ DỮ LIỆU: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 3. KẾT QUẢ
# ---------------------------------------------------------
print(f"\n[3] KẾT QUẢ TOP {TOP_K} LOÀI PHỔ BIẾN NHẤT:")
print("-" * 60)
print("Hãy copy list dưới đây thay thế vào biến 'tax_list' trong file 'protlib/models/prepocess.py':\n")
print(f"tax_list = {top_ids}")
print("-" * 60)

print("\nChi tiết tần suất (Top 10):")
for tax_id, count in top_taxons_series.head(10).items():
    print(f"   - ID {int(tax_id)}: {count:,} lần")