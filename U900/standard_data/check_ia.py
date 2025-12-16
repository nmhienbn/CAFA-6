import pandas as pd
import os
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN (Bạn hãy chỉnh lại cho đúng thư mục của bạn) ---
BASE_PATH = "/data/hien/CAFA-6/U900" 
IA_PATH = os.path.join(BASE_PATH, "IA.txt")
OBO_PATH = os.path.join(BASE_PATH, "Train/go-basic.obo")
# File dự đoán đang lỗi (Ví dụ CC, bạn có thể đổi thành bp/mf)
SUB_PATH = os.path.join(BASE_PATH, "models/gcn/cc/temp/sub.tsv") 

print(f"=== BẮT ĐẦU KIỂM TRA DỮ LIỆU ===")
print(f"Thư mục gốc: {BASE_PATH}")

# ---------------------------------------------------------
# KIỂM TRA 1: FILE IA.TXT CÓ BỊ LỖI KHÔNG?
# ---------------------------------------------------------
print("\n[1] Đang kiểm tra file IA.txt...")
ia_dict = {}
try:
    if not os.path.exists(IA_PATH):
        print(f"❌ LỖI: Không tìm thấy file {IA_PATH}")
    else:
        # Đọc file IA
        df_ia = pd.read_csv(IA_PATH, sep='\t', names=['term', 'ia'], header=None)
        print(f"   ✅ Đọc thành công {len(df_ia)} dòng.")
        
        # Chuyển thành dict để tra cứu nhanh
        ia_dict = dict(zip(df_ia['term'], df_ia['ia']))
        
        # Kiểm tra giá trị
        cnt_zero = (df_ia['ia'] == 0).sum()
        cnt_nan = df_ia['ia'].isna().sum()
        print(f"   - Số term có IA = 0: {cnt_zero}")
        print(f"   - Số term có IA = NaN: {cnt_nan}")
        print(f"   - Ví dụ 3 dòng đầu: {list(ia_dict.items())[:3]}")
        
        if len(df_ia) == 0:
            print("   ❌ CẢNH BÁO: File IA.txt bị rỗng!")
except Exception as e:
    print(f"❌ LỖI ĐỌC FILE IA: {e}")

# ---------------------------------------------------------
# KIỂM TRA 2: LỖI MAPPING ID (KHỚP GIỮA OBO VÀ IA)
# ---------------------------------------------------------
print("\n[2] Đang kiểm tra khớp ID giữa OBO và IA...")
try:
    # Parse nhanh file OBO để lấy danh sách ID
    obo_ids = set()
    with open(OBO_PATH, 'r') as f:
        for line in f:
            if line.startswith("id: GO:"):
                obo_ids.add(line.strip().split("id: ")[1])
    
    print(f"   ✅ Tìm thấy {len(obo_ids)} term trong file OBO.")
    
    # Kiểm tra xem ID trong OBO có nằm trong IA.txt không
    missing_in_ia = obo_ids - set(ia_dict.keys())
    print(f"   - Số term có trong OBO nhưng thiếu trong IA.txt: {len(missing_in_ia)}")
    
    if len(missing_in_ia) > 0:
        print(f"   ⚠️ Ví dụ term bị thiếu: {list(missing_in_ia)[:5]}")
        print("   -> Nếu con số này quá lớn, file IA.txt có thể bị sai phiên bản.")
except Exception as e:
    print(f"❌ LỖI ĐỌC FILE OBO: {e}")

# ---------------------------------------------------------
# KIỂM TRA 3: FILE DỰ ĐOÁN (SUB.TSV) VÀ ROOT TERM
# ---------------------------------------------------------
print("\n[3] Đang kiểm tra file dự đoán (sub.tsv)...")
try:
    if not os.path.exists(SUB_PATH) or os.path.getsize(SUB_PATH) == 0:
        print(f"❌ LỖI: File {SUB_PATH} không tồn tại hoặc RỖNG!")
        print("   -> Đây là nguyên nhân crash ở bước trước.")
    else:
        # Đọc file dự đoán
        df_sub = pd.read_csv(SUB_PATH, sep='\t', names=['EntryID', 'term', 'prob'], header=None)
        print(f"   ✅ File có {len(df_sub)} dòng dự đoán.")
        
        # Lấy các term duy nhất được dự đoán
        predicted_terms = df_sub['term'].unique()
        print(f"   - Tổng số term unique được dự đoán: {len(predicted_terms)}")
        
        # Phân tích các term này
        cnt_valid = 0
        cnt_root_zero = 0
        cnt_missing = 0
        missing_examples = []
        
        for term in predicted_terms:
            val = ia_dict.get(term, None)
            if val is None:
                cnt_missing += 1
                if len(missing_examples) < 5: missing_examples.append(term)
            elif val == 0:
                cnt_root_zero += 1
            else:
                cnt_valid += 1
        
        print(f"   📊 KẾT QUẢ PHÂN TÍCH TERM DỰ ĐOÁN:")
        print(f"      + Số term hợp lệ (IA > 0): {cnt_valid}  <-- Cần cái này > 0 để chạy được")
        print(f"      + Số term là Root hoặc IA=0: {cnt_root_zero}")
        print(f"      + Số term KHÔNG TÌM THẤY trong IA: {cnt_missing}")
        
        if cnt_missing > 0:
            print(f"      ⚠️ Ví dụ term lạ (không có trong IA): {missing_examples}")
            
        if cnt_valid == 0:
            print("\n❌ KẾT LUẬN: CODE CRASH VÌ KHÔNG CÓ DỰ ĐOÁN NÀO CÓ 'IA > 0'")
            if cnt_root_zero > 0:
                print("   -> Nguyên nhân: Mô hình chỉ dự đoán ra Root Term (xác suất cao nhất), các term con cụ thể bị loại bỏ.")
            if cnt_missing > 0:
                print("   -> Nguyên nhân: ID dự đoán bị sai lệch hoàn toàn so với file IA.")
        else:
            print("\n✅ KẾT LUẬN: Dữ liệu IA hợp lệ. Vấn đề có thể nằm ở bước lọc ngưỡng 'flg' trong code metric.")

except Exception as e:
    print(f"❌ LỖI ĐỌC SUB.TSV: {e}")