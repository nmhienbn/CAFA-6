import pandas as pd
import os

def normalize_aspect(file_path, col_name='aspect'):
    """
    Nhiệm vụ 1: Đổi C -> CCO, F -> MFO, P -> BPO
    """
    print(f"--- Đang xử lý file: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    # 1. Đọc file
    df = pd.read_csv(file_path, sep='\t')
    
    # Kiểm tra xem cột cần sửa có tồn tại không
    if col_name not in df.columns:
        print(f"Lỗi: File không có cột tên là '{col_name}'. Các cột hiện có: {list(df.columns)}")
        return

    # 2. Map giá trị cũ sang giá trị mới
    mapping = {
        'C': 'CCO',
        'F': 'MFO',
        'P': 'BPO'
    }
    
    # Chỉ thay thế những giá trị khớp chính xác
    df[col_name] = df[col_name].replace(mapping)
    
    # 3. Lưu ghi đè (index=False để không sinh thêm cột số thứ tự)
    df.to_csv(file_path, sep='\t', index=False)
    print(f"-> Đã chuẩn hóa {col_name} và lưu đè thành công.\n")


def add_header(file_path):
    """
    Nhiệm vụ 2: Thêm header EntryID, taxonomyID cho file chưa có header
    """
    print(f"--- Đang xử lý file: {file_path} ---")

    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    # 1. Đọc file với header=None (báo cho pandas biết dòng đầu là dữ liệu)
    df = pd.read_csv(file_path, sep='\t', header=None)
    
    # Kiểm tra số lượng cột để tránh lỗi gán sai
    if df.shape[1] != 2:
        print(f"Cảnh báo: File này có {df.shape[1]} cột, nhưng bạn chỉ yêu cầu gán 2 tên cột.")
        # Vẫn gán 2 cột đầu, các cột sau tự động đặt tên số
        df.columns = ['EntryID', 'taxonomyID'] + [f'col_{i}' for i in range(2, df.shape[1])]
    else:
        # 2. Gán tên cột chuẩn
        df.columns = ['EntryID', 'taxonomyID']

    # 3. Lưu ghi đè (header=True để ghi dòng tiêu đề mới vào)
    df.to_csv(file_path, sep='\t', index=False)
    print(f"-> Đã thêm header EntryID, taxonomyID và lưu đè thành công.\n")


# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN FILE CỦA BẠN TẠI ĐÂY
# ==========================================
if __name__ == '__main__':
    # File 1: Cần đổi tên Aspect (Thường là train_terms.tsv)
    # Lưu ý: Kiểm tra xem trong file đó cột chứa C,F,P tên là gì? 
    # Mặc định mình để là 'aspect', nếu tên khác thì sửa tham số col_name
    FILE_ASPECT = 'Train/train_terms.tsv' 
    
    # File 2: Cần thêm header (Thường là train_taxonomy.tsv)
    FILE_HEADER = 'Train/train_taxonomy.tsv'

    # --- CHẠY ---
    # Nhiệm vụ 1
    normalize_aspect(FILE_ASPECT, col_name='aspect')
    
    # Nhiệm vụ 2
    add_header(FILE_HEADER)