import numpy as np
import pandas as pd
import sys

def load_ids_from_fasta(fasta_path):
    """
    Đọc file FASTA và trích xuất Protein ID từ dòng header.
    Format header trong ảnh: >A0A0C5B5G6 9606 -> Lấy A0A0C5B5G6
    """
    print(f"Loading IDs from {fasta_path}...")
    ids = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Bỏ dấu '>', tách theo khoảng trắng và lấy phần tử đầu tiên
                # Ví dụ: ">A0A0C5B5G6 9606" -> "A0A0C5B5G6"
                protein_id = line.strip()[1:].split()[0]
                ids.append(protein_id)
    return np.array(ids)

def create_submission():
    # --- CẤU HÌNH ĐƯỜNG DẪN FILE ---
    FASTA_PATH = "Test/testsuperset.fasta"
    YSUBMIT_PATH = "models/nn_serg/Y_submit.npy"
    YLABELS_PATH = "models/nn_serg/Y_labels.npy"
    OUTPUT_PATH = "submission.tsv"
    
    # Ngưỡng lọc điểm số (giảm dung lượng file output)
    # CAFA cho phép nộp file lớn, nhưng 0.001 là ngưỡng an toàn để loại bỏ nhiễu
    THRESHOLD = 0.001 

    # 1. Load IDs từ FASTA
    target_ids = load_ids_from_fasta(FASTA_PATH)
    print(f"-> Found {len(target_ids)} protein IDs.")

    # 2. Load Predictions & Labels
    print("Loading numpy arrays...")
    y_submit = np.load(YSUBMIT_PATH)
    y_labels = np.load(YLABELS_PATH)

    print(f"-> Y_submit shape: {y_submit.shape}")
    print(f"-> Y_labels shape: {y_labels.shape}")

    # 3. Validate kích thước
    if len(target_ids) != y_submit.shape[0]:
        print(f"❌ LỖI: Số lượng ID ({len(target_ids)}) không khớp số hàng của Y_submit ({y_submit.shape[0]})!")
        print("Vui lòng kiểm tra xem file fasta có đúng là file dùng để generate Y_submit không.")
        sys.exit(1)
    else:
        print("✅ Kích thước khớp. Đang xử lý dữ liệu...")

    # 4. Convert Matrix to Sparse List (Tối ưu RAM)
    # Sử dụng np.where để tìm tọa độ các điểm số > threshold
    print(f"Filtering scores >= {THRESHOLD}...")
    row_idx, col_idx = np.where(y_submit >= THRESHOLD)

    # Map tọa độ sang giá trị thực
    final_ids = target_ids[row_idx]
    final_terms = y_labels[col_idx]
    final_scores = y_submit[row_idx, col_idx]

    # 5. Tạo DataFrame và xuất file
    print("Creating DataFrame...")
    df = pd.DataFrame({
        'id': final_ids,
        'term': final_terms,
        'score': final_scores
    })

    # Làm tròn điểm số (3 số lẻ) để giảm kích thước file
    df['score'] = df['score'].round(3)

    print(f"Saving to {OUTPUT_PATH}...")
    # Format CAFA: Tab-separated, No Header, index=False
    # Cột: ProteinID | GO Term | Score
    df.to_csv(OUTPUT_PATH, sep='\t', index=False, header=False)
    
    print("🎉 Done! File saved successfully.")

if __name__ == "__main__":
    create_submission()