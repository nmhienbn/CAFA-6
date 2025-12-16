import pandas as pd
import os


def normalize_aspect(file_path, col_name='aspect'):
    print(f"--- Đang xử lý file: {file_path} ---")

    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    df = pd.read_csv(file_path, sep='\t')

    if col_name not in df.columns:
        print(
            f"Lỗi: File không có cột tên là '{col_name}'. Các cột hiện có: {list(df.columns)}")
        return

    mapping = {
        'C': 'CCO',
        'F': 'MFO',
        'P': 'BPO'
    }

    df[col_name] = df[col_name].replace(mapping)

    df.to_csv(file_path, sep='\t', index=False)
    print(f"-> Đã chuẩn hóa {col_name} và lưu đè thành công.\n")


def add_header(file_path):
    print(f"--- Đang xử lý file: {file_path} ---")

    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    df = pd.read_csv(file_path, sep='\t', header=None)

    if df.shape[1] != 2:
        print(
            f"Cảnh báo: File này có {df.shape[1]} cột, nhưng bạn chỉ yêu cầu gán 2 tên cột.")

        df.columns = ['EntryID', 'taxonomyID'] + \
            [f'col_{i}' for i in range(2, df.shape[1])]
    else:

        df.columns = ['EntryID', 'taxonomyID']

    df.to_csv(file_path, sep='\t', index=False)
    print(f"-> Đã thêm header EntryID, taxonomyID và lưu đè thành công.\n")


if __name__ == '__main__':

    FILE_ASPECT = 'Train/train_terms.tsv'

    FILE_HEADER = 'Train/train_taxonomy.tsv'

    normalize_aspect(FILE_ASPECT, col_name='aspect')

    add_header(FILE_HEADER)
