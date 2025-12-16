import os
import subprocess
import multiprocessing
import time
import sys
import joblib
import numpy as np
import yaml

# --- CẤU HÌNH ---
BASE_PATH = "./"
CONFIG_PATH = "config.yaml"
LOG_DIR = "logs"
MODEL_NAME = "lin_t5_cond"  # <-- ĐỔI TÊN MODEL CẦN CHẠY Ở ĐÂY

# Python paths
RAPIDS_ENV = "rapids-env/bin/python" 
PYTORCH_ENV = "pytorch-env/bin/python"

# Script paths
TRAIN_WORKER_SCRIPT = f"{BASE_PATH}/protlib/scripts/train_lin_fold.py" # File worker mình vừa viết ở trên

# --- KHAI BÁO TÁC VỤ (Tự động sinh cho 5 GPU) ---
# Format: (GPU_ID, [(ENV, SCRIPT, MODEL_NAME, FOLD_ID)])
tasks = []
for fold in range(5):
    # Map fold 0 -> GPU 0, fold 1 -> GPU 1, ..., fold 4 -> GPU 4
    gpu_id = str(fold)
    tasks.append((gpu_id, [(RAPIDS_ENV, TRAIN_WORKER_SCRIPT, MODEL_NAME, fold)]))


def run_worker(gpu_id, command_list):
    """Chạy list lệnh trên GPU chỉ định và ghi log ra file."""
    
    print(f"🚀 [GPU {gpu_id}] Worker started.")

    for py_env, script, model_name, fold_id in command_list:
        # Xây dựng tên file log: tenmodel_foldX.log
        log_filename = f"{model_name}_fold{fold_id}.log"
        log_path = os.path.join(LOG_DIR, log_filename)
        
        # Tạo câu lệnh
        cmd = [
            py_env, script, 
            "--config-path", CONFIG_PATH,
            "--model-name", model_name,
            "--device", str(gpu_id),
            "--fold", str(fold_id)
        ]

        print(f"    ▶ [GPU {gpu_id}] Đang chạy Fold {fold_id} cho model: {model_name}")
        print(f"      📄 Logs -> {log_path}")
        
        # Mở file log để ghi
        with open(log_path, "w") as f_log:
            try:
                # Set unbuffered để log hiện ngay
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                
                subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, check=True, env=env)
            except subprocess.CalledProcessError:
                f_log.write(f"\n\n[ERROR] Process failed with exit code 1.\n")
                print(f"    ❌ [GPU {gpu_id}] Lỗi Fold {fold_id}! Kiểm tra file {log_path}.")
                return 

    print(f"✅ [GPU {gpu_id}] Hoàn thành Fold {fold_id}!")


def merge_results(model_name):
    """Gộp kết quả từ 5 fold lại thành file cuối cùng."""
    print(f"\n🔄 Đang gộp kết quả cho {model_name}...")
    
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    output_dir = os.path.join(config['base_path'], config['models_path'], model_name)
    
    try:
        # Merge OOF & Test
        # Load file mẫu để lấy shape
        first_oof = joblib.load(os.path.join(output_dir, 'temp_oof_fold_0.pkl'))
        final_oof = np.zeros_like(first_oof)
        
        first_test = joblib.load(os.path.join(output_dir, 'temp_test_fold_0.pkl'))
        final_test = np.zeros_like(first_test)

        for f in range(5):
            oof_path = os.path.join(output_dir, f'temp_oof_fold_{f}.pkl')
            test_path = os.path.join(output_dir, f'temp_test_fold_{f}.pkl')
            
            final_oof += joblib.load(oof_path)
            final_test += joblib.load(test_path)
            
            # Xóa file tạm (Clean up)
            if os.path.exists(oof_path): os.remove(oof_path)
            if os.path.exists(test_path): os.remove(test_path)

        final_test /= 5.0 # Chia trung bình cho test set

        joblib.dump(final_oof, os.path.join(output_dir, 'oof_pred.pkl'))
        joblib.dump(final_test, os.path.join(output_dir, 'test_pred.pkl'))
        
        print(f"🎉 Gộp xong! File lưu tại: {output_dir}")

    except Exception as e:
        print(f"❌ Lỗi khi gộp kết quả: {e}")
        print("Hãy kiểm tra xem tất cả các fold worker đã chạy xong chưa.")


def main():
    # 1. Tạo thư mục logs
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Lưu ý: Code cũ của bạn có bước "Tạo K-Folds" (create_gkf.py).
    # Trong logic mới này, việc chia fold được xử lý deterministically bằng seed trong train_worker.py 
    # nên không nhất thiết phải chạy create_gkf.py trước, TRỪ KHI script create_gkf.py làm việc khác quan trọng.
    # Nếu cần thì uncomment dòng dưới:
    # print("--- Bước 0: Tạo K-Folds (Optional) ---")
    # subprocess.run([RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/create_gkf.py", "--config-path", CONFIG_PATH])

    print(f"🚀 Bắt đầu train song song model: {MODEL_NAME} trên 5 GPU...\n")

    # 2. Khởi động các Workers
    processes = []
    for gpu_id, cmds in tasks:
        p = multiprocessing.Process(target=run_worker, args=(gpu_id, cmds))
        processes.append(p)
        p.start()
        time.sleep(1) # Delay nhỏ để tránh spam log cùng lúc

    # 3. Chờ hoàn thành
    failed = False
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed = True

    if not failed:
        # 4. Gộp kết quả nếu chạy thành công
        merge_results(MODEL_NAME)
        print("\n🎉 TẤT CẢ ĐÃ XONG! Kiểm tra thư mục 'logs/' để xem chi tiết.")
    else:
        print("\n❌ Có lỗi xảy ra ở một số GPU. Không gộp kết quả.")

if __name__ == "__main__":
    main()