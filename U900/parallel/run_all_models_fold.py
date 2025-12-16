import os
import subprocess
import multiprocessing
import time
import sys
import yaml
import numpy as np
import joblib

# --- CẤU HÌNH ---
BASE_PATH = "./"
CONFIG_PATH = "config.yaml"
LOG_DIR = "logs"

# Python Environments
RAPIDS_ENV = "rapids-env/bin/python" 
PYTORCH_ENV = "pytorch-env/bin/python"

# --- KHAI BÁO MODEL ---
# Cấu trúc: (Environment, Script Path, Model Name)
# Lưu ý: Script phải là các file "_fold.py" đã sửa đổi để chạy từng fold
models_to_run = [
    # # --- NHÓM 1: PY-BOOST (4 Models) ---
    # (RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_pb_fold.py", "pb_t54500_raw"),
    # (RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_pb_fold.py", "pb_t54500_cond"),
    # (RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_pb_fold.py", "pb_t5esm4500_raw"),
    # (RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_pb_fold.py", "pb_t5esm4500_cond"),
    
    # # --- NHÓM 2: LINEAR (2 Models) ---
    # (RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_lin_fold.py", "lin_t5_raw"),
    # (RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_lin_fold.py", "lin_t5_cond"),

    # --- NHÓM 3: NEURAL NETWORK (1 Model) ---
    # Thay thế toàn bộ pipeline tuần tự cũ bằng 1 script chạy fold song song
    (PYTORCH_ENV, f"{BASE_PATH}/nn_solution/train_nn_fold.py", "nn_pMLP"),
]

FOLDS = [0, 1, 2, 3, 4]
NUM_GPUS = 8  # Bạn có 8x A100

def worker(gpu_id, task_queue):
    """Worker chạy trên GPU được chỉ định"""
    print(f"🚀 [GPU {gpu_id}] Worker online.")
    
    while True:
        try:
            if task_queue.empty(): break
            
            # Lấy nhiệm vụ
            env, script, model_name, fold = task_queue.get_nowait()
            
            # Tạo file log riêng
            log_path = os.path.join(LOG_DIR, f"{model_name}_fold{fold}.log")
            
            # Câu lệnh chạy
            cmd = [
                env, script,
                "--config-path", CONFIG_PATH,
                "--model-name", model_name,
                "--device", str(gpu_id),
                "--fold", str(fold)
            ]
            
            print(f"    ▶ [GPU {gpu_id}] Running: {model_name} | Fold {fold}")
            
            with open(log_path, "w") as f_log:
                try:
                    subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError:
                    f_log.write(f"\n[ERROR] Process failed for Fold {fold}\n")
                    print(f"    ❌ [GPU {gpu_id}] FAILED: {model_name} | Fold {fold} (Check logs)")
        
        except Exception:
            break
            
    print(f"✅ [GPU {gpu_id}] Completed all assigned tasks.")

def merge_outputs():
    """Tự động gộp kết quả cho TẤT CẢ các model trong danh sách"""
    print("\n🔄 ĐANG GỘP KẾT QUẢ (MERGING)...")
    
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    models_root = os.path.join(config['base_path'], config['models_path'])

    # Lấy danh sách tên model duy nhất
    unique_models = sorted(list(set([m[2] for m in models_to_run])))

    for model_name in unique_models:
        model_dir = os.path.join(models_root, model_name)
        if not os.path.exists(model_dir):
            print(f"⚠️  Bỏ qua {model_name} (Chưa thấy thư mục output)")
            continue

        print(f"   -> Processing: {model_name}...", end=" ")
        
        try:
            # --- TRƯỜNG HỢP 1: NEURAL NETWORK (File .npy) ---
            if "nn_" in model_name:
                # Merge OOF
                oof_files = [os.path.join(model_dir, f'temp_oof_fold_{f}.npy') for f in FOLDS]
                if all(os.path.exists(f) for f in oof_files):
                    # Cộng dồn các fold lại (vì mỗi file chỉ chứa giá trị tại vị trí val)
                    full_oof = sum(np.load(f) for f in oof_files)
                    np.save(os.path.join(model_dir, 'Y_pred_oof_blend.npy'), full_oof)
                
                # Merge Test (Submission)
                test_files = [os.path.join(model_dir, f'temp_test_fold_{f}.npy') for f in FOLDS]
                if all(os.path.exists(f) for f in test_files):
                    # Test thì lấy trung bình cộng
                    full_test = sum(np.load(f) for f in test_files) / len(FOLDS)
                    np.save(os.path.join(model_dir, 'Y_submit.npy'), full_test)
                print("OK (Format .npy)")

            # --- TRƯỜNG HỢP 2: PY-BOOST & LINEAR (File .pkl) ---
            else:
                # Merge OOF
                oof_files = [os.path.join(model_dir, f'temp_oof_fold_{f}.pkl') for f in FOLDS]
                if all(os.path.exists(f) for f in oof_files):
                    full_oof = sum(joblib.load(f) for f in oof_files)
                    joblib.dump(full_oof, os.path.join(model_dir, 'oof_pred.pkl'))
                
                # Merge Test
                test_files = [os.path.join(model_dir, f'temp_test_fold_{f}.pkl') for f in FOLDS]
                if all(os.path.exists(f) for f in test_files):
                    full_test = sum(joblib.load(f) for f in test_files) / len(FOLDS)
                    joblib.dump(full_test, os.path.join(model_dir, 'test_pred.pkl'))
                print("OK (Format .pkl)")

        except Exception as e:
            print(f"\n      ❌ Lỗi khi gộp {model_name}: {e}")

    print("\n✨ TẤT CẢ ĐÃ HOÀN TẤT!")

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    task_queue = multiprocessing.Queue()
    
    # 1. Đẩy 35 tasks vào hàng đợi (7 models * 5 folds)
    for env, script, model_name in models_to_run:
        for fold in FOLDS:
            task_queue.put((env, script, model_name, fold))
            
    print(f"--- Bắt đầu Train Phân Tán trên {NUM_GPUS} GPU A100 ---")
    print(f"--- Tổng tác vụ: {len(models_to_run) * len(FOLDS)} ---")
    
    # 2. Khởi chạy 8 Workers
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = multiprocessing.Process(target=worker, args=(gpu_id, task_queue))
        processes.append(p)
        p.start()
        time.sleep(1) 
        
    # 3. Chờ xong
    for p in processes:
        p.join()

    # 4. Gộp kết quả
    merge_outputs()

if __name__ == "__main__":
    main()