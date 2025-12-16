import os
import subprocess
import multiprocessing
import time
import sys

# --- CẤU HÌNH ---
BASE_PATH = "./"
CONFIG_PATH = "config.yaml"
LOG_DIR = "logs"  # Thư mục chứa log

# Python paths
RAPIDS_ENV = "rapids-env/bin/python" 
PYTORCH_ENV = "pytorch-env/bin/python"

# --- KHAI BÁO TÁC VỤ ---
tasks = [
    # # NHÓM 1: PY-BOOST (GPU 0-3)
    # ("1", [(RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_pb.py", "pb_t54500_raw")]),
    # ("2", [(RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_pb.py", "pb_t54500_cond")]),
    # ("3", [(RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_pb.py", "pb_t5esm4500_raw")]),
    # ("4", [(RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_pb.py", "pb_t5esm4500_cond")]),

    # # # NHÓM 2: LINEAR (GPU 4-5)
    # ("5", [(RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_lin.py", "lin_t5_raw")]),
    # ("6", [(RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/train_lin.py", "lin_t5_cond")]),

    # NHÓM 3: NN PIPELINE (GPU 6)
    ("7", [
        (PYTORCH_ENV, f"{BASE_PATH}/nn_solution/train_models.py", None), 
        (PYTORCH_ENV, f"{BASE_PATH}/nn_solution/inference_models.py", None),
        (PYTORCH_ENV, f"{BASE_PATH}/nn_solution/make_pkl.py", None)
    ]),
]

def run_worker(gpu_id, command_list):
    """Chạy list lệnh trên GPU chỉ định và ghi log ra file."""
    
    print(f"🚀 [GPU {gpu_id}] Worker started.")

    for py_env, script, model_name in command_list:
        # Xây dựng tên file log
        # Nếu có model_name -> logs/pb_t54500_raw.log
        # Nếu không (NN) -> logs/train_models.log
        if model_name:
            log_filename = f"{model_name}.log"
        else:
            script_basename = os.path.basename(script).replace('.py', '')
            log_filename = f"{script_basename}.log"
            
        log_path = os.path.join(LOG_DIR, log_filename)
        
        # Tạo câu lệnh
        cmd = [py_env, script, "--config-path", CONFIG_PATH]
        if model_name:
            cmd.extend(["--model-name", model_name])
        
        if "pkl" not in script:
            cmd.extend(["--device", str(gpu_id)])
            print(f"    ▶ [GPU {gpu_id}] Đang chạy: {model_name or os.path.basename(script)}")
            
        print(f"      📄 Logs -> {log_path}")
        
        # Mở file log để ghi
        with open(log_path, "w") as f_log:
            try:
                # stdout=f_log: Ghi print vào file
                # stderr=subprocess.STDOUT: Ghi cả lỗi vào cùng file đó
                subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError:
                # Nếu lỗi, ghi thêm dòng báo lỗi vào cuối file log
                f_log.write(f"\n\n[ERROR] Process failed with exit code 1.\n")
                print(f"    ❌ [GPU {gpu_id}] Lỗi! Kiểm tra file {log_path} để xem chi tiết.")
                return 

    print(f"✅ [GPU {gpu_id}] Hoàn thành mọi tác vụ!")

def main():
    # 1. Tạo thư mục logs nếu chưa có
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 2. Chạy bước tạo Fold (Vẫn in ra màn hình console để tiện nhìn)
    print("--- Bước 0: Tạo K-Folds ---")
    try:
        subprocess.run([RAPIDS_ENV, f"{BASE_PATH}/protlib/scripts/create_gkf.py", "--config-path", CONFIG_PATH], check=True)
    except Exception:
        print("Lỗi khi tạo folds. Dừng chương trình.")
        return

    print("-> Tạo Folds xong. Bắt đầu train song song...\n")

    # 3. Khởi động các Workers
    processes = []
    for gpu_id, cmds in tasks:
        p = multiprocessing.Process(target=run_worker, args=(gpu_id, cmds))
        processes.append(p)
        p.start()
        time.sleep(2) 

    # 4. Chờ hoàn thành
    for p in processes:
        p.join()

    print("\n🎉 TẤT CẢ ĐÃ XONG! Kiểm tra thư mục 'logs/' để xem kết quả.")

if __name__ == "__main__":
    main()