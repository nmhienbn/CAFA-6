import os
import subprocess
import multiprocessing
import time
import sys

# --- CẤU HÌNH ---
BASE_PATH = "./"  # Thư mục gốc chứa cả 'protlib' và 'protnn'
CONFIG_PATH = "config.yaml"
LOG_DIR = "logs"
PYTORCH_ENV = "pytorch-env/bin/python"

# --- KHAI BÁO TÁC VỤ ---
tasks = [
    ("2", "bp"),
    ("4", "mf"),
    ("7", "cc")
]

def run_worker(gpu_id, ontology):
    script_path = f"{BASE_PATH}/protnn/scripts/train_gcn.py"
    log_filename = f"gcn_{ontology}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    print(f"🚀 [GPU {gpu_id}] Worker started for Ontology: {ontology.upper()}")

    # --- KHẮC PHỤC LỖI IMPORT ---
    # 1. Lấy biến môi trường hiện tại
    current_env = os.environ.copy()
    # 2. Thêm đường dẫn tuyệt đối của BASE_PATH vào PYTHONPATH
    # Điều này giúp python trong subprocess nhìn thấy folder 'protlib'
    abs_base_path = os.path.abspath(BASE_PATH)
    current_env["PYTHONPATH"] = f"{abs_base_path}:{current_env.get('PYTHONPATH', '')}"

    cmd = [
        PYTORCH_ENV, script_path,
        "--config-path", CONFIG_PATH,
        "--ontology", ontology,
        "--device", str(gpu_id)
    ]

    print(f"    ▶ [GPU {gpu_id}] Đang chạy: train_gcn.py --ontology {ontology}")
    print(f"      📄 Logs -> {log_path}")

    with open(log_path, "w") as f_log:
        try:
            # Truyền env=current_env vào subprocess
            subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, env=current_env, check=True)
            print(f"✅ [GPU {gpu_id}] Hoàn thành ontology {ontology.upper()}!")
        except subprocess.CalledProcessError:
            f_log.write(f"\n\n[ERROR] Process failed with exit code 1.\n")
            print(f"❌ [GPU {gpu_id}] Lỗi khi chạy {ontology}! Kiểm tra file {log_path}.")

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"--- Bắt đầu huấn luyện GCN song song trên GPUs: {[t[0] for t in tasks]} ---\n")

    processes = []
    for gpu_id, ont in tasks:
        p = multiprocessing.Process(target=run_worker, args=(gpu_id, ont))
        processes.append(p)
        p.start()
        time.sleep(1)

    for p in processes:
        p.join()

    print("\n🎉 TẤT CẢ ĐÃ XONG! Kiểm tra thư mục 'logs/' để xem kết quả.")

if __name__ == "__main__":
    main()