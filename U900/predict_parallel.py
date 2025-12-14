import os
import subprocess
import multiprocessing
import time
import yaml

# --- CẤU HÌNH ---
BASE_PATH = "./"
CONFIG_PATH = "config.yaml"
LOG_DIR = "logs_gcn"
WORKER_SCRIPT = "protnn/scripts/predict_gcn.py" # Tên file worker bạn vừa lưu ở trên

# Python path (Sử dụng môi trường chứa PyTorch/Protnn)
PYTORCH_ENV = "pytorch-env/bin/python" 

# Số lượng GPU tối đa muốn sử dụng
NUM_GPUS = 8
BATCH_SIZE = "1024"
NUM_WORKERS = "4"

def run_worker(gpu_id, command_list):
    """Worker process: Chạy danh sách các lệnh trên GPU được chỉ định."""
    print(f"🚀 [GPU {gpu_id}] Worker started with {len(command_list)} tasks.")

    for py_env, script, config_path, tta_idx in command_list:
        log_filename = f"gcn_tta_{tta_idx}.log"
        log_path = os.path.join(LOG_DIR, log_filename)
        
        cmd = [
            py_env, script, 
            "--config-path", config_path,
            "--device", str(gpu_id),
            "--run-index", str(tta_idx),
            "--batch-size", BATCH_SIZE,
            "--num-workers", NUM_WORKERS
        ]

        print(f"    ▶ [GPU {gpu_id}] Running TTA Index {tta_idx}")
        
        with open(log_path, "w") as f_log:
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, check=True, env=env)
            except subprocess.CalledProcessError:
                f_log.write(f"\n\n[ERROR] Process failed for TTA {tta_idx}.\n")
                print(f"    ❌ [GPU {gpu_id}] Failed TTA {tta_idx}! Check logs.")
                return 

    print(f"✅ [GPU {gpu_id}] All tasks completed.")

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Đọc Config để biết có bao nhiêu TTA steps
    print(f"🔍 Đang đọc config từ {CONFIG_PATH}...")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    # Lấy danh sách TTA từ ontology đầu tiên (thường giống nhau cho cả 3)
    # Giả định cấu trúc config['gcn']['bp']['tta'] tồn tại
    first_onto = list(config['gcn'].keys())[0] # thường là 'bp'
    tta_configs = config['gcn'][first_onto]['tta']
    
    num_tta_tasks = len(tta_configs)
    print(f"📊 Tìm thấy {num_tta_tasks} cấu hình TTA cần chạy.")

    # 2. Phân chia tác vụ (Round-robin)
    # tasks = { '0': [cmd1, cmd2], '1': [cmd3], ... }
    tasks = {str(i): [] for i in range(NUM_GPUS)}
    
    for k in range(num_tta_tasks):
        gpu_id = str(k % NUM_GPUS) # Chia đều theo modulo
        
        # Tạo command tuple
        task_info = (PYTORCH_ENV, WORKER_SCRIPT, CONFIG_PATH, k)
        tasks[gpu_id].append(task_info)

    # Lọc bỏ các GPU không có việc (nếu ít task hơn GPU)
    active_tasks = [(gid, cmds) for gid, cmds in tasks.items() if cmds]

    print(f"🚀 Bắt đầu chạy song song trên {len(active_tasks)} GPU...\n")

    # 3. Khởi chạy multiprocessing
    processes = []
    for gpu_id, cmds in active_tasks:
        p = multiprocessing.Process(target=run_worker, args=(gpu_id, cmds))
        processes.append(p)
        p.start()
        time.sleep(1) # Delay nhẹ để tránh load data ồ ạt cùng lúc

    # 4. Chờ hoàn thành
    for p in processes:
        p.join()

    print(f"\n🎉 Đã chạy xong tất cả các tiến trình! Kiểm tra kết quả trong folder models/gcn.")

if __name__ == "__main__":
    main()