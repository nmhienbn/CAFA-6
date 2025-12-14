import os
import subprocess
import argparse
import sys
import time

# --- CẤU HÌNH ---
BASE_PATH = "."
CONFIG_PATH = "config.yaml"
LOG_DIR = "logs_ddp"
SCRIPT_PATH = f"protnn/scripts/train_gcn_ddp.py"
ENV_PYTHON = "pytorch-env/bin/python" # Python env chứa pytorch

# Danh sách ontology cần chạy.
# Vì chạy DDP chiếm toàn bộ GPU, ta sẽ chạy TUẦN TỰ từng ontology.
ONTOLOGIES = ["bp"] #, "mf", "cc"]

def run_ddp_task(ontology, num_gpus, num_workers, batch_size, log_to_file):
    print(f"\n========================================================")
    print(f"🚀 BẮT ĐẦU TRAINING DDP: {ontology.upper()}")
    print(f"   GPUs: {num_gpus} | CPU Workers/GPU: {num_workers} | Batch/GPU: {batch_size}")
    print(f"========================================================\n")
    print(f"{BASE_PATH}/{ENV_PYTHON}")
    # Xây dựng lệnh torchrun
    # torchrun tự động quản lý biến môi trường cho DDP
    cmd = [
        f"{BASE_PATH}/{ENV_PYTHON}", "-m", "torch.distributed.run",
        "--nproc_per_node", str(num_gpus),
        "--master_port", "29500", # Port mặc định
        f"{BASE_PATH}/{SCRIPT_PATH}",
        "--config-path", CONFIG_PATH,
        "--ontology", ontology,
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers)
    ]

    if log_to_file:
        cmd.append("--log-to-file")

    # Setup Environment
    current_env = os.environ.copy()
    abs_base_path = os.path.abspath(BASE_PATH)
    current_env["PYTHONPATH"] = f"{abs_base_path}:{current_env.get('PYTHONPATH', '')}"
    # OMP_NUM_THREADS nên set thấp để tránh xung đột với PyTorch DataLoader workers
    current_env["OMP_NUM_THREADS"] = "1" 

    # Xử lý logging
    if log_to_file:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"ddp_train_{ontology}.log")
        print(f"📄 Logs đang được ghi vào: {log_file}")
        
        with open(log_file, "w") as f:
            # Chạy subprocess và redirect toàn bộ stdout/stderr vào file
            try:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=current_env, check=True)
                print(f"✅ Hoàn thành {ontology.upper()} thành công.")
            except subprocess.CalledProcessError:
                f.write("\n\n[FATAL ERROR] Training process crashed.\n")
                print(f"❌ Lỗi khi chạy {ontology.upper()}. Kiểm tra file log.")
                return False
    else:
        # In trực tiếp ra màn hình
        try:
            subprocess.run(cmd, env=current_env, check=True)
            print(f"✅ Hoàn thành {ontology.upper()} thành công.")
        except subprocess.CalledProcessError:
            print(f"❌ Lỗi khi chạy {ontology.upper()}.")
            return False
            
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=8, help="Số lượng GPU sử dụng (mặc định 8)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size trên mỗi GPU (A100 80GB -> 4096 ok)")
    parser.add_argument("--no-log", action="store_true", help="Nếu set flag này, sẽ in log ra màn hình thay vì ghi file")
    args = parser.parse_args()

    # Tính toán số lượng CPU worker tối ưu
    # Bạn có 256 core, 8 GPU => 32 core/GPU.
    # Tuy nhiên PyTorch dataloader có overhead, set khoảng 24 là an toàn và hiệu quả.
    total_cores = os.cpu_count()
    workers_per_gpu = min(32, total_cores // args.gpus) 
    
    print(f"Hệ thống có {total_cores} cores. Sử dụng {workers_per_gpu} workers cho mỗi trong số {args.gpus} GPU.")

    start_time = time.time()

    # Chạy tuần tự từng Ontology (BP -> MF -> CC)
    # Vì mỗi cái dùng Full 8 GPU nên phải chạy tuần tự
    for ont in ONTOLOGIES:
        success = run_ddp_task(
            ontology=ont,
            num_gpus=args.gpus,
            num_workers=workers_per_gpu,
            batch_size=args.batch_size,
            log_to_file=not args.no_log
        )
        
        if not success:
            print("Dừng pipeline do có lỗi xảy ra.")
            break
        
        # Nghỉ 5s để giải phóng VRAM hoàn toàn trước khi qua cái mới
        time.sleep(5)

    print(f"\n🎉 TỔNG THỜI GIAN: {(time.time() - start_time)/60:.2f} phút.")