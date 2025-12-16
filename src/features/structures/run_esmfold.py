import torch
from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from Bio import SeqIO
from tqdm.auto import tqdm  # <--- THÊM THƯ VIỆN NÀY
import sys
import os
import argparse

# Hàm convert output giữ nguyên
def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def main(fasta_file, output_dir, gpu_id):
    device = f"cuda:{gpu_id}"
    
    # [TỐI ƯU 1] Set biến môi trường để giảm phân mảnh bộ nhớ
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # [TỐI ƯU 2] Load model trực tiếp bằng FP16 để tránh spike bộ nhớ lúc khởi tạo
    # print(f"GPU {gpu_id}: Loading Model...") 
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", 
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 # Load thẳng FP16
    )
    model = model.eval().to(device)
    
    # [TỐI ƯU 3] Thử bật tính năng chunking (Lưu ý: Chỉ hoạt động nếu bản HF hỗ trợ hoặc bạn dùng bản gốc)
    # Nếu dòng này báo lỗi AttributeError thì comment lại, nhưng nó là chìa khóa giảm VRAM.
    try:
        model.trunk.set_chunk_size(64)
    except AttributeError:
        pass # Bản HF hiện tại có thể không có method này

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_records = list(SeqIO.parse(fasta_file, "fasta"))
    pbar = tqdm(all_records, desc=f"GPU {gpu_id}", unit="prot", position=0, leave=True)
    
    for record in pbar:
        pdb_path = os.path.join(output_dir, f"{record.id}.pdb")
        
        if os.path.exists(pdb_path): 
            continue 

        pbar.set_postfix_str(f"ID: {record.id[:10]}", refresh=False)

        # Cắt sequence (bạn đã làm đúng)
        seq = str(record.seq)[:1024] 
        
        try:
            with torch.no_grad():
                tokenized_input = tokenizer(
                    [seq], 
                    return_tensors="pt", 
                    add_special_tokens=False
                )['input_ids'].to(device)
                
                # [TỐI ƯU 4] Sử dụng Flash Attention (PyTorch 2.0+) nếu có thể
                # Giúp giảm bộ nhớ và tăng tốc độ đáng kể
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                        output = model(tokenized_input)
                else:
                    output = model(tokenized_input)

                pdb_str = convert_outputs_to_pdb(output)[0]
                
                with open(pdb_path, "w") as f:
                    f.write(pdb_str)
            
            # [QUAN TRỌNG] Xóa biến output và dọn cache sau mỗi lần chạy
            del output
            del tokenized_input
            torch.cuda.empty_cache() # Bắt buộc có để tránh OOM tích tụ

        except RuntimeError as e:
            if "out of memory" in str(e):
                tqdm.write(f"GPU {gpu_id} SKIPPED {record.id}: OOM (Length {len(seq)})")
                torch.cuda.empty_cache() # Dọn dẹp để cứu vãn cho protein sau
            else:
                tqdm.write(f"GPU {gpu_id} Error {record.id}: {e}")
        except Exception as e:
             tqdm.write(f"GPU {gpu_id} Error {record.id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    main(args.fasta, args.out, args.gpu)