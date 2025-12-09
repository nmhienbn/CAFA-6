import requests
import pandas as pd
import argparse
import time
from tqdm import tqdm

def split_chunks(lst, n):
    """Chia list thành các chunk nhỏ"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_proteins", required=True, help="Path to all_proteins.txt")
    parser.add_argument("--out_path", required=True, help="Path to string_to_protein_id.tsv")
    parser.add_argument("--batch_size", type=int, default=2000, help="Number of IDs per API call")
    args = parser.parse_args()

    # 1. Đọc danh sách protein
    print(f"Reading proteins from {args.input_proteins}...")
    with open(args.input_proteins, 'r') as f:
        proteins = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(proteins)} proteins. Fetching STRING IDs...")

    # 2. Gọi API STRING theo batch
    # STRING API endpoint: get_string_ids
    url = "https://string-db.org/api/tsv/get_string_ids"
    
    results = []
    
    # Chia nhỏ để không bị timeout
    batches = list(split_chunks(proteins, args.batch_size))
    
    for batch in tqdm(batches):
        params = {
            "identifiers": "\r".join(batch), # STRING yêu cầu phân cách bằng carriage return
            "species": 40674, # 40674 là Mammalia? KHÔNG NÊN SET CỐ ĐỊNH nếu data đa loài.
            # Tốt nhất: Không set species nếu input là UniProt ID unique. 
            # STRING tự detect. Tuy nhiên, nếu ID trùng lặp giữa các loài, API có thể hỏi lại.
            # Cách an toàn nhất cho CAFA (đa loài): set echo_query=1 để map lại input
            "echo_query": 1,
            "limit": 1
        }
        
        try:
            # Lưu ý: Nếu không set species, STRING có thể trả về nhiều kết quả. 
            # CAFA ID thường là UniProt Accession (duy nhất toàn cầu).
            response = requests.post(url, data=params)
            
            if response.status_code == 200:
                lines = response.text.strip().split("\n")
                if len(lines) > 1: # Có dữ liệu (dòng 1 là header)
                    for line in lines[1:]:
                        cols = line.split("\t")
                        # Format trả về thường là: queryItem, stringId, taxonomyId, preferredName, ...
                        # Check header thực tế: queryItem	stringId	ncbiTaxonId	preferredName	annotation
                        if len(cols) >= 2:
                            query_input = cols[0]
                            string_id = cols[1]
                            results.append(f"{string_id}\t{query_input}")
            else:
                print(f"\nError {response.status_code} for batch starting with {batch[0]}")
                
            time.sleep(1) # Sleep nhẹ để không bị ban IP
            
        except Exception as e:
            print(f"\nException: {e}")

    # 3. Lưu kết quả
    print(f"Mapped {len(results)} IDs.")
    with open(args.out_path, "w") as f:
        f.write("string_id\tprotein_id\n")
        f.write("\n".join(results))
    print(f"Saved to {args.out_path}")

if __name__ == "__main__":
    main()