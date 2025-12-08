#!/usr/bin/env python
import argparse
import json
import os
import subprocess

import pandas as pd


def run_cmd(cmd):
    print("[CMD]", " ".join(cmd))
    res = subprocess.run(cmd)
    return res.returncode == 0


def load_taxon_ids_from_protein_taxon(protein_taxon_path: str, top_k: int | None):
    print(f"[INFO] Loading protein_taxon from {protein_taxon_path}")
    df = pd.read_csv(protein_taxon_path, sep=None, engine="python")
    if "taxon" not in df.columns:
        raise ValueError(
            f"protein_taxon file must contain 'taxon_id' column, "
            f"got columns={df.columns.tolist()}"
        )

    # Đếm số protein mỗi taxon, sort giảm dần
    stats = df["taxon"].value_counts().sort_values(ascending=False)
    taxon_ids = stats.index.astype(int).tolist()

    if top_k is not None and top_k > 0 and len(taxon_ids) > top_k:
        print(f"[INFO] Using TOP-{top_k} taxa by #proteins (from protein_taxon.tsv)")
        taxon_ids = taxon_ids[:top_k]

    print("[INFO] Top species by #proteins:")
    print(stats.head(10))
    return taxon_ids


def load_taxon_ids_from_json(taxon_json_path: str, top_k: int | None):
    print(f"[INFO] Loading taxon json from {taxon_json_path}")
    with open(taxon_json_path, "r") as f:
        j = json.load(f)

    # 1) {"root": {"taxon2idx": {...}}}
    # 2) {"taxon2idx": {...}}
    if isinstance(j, dict) and "root" in j and isinstance(j["root"], dict) and "taxon2idx" in j["root"]:
        taxon2idx = j["root"]["taxon2idx"]
    elif isinstance(j, dict) and "taxon2idx" in j:
        taxon2idx = j["taxon2idx"]
    else:
        raise ValueError(
            "Cannot find 'taxon2idx' in JSON. Expected either "
            '{"root": {"taxon2idx": {...}}} or {"taxon2idx": {...}}.'
        )

    if not isinstance(taxon2idx, dict):
        raise ValueError("taxon2idx in JSON must be a dict")

    taxon_ids = sorted(int(k) for k in taxon2idx.keys())

    if top_k is not None and top_k > 0 and len(taxon_ids) > top_k:
        print(f"[INFO] Using first TOP-{top_k} taxa from taxon2idx (sorted by taxon_id)")
        taxon_ids = taxon_ids[:top_k]

    return taxon_ids


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download STRING v12 PPI (protein.links.full.v12.0) "
            "ONLY for species (taxon_id) used in CAFA6."
        )
    )

    # 2 nguồn: TSV hoặc JSON
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--protein_taxon_path",
        type=str,
        help="TSV/CSV with [protein_id, taxon_id] (train+test). "
             "Used to infer taxon_id list and rank by #proteins.",
    )
    group.add_argument(
        "--taxon_json_path",
        type=str,
        help="JSON containing taxon2idx (e.g. {\"root\": {\"taxon2idx\": {...}}}). "
             "Keys of taxon2idx are taxon_id.",
    )

    parser.add_argument(
        "--top_k_taxa",
        type=int,
        default=0,
        help="If >0, only download top-K taxa "
             "(by #proteins if using protein_taxon_path; "
             "otherwise first K sorted taxon_id for JSON). "
             "0 = download all taxa.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save *.protein.links.full.v12.0.txt.gz",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print URLs, do not actually download.",
    )

    args = parser.parse_args()

    # 1) lấy list taxon_id
    if args.taxon_json_path is not None:
        taxon_ids = load_taxon_ids_from_json(args.taxon_json_path, args.top_k_taxa)
    else:
        taxon_ids = load_taxon_ids_from_protein_taxon(args.protein_taxon_path, args.top_k_taxa)

    print(f"[INFO] Final #taxa to download: {len(taxon_ids)}")
    print("[INFO] Example taxa:", taxon_ids[:10])

    os.makedirs(args.out_dir, exist_ok=True)
    base = "https://stringdb-downloads.org/download/protein.links.full.v12.0"

    # 2) loop tải
    for i, tax in enumerate(taxon_ids):
        tax_str = str(tax)
        fname = f"{tax_str}.protein.links.full.v12.0.txt.gz"
        url = f"{base}/{fname}"
        dest = os.path.join(args.out_dir, fname)

        print(f"[INFO] [{i+1}/{len(taxon_ids)}] species={tax_str}")
        print(f"       URL:  {url}")
        print(f"       OUT:  {dest}")

        if args.dry_run:
            continue

        if os.path.exists(dest):
            print("[INFO] File already exists, skip.")
            continue

        cmd = ["wget", "-c", url, "-O", dest]
        ok = run_cmd(cmd)
        if not ok:
            print(f"[WARN] Failed to download {url} (maybe not in STRING v12).")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
