#!/usr/bin/env python
import os
import argparse
import math
from collections import defaultdict

import numpy as np
from Bio import SeqIO


def load_fasta_ids(fasta_path):
    ids = []
    for r in SeqIO.parse(fasta_path, "fasta"):
        ids.append(str(r.id))
    return ids


def parse_top_k(s):
    ks = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        ks.append(int(x))
    ks = sorted(set(k for k in ks if k > 0))
    if not ks:
        ks = [1, 3, 5, 10]
    return ks


def parse_foldseek_m8(m8_path, max_hits_per_query=0):
    """
    FoldSeek tabular output (.m8) with default columns:
    query,target,fident,alnlen,mismatch,gapopen,
    qstart,qend,tstart,tend,evalue,bits

    We dùng:
      identity  = fident (col 3)
      evalue    = evalue (col 11)
      bitscore  = bits (col 12)
    """
    hits = defaultdict(list)

    with open(m8_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 12:
                continue

            qid = parts[0]
            try:
                identity = float(parts[2])   # fident
                evalue   = float(parts[10])
                bitscore = float(parts[11])
            except ValueError:
                continue

            hits[qid].append((bitscore, identity, evalue))

            if max_hits_per_query > 0 and len(hits[qid]) > max_hits_per_query:
                # keep best by bitscore
                hits[qid].sort(key=lambda x: x[0], reverse=True)
                hits[qid] = hits[qid][:max_hits_per_query]

    return hits


def compute_features_for_query(hit_list, top_ks, log_min=-300.0):
    """
    hit_list: list[(bitscore, identity, evalue)]
    Layout feature:
        0: num_hits
        1: best_bitscore
        2: best_identity
        3: best_log10e
        then for each k in top_ks:
           identity_top{k}_mean
           bitscore_top{k}_mean
           log10e_top{k}_mean
    """
    num_hits = len(hit_list)
    feat_dim = 4 + 3 * len(top_ks)
    feat = np.zeros(feat_dim, dtype=np.float32)

    feat[0] = float(num_hits)
    if num_hits == 0:
        return feat

    # sort by bitscore desc
    hit_list = sorted(hit_list, key=lambda x: x[0], reverse=True)

    best_bits, best_id, best_eval = hit_list[0]
    best_log10e = math.log10(best_eval) if best_eval > 0 else log_min

    feat[1] = float(best_bits)
    feat[2] = float(best_id)
    feat[3] = float(best_log10e)

    idx = 4
    for k in top_ks:
        top = hit_list[:k]
        if not top:
            idx += 3
            continue

        bits = [h[0] for h in top]
        ids  = [h[1] for h in top]
        logs = [math.log10(h[2]) if h[2] > 0 else log_min for h in top]

        feat[idx]     = float(np.mean(ids))
        feat[idx + 1] = float(np.mean(bits))
        feat[idx + 2] = float(np.mean(logs))
        idx += 3

    return feat


def build_feature_names(top_ks):
    names = [
        "num_hits",
        "best_bitscore",
        "best_identity",
        "best_log10e",
    ]
    for k in top_ks:
        names.append(f"identity_top{k}_mean")
        names.append(f"bitscore_top{k}_mean")
        names.append(f"log10e_top{k}_mean")
    return names


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--foldseek_tsv", type=str, required=True,
                        help="FoldSeek .m8 output")
    parser.add_argument("--fasta_path", type=str, required=True,
                        help="FASTA used as query (để lấy order id)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Dir để lưu .npy")
    parser.add_argument("--short_name", type=str, required=True,
                        help="Ví dụ: afdb50, afdb_sprot")

    parser.add_argument("--top_k", type=str, default="1,3,5,10",
                        help="VD '1,3,5,10'")
    parser.add_argument("--max_hits_per_query", type=int, default=0,
                        help="Cap hit/query (0 = không cap)")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"FASTA:        {args.fasta_path}")
    print(f"FoldSeek TSV: {args.foldseek_tsv}")
    print(f"Save dir:     {args.save_dir}")
    print(f"Short name:   {args.short_name}")
    print(f"top_k:        {args.top_k}")
    print(f"max_hits/qry: {args.max_hits_per_query}")

    top_ks = parse_top_k(args.top_k)

    # 1) Load ids theo đúng thứ tự CAFA6
    fasta_ids = load_fasta_ids(args.fasta_path)
    print(f"Loaded {len(fasta_ids)} ids from FASTA")

    # 2) Load hits
    hits_by_q = parse_foldseek_m8(
        args.foldseek_tsv,
        max_hits_per_query=args.max_hits_per_query
    )
    print(f"Found hits for {len(hits_by_q)} queries")

    # 3) Build feature matrix
    feat_names = build_feature_names(top_ks)
    feat_dim = len(feat_names)
    n = len(fasta_ids)

    feats = np.zeros((n, feat_dim), dtype=np.float32)

    for i, qid in enumerate(fasta_ids):
        hit_list = hits_by_q.get(qid, [])
        feats[i] = compute_features_for_query(hit_list, top_ks)

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{n}", end="\r", flush=True)

    print(f"\nFeature matrix shape: {feats.shape}")

    base_name = os.path.splitext(os.path.basename(args.fasta_path))[0]
    save_path = os.path.join(args.save_dir, args.short_name)
    os.makedirs(save_path, exist_ok=True)

    feat_path = os.path.join(save_path, f"{base_name}_foldseek_feats.npy")
    ids_path  = os.path.join(save_path, f"{base_name}_ids.npy")
    names_path = os.path.join(save_path, f"{base_name}_feat_names.txt")

    np.save(feat_path, feats)
    np.save(ids_path, np.array(fasta_ids))

    with open(names_path, "w") as f:
        for name in feat_names:
            f.write(name + "\n")

    print(f"Saved features:   {feat_path}")
    print(f"Saved ids:        {ids_path}")
    print(f"Saved feat names: {names_path}")


if __name__ == "__main__":
    main()
