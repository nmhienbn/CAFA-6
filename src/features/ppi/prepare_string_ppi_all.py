#!/usr/bin/env python
import argparse
import glob
import os

import pandas as pd


def load_one_file(path: str, score_col: str, min_score: int):
    print(f"[INFO] Loading {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    # STRING thường có protein1, protein2, combined_score
    for c in ("protein1", "protein2", score_col):
        if c not in df.columns:
            raise ValueError(
                f"{os.path.basename(path)} must contain columns "
                f"protein1, protein2, {score_col}, got {df.columns.tolist()}"
            )

    df = df[["protein1", "protein2", score_col]].copy()
    df = df.rename(columns={score_col: "weight"})

    if min_score is not None and min_score > 0:
        df = df[df["weight"] >= min_score]

    # gộp edge trùng trong file, lấy max weight
    df = df.groupby(["protein1", "protein2"], as_index=False)["weight"].max()
    print(f"[INFO] {os.path.basename(path)}: |E|={len(df)} after filter (min_score={min_score})")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Merge STRING protein.links.full.* for ALL species into one PPI edge list"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing many STRING files *.protein.links.full.*.txt(.gz)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.protein.links.full*.txt*",
        help="Glob pattern inside input_dir (default: *.protein.links.full*.txt*)",
    )
    parser.add_argument(
        "--score_col",
        type=str,
        default="combined_score",
        help="Score column name in STRING files (usually combined_score)",
    )
    parser.add_argument(
        "--min_score",
        type=int,
        default=400,
        help="Min score to keep an edge (0 = keep all)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output TSV path with columns [protein1, protein2, weight] (STRING IDs)",
    )

    args = parser.parse_args()

    pattern_path = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob.glob(pattern_path))
    if not files:
        raise FileNotFoundError(f"No files matched pattern {pattern_path}")

    print(f"[INFO] Found {len(files)} STRING PPI files")

    dfs = []
    for i, f in enumerate(files):
        print(f"[INFO] [{i+1}/{len(files)}] {f}")
        df = load_one_file(f, score_col=args.score_col, min_score=args.min_score)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Chuẩn hóa edge vô hướng: sort (protein1, protein2)
    mask = df_all["protein1"] > df_all["protein2"]
    df_all.loc[mask, ["protein1", "protein2"]] = df_all.loc[
        mask, ["protein2", "protein1"]
    ].values

    # Gộp các edge trùng across species files (thực ra sẽ hiếm khi trùng nếu taxon khác)
    df_all = df_all.groupby(["protein1", "protein2"], as_index=False)["weight"].max()
    print(f"[INFO] After merging ALL species: |E|={len(df_all)} unique edges")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df_all.to_csv(args.out_path, sep="\t", index=False)
    print(f"[INFO] Saved unified PPI edge list to {args.out_path}")


if __name__ == "__main__":
    main()
