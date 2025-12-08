#!/usr/bin/env python
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remap PPI edge list IDs to CAFA protein_id namespace "
            "using a mapping table (e.g. STRING/Ensembl/UniProt -> protein_id)."
        )
    )

    parser.add_argument("--ppi_in", type=str, required=True)
    parser.add_argument("--ppi_protein1_col", type=str, default="protein1")
    parser.add_argument("--ppi_protein2_col", type=str, default="protein2")
    parser.add_argument("--ppi_weight_col", type=str, default="weight")

    parser.add_argument("--mapping_path", type=str, required=True)
    parser.add_argument("--map_src_col", type=str, required=True)
    parser.add_argument("--map_tgt_col", type=str, default="protein_id")

    parser.add_argument("--drop_self", action="store_true")
    parser.add_argument(
        "--agg_method",
        type=str,
        default="max",
        choices=["max", "mean", "sum"],
    )

    parser.add_argument("--ppi_out", type=str, required=True)

    args = parser.parse_args()

    print(f"[INFO] Loading PPI from {args.ppi_in}")
    ppi = pd.read_csv(args.ppi_in, sep=None, engine="python")

    for c in [args.ppi_protein1_col, args.ppi_protein2_col]:
        if c not in ppi.columns:
            raise ValueError(f"PPI file must contain '{c}', got {ppi.columns.tolist()}")

    if args.ppi_weight_col not in ppi.columns:
        print(f"[WARN] No weight column '{args.ppi_weight_col}', setting weight=1.0")
        ppi[args.ppi_weight_col] = 1.0

    ppi = ppi[[args.ppi_protein1_col, args.ppi_protein2_col, args.ppi_weight_col]].copy()
    ppi = ppi.rename(
        columns={
            args.ppi_protein1_col: "p1_src",
            args.ppi_protein2_col: "p2_src",
            args.ppi_weight_col: "weight",
        }
    )
    print(f"[INFO] Raw edges: {len(ppi)}")

    print(f"[INFO] Loading mapping from {args.mapping_path}")
    mapping = pd.read_csv(args.mapping_path, sep=None, engine="python")
    for c in [args.map_src_col, args.map_tgt_col]:
        if c not in mapping.columns:
            raise ValueError(
                f"Mapping file must contain '{c}', got {mapping.columns.tolist()}"
            )

    mapping = mapping[[args.map_src_col, args.map_tgt_col]].copy()
    mapping[args.map_src_col] = mapping[args.map_src_col].astype(str)
    mapping[args.map_tgt_col] = mapping[args.map_tgt_col].astype(str)

    map_dict = dict(zip(mapping[args.map_src_col], mapping[args.map_tgt_col]))

    ppi["p1_src"] = ppi["p1_src"].astype(str)
    ppi["p2_src"] = ppi["p2_src"].astype(str)

    ppi["protein1"] = ppi["p1_src"].map(map_dict)
    ppi["protein2"] = ppi["p2_src"].map(map_dict)

    before = len(ppi)
    ppi = ppi.dropna(subset=["protein1", "protein2"])
    after_mapped = len(ppi)
    print(
        f"[INFO] Edges mapped on both sides: {after_mapped}/{before} "
        f"({after_mapped / max(before,1):.2%})"
    )

    if args.drop_self:
        before_self = len(ppi)
        ppi = ppi[ppi["protein1"] != ppi["protein2"]]
        after_self = len(ppi)
        print(
            f"[INFO] Dropped self-loops: {before_self - after_self}, remain {after_self}"
        )

    mask = ppi["protein1"] > ppi["protein2"]
    ppi.loc[mask, ["protein1", "protein2"]] = ppi.loc[
        mask, ["protein2", "protein1"]
    ].values

    if args.agg_method == "max":
        agg_df = ppi.groupby(["protein1", "protein2"], as_index=False)["weight"].max()
    elif args.agg_method == "mean":
        agg_df = ppi.groupby(["protein1", "protein2"], as_index=False)["weight"].mean()
    else:
        agg_df = ppi.groupby(["protein1", "protein2"], as_index=False)["weight"].sum()

    print(f"[INFO] After aggregation: |E|={len(agg_df)}")

    os.makedirs(os.path.dirname(args.ppi_out), exist_ok=True)
    agg_df.to_csv(args.ppi_out, sep="\t", index=False)
    print(f"[INFO] Saved remapped PPI to {args.ppi_out}")


if __name__ == "__main__":
    main()
