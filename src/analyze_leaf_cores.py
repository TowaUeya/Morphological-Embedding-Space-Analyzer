from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_ids
from src.utils.labels import detect_cluster_column, infer_label_from_specimen_id, load_label_mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze leaf clusters as high-confidence cores")
    p.add_argument("--clusters", type=Path, required=True)
    p.add_argument("--ids", type=Path, default=None)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label-column", type=str, default=None)
    p.add_argument("--cluster-column", type=str, default=None)
    p.add_argument("--noise-label", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    cdf = pd.read_csv(args.clusters)
    cdf["specimen_id"] = cdf["specimen_id"].astype(str)
    cluster_col = detect_cluster_column(cdf, preferred=args.cluster_column)
    cdf["cluster_id"] = cdf[cluster_col].astype(int)

    if args.ids is not None:
        specimen_ids = load_ids(args.ids)
    else:
        specimen_ids = cdf["specimen_id"].tolist()

    label_df, used_label_column = load_label_mapping(specimen_ids, args.labels, args.label_column)
    df = cdf[["specimen_id", "cluster_id"]].merge(label_df, on="specimen_id", how="right")
    df["cluster_id"] = df["cluster_id"].fillna(args.noise_label).astype(int)
    df["label"] = df["label"].fillna(df["specimen_id"].map(infer_label_from_specimen_id)).astype(str)

    core_df = df[df["cluster_id"] != args.noise_label].copy()
    noise_df = df[df["cluster_id"] == args.noise_label].copy()

    n_total = int(len(df))
    n_core = int(len(core_df))
    n_noise = int(len(noise_df))
    coverage = float(n_core / max(1, n_total))

    cluster_rows: list[dict[str, object]] = []
    weighted_correct = 0
    for cid, g in core_df.groupby("cluster_id"):
        counts = g["label"].value_counts()
        majority_label = str(counts.index[0])
        majority_count = int(counts.iloc[0])
        size = int(len(g))
        purity = float(majority_count / max(1, size))
        weighted_correct += majority_count
        cluster_rows.append(
            {
                "cluster_id": int(cid),
                "cluster_size": size,
                "majority_label": majority_label,
                "majority_count": majority_count,
                "purity": purity,
                "label_distribution_json": json.dumps({k: int(v) for k, v in counts.items()}, ensure_ascii=False),
            }
        )

    cluster_purity_df = pd.DataFrame(cluster_rows).sort_values(["cluster_size", "cluster_id"], ascending=[False, True])

    weighted_purity = float(weighted_correct / max(1, n_core)) if n_core > 0 else 0.0
    macro_purity = float(cluster_purity_df["purity"].mean()) if not cluster_purity_df.empty else 0.0

    category_rows: list[dict[str, object]] = []
    noise_rows: list[dict[str, object]] = []
    for label, g in df.groupby("label"):
        n_label_total = int(len(g))
        n_label_core = int((g["cluster_id"] != args.noise_label).sum())
        n_label_noise = int((g["cluster_id"] == args.noise_label).sum())

        core_clusters = core_df[core_df["label"] == label]["cluster_id"].value_counts()
        best_cluster_id = int(core_clusters.index[0]) if not core_clusters.empty else args.noise_label
        best_cluster_purity = float(
            cluster_purity_df.loc[cluster_purity_df["cluster_id"] == best_cluster_id, "purity"].iloc[0]
        ) if best_cluster_id != args.noise_label else 0.0

        category_rows.append(
            {
                "label": str(label),
                "n_total": n_label_total,
                "n_core": n_label_core,
                "n_noise": n_label_noise,
                "coverage": float(n_label_core / max(1, n_label_total)),
                "n_clusters_containing_label": int(core_clusters.size),
                "best_cluster_id": best_cluster_id,
                "best_cluster_purity": best_cluster_purity,
            }
        )
        noise_rows.append(
            {
                "label": str(label),
                "n_noise": n_label_noise,
                "noise_fraction_within_label": float(n_label_noise / max(1, n_label_total)),
            }
        )

    category_coverage_df = pd.DataFrame(category_rows).sort_values("label")
    noise_by_category_df = pd.DataFrame(noise_rows).sort_values("label")

    representative_clusters_df = cluster_purity_df[["cluster_id", "cluster_size", "majority_label", "purity"]].rename(
        columns={"majority_label": "representative_label"}
    )

    size_series = cluster_purity_df["cluster_size"] if not cluster_purity_df.empty else pd.Series(dtype=float)
    summary = {
        "n_total": n_total,
        "n_core": n_core,
        "n_noise": n_noise,
        "coverage": coverage,
        "noise_ratio": float(n_noise / max(1, n_total)),
        "n_clusters": int(cluster_purity_df.shape[0]),
        "mean_cluster_size": float(size_series.mean()) if not size_series.empty else 0.0,
        "median_cluster_size": float(size_series.median()) if not size_series.empty else 0.0,
        "weighted_purity": weighted_purity,
        "macro_purity": macro_purity,
        "config": {
            "clusters": str(args.clusters),
            "ids": str(args.ids) if args.ids else None,
            "labels": str(args.labels) if args.labels else None,
            "label_column": used_label_column,
            "cluster_column": cluster_col,
            "noise_label": args.noise_label,
        },
    }

    (args.out / "leaf_core_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cluster_purity_df.to_csv(args.out / "cluster_purity.csv", index=False)
    category_coverage_df.to_csv(args.out / "category_coverage.csv", index=False)
    noise_by_category_df.to_csv(args.out / "noise_by_category.csv", index=False)
    representative_clusters_df.to_csv(args.out / "representative_clusters.csv", index=False)


if __name__ == "__main__":
    main()
