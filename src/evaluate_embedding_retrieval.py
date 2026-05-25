from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.evaluation import (
    aggregate_by_label,
    aggregate_by_subset,
    aggregate_overall,
    compute_retrieval,
    normalize_embeddings,
    parse_topk,
)
from src.utils.io import ensure_dir, load_ids
from src.utils.labels import detect_cluster_column, load_label_mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kNN retrieval evaluation for embedding space")
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--normalize", choices=["none", "l2"], default="l2")
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--label-column", type=str, default=None)
    p.add_argument("--include-self", action="store_true", help="Include self in nearest-neighbor ranking")
    p.add_argument("--exclude-same-label-column", type=str, default=None)
    p.add_argument("--by-cluster", type=Path, default=None, help="Optional clusters.csv for subset eval")
    p.add_argument("--cluster-column", type=str, default=None)
    p.add_argument("--noise-label", type=int, default=-1)
    p.add_argument(
        "--min-label-count",
        type=int,
        default=2,
        help="Minimum sample count per label for retrieval_by_label.csv export (default: 2).",
    )
    return p.parse_args()


def _subset_names(ids: list[str], by_cluster: Path | None, cluster_column: str | None, noise_label: int) -> np.ndarray:
    names = np.array(["all"] * len(ids), dtype=object)
    if by_cluster is None:
        return names

    cdf = pd.read_csv(by_cluster)
    ccol = detect_cluster_column(cdf, preferred=cluster_column)
    mapping = dict(zip(cdf["specimen_id"].astype(str), cdf[ccol].astype(int)))

    for i, sid in enumerate(ids):
        cid = mapping.get(sid, noise_label)
        names[i] = "leaf_core" if cid != noise_label else "leaf_noise"
    return names


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    emb = np.load(args.emb)
    ids = load_ids(args.ids)
    if emb.shape[0] != len(ids):
        raise ValueError("ids length mismatch with embeddings")

    labels_df, used_label_column = load_label_mapping(ids, args.labels, args.label_column)
    labels = labels_df["label"].astype(str).to_numpy()

    exclude_labels = None
    exclude_mask = None
    excluded_candidate_counts = None
    if args.exclude_same_label_column is not None:
        if args.labels is None:
            raise ValueError("--exclude-same-label-column を使う場合は --labels が必要です")
        labels_table = pd.read_csv(args.labels)
        if args.exclude_same_label_column not in labels_table.columns:
            raise ValueError(f"column '{args.exclude_same_label_column}' not found in labels file")
        exclude_df = labels_table[["specimen_id", args.exclude_same_label_column]].copy()
        exclude_df["specimen_id"] = exclude_df["specimen_id"].astype(str)
        exclude_df = exclude_df.drop_duplicates(subset=["specimen_id"], keep="first")
        aligned_exclude = pd.DataFrame({"specimen_id": ids}).merge(exclude_df, on="specimen_id", how="left")
        exclude_labels = aligned_exclude[args.exclude_same_label_column].fillna("unknown").astype(str).to_numpy()
        exclude_mask = exclude_labels[:, None] == exclude_labels[None, :]
        excluded_candidate_counts = exclude_mask.sum(axis=1).astype(int)
        if args.label_column == args.exclude_same_label_column:
            warnings.warn(
                "--label-column と --exclude-same-label-column が同じです。"
                "特に species 評価では正解候補を除外し、Top-k accuracy が0に近くなる可能性があります。",
                stacklevel=2,
            )

    topk = parse_topk(args.topk)
    emb = normalize_embeddings(emb.astype(np.float64), mode=args.normalize)

    results = compute_retrieval(
        emb=emb,
        labels=labels,
        topk=topk,
        metric=args.metric,
        exclude_self=not args.include_self,
        exclude_mask=exclude_mask,
    )

    overall_df = aggregate_overall(results.hit_matrix, labels, topk)
    by_label_df = aggregate_by_label(labels, results.hit_matrix, results.first_correct_rank, topk)
    label_counts = pd.Series(labels).value_counts().sort_index()
    excluded_counts = label_counts[label_counts < args.min_label_count]
    by_label_df = by_label_df[by_label_df["label"].astype(str).isin(label_counts[label_counts >= args.min_label_count].index)]
    by_label_df = by_label_df.reset_index(drop=True)

    subset_names = _subset_names(ids, args.by_cluster, args.cluster_column, args.noise_label)
    by_subset_df = aggregate_by_subset(subset_names, results.hit_matrix, topk)

    max_k = topk[-1]
    nn_rows: list[dict[str, object]] = []
    for i, sid in enumerate(ids):
        for rank in range(max_k):
            j = int(results.nearest_indices[i, rank])
            nn_rows.append(
                {
                    "query_id": sid,
                    "query_label": labels[i],
                    "rank": rank + 1,
                    "neighbor_id": ids[j],
                    "neighbor_label": labels[j],
                    "distance": float(results.nearest_distances[i, rank]),
                    "is_same_label": bool(labels[i] == labels[j]),
                    "query_subset": subset_names[i],
                    "query_exclude_label": exclude_labels[i] if exclude_labels is not None else None,
                    "neighbor_exclude_label": exclude_labels[j] if exclude_labels is not None else None,
                }
            )
    nn_df = pd.DataFrame(nn_rows)

    overall_topk = {f"top{k}": float(overall_df.loc[overall_df["k"] == k, "topk_accuracy"].iloc[0]) for k in topk}
    macro_topk = {f"top{k}": float(overall_df.loc[overall_df["k"] == k, "macro_topk_accuracy"].iloc[0]) for k in topk}

    summary = {
        "n_samples": int(emb.shape[0]),
        "label_column": used_label_column,
        "metric": args.metric,
        "normalize": args.normalize,
        "topk": topk,
        "overall_topk_accuracy": overall_topk,
        "macro_mean_topk_accuracy": macro_topk,
        "subsets_available": sorted(pd.unique(subset_names).tolist()),
        "excluded_labels_from_by_label": [
            {"label": str(label), "count": int(count)} for label, count in excluded_counts.items()
        ],
        "config": {
            "emb": str(args.emb),
            "ids": str(args.ids),
            "labels": str(args.labels) if args.labels else None,
            "by_cluster": str(args.by_cluster) if args.by_cluster else None,
            "exclude_self": not args.include_self,
            "min_label_count": int(args.min_label_count),
            "exclude_same_label_column": args.exclude_same_label_column,
        },
        "exclude_same_label_column": args.exclude_same_label_column,
        "excluded_pairs_total": int(np.sum(excluded_candidate_counts)) if excluded_candidate_counts is not None else 0,
        "excluded_candidates_per_query_mean": float(np.mean(excluded_candidate_counts)) if excluded_candidate_counts is not None else 0.0,
    }

    overall_df.to_csv(args.out / "retrieval_overall.csv", index=False)
    by_label_df.to_csv(args.out / "retrieval_by_label.csv", index=False)
    by_subset_df.to_csv(args.out / "retrieval_by_subset.csv", index=False)
    nn_df.to_csv(args.out / "nearest_neighbors.csv", index=False)
    (args.out / "retrieval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
