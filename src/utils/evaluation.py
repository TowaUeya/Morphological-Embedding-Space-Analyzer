from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def parse_topk(topk_text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in topk_text.split(",") if x.strip()})
    if not values or any(v <= 0 for v in values):
        raise ValueError("topk must contain positive integers")
    return values


def normalize_embeddings(emb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return emb
    if mode != "l2":
        raise ValueError(f"unsupported normalize mode: {mode}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def pairwise_distance_matrix(emb: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        sq = np.sum(emb * emb, axis=1, keepdims=True)
        d2 = np.maximum(sq + sq.T - 2.0 * emb @ emb.T, 0.0)
        return np.sqrt(d2, dtype=np.float64)
    if metric == "cosine":
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        denom = np.maximum(norms @ norms.T, 1e-12)
        cosine_sim = (emb @ emb.T) / denom
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        return 1.0 - cosine_sim
    raise ValueError(f"unsupported metric: {metric}")


@dataclass
class RetrievalResults:
    nearest_indices: np.ndarray
    nearest_distances: np.ndarray
    hit_matrix: np.ndarray
    first_correct_rank: np.ndarray


def compute_retrieval(
    emb: np.ndarray,
    labels: np.ndarray,
    topk: list[int],
    metric: str,
    exclude_self: bool,
    exclude_mask: np.ndarray | None = None,
) -> RetrievalResults:
    distances = pairwise_distance_matrix(emb, metric=metric)
    n = distances.shape[0]

    if exclude_self:
        np.fill_diagonal(distances, np.inf)

    if exclude_mask is not None:
        if exclude_mask.shape != distances.shape:
            raise ValueError("exclude_mask shape mismatch with distance matrix")
        distances = distances.copy()
        distances[exclude_mask] = np.inf

    order = np.argsort(distances, axis=1)
    max_k = topk[-1]
    nearest_indices = order[:, :max_k]
    nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)

    nn_labels = labels[nearest_indices]
    same = nn_labels == labels[:, None]

    hit_matrix = np.zeros((n, len(topk)), dtype=bool)
    for i, k in enumerate(topk):
        hit_matrix[:, i] = np.any(same[:, :k], axis=1)

    first_correct_rank = np.full(n, np.nan, dtype=float)
    for i in range(n):
        idx = np.where(same[i])[0]
        if idx.size > 0:
            first_correct_rank[i] = float(idx[0] + 1)

    return RetrievalResults(
        nearest_indices=nearest_indices,
        nearest_distances=nearest_distances,
        hit_matrix=hit_matrix,
        first_correct_rank=first_correct_rank,
    )


def aggregate_overall(hit_matrix: np.ndarray, labels: np.ndarray, topk: list[int]) -> pd.DataFrame:
    rows = []
    label_series = pd.Series(labels)
    per_label_counts = label_series.value_counts()

    for j, k in enumerate(topk):
        topk_acc = float(np.mean(hit_matrix[:, j])) if hit_matrix.shape[0] else 0.0
        macro_parts = []
        for label in per_label_counts.index:
            mask = labels == label
            if np.any(mask):
                macro_parts.append(float(np.mean(hit_matrix[mask, j])))
        macro_acc = float(np.mean(macro_parts)) if macro_parts else 0.0
        rows.append({"k": int(k), "topk_accuracy": topk_acc, "macro_topk_accuracy": macro_acc})

    return pd.DataFrame(rows)


def aggregate_by_label(
    labels: np.ndarray,
    hit_matrix: np.ndarray,
    first_correct_rank: np.ndarray,
    topk: list[int],
) -> pd.DataFrame:
    rows = []
    for label in sorted(pd.unique(labels)):
        mask = labels == label
        row: dict[str, object] = {"label": str(label), "n_queries": int(mask.sum())}
        for j, k in enumerate(topk):
            row[f"top{k}_accuracy"] = float(np.mean(hit_matrix[mask, j])) if np.any(mask) else 0.0
        values = first_correct_rank[mask]
        row["mean_rank_first_correct"] = float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_subset(
    subset_names: np.ndarray,
    hit_matrix: np.ndarray,
    topk: list[int],
) -> pd.DataFrame:
    rows = []
    for subset in ["all", "leaf_core", "leaf_noise"]:
        mask = subset_names == subset
        if not np.any(mask):
            continue
        row: dict[str, object] = {"subset": subset, "n_queries": int(mask.sum())}
        for j, k in enumerate(topk):
            row[f"top{k}_accuracy"] = float(np.mean(hit_matrix[mask, j]))
        rows.append(row)
    return pd.DataFrame(rows)
