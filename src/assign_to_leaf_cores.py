from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.evaluation import normalize_embeddings
from src.utils.io import ensure_dir, load_ids
from src.utils.labels import detect_cluster_column


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assign leaf-noise points to nearest high-confidence cores")
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--clusters", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--normalize", choices=["none", "l2"], default="l2")
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--cluster-column", type=str, default=None)
    p.add_argument("--noise-label", type=int, default=-1)
    p.add_argument("--core-repr", choices=["medoid", "centroid"], default="medoid")
    p.add_argument("--distance-quantile", type=float, default=0.95)
    p.add_argument("--margin-ratio", type=float, default=1.15)
    p.add_argument("--no-assign-core-points", action="store_true")
    return p.parse_args()


def _pairwise_distance(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        aa = np.sum(a * a, axis=1, keepdims=True)
        bb = np.sum(b * b, axis=1, keepdims=True).T
        d2 = np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)
        return np.sqrt(d2, dtype=np.float64)
    if metric == "cosine":
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        denom = np.maximum(a_norm @ b_norm.T, 1e-12)
        cos = (a @ b.T) / denom
        cos = np.clip(cos, -1.0, 1.0)
        return 1.0 - cos
    raise ValueError(f"unsupported metric: {metric}")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    emb = np.load(args.emb).astype(np.float64)
    ids = load_ids(args.ids)
    if emb.shape[0] != len(ids):
        raise ValueError("ids length mismatch with embeddings")

    emb = normalize_embeddings(emb, args.normalize)

    cdf = pd.read_csv(args.clusters)
    ccol = detect_cluster_column(cdf, preferred=args.cluster_column)
    mapping = dict(zip(cdf["specimen_id"].astype(str), cdf[ccol].astype(int)))

    original_cluster = np.array([mapping.get(sid, args.noise_label) for sid in ids], dtype=int)
    core_mask = original_cluster != args.noise_label
    noise_mask = ~core_mask

    core_ids = sorted(int(c) for c in np.unique(original_cluster[core_mask]))
    if not core_ids:
        raise ValueError("no non-noise cores were found")

    core_rep_vectors = []
    core_rep_sid = []
    core_sizes = []
    core_radii = []

    for cid in core_ids:
        idx = np.where(original_cluster == cid)[0]
        core_sizes.append(int(idx.size))
        points = emb[idx]

        if args.core_repr == "centroid":
            rep = np.mean(points, axis=0)
            rep = rep.reshape(1, -1)
            dist = _pairwise_distance(points, rep, args.metric).reshape(-1)
            rep_sid = ids[int(idx[np.argmin(dist)])]
            rep_vec = rep.reshape(-1)
        else:
            dm = _pairwise_distance(points, points, args.metric)
            medoid_local = int(np.argmin(dm.sum(axis=1)))
            rep_idx = int(idx[medoid_local])
            rep_sid = ids[rep_idx]
            rep_vec = emb[rep_idx]
            dist = dm[:, medoid_local]

        radius = float(np.quantile(dist, args.distance_quantile))
        core_rep_vectors.append(rep_vec)
        core_rep_sid.append(rep_sid)
        core_radii.append(radius)

    core_rep_matrix = np.stack(core_rep_vectors)

    assigned_cluster = np.full(len(ids), args.noise_label, dtype=int)
    assignment_status = np.array(["unassigned_noise"] * len(ids), dtype=object)
    accepted = np.zeros(len(ids), dtype=bool)

    d1_all = np.full(len(ids), np.nan, dtype=float)
    d2_all = np.full(len(ids), np.nan, dtype=float)
    ratio_all = np.full(len(ids), np.nan, dtype=float)

    if not args.no_assign_core_points:
        assigned_cluster[core_mask] = original_cluster[core_mask]
        assignment_status[core_mask] = "core"
        accepted[core_mask] = True

    noise_indices = np.where(noise_mask)[0]
    if noise_indices.size > 0:
        noise_to_core = _pairwise_distance(emb[noise_indices], core_rep_matrix, args.metric)
        nearest_order = np.argsort(noise_to_core, axis=1)

        for row_i, global_i in enumerate(noise_indices):
            n1 = int(nearest_order[row_i, 0])
            n2 = int(nearest_order[row_i, 1]) if len(core_ids) > 1 else n1

            d1 = float(noise_to_core[row_i, n1])
            d2 = float(noise_to_core[row_i, n2])

            if d1 == 0.0:
                ratio = float("inf") if d2 > 0 else 1.0
            else:
                ratio = float(d2 / d1)

            d1_all[global_i] = d1
            d2_all[global_i] = d2
            ratio_all[global_i] = ratio

            target_radius = core_radii[n1]
            ok = (d1 <= target_radius) and (ratio >= args.margin_ratio)
            if ok:
                assigned_cluster[global_i] = core_ids[n1]
                assignment_status[global_i] = "assigned_from_noise"
                accepted[global_i] = True

    assigned_df = pd.DataFrame(
        {
            "specimen_id": ids,
            "original_cluster_id": original_cluster,
            "assigned_cluster_id": assigned_cluster,
            "assignment_status": assignment_status,
            "distance_to_nearest_core": d1_all,
            "distance_to_second_core": d2_all,
            "margin_ratio": ratio_all,
            "accepted": accepted,
        }
    )

    core_rep_df = pd.DataFrame(
        {
            "cluster_id": core_ids,
            "core_repr": [args.core_repr] * len(core_ids),
            "representative_specimen_id": core_rep_sid,
            "core_size": core_sizes,
            "acceptance_radius": core_radii,
        }
    )

    # noise-only distance details
    noise_dist_rows = []
    for i in np.where(noise_mask)[0]:
        nearest_core = int(assigned_cluster[i]) if assignment_status[i] == "assigned_from_noise" else None
        if np.isnan(d1_all[i]):
            continue
        # Recover nearest and second by recomputing on representatives for readability
        drow = _pairwise_distance(emb[i : i + 1], core_rep_matrix, args.metric).reshape(-1)
        order = np.argsort(drow)
        n1 = core_ids[int(order[0])]
        n2 = core_ids[int(order[1])] if len(core_ids) > 1 else core_ids[int(order[0])]
        noise_dist_rows.append(
            {
                "specimen_id": ids[i],
                "nearest_core": n1,
                "second_core": n2,
                "d1": float(d1_all[i]),
                "d2": float(d2_all[i]),
                "ratio": float(ratio_all[i]),
                "accepted": bool(accepted[i]),
                "assigned_cluster_id": nearest_core,
            }
        )
    noise_dist_df = pd.DataFrame(noise_dist_rows)

    n_core_points = int(core_mask.sum())
    n_noise_points = int(noise_mask.sum())
    n_assigned_noise = int((assignment_status == "assigned_from_noise").sum())
    n_unassigned_noise = int((assignment_status == "unassigned_noise").sum())

    summary = {
        "n_total": int(len(ids)),
        "n_core_points": n_core_points,
        "n_noise_points": n_noise_points,
        "n_assigned_from_noise": n_assigned_noise,
        "n_unassigned_noise": n_unassigned_noise,
        "expanded_coverage": float((n_core_points + n_assigned_noise) / max(1, len(ids))),
        "original_coverage": float(n_core_points / max(1, len(ids))),
        "n_cores": int(len(core_ids)),
        "distance_quantile": args.distance_quantile,
        "margin_ratio": args.margin_ratio,
        "core_repr": args.core_repr,
        "config": {
            "emb": str(args.emb),
            "ids": str(args.ids),
            "clusters": str(args.clusters),
            "normalize": args.normalize,
            "metric": args.metric,
            "cluster_column": ccol,
            "noise_label": args.noise_label,
            "assign_core_points": not args.no_assign_core_points,
        },
    }

    assigned_df.to_csv(args.out / "assigned_clusters.csv", index=False)
    core_rep_df.to_csv(args.out / "core_representatives.csv", index=False)
    noise_dist_df.to_csv(args.out / "assignment_distances.csv", index=False)
    (args.out / "assignment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
