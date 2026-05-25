from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-ready tables/figures from retrieval/leaf analysis outputs.")
    p.add_argument("--retrieval-overall", type=Path, default=None, help="Path to retrieval_overall.csv")
    p.add_argument("--retrieval-by-label", type=Path, default=None, help="Path to retrieval_by_label.csv")
    p.add_argument("--leaf-summary", type=Path, default=None, help="Path to leaf_core_summary.json")
    p.add_argument("--leaf-coverage", type=Path, default=None, help="Path to category_coverage.csv")
    p.add_argument(
        "--retrieval-by-subset",
        type=Path,
        default=None,
        help="Path to retrieval_by_subset.csv (leaf_core vs leaf_noise)",
    )
    p.add_argument("--cluster-purity", type=Path, default=None, help="Optional path to cluster_purity.csv")
    p.add_argument("--out", type=Path, required=True, help="Output directory for generated tables/figures")
    return p.parse_args()


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_to_original = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_original:
            return lower_to_original[c.lower()]
    if required:
        raise KeyError(f"Missing required column. candidates={candidates}, available={list(df.columns)}")
    return None


def _topk_from_long_format(
    df: pd.DataFrame,
    value_candidates: list[str],
    required_k: list[int],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    k_col = _pick_col(df, ["k"])
    v_col = _pick_col(df, value_candidates)
    working = df.copy()
    working[k_col] = pd.to_numeric(working[k_col], errors="coerce")
    missing_k = [k for k in required_k if k not in set(working[k_col].dropna().astype(int).tolist())]
    if missing_k:
        raise KeyError(f"Missing required k values {missing_k} in long-format table.")

    pivot = working.pivot_table(index=group_cols, columns=k_col, values=v_col, aggfunc="first")
    for k in required_k:
        if k not in pivot.columns:
            raise KeyError(f"Missing required k={k} after pivot.")
    pivot = pivot[required_k].rename(columns={k: f"top{k}" for k in required_k})
    return pivot.reset_index()


def _coerce_topk_columns(df: pd.DataFrame, required_k: list[int], group_cols: list[str] | None = None) -> pd.DataFrame:
    # Wide format support: top1/top_1/top1_accuracy/top_1_accuracy ...
    wide_candidates = {
        k: [f"top{k}", f"top_{k}", f"top{k}_accuracy", f"top_{k}_accuracy"]
        for k in required_k
    }
    if all(any(any(c.lower() == col.lower() for col in df.columns) for c in cols) for cols in wide_candidates.values()):
        out = df.copy()
        rename_map = {}
        for k, candidates in wide_candidates.items():
            src = _pick_col(df, candidates)
            rename_map[src] = f"top{k}"
        out = out.rename(columns=rename_map)
        keep_cols = (group_cols or []) + [f"top{k}" for k in required_k]
        return out[keep_cols].copy()

    # Long format support: k + topk_accuracy
    return _topk_from_long_format(
        df=df,
        value_candidates=["topk_accuracy", "accuracy"],
        required_k=required_k,
        group_cols=group_cols,
    )


def save_table1_overall(retrieval_overall_path: Path, out_tables: Path) -> Path:
    retrieval_overall = pd.read_csv(retrieval_overall_path)
    overall_topk = _coerce_topk_columns(retrieval_overall, required_k=[1, 5, 10])
    table1 = pd.DataFrame(
        [
            {
                "evaluation": "all_samples",
                "top1": overall_topk.iloc[0]["top1"],
                "top5": overall_topk.iloc[0]["top5"],
                "top10": overall_topk.iloc[0]["top10"],
            }
        ]
    )
    out_path = out_tables / "table1_knn_overall.csv"
    table1.to_csv(out_path, index=False)
    return out_path


def save_table2_leaf_summary(leaf_summary_path: Path, out_tables: Path) -> Path:
    with leaf_summary_path.open("r", encoding="utf-8") as f:
        leaf_summary = json.load(f)
    table2 = pd.DataFrame(
        [
            {
                "n_total": leaf_summary.get("n_total"),
                "n_core": leaf_summary.get("n_core"),
                "n_noise": leaf_summary.get("n_noise"),
                "coverage": leaf_summary.get("coverage"),
                "noise_ratio": leaf_summary.get("noise_ratio"),
                "n_clusters": leaf_summary.get("n_clusters"),
                "weighted_purity": leaf_summary.get("weighted_purity"),
                "macro_purity": leaf_summary.get("macro_purity"),
            }
        ]
    )
    out_path = out_tables / "table2_leaf_core_summary.csv"
    table2.to_csv(out_path, index=False)
    return out_path


def save_table3_subset(retrieval_by_subset_path: Path, out_tables: Path) -> Path:
    retrieval_subset = pd.read_csv(retrieval_by_subset_path)
    subset_col = _pick_col(retrieval_subset, ["subset", "group"])
    table3 = _coerce_topk_columns(retrieval_subset, required_k=[1, 5, 10], group_cols=[subset_col]).copy()
    table3 = table3.rename(columns={subset_col: "subset"})
    out_path = out_tables / "table3_leaf_core_vs_noise_retrieval.csv"
    table3.to_csv(out_path, index=False)
    return out_path


def _existing_input(path: Path | None, label: str) -> Path | None:
    if path is None:
        print(f"[INFO] skip {label}: input not specified.")
        return None
    if not path.exists():
        print(f"[INFO] skip {label}: input does not exist: {path}")
        return None
    return path


def _log_generated(path: Path) -> None:
    print(f"[INFO] generated: {path}")


def plot_category_topk(retrieval_by_label: pd.DataFrame, out_figs: Path) -> None:
    label_col = _pick_col(retrieval_by_label, ["label", "category", "class_name"])
    df = _coerce_topk_columns(retrieval_by_label, required_k=[1, 5, 10], group_cols=[label_col]).copy()
    df = df.rename(columns={label_col: "label"})
    df = df.sort_values("top1", ascending=False)

    x = range(len(df))
    width = 0.28
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.35), 6))
    ax.bar([i - width for i in x], df["top1"], width=width, label="Top-1")
    ax.bar(x, df["top5"], width=width, label="Top-5")
    ax.bar([i + width for i in x], df["top10"], width=width, label="Top-10")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=75, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Category-wise kNN Top-k Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_figs / "figure2_category_topk_accuracy.png", dpi=200)
    plt.close(fig)


def plot_leaf_coverage(coverage_df: pd.DataFrame, out_figs: Path) -> None:
    label_col = _pick_col(coverage_df, ["label", "category", "class_name"])
    cov_col = _pick_col(coverage_df, ["coverage", "core_coverage", "leaf_coverage"])
    df = coverage_df[[label_col, cov_col]].copy()
    df.columns = ["label", "coverage"]
    df = df.sort_values("coverage", ascending=False)

    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.35), 6))
    ax.bar(df["label"], df["coverage"])
    ax.tick_params(axis="x", labelrotation=75)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Coverage")
    ax.set_title("Category-wise Leaf Coverage")
    fig.tight_layout()
    fig.savefig(out_figs / "figure3_category_leaf_coverage.png", dpi=200)
    plt.close(fig)


def plot_coverage_vs_top1(retrieval_by_label: pd.DataFrame, coverage_df: pd.DataFrame, out_figs: Path) -> None:
    r_label = _pick_col(retrieval_by_label, ["label", "category", "class_name"])
    retrieval_topk = _coerce_topk_columns(retrieval_by_label, required_k=[1], group_cols=[r_label])
    c_label = _pick_col(coverage_df, ["label", "category", "class_name"])
    c_cov = _pick_col(coverage_df, ["coverage", "core_coverage", "leaf_coverage"])

    merged = pd.merge(
        retrieval_topk[[r_label, "top1"]].rename(columns={r_label: "label"}),
        coverage_df[[c_label, c_cov]].rename(columns={c_label: "label", c_cov: "coverage"}),
        on="label",
        how="inner",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["coverage"], merged["top1"], alpha=0.85)
    for _, row in merged.iterrows():
        ax.annotate(str(row["label"]), (row["coverage"], row["top1"]), fontsize=8, alpha=0.75)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Leaf coverage")
    ax.set_ylabel("Top-1 retrieval accuracy")
    ax.set_title("Coverage × Top-1 Retrieval Accuracy")
    fig.tight_layout()
    fig.savefig(out_figs / "figure4_coverage_vs_top1_scatter.png", dpi=220)
    plt.close(fig)

    merged.to_csv(out_figs / "figure4_coverage_vs_top1_data.csv", index=False)


def plot_cluster_purity(cluster_purity_path: Path, out_figs: Path) -> Path | None:
    purity_df = pd.read_csv(cluster_purity_path)
    cluster_col = _pick_col(purity_df, ["cluster_id", "cluster"])
    purity_col = _pick_col(purity_df, ["purity"], required=False)
    label_col = _pick_col(purity_df, ["majority_label", "label", "major_label"], required=False)
    if purity_col is None:
        return None

    plot_df = purity_df[[cluster_col, purity_col]].copy()
    if label_col is not None:
        plot_df["label"] = purity_df[label_col].astype(str)
        plot_df["cluster_text"] = plot_df[cluster_col].astype(str) + ":" + plot_df["label"]
    else:
        plot_df["cluster_text"] = plot_df[cluster_col].astype(str)
    plot_df = plot_df.sort_values(purity_col, ascending=False).head(40)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["cluster_text"], plot_df[purity_col])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Purity")
    ax.set_title("Cluster Purity (Top 40 clusters by purity)")
    ax.tick_params(axis="x", labelrotation=80, labelsize=8)
    fig.tight_layout()
    out_path = out_figs / "figure5_cluster_purity.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)
    out_tables = args.out / "tables"
    out_figs = args.out / "figures"
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    retrieval_overall_path = _existing_input(args.retrieval_overall, "table1_knn_overall.csv")
    if retrieval_overall_path is not None:
        _log_generated(save_table1_overall(retrieval_overall_path, out_tables))

    leaf_summary_path = _existing_input(args.leaf_summary, "table2_leaf_core_summary.csv")
    if leaf_summary_path is not None:
        _log_generated(save_table2_leaf_summary(leaf_summary_path, out_tables))

    retrieval_by_subset_path = _existing_input(args.retrieval_by_subset, "table3_leaf_core_vs_noise_retrieval.csv")
    if retrieval_by_subset_path is not None:
        _log_generated(save_table3_subset(retrieval_by_subset_path, out_tables))

    retrieval_by_label_path = _existing_input(args.retrieval_by_label, "figure2_category_topk_accuracy.png")
    retrieval_by_label = None
    if retrieval_by_label_path is not None:
        retrieval_by_label = pd.read_csv(retrieval_by_label_path)
        print("[INFO] per-label plots exclude labels with n < min_label_count (based on retrieval_by_label.csv).")
        plot_category_topk(retrieval_by_label, out_figs)
        _log_generated(out_figs / "figure2_category_topk_accuracy.png")

    leaf_coverage_path = _existing_input(args.leaf_coverage, "figure3_category_leaf_coverage.png")
    coverage_df = None
    if leaf_coverage_path is not None:
        coverage_df = pd.read_csv(leaf_coverage_path)
        plot_leaf_coverage(coverage_df, out_figs)
        _log_generated(out_figs / "figure3_category_leaf_coverage.png")

    if retrieval_by_label is not None and coverage_df is not None:
        plot_coverage_vs_top1(retrieval_by_label, coverage_df, out_figs)
        _log_generated(out_figs / "figure4_coverage_vs_top1_scatter.png")
        _log_generated(out_figs / "figure4_coverage_vs_top1_data.csv")
    else:
        print(
            "[INFO] skip figure4_coverage_vs_top1_scatter.png and figure4_coverage_vs_top1_data.csv: "
            "requires both --retrieval-by-label and --leaf-coverage."
        )

    cluster_purity_path = _existing_input(args.cluster_purity, "figure5_cluster_purity.png")
    if cluster_purity_path is not None:
        cluster_purity_out = plot_cluster_purity(cluster_purity_path, out_figs)
        if cluster_purity_out is not None:
            _log_generated(cluster_purity_out)
        else:
            print("[INFO] skip figure5_cluster_purity.png: purity column not found.")

    print(f"[OK] tables -> {out_tables}")
    print(f"[OK] figures -> {out_figs}")


if __name__ == "__main__":
    main()
