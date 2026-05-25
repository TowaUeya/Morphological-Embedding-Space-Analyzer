from __future__ import annotations

from pathlib import Path

import pandas as pd

def infer_label_from_specimen_id(specimen_id: str) -> str:
    text = str(specimen_id).strip().replace('\\', '/')
    if not text:
        return "unknown"
    return text.split('/', 1)[0] or "unknown"


def _read_table(labels_path: Path) -> pd.DataFrame:
    suffix = labels_path.suffix.lower()
    if suffix != ".csv":
        raise ValueError("labels must be CSV format (.csv) and include a 'specimen_id' column")
    table = pd.read_csv(labels_path)
    if "specimen_id" not in table.columns:
        raise ValueError("labels.csv must contain 'specimen_id' column")
    table["specimen_id"] = table["specimen_id"].astype(str)
    return table


def load_label_mapping(
    specimen_ids: list[str],
    labels_path: Path | None,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return DataFrame(specimen_id,label) aligned to specimen_ids order."""
    ids_df = pd.DataFrame({"specimen_id": specimen_ids})

    if labels_path is None:
        out = ids_df.copy()
        out["label"] = out["specimen_id"].map(infer_label_from_specimen_id)
        return out, "inferred_from_ids"

    table = _read_table(labels_path)
    if label_column is None:
        used_label_column = "label" if "label" in table.columns else None
    else:
        used_label_column = label_column

    if used_label_column is None:
        raise ValueError(
            "labels.csv must include a label column. Pass --label-column when your labels are stored in another column."
        )
    if used_label_column not in table.columns:
        raise ValueError(f"label column '{used_label_column}' not found in labels.csv")

    reduced = table[["specimen_id", used_label_column]].copy()
    reduced = reduced.rename(columns={used_label_column: "label"})
    reduced = reduced.drop_duplicates(subset=["specimen_id"], keep="first")
    reduced["label"] = reduced["label"].astype(str)

    out = ids_df.merge(reduced, on="specimen_id", how="left")
    out["label"] = out["label"].fillna(out["specimen_id"].map(infer_label_from_specimen_id))
    return out, used_label_column


def detect_cluster_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred is not None:
        if preferred not in df.columns:
            raise ValueError(f"cluster column '{preferred}' not found")
        return preferred

    for col in ["cluster_id", "cluster", "label"]:
        if col in df.columns:
            return col
    raise ValueError("cluster column not found. expected one of: cluster_id, cluster, label")
