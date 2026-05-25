import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_exclude_same_label_not_in_neighbors(tmp_path: Path) -> None:
    emb = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=np.float64,
    )
    ids = ["a1", "a2", "b1", "b2"]

    emb_path = tmp_path / "emb.npy"
    ids_path = tmp_path / "ids.txt"
    labels_path = tmp_path / "labels.csv"
    out_dir = tmp_path / "out"

    np.save(emb_path, emb)
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    labels_path.write_text(
        "specimen_id,label_genus,label_species\n"
        "a1,G1,S1\n"
        "a2,G1,S1\n"
        "b1,G1,S2\n"
        "b2,G2,S2\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluate_embedding_retrieval",
            "--emb",
            str(emb_path),
            "--ids",
            str(ids_path),
            "--labels",
            str(labels_path),
            "--label-column",
            "label_genus",
            "--exclude-same-label-column",
            "label_species",
            "--out",
            str(out_dir),
            "--topk",
            "1,2",
        ],
        check=True,
    )

    with (out_dir / "nearest_neighbors.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        assert row["query_exclude_label"] != row["neighbor_exclude_label"]

    summary = json.loads((out_dir / "retrieval_summary.json").read_text(encoding="utf-8"))
    assert summary["exclude_same_label_column"] == "label_species"
    assert summary["excluded_pairs_total"] > 0
