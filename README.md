# Morphological Embedding Space Analyzer: Retrieval, clustering, and visualization of DINOv2-based morphological embeddings
Analysis toolkit for evaluating nearest-neighbor retrieval, HDBSCAN-based auxiliary structure, leaf-core regions, residual samples, and publication figures.

## Requirements
- Python 3.10+
- See `requirements.txt` for analysis-focused dependencies.

## Installation
```bash
pip install -r requirements.txt
```

## Input Files
Expected inputs are precomputed embeddings and metadata:
- `embeddings.npy` (required)
- `ids.txt` (required)
- `labels.csv` (optional; needed for label-aware evaluation)
- `clusters.csv` (used by selected downstream analyses)

This repository does **not** generate embeddings. It analyzes externally prepared embedding files.

## Usage
### 1) Retrieval evaluation
```bash
python -m src.evaluate_embedding_retrieval \
  --embeddings embeddings.npy \
  --ids ids.txt \
  --labels labels.csv \
  --outdir results/retrieval
```

### 2) Clustering (baseline / optional auxiliary)
```bash
python -m src.cluster_baseline \
  --embeddings embeddings.npy \
  --ids ids.txt \
  --outdir results/cluster_baseline

python -m src.cluster_recursive_hdbscan \
  --embeddings embeddings.npy \
  --ids ids.txt \
  --outdir results/cluster_recursive

python -m src.cluster_branch_detector \
  --embeddings embeddings.npy \
  --ids ids.txt \
  --outdir results/cluster_branch_aux
```

### 3) Leaf core / residual analysis and figure generation
```bash
python -m src.analyze_leaf_cores \
  --embeddings embeddings.npy \
  --ids ids.txt \
  --clusters results/cluster_baseline/clusters.csv \
  --outdir results/leaf_cores

python -m src.assign_to_leaf_cores \
  --embeddings embeddings.npy \
  --ids ids.txt \
  --leaf-cores results/leaf_cores/leaf_cores.csv \
  --outdir results/leaf_assignments

python -m src.plot_publication_figures \
  --input-dir results \
  --outdir results/figures
```

## Outputs
Depending on the command, outputs include:
- Retrieval metrics and neighbor reports
- Cluster assignments and auxiliary HDBSCAN summaries
- Leaf-core and residual analysis tables
- Publication-ready figures and embedding-space visualizations

## Notes
- Prefer running modules via `python -m src.<module_name>`.
- `configs/visualization.yaml` can be used to tune visualization behavior.
- `tests/test_retrieval_exclude_same_label.py` is included for retrieval evaluation logic.

## Citation
```bibtex
@software{morphological_embedding_space_analyzer,
  title  = {Morphological Embedding Space Analyzer},
  author = {Your Name or Team},
  year   = {2026},
  url    = {https://github.com/your-org/morphological-embedding-space-analyzer}
}
```

## Links
* Source code: [https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer](https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer)
* Archived version: [https://doi.org/10.5281/zenodo.20258408](https://doi.org/10.5281/zenodo.20258408)

## Related Repositories

This repository provides the embedding-generation pipeline. Downstream analyses and related tools are maintained in the following repositories:

- **MultiView3D-DINOv2**  
  [https://github.com/TowaUeya/MultiView3D-DINOv2](https://github.com/TowaUeya/MultiView3D-DINOv2)  
  Software pipeline for rendering multi-view images from 3D specimens and extracting frozen DINOv2 embeddings.

- **Repository-Name-2**  
  https://github.com/TowaUeya/Repository-Name-2  
  Briefly describe what this repository is used for and how it relates to MultiView3D-DINO
