# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリの概要

3D標本のDINOv2特徴量(形態学的埋め込み)に対する**下流の解析**ツールキット。埋め込み自体は**生成しない** — 事前計算された `embeddings.npy` + `ids.txt`(姉妹リポジトリ MultiView3D-DINOv2 が生成)を入力として消費し、検索評価(retrieval evaluation)、HDBSCANクラスタリング、リーフコア/残差(leaf-core/residual)抽出、図の生成を行う。

## コマンド

すべてのスクリプトはリポジトリのルートからモジュールとして実行する(import は `src.` パッケージ接頭辞を使う。例: `from src.utils.io import ...`):

```bash
python -m src.<module_name> --emb embeddings.npy --ids ids.txt --out results/<name>
```

- 依存パッケージのインストール: `pip install -r requirements.txt`(Python 3.10以上)
- テスト実行: `pytest`
- 単一テストの実行: `pytest tests/test_retrieval_exclude_same_label.py::test_exclude_same_label_not_in_neighbors`

注意: テストスイートは `subprocess` + `python -m src.evaluate_embedding_retrieval` 経由でCLIを起動するため、`src` がimport可能なリポジトリルートから実行する必要がある。

## データ契約(全スクリプト共通)

- `embeddings.npy` — 浮動小数点行列。行 `i` は `ids.txt` の `i` 行目に対応する。長さの不一致は致命的エラー(hard error)。
- `ids.txt` — 1行に1つの specimen_id(`load_ids` が空行を除去する)。
- `labels.csv` — `specimen_id` 列が必須。ラベル列のデフォルトは `label`(`--label-column` で上書き可)。存在しない場合、ラベルは specimen_id から**推論**される(最初の `/` より前のパス接頭辞を取る = `infer_label_from_specimen_id`)。
- `clusters.csv` — クラスタリングスクリプトが生成する。クラスタ列は優先順 `cluster_id` → `cluster` → `label` で自動検出される(`detect_cluster_column`)。`-1` はノイズを意味する。

## アーキテクチャ

`src/utils/` が共有プリミティブを持ち、トップレベルの `src/*.py` はそれらを組み合わせる独立したCLIエントリポイント。データ規約を変更するときは各スクリプトではなく util を編集すること。

- `utils/io.py` — ロギング、`ensure_dir`、`load_ids`/`save_ids`、シード設定、YAML、レンダーパス→specimen_id のグルーピング、`resolve_file_or_recursive_search`(ファイルまたは検索対象ディレクトリを受け付ける)。
- `utils/labels.py` — ラベルのマッピング/推論、クラスタ列の検出。
- `utils/evaluation.py` — 検索の計算: `normalize_embeddings`、`pairwise_distance_matrix`(cosine/euclidean)、`compute_retrieval`(`RetrievalResults` データクラスを返す)、各種 `aggregate_*` レポーター。
- `utils/vision.py` — `l2_normalize`。

### リーフコア手法(中心的な設計思想)

この解析は HDBSCAN の**リーフクラスタを高信頼な「コア」**として扱い、それ以外をすべて残差「ノイズ」として扱う:

1. `cluster_baseline` が HDBSCAN を実行し `clusters.csv` を生成。`--auxiliary_method` により補助検出器へディスパッチする: `run_recursive_hdbscan`(`cluster_recursive_hdbscan.py`)または `run_branch_detector`(`cluster_branch_detector.py`)。
2. `analyze_leaf_cores` がリーフクラスタをコアとして特性評価する(純度 purity、被覆率 coverage)。
3. `assign_to_leaf_cores` がリーフノイズ点を最近傍のコア(medoid/centroid)へ割り当てる。距離分位点(distance-quantile)+ マージン比(margin-ratio)によるゲーティング付き。
4. `evaluate_embedding_retrieval` が Top-k 精度を報告する。`--by-cluster clusters.csv` を与えると `leaf_core` 対 `leaf_noise` のサブセット別に内訳を出せる。
5. `plot_publication_figures` / `evaluate_with_labels` が上記の CSV/JSON 出力を消費する(`evaluate_with_labels` は**事後検証(post-hoc)専用** — ラベルはクラスタリングに一切フィードバックされない)。

`report_neighbors.py` はリーフコアのパイプライン外の独立ヘルパー: 各標本の埋め込み空間における上位 `k` 近傍をダンプする(`--k`、デフォルト10)。

**可視化や木(tree)プロットは説明用のみであり、モデル/パラメータ選択の入力には決してしない。** これは `configs/visualization.yaml`、および `visualize_embedding_space.py` と `plot_hdbscan_trees.py` の `--description` で明示されている。クラスタリングのハイパーパラメータは、UMAP やデンドログラムの「見た目」とは独立に固定/正当化すること。

### 検索評価の詳細

`evaluate_embedding_retrieval.py` はデフォルトで L2 正規化 cosine、自己除外 kNN を用いる。`--exclude-same-label-column` は副次ラベルを共有する候補をマスクする(例: 属 genus の検索評価時に同種 same-species を除外)— `--labels` が必要。`--min-label-count`(デフォルト2)はラベル別レポートから小さすぎるラベルを除外する。出力: `retrieval_overall.csv`、`retrieval_by_label.csv`、`retrieval_by_subset.csv`、`nearest_neighbors.csv`、`retrieval_summary.json`。

## 規約

- 一部のユーザー向け検証メッセージやコードコメントは日本語。それらのスクリプトを編集するときは周囲のスタイルに合わせること。
- CLIフラグの命名はスクリプト間で一貫していない(クラスタリングスクリプトは `--min_cluster_size` のような snake_case、検索/リーフスクリプトは `--label-column` のような kebab-case)。編集対象ファイル内で既に使われている規約に従うこと。
