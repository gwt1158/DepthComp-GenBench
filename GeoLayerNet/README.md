# GeoLayerNet

This directory contains the compact GeoLayerNet implementation used for depth
completion experiments. It includes the model, training and evaluation entry
points, final experiment configurations, and the dataset preparation scripts
required by the reference workflow.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Prepare the retained NYUv2-48K 1% sparse protocol from the FastDepth/Hugging
Face tar files:

```bash
python tools/build_nyu48k_sparse_sets.py \
  --tar-root data/nyu_depth_v2_hf \
  --protocols 1pct \
  --overwrite --verify
```

Prepare KITTI JSONL manifests without copying image files:

```bash
python tools/prepare_kitti_geolayernet.py \
  --src data/kitti_official_raw \
  --dst data/kitti_geolayernet
```

Run `python tools/prepare_kitti_geolayernet.py --help` if the local KITTI layout
uses different argument names.

For OOD evaluation, edit only `dataset.dataroot` in
`configs/ood_example.yaml`. Each NPZ sample must contain `rgb`, `sparse_depth`,
and `target_depth`; all paths remain relative to the project root.

## Training

The default settings at the top of `train.py` target NYUv2-48K with 1% sparse
input. They may be edited directly or overridden on the command line:

```bash
python train.py --dry-run
python train.py
```

Run the retained NYUv2-48K protocol with:

```bash
GPU=0 bash scripts/run_nyu48k_train_eval.sh
```

## Evaluation

`eval.py` requires explicit dataset, checkpoint, and result paths unless the
user settings block at the top of the file is filled in.

```bash
python eval.py \
  --dataset-name nyu48k_1pct \
  --config configs/geolayernet_final_nyu48k_1pct_fullval.yaml \
  --dataroot data/nyu48k_1pct \
  --checkpoint checkpoints/nyu48k_1pct/best_net_G.pth \
  --output-dir results/nyu48k_1pct_safe_eval \
  --eval-name nyu48k_1pct_safe_eval
```

For KITTI:

```bash
KITTI_ROOT=data/kitti_geolayernet \
FINAL_CK=checkpoints/kitti/best_net_G.pth \
bash scripts/run_kitti_eval.sh
```

Datasets, checkpoints, outputs, caches, and generated figures are intentionally
excluded from this source-only directory.
