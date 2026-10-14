# DepthComp-GenBench

**A Unified Generalization Benchmark for Depth Completion**

DepthComp-GenBench is a compact benchmark for training and evaluating eight
depth completion methods under one data and metric protocol. Each method keeps
its main network and necessary native scripts, while `unified_evaluation/`
provides shared loading, metrics, visualization, checkpoint handling, and OOD
testing.

## Quick start

```bash
git clone https://github.com/gwt1158/DepthComp-GenBench.git
cd DepthComp-GenBench
python -m pip install -r GeoLayerNet/requirements.txt
python -m unified_evaluation.evaluate_geolayernet --help
python -m unified_evaluation.eval_ood16 --help
```

Download datasets and optional checkpoints separately. Dataset links and the
required directory structure are documented in [`DATASETS.md`](DATASETS.md).
No datasets, pretrained weights, or method checkpoints are distributed in this
repository.

## Methods

| Method | Source directory | Unified evaluation entry |
| --- | --- | --- |
| GeoLayerNet | `GeoLayerNet/` | `evaluate_geolayernet.py` |
| SGNet | `SGNet-main/` | `evaluate_sgnet.py` |
| SPNet | `SPNet-main/` | `evaluate_spnet.py` |
| G2-MonoDepth | `G2-MonoDepth-main/` | `evaluate_g2_monodepth.py` |
| CompletionFormer | `CompletionFormer-main/` | `evaluate_completionformer.py` |
| NLSPN | `NLSPN_ECCV20-master/` | `evaluate_nlspn.py` |
| PSD | `PSD-main/` | `evaluate_psd.py` |
| Marigold-SSD | `marigold-ssd-main/` | `evaluate_marigold_ssd.py` |

Run all unified commands from the repository root.

GeoLayerNet uses four geometric layers by default (`model.num_layers: 4`). The
implementation also supports `num_layers: 1` and other positive values through
the same configuration field. Layer count is part of the architecture, so use
a checkpoint trained with the same value; changing it at evaluation time is
not checkpoint-compatible.

## Datasets

The shared protocol uses NYUv2 and KITTI as source-domain training and
in-domain test sets. Generalization follows the PSD OOD-16 protocol.

- [NYUv2 48K train and 654 validation images on Hugging Face](https://huggingface.co/datasets/heuliuh/nyu_depth_v2)
- [KITTI depth data and raw images on Hugging Face](https://huggingface.co/datasets/dangth2004/KITTI-Depth-Estimation)
- [PSD OOD validation collection on Hugging Face](https://huggingface.co/datasets/cslxx/PSD-val)

The complete source, download, access, and attribution list is maintained in
[`DATASETS.md`](DATASETS.md).

The PSD collection covers VOID1500, SUNRGBD, TOFDC, DIML indoor, DIODE indoor,
Middlebury, ETH3D, DrivingStereo, Argoverse, Cityscapes, DIODE outdoor,
Hypersim, Virtual KITTI 2, KITTI-360, and Stanford2D3D. ScanNet is the sixteenth
domain. It is license-gated: accept the terms before downloading it from the
[gated Hugging Face mirror](https://huggingface.co/datasets/Pointcept/scannet-compressed)
or the [official ScanNet site](http://www.scan-net.org/). Do not redistribute
ScanNet outside its license.

### Unified sample definition

Every sample has the same four logical inputs/labels:

| Field | Meaning | Required rule |
| --- | --- | --- |
| `rgb` | Aligned color image | Three channels, spatially aligned with depth |
| `sparse_depth` | Sparse depth given to the model | Metric depth in metres before loader normalization; invalid pixels are zero |
| `target_depth` | Reference depth used only for supervision or metrics | Metric depth in metres before loader normalization; invalid or unavailable pixels are zero |
| `valid_mask` | Sparse-input validity mask | Binary mask with `valid_mask = sparse_depth > 0` |

The evaluation code additionally derives `target_mask = target_depth > 0`,
restricted to the configured metric depth range. `valid_mask` and
`target_mask` are different: the former tells the model where sparse input is
present, while the latter tells the evaluator where reference depth is valid.

Use the following source-domain structure. Matching files must have the same
stem in all four folders:

```text
GeoLayerNet/data/
├── nyu48k_1pct/
│   ├── train/
│   │   ├── rgb/00000000.png
│   │   ├── sparse_depth/00000000.png
│   │   ├── target_depth/00000000.png
│   │   └── valid_mask/00000000.png
│   ├── val/
│   │   └── {rgb,sparse_depth,target_depth,valid_mask}/...
│   └── test/
│       └── {rgb,sparse_depth,target_depth,valid_mask}/...
└── kitti_geolayernet/
    ├── train/
    │   ├── rgb/2011_09_26_drive_0001_sync_0000000005.png
    │   ├── sparse_depth/2011_09_26_drive_0001_sync_0000000005.png
    │   ├── target_depth/2011_09_26_drive_0001_sync_0000000005.png
    │   └── valid_mask/2011_09_26_drive_0001_sync_0000000005.png
    ├── val/
    │   └── {rgb,sparse_depth,target_depth,valid_mask}/...
    └── test/
        └── {rgb,sparse_depth,target_depth,valid_mask}/...
```

### Depth and mask construction

- **NYUv2:** use the aligned dense depth image (with invalid pixels kept at
  zero) as `target_depth`. Generate
  `sparse_depth` by sampling only valid target pixels with the fixed protocol
  seed and requested density (the reference setup uses one percent). Store the
  sampled metric values at the selected pixels and zero elsewhere. Generate
  `valid_mask` as `(sparse_depth > 0)`, encoded as binary `0/255` PNG when it is
  saved as an image. The committed preparation script is
  `GeoLayerNet/tools/build_nyu48k_sparse_sets.py`.
- **KITTI:** use the projected single-frame `velodyne_raw` map as
  `sparse_depth`. For the official depth-completion benchmark, use the provided
  accumulated/semi-dense `groundtruth` map as `target_depth`; do not treat it as
  a fully dense camera-depth image. If a KITTI-like dataset provides only one
  original sparse LiDAR depth map and no separate annotation, copy that map to
  `target_depth` for evaluation and clearly report that metrics cover only its
  nonzero LiDAR pixels. Generate `valid_mask = (sparse_depth > 0)`. KITTI
  16-bit depth PNG values use `depth_m = png_value / 256`.
- **OOD datasets:** preserve an official sparse input and official validity
  mask when both are supplied by zeroing sparse values outside that mask. The
  unified loader then reproduces it as `valid_mask = (sparse_depth > 0)`. If
  sparse depth is supplied without a mask, use the same expression. If only
  reference depth is
  supplied, generate sparse input once from valid reference pixels using the
  declared sampling pattern, point count or density, and a fixed seed; save the
  generated sparse input and reuse it for every method. Never resample it per
  method or per evaluation run. The reference mask is always
  `target_mask = (target_depth > 0)` plus the configured depth-range check.

The source download may use its native layout. Convert it once to the layout
above, or use the KITTI official-layout/JSONL support in
`GeoLayerNet/tools/prepare_kitti_geolayernet.py`. RGB, sparse depth, target
depth, and masks must undergo exactly the same crop/resize; use nearest-neighbor
interpolation for depth and masks.

### OOD-16 storage layout

OOD datasets use one NPZ file per sample so all domains share one loader. An
NPZ stores `rgb`, `sparse_depth`, and `target_depth`; the loader deterministically
constructs `valid_mask` and `target_mask` with the rules above. Optional camera
matrices may be stored as `K` or `intrinsics`.

Complete example:

```text
GeoLayerNet/data/
└── ood/
    ├── VOID1500/
    │   ├── test.txt
    │   └── test/
    │       ├── 000000.npz
    │       └── 000001.npz
    ├── SUNRGBD/{test.txt,test/*.npz}
    ├── TOFDC/{test.txt,test/*.npz}
    ├── DIML/{test.txt,test/*.npz}
    ├── DIODE_indoor/{test.txt,test/*.npz}
    ├── Middlebury/{test.txt,test/*.npz}
    ├── ETH3D/{test.txt,test/*.npz}
    ├── DrivingStereo/{test.txt,test/*.npz}
    ├── Argoverse/{test.txt,test/*.npz}
    ├── Cityscape/{test.txt,test/*.npz}
    ├── DIODE_outdoor/{test.txt,test/*.npz}
    ├── Hypersim/{test.txt,test/*.npz}
    ├── VKITTI2/{test.txt,test/*.npz}
    ├── KITTI360/{test.txt,test/*.npz}
    ├── Stanford/{test.txt,test/*.npz}
    └── ScanNet/{test.txt,test/*.npz}
```

For example, `VOID1500/test.txt` contains paths relative to `VOID1500/`:

```text
test/000000.npz
test/000001.npz
```

Each array uses `rgb: HxWx3 or 3xHxW`, `sparse_depth: HxW or 1xHxW`, and
`target_depth: HxW or 1xHxW`. RGB may be `uint8` in `[0,255]` or floating point
in `[0,1]`; NPZ depth arrays must be `float32` metres with zero denoting
invalid pixels. Every OOD directory uses a fixed `test.txt` manifest and must
not depend on directory scan order. The single configuration template is
`GeoLayerNet/configs/ood_example.yaml`; the OOD runner reuses it for all domain
roots. The same source checkpoint and inference settings must be used for all
OOD domains.

## Setup

Install the shared loader dependencies and then the dependencies for the method
you intend to run:

```bash
python -m pip install -r GeoLayerNet/requirements.txt
python -m pip install -r METHOD_DIRECTORY/requirements.txt
```

Some methods require CUDA extensions. Their compact READMEs link to the
official extension source and model weights. No binary weights are committed.

## Training

GeoLayerNet provides the reference source-domain workflow:

```bash
python GeoLayerNet/train.py --dry-run
GPU=0 bash GeoLayerNet/scripts/run_nyu48k_train_eval.sh
```

For another method, enter its directory and use the native training command in
its README. Keep the registered NYUv2 or KITTI split, fixed sparse input, seed,
and source-domain depth scale. Store generated checkpoints under that method's
ignored `checkpoints/` or `weights/` directory.

## In-domain evaluation

Each adapter exposes the same evaluation interface:

```bash
python -m unified_evaluation.evaluate_geolayernet --help
python -m unified_evaluation.evaluate_geolayernet \
  --dataset-config GeoLayerNet/configs/geolayernet_final_nyu48k_1pct_fullval.yaml \
  --test-dataroot GeoLayerNet/data/nyu48k_1pct \
  --checkpoint GeoLayerNet/checkpoints/model.pth \
  --output-dir results/geolayernet_nyu
```

Replace `evaluate_geolayernet` with the entry shown in the method table and add
the method-specific arguments displayed by `--help`.

## OOD-16 evaluation

Evaluate one unchanged checkpoint over every domain:

```bash
python -m unified_evaluation.eval_ood16 \
  --eval-script unified_evaluation/evaluate_geolayernet.py \
  --checkpoint GeoLayerNet/checkpoints/model.pth \
  --ood-config-dir GeoLayerNet/configs/ood_example.yaml \
  --ood-root GeoLayerNet/data/ood \
  --output-root results/geolayernet_ood16 \
  --strict-checkpoint --require-ood16
```

`eval_ood16` is the canonical evaluation entry. A complete run reports
Macro-16 only when all 16 registered domains pass. If ScanNet access has not
yet been approved, the retained `eval_ood15` compatibility entry may be used
with the same template and root; its result is explicitly reported as OOD-15
partial and must not be labeled Macro-16.

## Metrics and comparison rule

The evaluator reports metric-depth RMSE, MAE, REL, inverse-depth errors,
delta thresholds, observed-region error, and missing-region error. Macro scores
are unweighted means of finalized per-domain metrics. OOD samples are never
used for checkpoint selection, tuning, or test-time hyperparameter changes.

## Acknowledgements and references

### Datasets

We thank the creators and maintainers of NYU Depth V2, KITTI Depth Completion,
VOID, SUN RGB-D, TOFDC, DIML, DIODE, Middlebury, ETH3D, ScanNet,
DrivingStereo, Argoverse, Cityscapes, Hypersim, Virtual KITTI 2, KITTI-360,
and Stanford2D3D for making their data available to the research community.

DepthComp-GenBench does not redistribute these datasets. Users must download
them from the linked Hugging Face collection or official source and comply
with each dataset's license and terms. ScanNet requires separate approval and
must not be redistributed. Please cite the original dataset papers when using
benchmark results. See [`DATASETS.md`](DATASETS.md) for all links and access
notes.

### Methods

GeoLayerNet evaluation benefits from the following official implementations.
Please cite the corresponding papers when reporting results:

1. [SGNet: Structure Guided Network via Gradient-Frequency Awareness](https://github.com/yanzq95/SGNet), AAAI 2024.
2. [SPNet: Scale Propagation Network for Generalizable Depth Completion](https://github.com/Wang-xjtu/SPNet), IEEE TPAMI 2025.
3. [G2-MonoDepth: A General Framework of Generalized Depth Inference](https://github.com/Wang-xjtu/G2-MonoDepth), IEEE TPAMI 2024.
4. [CompletionFormer: Depth Completion with Convolutions and Vision Transformers](https://github.com/youmi-zym/CompletionFormer), CVPR 2023.
5. [NLSPN: Non-Local Spatial Propagation Network for Depth Completion](https://github.com/zzangjinsun/NLSPN_ECCV20), ECCV 2020.
6. [PSD: Propagating Sparse Depth via Depth Foundation Model](https://github.com/ShenglunChen/PSD), IEEE TIP 2025.
7. [Marigold-SSD: Need for Speed—Zero-Shot Depth Completion with Single-Step Diffusion](https://github.com/DTU-PAS/marigold-ssd), CVPRW 2026.

Copyright and licenses remain with the original authors. Consult each method
directory and upstream repository before redistribution or commercial use.

## License and third-party notice

Each retained comparison method remains subject to the license in its own
directory and upstream repository. Dataset licenses and access restrictions
remain with their respective owners. Model weights and datasets are not
redistributed here. A license for the original DepthComp-GenBench and
GeoLayerNet code will be added before the public release.
