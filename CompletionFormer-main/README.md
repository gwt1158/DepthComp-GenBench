# CompletionFormer

Minimal source release for CompletionFormer depth completion.

## Contents

- `src/model/`: CompletionFormer, PVT/JCAT backbone, and NLSPN modules.
- `src/data/`: native NYU, KITTI, and LiDAR data loaders.
- `src/loss/`, `src/metric/`, and `src/summary/`: training support.
- `src/main.py`: native training and evaluation entry point.
- `utils/`: NYU and KITTI manifest preparation scripts.
- `data_json/example.json`: one manifest-format example.

Datasets, manifests, checkpoints, and results use project-relative paths.

## Setup

```bash
pip install -r requirements.txt
```

## Weights

Weights are not stored in this repository.

- NYU checkpoints: https://drive.google.com/drive/folders/1GlMVhI1Auo9noimR6NN0S-QLwL04ypCb
- KITTI checkpoints: https://drive.google.com/drive/folders/1Tp1XAU7D7HOMq_iLEGzvt4I15g_HGBeM
- Backbone initialization files: https://github.com/youmi-zym/CompletionFormer/tree/main/src/pretrained
- DCNv2 extension: https://github.com/youmi-zym/CompletionFormer/tree/main/src/model/deformconv

Place backbone files in `src/pretrained/` and task checkpoints in `checkpoints/`.
Download DCNv2 to `src/model/deformconv/`, then build it with its `setup.py`.

## Native usage

```bash
python src/main.py --help
```

## Unified evaluation

Run from the workspace root:

```bash
python -m unified_evaluation.evaluate_completionformer --help
```

The shared adapter preserves the released PVT/JCAT network, six-step NLSPN
refinement, strict checkpoint loading, and source-domain depth scaling.

## License

See `LICENSE`.
