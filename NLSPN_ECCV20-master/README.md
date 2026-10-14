# NLSPN

Minimal source release for Non-Local Spatial Propagation Network depth completion.

## Contents

- `src/model/nlspnmodel.py`: the NLSPN network.
- `src/model/common.py`: encoder and decoder components.
- `src/model/modulated_deform_conv_func.py`: NLSPN deformation interface.
- `src/data/`: native NYU and KITTI loaders.
- `src/loss/`, `src/metric/`, and `src/summary/`: training support.
- `src/main.py`: native training and evaluation entry point.
- `utils/`: dataset manifest preparation scripts.
- `data_json/example.json`: one manifest-format example.

Datasets, manifests, checkpoints, and results use project-relative paths.

## Setup

```bash
pip install -r requirements.txt
```

## External files

Weights and DCNv2 source are not stored in this repository.

- NYU checkpoint: https://drive.google.com/file/d/1LTdZI36zdOeVmKRbG3orzvwXctg-OjGC
- KITTI checkpoint: https://drive.google.com/file/d/11by_1oglcncSHFeF3S2ldcysQYjyDcQh
- ResNet initialization: https://github.com/zzangjinsun/NLSPN_ECCV20/tree/master/src/pretrained
- DCNv2 extension: https://github.com/zzangjinsun/NLSPN_ECCV20/tree/master/src/model/deformconv

Place `resnet34.pth` in `src/pretrained/`. Download DCNv2 to
`src/model/deformconv/`, then build it with its `setup.py`.

## Native usage

```bash
python src/main.py --help
```

## Unified evaluation

Run from the workspace root:

```bash
python -m unified_evaluation.evaluate_nlspn --help
```

The shared adapter preserves the released ResNet-34 model, 18 NLSPN propagation
steps, native depth scaling, and strict checkpoint loading.

## License

See `LICENSE`.
