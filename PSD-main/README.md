# PSD-D

Compact release of **Propagating Sparse Depth via Depth Foundation Model for
Out-of-Distribution Depth Completion** (IEEE TIP 2025).

This tree keeps the published PSD network, the Depth Anything V1 ViT-L prior,
the native training manager, dataset loaders, one-line manifest examples, and
root-relative configuration. Experimental foundation-model alternatives,
ablation entry points, audit bundles, generated media, caches, and binary
weights are intentionally excluded.

## Setup

```bash
pip install -r requirements.txt
```

Create `weights/` in this project and download:

- [PSD-NK-D completion checkpoint](https://drive.google.com/file/d/1ynNWqq1POKF0g2JbvroMksPTKdBvP10F/view?usp=drive_link)
- [Depth Anything V1 ViT-L checkpoint](https://huggingface.co/LiheYoung/depth_anything_vitl14/resolve/main/pytorch_model.bin)
- [Torchvision ResNet-18 initializer](https://download.pytorch.org/models/resnet18-f37072fd.pth)

Rename the initializers to `weights/depth_anything_vitl14.pth` and
`weights/resnet18.pth`. Set `load_ckpt` in `configs/cfg.yml` when resuming or
evaluating with the native entry point. DINOv2 source code is fetched from its
official Torch Hub repository on first use instead of being duplicated here.

## Native entry point

```bash
python main.py --cfg configs/cfg.yml
```

All dataset and weight paths in `configs/cfg.yml` are relative to this project
root. Each file under `filename/` contains one format example; replace it with
your own split manifest. `configs/ood_example.yml` shows how one fixed PSD
checkpoint can reference several OOD datasets.

## Unified evaluation

Run from the workspace root:

```bash
python -m unified_evaluation.evaluate_psd \
  --training-domain N \
  --depth-unit-scale 10 \
  --checkpoint PSD-main/weights/psd_nk_depthanything.pth \
  --dataset-config GeoLayerNet/configs/example.yaml \
  --test-dataroot GeoLayerNet/data/example \
  --output-dir results/psd
```

Use `--training-domain K --depth-unit-scale 80` for a KITTI-trained checkpoint.
The shared OOD runner can reuse the same source-domain checkpoint across the
datasets listed in `configs/ood_example.yml`.

## Citation

```bibtex
@article{chen2025psd,
  title={Propagating Sparse Depth via Depth Foundation Model for Out-of-Distribution Depth Completion},
  author={Chen, Shenglun and Ma, Xinzhu and Zhang, Hong and Li, Haojie and Wang, Zhihui},
  journal={IEEE Transactions on Image Processing},
  volume={34},
  pages={5285--5299},
  year={2025}
}
```
