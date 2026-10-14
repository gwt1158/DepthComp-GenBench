# Marigold-SSD

Compact release of **Need for Speed: Zero-Shot Depth Completion with
Single-Step Diffusion**.

This tree retains the single-step diffusion model, conditional depth encoder,
training and inference code, dataset loaders, preprocessing, postprocessing,
and evaluation metrics. Local model snapshots, documentation assets, audit
packages, repeated sparsity experiments, interpolation baselines, caches, and
duplicate comparison scripts are excluded.

## Installation

```bash
conda env create -f environment.yml
conda activate ssd
pip install -r requirements.txt
```

## Weights

Weights are not stored in this repository:

- [Marigold E2E foundation model](https://huggingface.co/GonzaloMG/marigold-e2e-ft-depth)
- [Official Marigold-SSD checkpoints](https://drive.google.com/drive/folders/1xvQ1_pXGCI0iIp7xyajgFhqFmLb9_cxX?usp=share_link)

Place the foundation snapshot at
`weights/GonzaloMG_marigold-e2e-ft-depth` and the task checkpoint under
`weights/ssd`. Both paths are relative to this project root.

## Training

```bash
./scripts/ssd_train
```

The launcher defaults to `data/` and `weights/` under this project. Override
`DATASETS_DIR` or `MODELS_DIR` only when a different local layout is needed.

## Native inference

```bash
python -m ssd.infer \
  --seed 1234 \
  --model weights/ssd \
  --model_type vae_mid \
  --eval_dataset ssd.datasets.Nyuv2 \
  --eval_dataset_path data/nyu-v2 \
  --eval_dataset_subset test \
  --eval_preprocessing ssd.preprocessing.nyuv2_default \
  --output_dir outputs/nyu
```

The one-line files under `ssd/configs/` and `ssd/datasets/` are manifest format
examples. `ssd/configs/ood_example.yml` shows one fixed checkpoint referencing
several datasets without maintaining point-count-specific configurations.

## Unified evaluation

Run from the workspace root:

```bash
python -m unified_evaluation.evaluate_marigold_ssd \
  --marigold-ssd-training-domain N \
  --marigold-ssd-foundation marigold-ssd-main/weights/GonzaloMG_marigold-e2e-ft-depth \
  --checkpoint marigold-ssd-main/weights/ssd/model.pth \
  --depth-unit-scale 10 \
  --dataset-config GeoLayerNet/configs/example.yaml \
  --test-dataroot GeoLayerNet/data/example \
  --output-dir results/marigold_ssd
```

Use training domain `K` and depth scale `80` with a KITTI-trained checkpoint.

## References

- [Project page](https://dtu-pas.github.io/marigold-ssd)
- [Paper](https://arxiv.org/abs/2603.10584v2)
