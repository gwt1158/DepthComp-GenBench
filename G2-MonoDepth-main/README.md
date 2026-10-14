# G2-MonoDepth

Minimal source release for generalized monocular RGB+X depth inference.

## Contents

- `src/networks.py`: the ReZero UNet.
- `src/modules.py` and `src/custom_blocks.py`: required network blocks.
- `src/data_tools.py`: native training data pipeline.
- `src/losses.py`: published training losses.
- `src/src_main.py`: native trainer.
- `train.py` and `test.py`: native entry points.
- `config.py` and `test_utils.py`: configuration and evaluation helpers.

Datasets, checkpoints, and outputs use paths relative to this project.

## Setup

```bash
pip install -r requirements.txt
```

## Weights

Weights are not stored in this repository. Download the official test model:

https://drive.google.com/file/d/1Cp0tRkQE0AAtvtMQcYVnb-cOj9J4CWdZ

Place it at `checkpoints/models/epoch_100.pth`.

## Native usage

```bash
python train.py
python test.py --rgbd_dir Test_Datasets/Ibims \
  --model_dir checkpoints/models/epoch_100.pth
```

## Unified evaluation

Run from the workspace root:

```bash
python -m unified_evaluation.evaluate_g2_monodepth --help
```

The shared adapter preserves the published ReZero UNet, native depth scaling,
input-validity mask, inference-domain adjustment, and strict checkpoint loading.
