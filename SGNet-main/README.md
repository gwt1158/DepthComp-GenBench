# SGNet

Minimal SGNet source release for guided depth super-resolution.

## Contents

- `models/`: SGNet and its required network modules.
- `data/`: NYU-v2, RGB-D-D, Middlebury, and Lu dataset loaders.
- `train.py`: native training entry point.
- `test.py`: native evaluation entry point.
- `utils.py` and `thops.py`: required metrics and tensor helpers.

Dataset, checkpoint, and output paths are relative to this project directory.

## Setup

```bash
pip install -r requirements.txt
```

## Native usage

```bash
python train.py --root_dir data/nyu_data --result results
python test.py --root_dir data/RGB-D-D \
  --model_dir checkpoints/SGNet_X8_R.pth --results_dir results
```

## Unified evaluation

Run the shared evaluator from the workspace root:

```bash
python -m unified_evaluation.evaluate_sgnet --help
```

The shared adapter preserves the SGNet network and applies deterministic,
mask-aware pooling to form the low-resolution sparse-depth input.

## License

See `LICENSE`.
