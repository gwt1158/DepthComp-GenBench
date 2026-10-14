# SPNet

Minimal source release for Scale Propagation Network depth completion.

## Contents

- `src/networks.py`: V2Net definition.
- `src/modules.py` and `src/custom_blocks.py`: encoder, decoder, and SP-Norm blocks.
- `src/data_tools.py`: native training data pipeline.
- `src/losses.py`: published training losses.
- `src/src_main.py`: native trainer.
- `train.py` and `test.py`: native command-line entry points.
- `config.py` and `test_utils.py`: configuration and evaluation helpers.

All datasets, checkpoints, and outputs use paths relative to this project.

## Setup

```bash
pip install -r requirements.txt
```

## Native usage

```bash
python train.py --model_type Large
python test.py --rgbd_dir Test_Datasets/Ibims \
  --model_dir checkpoints/models/Large_300.pth --model_type Large
```

## Unified evaluation

Run the shared evaluator from the workspace root:

```bash
python -m unified_evaluation.evaluate_spnet --help
```

The shared adapter uses the published SPNet-Large architecture, strict checkpoint
loading, fixed evaluation geometry, and the selected sparse-depth input only.
