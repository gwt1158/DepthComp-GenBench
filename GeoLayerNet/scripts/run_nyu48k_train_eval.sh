#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-data/nyu48k_1pct}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/nyu48k_1pct}
RESULT_ROOT=${RESULT_ROOT:-results}
CHECKPOINT="${CHECKPOINT_DIR}/best_net_G.pth"
TRAIN_OUTPUT="${RESULT_ROOT}/nyu48k_1pct_train"
EVAL_OUTPUT="${RESULT_ROOT}/nyu48k_1pct_safe_eval"
CONFIG="configs/geolayernet_final_nyu48k_1pct_fullval.yaml"

run() {
  echo "+ CUDA_VISIBLE_DEVICES=$GPU $PY $*"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$@"
}

run train.py \
  --dataset-name nyu48k_1pct \
  --config "$CONFIG" \
  --train-dataroot "$DATA_ROOT" \
  --val-dataroot "$DATA_ROOT" \
  --name nyu48k_1pct \
  --output-dir "$TRAIN_OUTPUT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --batch-size 1

run eval.py \
  --dataset-name nyu48k_1pct \
  --config "$CONFIG" \
  --dataroot "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EVAL_OUTPUT" \
  --eval-name nyu48k_1pct_safe_eval \
  --output-mode safe \
  --num-test 100000 \
  --extra-metrics \
  --save-visuals \
  --overwrite

echo "Done: $EVAL_OUTPUT"
