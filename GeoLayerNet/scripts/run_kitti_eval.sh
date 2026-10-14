#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
PY=${PY:-python}
KITTI_ROOT=${KITTI_ROOT:-data/kitti_geolayernet}
FINAL_CK=${FINAL_CK:-checkpoints/kitti/best_net_G.pth}
RESULT_ROOT=${RESULT_ROOT:-results}
OUTPUT_DIR="${RESULT_ROOT}/kitti_safe_eval"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" eval.py \
  --dataset-name kitti \
  --config configs/geolayernet_kitti_psd_fullval.yaml \
  --dataroot "$KITTI_ROOT" \
  --checkpoint "$FINAL_CK" \
  --output-dir "$OUTPUT_DIR" \
  --eval-name kitti_safe_eval \
  --output-mode safe \
  --num-test 100000 \
  --extra-metrics \
  --save-visuals \
  --overwrite

"$PY" tools/summarize_results.py \
  --runs GeoLayerNet_kitti="$OUTPUT_DIR" \
  --out "${RESULT_ROOT}/summary_kitti.csv"

echo "Done: ${RESULT_ROOT}/summary_kitti.csv"
