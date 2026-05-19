#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/lyn/miniconda3/etc/profile.d/conda.sh
conda activate sk_vit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG_NAME="${CONFIG_NAME:-imigue_prune70_2mean_merge_rgb_k400_ep40}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/output}"
MASTER_PORT="${MASTER_PORT:-29604}"
mkdir -p "${SAVE_DIR}/rgb/logs" "${SAVE_DIR}/rgb/checkpoints" "${SAVE_DIR}/rgb/val_log"

CKPT_DIR="${SAVE_DIR}/rgb/checkpoints/train/sk_vit/${CONFIG_NAME}"
if ls "${CKPT_DIR}"/*_ep0040.pth.tar >/dev/null 2>&1; then
  echo "[SKIP] rgb+prune already finished: ${CONFIG_NAME}"
  exit 0
fi

if [[ ! -f "${PROJECT_ROOT}/experiments/sk_vit/${CONFIG_NAME}.yaml" ]]; then
  echo "[ERR] config missing: ${PROJECT_ROOT}/experiments/sk_vit/${CONFIG_NAME}.yaml"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
python -m torch.distributed.run --nproc_per_node=2 --master_port "${MASTER_PORT}" \
  lib/train/run_training.py \
  --script sk_vit \
  --config "${CONFIG_NAME}" \
  --save_dir "${SAVE_DIR}"
