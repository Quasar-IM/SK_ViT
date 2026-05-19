#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/lyn/miniconda3/etc/profile.d/conda.sh
conda activate sk_vit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG_NAME="imigue_sk_bone_ie_officialvaltrain_ep20"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/output}"
MASTER_PORT="${MASTER_PORT:-29602}"
mkdir -p "${SAVE_DIR}/sk/logs" "${SAVE_DIR}/sk/checkpoints" "${SAVE_DIR}/sk/val_log"

if [[ ! -f "${PROJECT_ROOT}/experiments/sk_maga/${CONFIG_NAME}.yaml" ]]; then
  echo "[ERR] config not found: ${PROJECT_ROOT}/experiments/sk_maga/${CONFIG_NAME}.yaml"
  exit 1
fi

CKPT_DIR="${SAVE_DIR}/sk/checkpoints/train/sk_maga/${CONFIG_NAME}"
if ls "${CKPT_DIR}"/*_ep0020.pth.tar >/dev/null 2>&1; then
  echo "[SKIP] bone+ie already finished: ${CONFIG_NAME}"
  exit 0
fi

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
python -m torch.distributed.run --nproc_per_node=2 --master_port "${MASTER_PORT}" \
  lib/train/run_training.py \
  --script sk_maga \
  --config "${CONFIG_NAME}" \
  --save_dir "${SAVE_DIR}"
