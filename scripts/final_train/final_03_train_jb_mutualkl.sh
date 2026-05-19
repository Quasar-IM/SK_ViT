#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/lyn/miniconda3/etc/profile.d/conda.sh
conda activate sk_vit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG_NAME="imigue_joint_bone_mutualkl_ie_official_j25_b25_ep20"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/output}"
MASTER_PORT="${MASTER_PORT:-29603}"
mkdir -p "${SAVE_DIR}/jb_fusion/logs" "${SAVE_DIR}/jb_fusion/checkpoints" "${SAVE_DIR}/jb_fusion/val_log"

CKPT_DIR="${SAVE_DIR}/jb_fusion/checkpoints/train/jb_fusion/${CONFIG_NAME}"
if ls "${CKPT_DIR}"/*_ep0020.pth.tar >/dev/null 2>&1; then
  echo "[SKIP] jb mutualkl already finished: ${CONFIG_NAME}"
  exit 0
fi

JOINT_CKPT="${SAVE_DIR}/sk/checkpoints/train/sk_maga/imigue_sk_joint_ie_officialvaltrain_ep20/SKMagaModel_ep0020.pth.tar"
BONE_CKPT="${SAVE_DIR}/sk/checkpoints/train/sk_maga/imigue_sk_bone_ie_officialvaltrain_ep20/SKMagaModel_ep0020.pth.tar"
if [[ ! -f "${JOINT_CKPT}" || ! -f "${BONE_CKPT}" ]]; then
  echo "[ERR] final_03 dependency missing."
  echo "       need: ${JOINT_CKPT}"
  echo "       need: ${BONE_CKPT}"
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/experiments/jb_fusion/${CONFIG_NAME}.yaml" ]]; then
  echo "[ERR] config missing: ${PROJECT_ROOT}/experiments/jb_fusion/${CONFIG_NAME}.yaml"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
python -m torch.distributed.run --nproc_per_node=2 --master_port "${MASTER_PORT}" \
  lib/train/run_training.py \
  --script jb_fusion \
  --config "${CONFIG_NAME}" \
  --save_dir "${SAVE_DIR}"
