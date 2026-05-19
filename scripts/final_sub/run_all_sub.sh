#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/lyn/miniconda3/etc/profile.d/conda.sh
conda activate sk_vit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/output}"
TMP_DIR="${TMP_DIR:-${PROJECT_ROOT}/scripts/final_sub/tmp_run}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/scripts/final_sub/submission}"
mkdir -p "${TMP_DIR}" "${OUT_DIR}"

RGB_CONFIG="${RGB_CONFIG:-imigue_prune70_2mean_merge_rgb_k400_ep40}"
RGB_EPOCH="${RGB_EPOCH:-26}"
JB_CONFIG="${JB_CONFIG:-imigue_joint_bone_mutualkl_ie_official_j25_b25_ep20}"
JB_EPOCH="${JB_EPOCH:-20}"

# Fixed fusion weights (not model ckpt), hard-coded.
ALPHA_RGB="${ALPHA_RGB:-0.6352739334106445}"
BETA_SK="${BETA_SK:-0.3647260367870331}"

RGB_TEST_NPZ="${TMP_DIR}/rgb_test.npz"
JB_TEST_NPZ="${TMP_DIR}/jb_test.npz"
DUAL_TEST_NPZ="${TMP_DIR}/dual_test.npz"
OUT_BASENAME="${OUT_BASENAME:-Submission_train_rgb_jb_fixedweight}"
FINAL_CSV="${OUT_DIR}/${OUT_BASENAME}.csv"

RGB_CKPT_GLOB="${SAVE_DIR}/rgb/checkpoints/train/sk_vit/${RGB_CONFIG}/*_ep$(printf "%04d" "${RGB_EPOCH}").pth.tar"
JB_CKPT_GLOB="${SAVE_DIR}/jb_fusion/checkpoints/train/jb_fusion/${JB_CONFIG}/*_ep$(printf "%04d" "${JB_EPOCH}").pth.tar"
RGB_CKPT="$(ls ${RGB_CKPT_GLOB} 2>/dev/null | head -n 1 || true)"
JB_CKPT="$(ls ${JB_CKPT_GLOB} 2>/dev/null | head -n 1 || true)"

if [[ -z "${RGB_CKPT}" || -z "${JB_CKPT}" ]]; then
  echo "[ERR] required trained checkpoints not found."
  echo "      need RGB: ${RGB_CKPT_GLOB}"
  echo "      need JB : ${JB_CKPT_GLOB}"
  echo "      run training first: bash ${PROJECT_ROOT}/scripts/final_train/run_all_train.sh"
  exit 1
fi


echo "== [1/3] use preset fusion weights =="
echo "alpha_rgb=${ALPHA_RGB}"
echo "beta_sk=${BETA_SK}"

echo "== [2/3] export test logits (rgb/jb) =="
if [[ -f "${RGB_TEST_NPZ}" ]]; then
  echo "[SKIP] rgb test logits already exist: ${RGB_TEST_NPZ}"
else
  python "${PROJECT_ROOT}/tools/test.py" \
    --script_name sk_vit \
    --config_name "${RGB_CONFIG}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint_epoch "${RGB_EPOCH}" \
    --save_logits_npz \
    --no_zip
  cp -f "${PROJECT_ROOT}/submission/rgb/sk_vit/${RGB_CONFIG}/Submission_ep$(printf "%04d" "${RGB_EPOCH}")_logits.npz" "${RGB_TEST_NPZ}"
fi

if [[ -f "${JB_TEST_NPZ}" ]]; then
  echo "[SKIP] jb test logits already exist: ${JB_TEST_NPZ}"
else
  python "${PROJECT_ROOT}/tools/export_jb_fusion_logits_npz.py" \
    --script_name jb_fusion \
    --config_name "${JB_CONFIG}" \
    --save_dir "${SAVE_DIR}" \
    --dataset_name iMiGUE_test \
    --checkpoint_epoch "${JB_EPOCH}" \
    --output_npz "${JB_TEST_NPZ}"
fi

echo "== [3/3] apply fixed fusion weights and generate submission =="
if [[ -f "${FINAL_CSV}" ]]; then
  echo "[SKIP] final submission already exists: ${FINAL_CSV}"
else
python - << PY
import json
import numpy as np
rgb = np.load("${RGB_TEST_NPZ}", allow_pickle=True)
jb = np.load("${JB_TEST_NPZ}", allow_pickle=True)
assert np.array_equal(rgb["ids"], jb["ids"])
np.savez_compressed("${DUAL_TEST_NPZ}", ids=rgb["ids"], rgb_logits=rgb["logits"], sk_logits=jb["logits"])
print("saved ${DUAL_TEST_NPZ}")
print("alpha_rgb=", float("${ALPHA_RGB}"), "beta_sk=", float("${BETA_SK}"))
PY

python - << PY
import subprocess, sys
cmd = [
  "python", "${PROJECT_ROOT}/tools/apply_internal_rgb_sk_logits_fusion.py",
  "--input_npz", "${DUAL_TEST_NPZ}",
  "--alpha_rgb", str(float("${ALPHA_RGB}")),
  "--beta_sk", str(float("${BETA_SK}")),
  "--output_dir", "${OUT_DIR}",
  "--output_basename", "${OUT_BASENAME}",
]
print("RUN:", " ".join(cmd))
sys.exit(subprocess.call(cmd))
PY
fi

echo "All submission stages completed."
