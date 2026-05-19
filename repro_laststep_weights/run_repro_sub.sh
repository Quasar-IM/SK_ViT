#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
source /home/lyn/miniconda3/etc/profile.d/conda.sh
conda activate sk_vit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "== [0/4] check preprocessed data (required) =="
bash "${PROJECT_ROOT}/scripts/final_train/check_preprocessed_data.sh"

REPRO_DIR="${SCRIPT_DIR}"
SAVE_DIR="${REPRO_DIR}/repro_output"
TMP_DIR="${REPRO_DIR}/tmp_run"
OUT_DIR="${REPRO_DIR}/submission"
mkdir -p "${TMP_DIR}" "${OUT_DIR}"

RGB_CKPT_SRC="$(find "${REPRO_DIR}" -maxdepth 1 -type f -name 'VisionTransformer_ep0026.pth.tar' | head -n 1)"
JB_CKPT_SRC="$(find "${REPRO_DIR}" -maxdepth 1 -type f -name 'SKJointBoneFeatureFusionModel_ep0020.pth.tar' | head -n 1)"

if [[ -z "${RGB_CKPT_SRC}" || -z "${JB_CKPT_SRC}" || ! -f "${RGB_CKPT_SRC}" || ! -f "${JB_CKPT_SRC}" ]]; then
  echo "[ERR] missing local repro weights in ${REPRO_DIR}"
  echo "need file: VisionTransformer_ep0026.pth.tar"
  echo "need file: SKJointBoneFeatureFusionModel_ep0020.pth.tar"
  exit 1
fi

RGB_CONFIG="imigue_prune70_2mean_merge_rgb_k400_ep40"
RGB_EPOCH=26

# Use a dedicated temp jb config adapted to currently available stream checkpoints.
JB_CONFIG="imigue_joint_bone_mutualkl_repro_rgb26jb20"
JB_EPOCH=20
JB_CFG_FILE="${PROJECT_ROOT}/experiments/jb_fusion/${JB_CONFIG}.yaml"

cat > "${JB_CFG_FILE}" << 'YAML'
DATA:
  SAMPLE_FRAMES: 32
  SIZE: 256
  CROP_SIZE: 256
  MODALITY: 'SK_MAGA_SKELETON'
  TRAIN:
    SAMPLER_MODE: 'RANDOMLY'
    DATASET_NAME: iMiGUE_train_full_plus_official_val
  VAL:
    SAMPLER_MODE: 'UNIFORMLY'
    DATASET_NAME: iMiGUE_official_val
  NUM_CLASSES: 32

MODEL:
  JOINT_BONE_FEATURE_FUSION: True
  FUSION_JOINT_CONFIG: 'imigue_sk_joint_ie_officialvaltrain_ep20'
  FUSION_BONE_CONFIG: 'imigue_sk_bone_ie_officialvaltrain_ep20'
  FUSION_JOINT_CHECKPOINT: 'output/sk/checkpoints/train/sk_maga/imigue_sk_maga_ep30/SKMagaModel_ep0025.pth.tar'
  FUSION_BONE_CHECKPOINT: 'output/sk/checkpoints/train/sk_maga/imigue_sk_maga_bone_ep30/SKMagaModel_ep0025.pth.tar'
  JB_FUSION_PROJ_DIM: 256
  FUSION_DROPOUT: 0.2

TRAIN:
  BATCH_SIZE: 8
  NUM_WORKER: 8
  LR: 1e-5
  EPOCH: 20
  WARM_UP_EPOCH: 5
  OPTIMIZER: 'ADAMW'
  WEIGHT_DECAY: 0.05
  SAVE_EVERY_EPOCH: True
  LOAD_LATEST: False
  FUSION_FREEZE_BACKBONES: False
  FUSION_FREEZE_EPOCHS: 0
  FUSION_BACKBONE_LR_MULT: 0.1
  JB_FUSION_WEIGHT: 1.0
  JB_JOINT_WEIGHT: 1.0
  JB_BONE_WEIGHT: 1.0
  JB_KL_JOINT_FROM_BONE_WEIGHT: 1.0
  JB_KL_BONE_FROM_JOINT_WEIGHT: 1.0
  JB_KD_TEMPERATURE: 2.0
  SCHEDULER:
    TYPE: 'warmup_and_cosine'
    INITIAL_LR: 1e-6
    MIN_LR: 1e-7
YAML

# Link the two repro checkpoints into expected save_dir structure.
RGB_CKPT_DST="${SAVE_DIR}/rgb/checkpoints/train/sk_vit/${RGB_CONFIG}/VisionTransformer_ep0026.pth.tar"
JB_CKPT_DST="${SAVE_DIR}/jb_fusion/checkpoints/train/jb_fusion/${JB_CONFIG}/SKJointBoneFeatureFusionModel_ep0020.pth.tar"
mkdir -p "$(dirname "${RGB_CKPT_DST}")" "$(dirname "${JB_CKPT_DST}")"
ln -sfn "${RGB_CKPT_SRC}" "${RGB_CKPT_DST}"
ln -sfn "${JB_CKPT_SRC}" "${JB_CKPT_DST}"

ALPHA_RGB="${ALPHA_RGB:-0.6352739334106445}"
BETA_SK="${BETA_SK:-0.3647260367870331}"

RGB_TEST_NPZ="${TMP_DIR}/rgb_test.npz"
JB_TEST_NPZ="${TMP_DIR}/jb_test.npz"
DUAL_TEST_NPZ="${TMP_DIR}/dual_test.npz"

OUT_BASENAME="Submission_rgb13_linear_rgb26_jbmutual20_repro"
FINAL_CSV="${OUT_DIR}/${OUT_BASENAME}.csv"
if [[ -f "${FINAL_CSV}" ]]; then
  echo "[SKIP] already exists: ${FINAL_CSV}"
  exit 0
fi

echo "== [1/4] use preset fusion weights =="
echo "alpha_rgb=${ALPHA_RGB}"
echo "beta_sk=${BETA_SK}"

echo "== [2/4] export test logits (rgb/jb) =="
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
  cp -f "${PROJECT_ROOT}/submission/rgb/sk_vit/${RGB_CONFIG}/Submission_ep0026_logits.npz" "${RGB_TEST_NPZ}"
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

echo "== [3/4] apply fixed weight and generate submission =="
if [[ -f "${FINAL_CSV}" ]]; then
  echo "[SKIP] final submission already exists: ${FINAL_CSV}"
else
python - << PY
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

  cat > "${OUT_DIR}/${OUT_BASENAME}_weights.json" << EOF
{
  "alpha_rgb": ${ALPHA_RGB},
  "beta_sk": ${BETA_SK}
}
EOF
  cat > "${OUT_DIR}/${OUT_BASENAME}_meta.txt" << EOF
RGB_WEIGHT=${RGB_CKPT_SRC}
JB_WEIGHT=${JB_CKPT_SRC}
RGB_CONFIG=${RGB_CONFIG}
RGB_EPOCH=${RGB_EPOCH}
JB_CONFIG=${JB_CONFIG}
JB_EPOCH=${JB_EPOCH}
VAL_DATASET=SKIPPED_USE_FIXED_ALPHA_BETA
EOF
fi

echo "== [done] =="
echo "csv : ${OUT_DIR}/${OUT_BASENAME}.csv"
echo "zip : ${OUT_DIR}/${OUT_BASENAME}.zip"
echo "npz : ${OUT_DIR}/${OUT_BASENAME}.npz"
echo "wgt : ${OUT_DIR}/${OUT_BASENAME}_weights.json"
