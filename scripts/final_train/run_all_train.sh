#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT="${PROJECT_ROOT}/scripts/final_train"

bash "${ROOT}/check_preprocessed_data.sh"
bash "${ROOT}/final_01_train_joint_ie.sh"
bash "${ROOT}/final_02_train_bone_ie.sh"
bash "${ROOT}/final_03_train_jb_mutualkl.sh"
bash "${ROOT}/final_04_train_rgb_prune.sh"

echo "All training stages completed."

# Continue directly to test + late-fusion submission
SUB_ROOT="${PROJECT_ROOT}/scripts/final_sub"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/output}" bash "${SUB_ROOT}/run_all_sub.sh"

echo "All train+submission stages completed."
