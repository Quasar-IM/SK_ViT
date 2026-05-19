#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/lyn/miniconda3/etc/profile.d/conda.sh
conda activate sk_vit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${RGB_TRAINVAL_ROOT:-}" || -z "${RGB_TEST_ROOT:-}" ]]; then
  readarray -t __rgb_env_lines < <(python - << 'PY'
import importlib.util
import pathlib

local_py = pathlib.Path("lib/train/admin/local.py")
spec = importlib.util.spec_from_file_location("local_env", str(local_py))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
env = mod.EnvironmentSettings()
print(getattr(env, "imigue_rgb_trainval_root", ""))
print(getattr(env, "imigue_rgb_test_root", ""))
PY
)
  RGB_TRAINVAL_ROOT_DEFAULT="${__rgb_env_lines[0]:-}"
  RGB_TEST_ROOT_DEFAULT="${__rgb_env_lines[1]:-}"
fi

RGB_TRAINVAL_ROOT="${RGB_TRAINVAL_ROOT:-${RGB_TRAINVAL_ROOT_DEFAULT:-${PROJECT_ROOT}/data/IMIGUE2026_RGB}}"
RGB_TEST_ROOT="${RGB_TEST_ROOT:-${RGB_TEST_ROOT_DEFAULT:-/mnt/sda/Datasets/MIGA3_track3_Phase_2}}"
SPLITS="${SPLITS:-train,val,test}"
SIZE="${SIZE:-256}"
FORCE="${FORCE:-0}"

need_run=0
IFS=',' read -r -a __splits <<< "${SPLITS}"
for s in "${__splits[@]}"; do
  s="$(echo "${s}" | xargs)"
  case "${s}" in
    train) d="${RGB_TRAINVAL_ROOT}/train_cash" ;;
    val) d="${RGB_TRAINVAL_ROOT}/valid_cash" ;;
    test) d="${RGB_TEST_ROOT}/test_cash" ;;
    *) continue ;;
  esac
  if [[ ! -d "${d}" ]]; then
    need_run=1
    break
  fi
  if [[ -z "$(ls -A "${d}" 2>/dev/null)" ]]; then
    need_run=1
    break
  fi
done

if [[ "${FORCE}" != "1" && "${need_run}" == "0" ]]; then
  echo "[SKIP] RGB preprocess already ready for splits=${SPLITS}"
  exit 0
fi

python "${PROJECT_ROOT}/tools/prebuild_rgb_frames.py" \
  --rgb_trainval_root "${RGB_TRAINVAL_ROOT}" \
  --rgb_test_root "${RGB_TEST_ROOT}" \
  --splits "${SPLITS}" \
  --size "${SIZE}"

echo "[DONE] RGB preprocess checked/built:"
echo "       trainval root: ${RGB_TRAINVAL_ROOT}"
echo "       test root:     ${RGB_TEST_ROOT}"
