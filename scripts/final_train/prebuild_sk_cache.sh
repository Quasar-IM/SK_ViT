#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/lyn/miniconda3/etc/profile.d/conda.sh
conda activate sk_vit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${SKELETON_TRAINVAL_ROOT:-}" || -z "${SKELETON_TEST_ROOT:-}" ]]; then
  readarray -t __sk_env_lines < <(python - << 'PY'
import importlib.util
import pathlib

local_py = pathlib.Path("lib/train/admin/local.py")
spec = importlib.util.spec_from_file_location("local_env", str(local_py))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
env = mod.EnvironmentSettings()
print(getattr(env, "imigue_sk_trainval_root", ""))
print(getattr(env, "imigue_sk_test_root", ""))
PY
)
  SKELETON_TRAINVAL_ROOT_DEFAULT="${__sk_env_lines[0]:-}"
  SKELETON_TEST_ROOT_DEFAULT="${__sk_env_lines[1]:-}"
fi

SKELETON_TRAINVAL_ROOT="${SKELETON_TRAINVAL_ROOT:-${SKELETON_TRAINVAL_ROOT_DEFAULT:-${PROJECT_ROOT}/data/IMIGUE2026_SKELETON}}"
SKELETON_TEST_ROOT="${SKELETON_TEST_ROOT:-${SKELETON_TEST_ROOT_DEFAULT:-/mnt/sda/Datasets/imigue_data_phase2}}"
SPLITS="${SPLITS:-train,val,test}"
FORCE="${FORCE:-0}"

FORCE_FLAG=""
if [[ "${FORCE}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

resolve_split_root_py() {
  local split="$1"
  python - "$split" << 'PY'
import sys, os, importlib.util, pathlib
split = sys.argv[1]
local_py = pathlib.Path("lib/train/admin/local.py")
spec = importlib.util.spec_from_file_location("local_env", str(local_py))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
env = mod.EnvironmentSettings()
sk_trainval = getattr(env, "imigue_sk_trainval_root", "")
sk_test = getattr(env, "imigue_sk_test_root", "")

def pick(cands):
    for p in cands:
        if p and os.path.isdir(p):
            return p
    return ""

if split == "test":
    print(pick([
        os.path.join(sk_test, "imigue_skeleton_test"),
        os.path.join(sk_test, "datasets", "imigue_skeleton_test"),
        sk_test,
    ]))
elif split == "train":
    print(pick([
        os.path.join(sk_trainval, "imigue_data_phase1", "datasets", "imigue_skeleton_train"),
        os.path.join(sk_trainval, "datasets", "imigue_skeleton_train"),
        os.path.join(sk_trainval, "imigue_skeleton_train"),
    ]))
else:
    print(pick([
        os.path.join(sk_trainval, "imigue_data_phase1", "datasets", "imigue_skeleton_validate"),
        os.path.join(sk_trainval, "datasets", "imigue_skeleton_validate"),
        os.path.join(sk_trainval, "imigue_skeleton_validate"),
    ]))
PY
}

need_run=0
IFS=',' read -r -a __splits <<< "${SPLITS}"
for s in "${__splits[@]}"; do
  s="$(echo "${s}" | xargs)"
  case "${s}" in
    train|val|test) ;;
    *) continue ;;
  esac
  split_root="$(resolve_split_root_py "${s}")"
  d="${split_root}/_sk_maga_preprocessed"
  if [[ -z "${split_root}" || ! -d "${d}" ]]; then
    need_run=1
    break
  fi
  if [[ -z "$(ls -A "${d}" 2>/dev/null)" ]]; then
    need_run=1
    break
  fi
done

if [[ "${FORCE}" != "1" && "${need_run}" == "0" ]]; then
  echo "[SKIP] skeleton preprocess already ready for splits=${SPLITS}"
  exit 0
fi

python "${PROJECT_ROOT}/tools/prebuild_sk_maga_cache.py" \
  --skeleton_trainval_root "${SKELETON_TRAINVAL_ROOT}" \
  --skeleton_test_root "${SKELETON_TEST_ROOT}" \
  --splits "${SPLITS}" \
  ${FORCE_FLAG}

echo "[DONE] skeleton data preprocess ready at split roots: _sk_maga_preprocessed/"
