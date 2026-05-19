#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

source /home/lyn/miniconda3/etc/profile.d/conda.sh
conda activate sk_vit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python - << 'PY'
import os
from lib.train.admin.environment import env_settings

env = env_settings()
rgb_trainval_root = getattr(env, "imigue_rgb_trainval_root", "")
rgb_test_root = getattr(env, "imigue_rgb_test_root", "")

sk_trainval_root = getattr(env, "imigue_sk_trainval_root", "")
sk_test_root = getattr(env, "imigue_sk_test_root", "")

def pick_existing(cands):
    for p in cands:
        if p and os.path.isdir(p):
            return p
    return None

def resolve_sk_split(split):
    if split == "test":
        return pick_existing([
            os.path.join(sk_test_root, "imigue_skeleton_test"),
            os.path.join(sk_test_root, "datasets", "imigue_skeleton_test"),
            sk_test_root,
        ])
    name = "imigue_skeleton_train" if split == "train" else "imigue_skeleton_validate"
    return pick_existing([
        os.path.join(sk_trainval_root, "datasets", name),
        os.path.join(sk_trainval_root, "imigue_data_phase1", "datasets", name),
        os.path.join(sk_trainval_root, name),
    ])

checks = [
    ("rgb train preprocess", os.path.join(rgb_trainval_root, "train_cash")),
    ("rgb val preprocess", os.path.join(rgb_trainval_root, "valid_cash")),
    ("rgb test preprocess", os.path.join(rgb_test_root, "test_cash")),
    ("skeleton train preprocess", os.path.join(resolve_sk_split("train") or "", "_sk_maga_preprocessed")),
    ("skeleton val preprocess", os.path.join(resolve_sk_split("val") or "", "_sk_maga_preprocessed")),
    ("skeleton test preprocess", os.path.join(resolve_sk_split("test") or "", "_sk_maga_preprocessed")),
]

missing = []
for name, path in checks:
    ok = os.path.isdir(path) and len(os.listdir(path)) > 0
    print(f"[{'OK' if ok else 'MISS'}] {name}: {path}")
    if not ok:
        missing.append((name, path))

if missing:
    raise RuntimeError(
        "Preprocessed data missing. Please run: "
        "python tools/pipeline.py --config configs/pipeline.yaml --stage preprocess_data"
    )

print("[OK] preprocessed data check passed.")
PY
