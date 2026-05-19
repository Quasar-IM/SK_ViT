#!/usr/bin/env bash
set -euo pipefail

# One-shot environment bootstrap for SK_ViT final pipeline.
# Usage:
#   bash install.sh
# Optional:
#   ENV_NAME=sk_vit PY_VER=3.8 TORCH_CUDA=11.3 bash install.sh

ENV_NAME="${ENV_NAME:-sk_vit}"
PY_VER="${PY_VER:-3.8}"
TORCH_CUDA="${TORCH_CUDA:-11.3}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERR] conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

if [[ -f "/home/lyn/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source /home/lyn/miniconda3/etc/profile.d/conda.sh
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "== [1/5] create/activate conda env: ${ENV_NAME} =="
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[OK] env exists: ${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi
conda activate "${ENV_NAME}"

echo "== [2/5] install pytorch stack =="
conda install -y \
  pytorch==1.12.1 \
  torchvision==0.13.1 \
  torchaudio==0.12.1 \
  cudatoolkit="${TORCH_CUDA}" \
  -c pytorch

echo "== [3/5] install pip deps =="
python -m pip install --upgrade pip
python -m pip install \
  opencv-python \
  einops \
  matplotlib \
  pyyaml \
  jpeg4py \
  scipy \
  tensorboardX \
  easydict \
  timm==0.4.12 \
  tqdm

echo "== [4/5] quick import check =="
python - <<'PY'
import torch, torchvision, torchaudio
import cv2, einops, matplotlib, yaml
import scipy, timm, easydict
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

echo "== [5/5] done =="
echo "[OK] Environment is ready: ${ENV_NAME}"
