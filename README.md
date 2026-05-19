# SK_ViT Guide (English + 中文)

## English

### 1. Current Core Pipeline
The current main pipeline is:

1. Train `SK-joint (IE)`  
2. Train `SK-bone (IE)`  
3. Train `joint+bone mutual KL (jb_fusion)`  
4. Train `RGB prune`  
5. Export `RGB/JB` logits on test, then use fixed `alpha/beta` for late fusion to generate submission files

Unified entry: `tools/pipeline.py`.

---

### 2. Environment Setup
Recommended one-shot installer:

```bash
bash install.sh
```

By default, it will:
- create/activate `sk_vit` (Python 3.8)
- install PyTorch 1.12.1 + CUDA 11.3
- install required dependencies and run import checks

---

### 3. Dataset Path Configuration
Edit `configs/pipeline.yaml`:

```yaml
project_root: /absolute/path/to/SK_ViT

env:
  imigue_rgb_trainval_root: /absolute/path/to/<RGB_trainval_parent>
  imigue_sk_trainval_root: /absolute/path/to/<skeleton_phase1_parent>
  imigue_rgb_test_root: /absolute/path/to/<RGB_test_parent>
  imigue_sk_test_root: /absolute/path/to/<skeleton_phase2_parent>
```

Fill **these exact levels**:
- `imigue_rgb_trainval_root`: RGB train/val parent directory (contains `train/`, `val/`, and later generated `train_cash/`, `valid_cash/`).
- `imigue_sk_trainval_root`: **`.../imigue_data_phase1`** (this level, not one level above).
- `imigue_rgb_test_root`: RGB test parent directory (later generated `test_cash/` under it).
- `imigue_sk_test_root`: **`.../imigue_data_phase2`** (code will resolve `imigue_skeleton_test` under it).

Expected skeleton layout (train/val input):

```text
imigue_sk_trainval_root/
└── datasets/
    ├── imigue_skeleton_train/
    │   ├── 0001/
    │   └── ...
    └── imigue_skeleton_validate/
        ├── 0001/
        └── ...
```

Expected skeleton layout (test input):

```text
imigue_sk_test_root/
└── imigue_skeleton_test/
    ├── 0001/
    └── ...
```

Expected RGB layout (train/val input):

```text
imigue_rgb_trainval_root/
├── train/
│   ├── 0001/
│   └── ...
└── val/
    ├── 0001/
    └── ...
```

Expected RGB layout (test input):

```text
imigue_rgb_test_root/
├── 0001/
└── ...
```

After `--stage preprocess_data`, generated skeleton preprocess files will be:

```text
.../imigue_skeleton_train/_sk_maga_preprocessed/
.../imigue_skeleton_validate/_sk_maga_preprocessed/
.../imigue_skeleton_test/_sk_maga_preprocessed/
```

After `--stage preprocess_data`, generated RGB preprocess files will be:

```text
imigue_rgb_trainval_root/train_cash/
imigue_rgb_trainval_root/valid_cash/
imigue_rgb_test_root/test_cash/
```

---

### 4. One-Command Pipeline Full Run
After environment setup and `configs/pipeline.yaml` configuration, run:

```bash
python tools/pipeline.py --config configs/pipeline.yaml --stage all
```

This `all` stage executes in order:

1. `init_local` (writes `lib/train/admin/local.py`)
2. `check_env`
3. `preprocess_data` (required data preprocessing)

After this, you can run training/reproduction directly.

---

### 5. Training & Reproduction (Adjacent)
- One-shot training (automatically enters submission stage after training):

```bash
bash scripts/final_train/run_all_train.sh
```

This script does **not** run preprocessing. It first checks whether preprocessing outputs already exist; if not, it fails immediately.

Before training, download `pretrained_models` from:
- https://1drv.ms/f/c/9c72e1b730033315/IgCLN-0_h4OLSbsap_zqc-KBAa2IJhwVtin7SZ0jacI0u-0

Place the downloaded folder at:

```text
SK_ViT/pretrained_models/
```

- Reproduce only the final step (when both required weights are already prepared):

```bash
bash repro_laststep_weights/run_repro_sub.sh
```

This script also does **not** run preprocessing. It first checks preprocessing outputs; if missing, it fails immediately.

Before reproduction, download repro weights from:
- https://1drv.ms/f/c/9c72e1b730033315/IgBSaRtgdrm5QoDZcKljNjHPAQ4yNLVrPj98KHt0Ckl_eGs?e=VKMPBf

Place required files at:

```text
SK_ViT/repro_laststep_weights/
├── VisionTransformer_ep0026.pth.tar
└── SKJointBoneFeatureFusionModel_ep0020.pth.tar
```

---

### 6. Key Script Locations
- Training orchestrator: `scripts/final_train/run_all_train.sh`
- Submission orchestrator: `scripts/final_sub/run_all_sub.sh`
- Final-step-only reproduction (with ready dual weights): `repro_laststep_weights/run_repro_sub.sh`

---

### 7. Weight Placement for Final-Step Reproduction
Before running `bash repro_laststep_weights/run_repro_sub.sh`, place the following two weight files in:

- Directory: `repro_laststep_weights/`
- Filenames (must match exactly):
  - `VisionTransformer_ep0026.pth.tar`
  - `SKJointBoneFeatureFusionModel_ep0020.pth.tar`

The script will automatically search for these two files in this directory and run the fusion submission process.

---

### 8. Output Inspection
Training pipeline (`bash scripts/final_train/run_all_train.sh`):
- Checkpoints: `output/sk/checkpoints/`, `output/jb_fusion/checkpoints/`, `output/rgb/checkpoints/`
- Final submission: `scripts/final_sub/submission/`

Reproduction pipeline (`bash repro_laststep_weights/run_repro_sub.sh`):
- Repro artifacts: `repro_laststep_weights/repro_output/`
- Repro submission: `repro_laststep_weights/submission/`

---

### 9. Common Issues
#### 9.1 Port Conflict
If you see `Address already in use` (commonly port `29500`), it means another distributed job is using that port.  
Solution: stop the old process, or change `MASTER_PORT` in the scripts.

#### 9.2 Dataset Path Check
Run:

```bash
python tools/pipeline.py --config configs/pipeline.yaml --stage check_data
```

Confirm that all configured paths are accessible and directory structures can be resolved correctly.

---

## 中文

### 1. 当前保留的核心流程
当前代码主流程是：

1. 训练 `SK-joint(IE)`  
2. 训练 `SK-bone(IE)`  
3. 训练 `joint+bone mutual KL (jb_fusion)`  
4. 训练 `RGB prune`  
5. 在 test 上导出 `RGB/JB` logits，使用固定 `alpha/beta` 做 late-fusion 生成提交文件

统一入口：`tools/pipeline.py`。

---

### 2. 环境安装
建议直接使用一键脚本：

```bash
bash install.sh
```

默认会：
- 创建/激活 `sk_vit`（Python 3.8）
- 安装 PyTorch 1.12.1 + CUDA 11.3
- 安装当前流程依赖并做导入校验

---

### 3. 数据路径配置
编辑 `configs/pipeline.yaml`：

```yaml
project_root: /绝对路径/SK_ViT

env:
  imigue_rgb_trainval_root: /绝对路径/<RGB训练验证总目录>
  imigue_sk_trainval_root: /绝对路径/<skeleton_phase1总目录>
  imigue_rgb_test_root: /绝对路径/<RGB测试总目录>
  imigue_sk_test_root: /绝对路径/<skeleton_phase2总目录>
```

四个路径建议按下面层级填写：
- `imigue_rgb_trainval_root`：RGB 训练/验证总目录（内部有 `train/`、`val/`，预处理后会生成 `train_cash/`、`valid_cash/`）。
- `imigue_sk_trainval_root`：**填到 `.../imigue_data_phase1` 这一层**（不要再高一级）。
- `imigue_rgb_test_root`：RGB 测试总目录（预处理后会在这里生成 `test_cash/`）。
- `imigue_sk_test_root`：**填到 `.../imigue_data_phase2` 这一层**（代码会自动解析其下的 `imigue_skeleton_test`）。

Skeleton 原始目录结构（train/val）：

```text
imigue_sk_trainval_root/
└── datasets/
    ├── imigue_skeleton_train/
    │   ├── 0001/
    │   └── ...
    └── imigue_skeleton_validate/
        ├── 0001/
        └── ...
```

Skeleton 原始目录结构（test）：

```text
imigue_sk_test_root/
└── imigue_skeleton_test/
    ├── 0001/
    └── ...
```

RGB 原始目录结构（train/val）：

```text
imigue_rgb_trainval_root/
├── train/
│   ├── 0001/
│   └── ...
└── val/
    ├── 0001/
    └── ...
```

RGB 原始目录结构（test）：

```text
imigue_rgb_test_root/
├── 0001/
└── ...
```

执行 `--stage preprocess_data` 后，会生成：

```text
.../imigue_skeleton_train/_sk_maga_preprocessed/
.../imigue_skeleton_validate/_sk_maga_preprocessed/
.../imigue_skeleton_test/_sk_maga_preprocessed/
```

执行 `--stage preprocess_data` 后，RGB 会生成：

```text
imigue_rgb_trainval_root/train_cash/
imigue_rgb_trainval_root/valid_cash/
imigue_rgb_test_root/test_cash/
```

---

### 4. pipeline 一键全流程
先安装环境并配置好 `configs/pipeline.yaml`，然后执行：

```bash
python tools/pipeline.py --config configs/pipeline.yaml --stage all
```

这个 `all` 会顺序执行：

1. `init_local`（写入 `lib/train/admin/local.py`）
2. `check_env`
3. `preprocess_data`（必需的数据预处理）

完成后即可直接运行训练/复现脚本。

---

### 5. 训练与复现（相邻）
- 训练一口气（训练结束自动进入提交流程）：

```bash
bash scripts/final_train/run_all_train.sh
```

该脚本**不会执行预处理**。会先检测预处理结果是否已存在；若不存在会立即报错退出。

训练前请先下载 `pretrained_models`：
- https://1drv.ms/f/c/9c72e1b730033315/IgCLN-0_h4OLSbsap_zqc-KBAa2IJhwVtin7SZ0jacI0u-0

下载后放到：

```text
SK_ViT/pretrained_models/
```

- 仅复现最后一步（已有双权重）：

```bash
bash repro_laststep_weights/run_repro_sub.sh
```

该脚本同样**不会执行预处理**。会先检测预处理结果；缺失则立即报错退出。

复现前请先下载复现权重：
- https://1drv.ms/f/c/9c72e1b730033315/IgBSaRtgdrm5QoDZcKljNjHPAQ4yNLVrPj98KHt0Ckl_eGs?e=VKMPBf

下载后将这两个文件放到：

```text
SK_ViT/repro_laststep_weights/
├── VisionTransformer_ep0026.pth.tar
└── SKJointBoneFeatureFusionModel_ep0020.pth.tar
```

---

### 6. 关键脚本位置
- 训练总控：`scripts/final_train/run_all_train.sh`
- 提交总控：`scripts/final_sub/run_all_sub.sh`
- 仅复现最后一步（已有双权重）：`repro_laststep_weights/run_repro_sub.sh`

---

### 7. 复现最后一步的权重放置
运行 `bash repro_laststep_weights/run_repro_sub.sh` 前，请将以下两个权重文件放到目录：

- 目录：`repro_laststep_weights/`
- 文件名（需一致）：
  - `VisionTransformer_ep0026.pth.tar`
  - `SKJointBoneFeatureFusionModel_ep0020.pth.tar`

脚本会在该目录自动查找这两个文件并执行融合提交流程。

---

### 8. 输出结果查看
训练流程（`bash scripts/final_train/run_all_train.sh`）：
- 权重目录：`output/sk/checkpoints/`、`output/jb_fusion/checkpoints/`、`output/rgb/checkpoints/`
- 最终提交文件：`scripts/final_sub/submission/`

复现流程（`bash repro_laststep_weights/run_repro_sub.sh`）：
- 复现中间产物：`repro_laststep_weights/repro_output/`
- 复现提交文件：`repro_laststep_weights/submission/`

---

### 9. 常见问题
#### 9.1 端口冲突
报错 `Address already in use`（常见是 29500）时，表示已有分布式任务占端口。  
处理方式：结束旧进程或改脚本里的 `MASTER_PORT`。

#### 9.2 数据路径检查
先跑：

```bash
python tools/pipeline.py --config configs/pipeline.yaml --stage check_data
```

确认路径都能访问，且目录结构可被自动解析。
