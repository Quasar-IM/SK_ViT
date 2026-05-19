import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cmd(cmd: str, cwd: str):
    print(f"[RUN] {cmd}")
    proc = subprocess.run(cmd, shell=True, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")


def cmd_check_env(project_root: str):
    checks = [
        os.path.join(project_root, "environment.txt"),
        os.path.join(project_root, "lib", "train", "admin", "local.py"),
        os.path.join(project_root, "scripts", "final_train", "run_all_train.sh"),
        os.path.join(project_root, "scripts", "final_sub", "run_all_sub.sh"),
    ]
    ok = True
    for p in checks:
        exists = os.path.exists(p)
        print(f"[{'OK' if exists else 'MISS'}] {p}")
        ok = ok and exists
    if not ok:
        raise RuntimeError("Environment check failed.")
    print("[OK] environment check passed.")


def cmd_check_data(project_root: str):
    from lib.train.admin.environment import env_settings

    env = env_settings()
    paths = {
        "imigue_rgb_trainval_root": getattr(env, "imigue_rgb_trainval_root", ""),
        "imigue_sk_trainval_root": getattr(env, "imigue_sk_trainval_root", ""),
        "imigue_rgb_test_root": getattr(env, "imigue_rgb_test_root", ""),
        "imigue_sk_test_root": getattr(env, "imigue_sk_test_root", ""),
    }
    ok = True
    for k, p in paths.items():
        exists = bool(p) and os.path.isdir(p)
        print(f"[{'OK' if exists else 'MISS'}] {k}: {p}")
        ok = ok and exists
    if not ok:
        raise RuntimeError("Data check failed. Update local.py paths first.")
    print("[OK] data check passed.")


def cmd_preprocess_data(project_root: str):
    run_cmd("bash scripts/final_train/prebuild_rgb_cache.sh", cwd=project_root)
    run_cmd("bash scripts/final_train/prebuild_sk_cache.sh", cwd=project_root)
    print("[OK] data preprocess completed.")


def cmd_init_local(project_root: str, cfg: dict):
    env_cfg = cfg.get("env", {})
    if not env_cfg:
        raise RuntimeError("No 'env' section found in pipeline yaml.")
    required = ["imigue_rgb_trainval_root", "imigue_sk_trainval_root", "imigue_rgb_test_root", "imigue_sk_test_root"]
    miss = [k for k in required if not env_cfg.get(k, "")]
    if miss:
        raise RuntimeError(f"Missing required env keys in pipeline yaml: {miss}")

    local_py = Path(project_root) / "lib" / "train" / "admin" / "local.py"
    local_py.parent.mkdir(parents=True, exist_ok=True)

    keys_order = [
        "workspace_dir",
        "tensorboard_dir",
        "pretrained_networks",
        "ijcai_miga_track1_dir",
        "imigue_rgb_trainval_root",
        "imigue_sk_trainval_root",
        "imigue_rgb_test_root",
        "imigue_sk_test_root",
        "imigue_depth_dir",
    ]

    derived_env = dict(env_cfg)
    derived_env.setdefault("workspace_dir", project_root)
    derived_env.setdefault("tensorboard_dir", os.path.join(project_root, "output", "tensorboard"))
    derived_env.setdefault("pretrained_networks", os.path.join(project_root, "pretrained_networks"))
    derived_env.setdefault("ijcai_miga_track1_dir", "")
    derived_env.setdefault("imigue_depth_dir", "")
    # Project convention: keep skeleton test root fixed at parent dataset root.
    derived_env["imigue_sk_test_root"] = "/mnt/sda/Datasets/imigue_data_phase2"
    lines = []
    lines.append("class EnvironmentSettings:")
    lines.append("    def __init__(self):")
    for k in keys_order:
        v = derived_env.get(k, "")
        v = "" if v is None else str(v)
        lines.append(f"        self.{k} = {v!r}")
    lines.append("")
    local_py.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote local settings: {local_py}")


def main():
    parser = argparse.ArgumentParser("Unified pipeline entry")
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Pipeline yaml config path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["init_local", "preprocess_data", "train_all", "sub_all", "repro_sub", "check_env", "check_data", "all"],
        help="Stage to run.",
    )
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        # assume run from project root
        cfg_path = os.path.abspath(cfg_path)
    cfg = load_cfg(cfg_path)
    project_root = cfg["project_root"]

    if args.stage == "init_local":
        cmd_init_local(project_root, cfg)
        return
    if args.stage == "check_env":
        cmd_check_env(project_root)
        return
    if args.stage == "check_data":
        sys.path.insert(0, project_root)
        cmd_check_data(project_root)
        return
    if args.stage == "preprocess_data":
        cmd_preprocess_data(project_root)
        return
    if args.stage == "all":
        cmd_init_local(project_root, cfg)
        cmd_check_env(project_root)
        cmd_preprocess_data(project_root)
        print("[DONE] stage=all (init_local + check_env + preprocess_data)")
        return

    commands = cfg.get("commands", {}).get(args.stage, [])
    if not commands:
        raise RuntimeError(f"No commands configured for stage={args.stage}")

    for cmd in commands:
        run_cmd(cmd, cwd=project_root)
    print(f"[DONE] stage={args.stage}")


if __name__ == "__main__":
    main()
