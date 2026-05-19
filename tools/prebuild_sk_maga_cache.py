#!/usr/bin/env python3
import argparse
import os
import csv
import numpy as np
import torch
from tqdm import tqdm

from lib.train.dataset.sk_maga_schema import (
    project_to_canonical_22,
    append_hand_centers,
    reorder_canonical_24_to_model_24,
)


SPLIT_CANDIDATES = {
    "train": ["imigue_skeleton_train"],
    "val": ["imigue_skeleton_validate"],
    "test": [""],  # strict: use provided test root directly
}


def pick_existing(paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


def resolve_split_root(sk_trainval_root, sk_test_root, split):
    if split == "test":
        cands = [
            os.path.join(sk_test_root, "imigue_skeleton_test"),
            os.path.join(sk_test_root, "datasets", "imigue_skeleton_test"),
            sk_test_root,
        ]
        return pick_existing(cands)
    # strict fixed layout under trainval root
    cands = [
        os.path.join(sk_trainval_root, "datasets", SPLIT_CANDIDATES[split][0]),
        os.path.join(sk_trainval_root, "imigue_data_phase1", "datasets", SPLIT_CANDIDATES[split][0]),
        os.path.join(sk_trainval_root, SPLIT_CANDIDATES[split][0]),
    ]
    return pick_existing(cands)


def load_csv_rows(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) <= 1:
        return np.zeros((0, 106), dtype=np.float32)
    return np.asarray(rows[1:], dtype=np.float32)


def convert_row_to_model24(frame_values):
    raw_points = torch.from_numpy(frame_values[1:].reshape(35, 3).astype(np.float32))
    canonical_22 = project_to_canonical_22(raw_points)
    canonical_24 = append_hand_centers(canonical_22)
    return reorder_canonical_24_to_model_24(canonical_24).numpy()


def build_one_video_cache(video_dir, out_path):
    video_id = os.path.basename(video_dir)
    csv_path = os.path.join(video_dir, f"{video_id}_light_hand.csv")
    if not os.path.exists(csv_path):
        return False, "missing_csv"
    rows = load_csv_rows(csv_path)
    if rows.shape[0] == 0:
        arr = np.zeros((0, 24, 3), dtype=np.float32)
        np.save(out_path, arr)
        return True, "empty"
    arr = np.zeros((rows.shape[0], 24, 3), dtype=np.float32)
    for t in range(rows.shape[0]):
        arr[t] = convert_row_to_model24(rows[t])
    np.save(out_path, arr)
    return True, "ok"


def build_split(sk_trainval_root, sk_test_root, split, force=False):
    split_root = resolve_split_root(sk_trainval_root, sk_test_root, split)
    if split_root is None:
        print(f"[CACHE] split={split} not found, skip.")
        return
    out_dir = os.path.join(split_root, "_sk_maga_preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    video_ids = sorted([x for x in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, x))])
    total = len(video_ids)
    built, skipped, missed = 0, 0, 0

    print(f"[CACHE] split={split} src={split_root}")
    print(f"[CACHE] split={split} dst={out_dir}")
    pbar = tqdm(video_ids, desc=f"SK preprocess [{split}]", unit="video")
    for i, vid in enumerate(pbar, 1):
        video_dir = os.path.join(split_root, vid)
        out_path = os.path.join(out_dir, f"{vid}_sk_maga.npy")
        if (not force) and os.path.exists(out_path):
            skipped += 1
            continue
        ok, tag = build_one_video_cache(video_dir, out_path)
        if ok:
            built += 1
        else:
            missed += 1
        pbar.set_postfix(built=built, skipped=skipped, missed=missed)

    print(f"[CACHE] split={split} done total={total} built={built} skipped={skipped} missed={missed}")


def main():
    parser = argparse.ArgumentParser("Prebuild SK_MAGA cache from *_light_hand.csv")
    parser.add_argument("--skeleton_trainval_root", type=str, default="/home/lyn/wky/SK_ViT/data/IMIGUE2026_SKELETON")
    parser.add_argument("--skeleton_test_root", type=str, default="/mnt/sda/Datasets/imigue_data_phase2")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.skeleton_trainval_root):
        raise FileNotFoundError(f"skeleton_trainval_root not found: {args.skeleton_trainval_root}")
    if not os.path.isdir(args.skeleton_test_root):
        raise FileNotFoundError(f"skeleton_test_root not found: {args.skeleton_test_root}")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        if split not in SPLIT_CANDIDATES:
            raise ValueError(f"Invalid split: {split}")
        build_split(args.skeleton_trainval_root, args.skeleton_test_root, split, force=args.force)


if __name__ == "__main__":
    main()
