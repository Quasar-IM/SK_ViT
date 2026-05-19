import argparse
import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


SPLIT_TO_CACHE = {
    "train": "train_cash",
    "val": "valid_cash",
    "test": "test_cash",
}


def _is_jpg_name(name: str) -> bool:
    low = name.lower()
    return low.endswith(".jpg") or low.endswith(".jpeg")


def _find_all_mp4(video_dir: Path):
    mp4s = []
    for n in sorted(os.listdir(video_dir)):
        if n.lower().endswith(".mp4"):
            mp4s.append(video_dir / n)
    return mp4s


def _extract_mp4_to_jpgs(mp4_path: Path, out_dir: Path, size: int, start_idx: int) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    ok, frame = cap.read()
    idx = start_idx
    while ok:
        if size > 0:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)
        idx += 1
        ok, frame = cap.read()
    cap.release()
    return idx - start_idx


def _copy_label_if_exists(src_video_dir: Path, dst_video_dir: Path, video_id: str):
    cand = [
        src_video_dir / f"{video_id}_label.csv",
        src_video_dir / "label.csv",
    ]
    for p in cand:
        if p.is_file():
            shutil.copy2(str(p), str(dst_video_dir / f"{video_id}_label.csv"))
            return True
    return False


def _prepare_from_src_split(src_split_dir: Path, cache_split_dir: Path, size: int):
    if not src_split_dir.is_dir():
        print(f"[RGB] split missing, skip: {src_split_dir}")
        return
    cache_split_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    cache_name = cache_split_dir.name
    for p in sorted(src_split_dir.iterdir()):
        if not p.is_dir():
            continue
        # Avoid recursively treating cache dirs as source videos.
        if p.name in {"train_cash", "valid_cash", "test_cash", cache_name}:
            continue
        videos.append(p)
    built, skipped, missed = 0, 0, 0
    pbar = tqdm(videos, desc=f"RGB preprocess [{src_split_dir.name}]", unit="video")
    for i, src_v in enumerate(pbar, 1):
        vid = src_v.name
        dst_v = cache_split_dir / vid
        dst_v.mkdir(parents=True, exist_ok=True)

        mp4s = _find_all_mp4(src_v)
        if len(mp4s) == 0:
            # If source already has jpgs, copy them as fallback to video root.
            has_jpg = any(_is_jpg_name(n) for n in os.listdir(dst_v))
            if has_jpg:
                skipped += 1
            else:
                src_jpgs = [n for n in sorted(os.listdir(src_v)) if _is_jpg_name(n)]
                if len(src_jpgs) == 0:
                    missed += 1
                else:
                    for n in src_jpgs:
                        shutil.copy2(str(src_v / n), str(dst_v / n))
                    built += 1
        else:
            # Per-mp4 cache to avoid frame-index mixing.
            wrote_any = False
            for j, mp4 in enumerate(mp4s):
                clip_dir = dst_v / f"{vid}_{j}"
                clip_dir.mkdir(parents=True, exist_ok=True)
                has_clip_jpg = any(_is_jpg_name(n) for n in os.listdir(clip_dir))
                if has_clip_jpg:
                    continue
                wrote = _extract_mp4_to_jpgs(mp4, clip_dir, size=size, start_idx=0)
                if wrote > 0:
                    wrote_any = True
            if wrote_any or any((dst_v / f"{vid}_{j}").is_dir() for j in range(len(mp4s))):
                built += 1
            else:
                missed += 1

        _copy_label_if_exists(src_v, dst_v, vid)

        pbar.set_postfix(built=built, skipped=skipped, missed=missed)

    print(
        f"[RGB] split={src_split_dir.name} done total={len(videos)} "
        f"built={built} skipped={skipped} missed={missed} -> cache={cache_split_dir}"
    )


def _resolve_split_dir(root: Path, split: str):
    alias = {
        "train": ["train"],
        "val": ["val", "valid", "validate"],
        "test": ["test"],
    }[split]
    # For test, strictly use the given root directly.
    if split == "test":
        return root if root.is_dir() else None
    for n in alias:
        c = root / n
        if c.is_dir():
            return c
    for n in alias:
        c = root / "RGB" / n
        if c.is_dir():
            return c
    return None


def main():
    parser = argparse.ArgumentParser("Prebuild RGB frame caches")
    parser.add_argument("--rgb_trainval_root", required=True, type=str)
    parser.add_argument("--rgb_test_root", required=True, type=str)
    parser.add_argument("--splits", default="train,val,test", type=str)
    parser.add_argument("--size", default=256, type=int)
    args = parser.parse_args()

    trainval_root = Path(args.rgb_trainval_root)
    test_root = Path(args.rgb_test_root)
    if not trainval_root.is_dir():
        raise FileNotFoundError(f"rgb trainval root not found: {trainval_root}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"rgb test root not found: {test_root}")

    req_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in req_splits:
        if split not in SPLIT_TO_CACHE:
            print(f"[RGB] unknown split={split}, skip")
            continue
        if split in ("train", "val"):
            src_root = trainval_root
        else:
            src_root = test_root
        src_split_dir = _resolve_split_dir(src_root, split)
        if src_split_dir is None:
            print(f"[RGB] cannot resolve split={split} under {src_root}, skip")
            continue
        cache_split_dir = src_root / SPLIT_TO_CACHE[split]
        _prepare_from_src_split(src_split_dir, cache_split_dir, size=args.size)


if __name__ == "__main__":
    main()
