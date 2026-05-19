import csv
import json
import os
from functools import lru_cache

import cv2
import numpy as np
import torch

from lib.train.admin import env_settings
from lib.train.data import opencv_loader
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.dataset.sk_maga_schema import (
    RGB_SK_MAGA_MODALITY,
    SK_MAGA_MODALITY,
    append_hand_centers,
    project_to_canonical_22,
    reorder_canonical_24_to_model_24,
)


class IJCAI_MiGA_Track1(BaseVideoDataset):
    def __init__(self, root=None, split=None, modality='RGB', image_loader=opencv_loader,
                 split_ids_file=None, split_ids_key=None,
                 heatmap_size=(56, 56), sigma=1.0, bone_mode=False,
                 pseudo_labels_file=None):
        super().__init__(name='IJCAI_MiGA_Track1', root=root, image_loader=image_loader)
        env = env_settings()
        self.rgb_trainval_root = getattr(env, "imigue_rgb_trainval_root", "")
        self.sk_trainval_root = getattr(env, "imigue_sk_trainval_root", "")
        self.rgb_test_root = getattr(env, "imigue_rgb_test_root", "")
        self.sk_test_root = getattr(env, "imigue_sk_test_root", "")
        self.depth_root = getattr(env, "imigue_depth_dir", "")
        self.modality = modality
        if root is not None:
            # root argument is ignored in locked mode
            pass
        if split in ("train", "val"):
            if self.modality in [SK_MAGA_MODALITY, RGB_SK_MAGA_MODALITY] and not os.path.isdir(self.sk_trainval_root):
                raise FileNotFoundError(f"Locked sk trainval root not found: {self.sk_trainval_root}")
            if self.modality in ["RGB", "RGBD", RGB_SK_MAGA_MODALITY] and not os.path.isdir(self.rgb_trainval_root):
                raise FileNotFoundError(f"Locked rgb trainval root not found: {self.rgb_trainval_root}")

        if split not in ['train', 'val', 'test']:
            raise NotImplementedError

        self.split_path = None
        self.rgb_split_root = None
        if self.modality in [SK_MAGA_MODALITY, RGB_SK_MAGA_MODALITY]:
            self.split_path = self._resolve_skeleton_split_path(split)
        elif split == "test":
            # RGB test still needs skeleton label csv metadata.
            self.split_path = self._resolve_skeleton_split_path(split)
        if self.modality in ["RGB", "RGBD", RGB_SK_MAGA_MODALITY]:
            self.rgb_split_root = self._resolve_rgb_split_root(split)
        self.rgb_uses_mp4 = self._detect_rgb_mp4_layout() if self.rgb_split_root is not None else False
        if self.modality == RGB_SK_MAGA_MODALITY and self.rgb_uses_mp4:
            raise NotImplementedError(
                "RGB_SK_MAGA with mp4-only RGB layout is not supported by this loader. "
                "Use extracted RGB frames layout or RGB-only training."
            )
        self.split = split
        self.bone_mode = bool(bone_mode)
        self.heatmap_size = tuple(heatmap_size) if not isinstance(heatmap_size, tuple) else heatmap_size
        self.sigma = float(sigma)
        self.skeleton_cache_suffix = "_sk_maga.npy"
        self.skeleton_preprocess_dirname = "_sk_maga_preprocessed"
        self.split_ids = None
        if split_ids_file is not None and split_ids_key is not None:
            with open(split_ids_file, 'r') as f:
                split_meta = json.load(f)
            self.split_ids = set(split_meta[split_ids_key])
        self.pseudo_labels_map = None
        if pseudo_labels_file is not None and pseudo_labels_file != "":
            self.pseudo_labels_map = {}
            with open(pseudo_labels_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row["sample_id"]
                    self.pseudo_labels_map[sid] = int(row["pseudo_label"])
            print(f"[IJCAI_MiGA_Track1] loaded pseudo labels: {len(self.pseudo_labels_map)} from {pseudo_labels_file}")
        if self.modality == "RGB" and self.rgb_uses_mp4:
            self.sequence_list, self.seq_lens, self.labels = self._get_sequence_list_and_label_rgb_mp4()
        elif self.modality == "RGB" and not self.rgb_uses_mp4:
            self.sequence_list, self.seq_lens, self.labels = self._get_sequence_list_and_label(self.rgb_split_root)
        else:
            self.sequence_list, self.seq_lens, self.labels = self._get_sequence_list_and_label(self.split_path)
        if self.modality in (SK_MAGA_MODALITY, RGB_SK_MAGA_MODALITY):
            cache_split_dir = os.path.join(self.split_path, self.skeleton_preprocess_dirname)
            if not os.path.isdir(cache_split_dir):
                raise FileNotFoundError(
                    f"Missing required skeleton cache dir: {cache_split_dir}. "
                    "Please run prebuild_sk_cache.sh first."
                )

    @staticmethod
    def _pick_existing(candidates):
        for p in candidates:
            if p and os.path.isdir(p):
                return p
        return None

    @staticmethod
    def _resolve_sk_test_root(base_root):
        cands = [
            os.path.join(base_root, "imigue_skeleton_test"),
            os.path.join(base_root, "datasets", "imigue_skeleton_test"),
            base_root,
        ]
        resolved = IJCAI_MiGA_Track1._pick_existing(cands)
        return resolved

    def _resolve_skeleton_split_path(self, split):
        if split == "test":
            resolved = self._resolve_sk_test_root(self.sk_test_root)
            if resolved is None:
                raise FileNotFoundError(
                    f"Cannot resolve skeleton split dir for 'test'. base={self.sk_test_root}"
                )
            return resolved

        # strict fixed layout for train/val
        name = "imigue_skeleton_train" if split == "train" else "imigue_skeleton_validate"
        cands = [
            os.path.join(self.sk_trainval_root, "datasets", name),
            os.path.join(self.sk_trainval_root, "imigue_data_phase1", "datasets", name),
            os.path.join(self.sk_trainval_root, name),
        ]
        resolved = self._pick_existing(cands)
        if resolved is None:
            raise FileNotFoundError(f"Cannot resolve skeleton split dir for '{split}'. tried: {cands}")
        return resolved

    def _resolve_rgb_split_root(self, split):
        if split in ("train", "val"):
            base = self.rgb_trainval_root
            if split == "train":
                resolved = os.path.join(base, "train_cash")
            else:
                resolved = os.path.join(base, "valid_cash")
        else:
            base = self.rgb_test_root
            resolved = os.path.join(base, "test_cash")
        if not resolved or not os.path.isdir(resolved):
            raise FileNotFoundError(
                f"Missing required RGB cache dir for split='{split}': {resolved}. "
                "Please run prebuild_rgb_cache.sh first."
            )
        return resolved

    def _detect_rgb_mp4_layout(self):
        if not os.path.isdir(self.rgb_split_root):
            return False
        for video_id in os.listdir(self.rgb_split_root):
            if video_id in {"train_cash", "valid_cash", "test_cash"}:
                continue
            video_dir = os.path.join(self.rgb_split_root, video_id)
            if not os.path.isdir(video_dir):
                continue
            # Cache clip-dir layout: <video_id>/<video_id>_<ms_id>/<frames>.jpg
            for sub in os.listdir(video_dir):
                sub_dir = os.path.join(video_dir, sub)
                if not os.path.isdir(sub_dir):
                    continue
                if sub.startswith(f"{video_id}_"):
                    for fn in os.listdir(sub_dir):
                        if fn.lower().endswith(".jpg") or fn.lower().endswith(".jpeg"):
                            return True
            # Support both "{video_id}_0.mp4" and "{video_id}.mp4" layouts.
            probe1 = os.path.join(video_dir, f"{video_id}_0.mp4")
            probe2 = os.path.join(video_dir, f"{video_id}.mp4")
            if os.path.exists(probe1) or os.path.exists(probe2):
                return True
            for name in os.listdir(video_dir):
                if name.lower().endswith(".mp4"):
                    return True
            return False
        return False

    def get_dataset_len(self):
        return len(self.sequence_list)

    def _get_sequence_list_and_label(self, ground_true_path):
        base_path = ground_true_path
        all_sequence_list = []
        seq_lens = []
        labels = []
        labels_dic = {}
        for i in range(32):
            labels_dic[i] = 0
        for video_id in os.listdir(base_path):
            labels_path = os.path.join(base_path, video_id, f'{video_id}_label.csv')
            if not os.path.exists(labels_path):
                if self.split == "test":
                    raise FileNotFoundError(
                        f"Missing required test label file: {labels_path}. "
                        "Strict mode: test split requires explicit labels metadata."
                    )
                print(f'Skipping video {video_id}')
                continue
            f = open(labels_path, "r")
            label_List = f.read().splitlines()
            for i, label_info in enumerate(label_List):
                class_id, start_frame, end_frame = label_info.split(',')
                sample_id = f"{video_id}_{i}"
                if self.split_ids is not None and sample_id not in self.split_ids:
                    continue
                if self.pseudo_labels_map is not None and sample_id not in self.pseudo_labels_map:
                    continue

                class_id = int(class_id) % 99  # illustrative class id is 0
                if self.pseudo_labels_map is not None:
                    class_id = int(self.pseudo_labels_map[sample_id])
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                sequence_list = {'video_id': video_id,
                                 'ms_id': i,
                                 'sample_id': sample_id,
                                 'start_frame': start_frame,
                                 'end_frame': end_frame}

                # Verify whether the data is missing
                for frame_id in range(start_frame, end_frame + 1):
                    if self.modality == SK_MAGA_MODALITY:
                        skeleton_path = os.path.join(base_path, video_id, f'{video_id}_light_hand.csv')
                        assert os.path.exists(skeleton_path), 'Path does not exist: {}'.format(skeleton_path)
                        break
                    elif self.modality == RGB_SK_MAGA_MODALITY:
                        rgb_frame_path = os.path.join(self.rgb_split_root, video_id, '{:06}'.format(frame_id) + '.jpg')
                        skeleton_path = os.path.join(base_path, video_id, f'{video_id}_light_hand.csv')
                        assert os.path.exists(rgb_frame_path), 'Path does not exist: {}'.format(rgb_frame_path)
                        assert os.path.exists(skeleton_path), 'Path does not exist: {}'.format(skeleton_path)
                    elif self.modality != 'RGBD':
                        frame_path = os.path.join(self.rgb_split_root, video_id, '{:06}'.format(frame_id) + '.jpg')
                        assert os.path.exists(frame_path), 'Path does not exist: {}'.format(frame_path)
                    else:
                        rgb_frame_path = os.path.join(self.rgb_split_root, video_id, '{:06}'.format(frame_id) + '.jpg')
                        depth_frame_path = os.path.join(self.depth_root, self.split, video_id, '{:06}'.format(frame_id) + '.jpg')
                        assert os.path.exists(rgb_frame_path) and os.path.exists(depth_frame_path), 'Path does not exist: {}'.format(rgb_frame_path)


                all_sequence_list.append(sequence_list)
                seq_lens.append(end_frame - start_frame + 1)
                labels.append(class_id)
                labels_dic[class_id] = labels_dic[class_id] + 1
            print('video {} has {} gesture'.format(video_id, i + 1))
        print(labels_dic)
        print('{} Length {}'.format(self.split, len(all_sequence_list)))
        return all_sequence_list, seq_lens, labels

    def _get_sequence_list_and_label_rgb_mp4(self):
        all_sequence_list = []
        seq_lens = []
        labels = []
        labels_dic = {i: 0 for i in range(32)}

        for video_id in os.listdir(self.rgb_split_root):
            if video_id in {"train_cash", "valid_cash", "test_cash"} or video_id.endswith("_cash"):
                continue
            video_dir = os.path.join(self.rgb_split_root, video_id)
            if not os.path.isdir(video_dir):
                continue
            if self.split == "test":
                # Use resolved skeleton test split path, not raw env root.
                labels_path = os.path.join(self.split_path, video_id, f"{video_id}_label.csv")
            else:
                labels_path = os.path.join(video_dir, f"{video_id}_label.csv")
            if not os.path.exists(labels_path):
                if self.split == "test":
                    # Skip non-video or cache dirs under test root.
                    continue
                print(f"Skipping video {video_id} (missing label csv)")
                continue
            with open(labels_path, "r") as f:
                label_list = f.read().splitlines()
            for i, label_info in enumerate(label_list):
                parts = label_info.split(",")
                if len(parts) < 1:
                    continue
                class_id = int(parts[0]) % 99
                sample_id = f"{video_id}_{i}"
                if self.split_ids is not None and sample_id not in self.split_ids:
                    continue
                if self.pseudo_labels_map is not None and sample_id not in self.pseudo_labels_map:
                    continue
                if self.pseudo_labels_map is not None:
                    class_id = int(self.pseudo_labels_map[sample_id])
                if self.split == "test":
                    clip_candidates = [
                        os.path.join(video_dir, f"{video_id}_{i}"),
                        os.path.join(video_dir, f"{video_id}_{i}.mp4"),
                        os.path.join(video_dir, f"{video_id}.mp4"),
                    ]
                else:
                    clip_candidates = [
                        os.path.join(video_dir, f"{video_id}_{i}"),
                        os.path.join(video_dir, f"{video_id}_{i}.mp4"),
                    ]
                clip_path = None
                for p in clip_candidates:
                    if os.path.exists(p):
                        clip_path = p
                        break
                if clip_path is None and self.split == "test":
                    # Fallback for cached test layout when label ms_id and cached clip ids are not perfectly aligned.
                    clip_dirs = []
                    if os.path.isdir(video_dir):
                        for n in sorted(os.listdir(video_dir)):
                            p = os.path.join(video_dir, n)
                            if not os.path.isdir(p):
                                continue
                            if not n.startswith(f"{video_id}_"):
                                continue
                            has_jpg = any(fn.lower().endswith(".jpg") or fn.lower().endswith(".jpeg")
                                          for fn in os.listdir(p))
                            if has_jpg:
                                clip_dirs.append(p)
                    if len(clip_dirs) > 0:
                        pick_idx = min(i, len(clip_dirs) - 1)
                        clip_path = clip_dirs[pick_idx]
                if clip_path is None:
                    if self.split == "test":
                        raise FileNotFoundError(
                            f"Missing required test mp4 for video {video_id}. tried: {clip_candidates}"
                        )
                    continue
                if os.path.isdir(clip_path):
                    frame_count = len([n for n in os.listdir(clip_path) if n.lower().endswith(".jpg") or n.lower().endswith(".jpeg")])
                    frame_count = max(1, frame_count)
                else:
                    frame_count = self._get_mp4_frame_count(clip_path)
                sequence_list = {
                    "video_id": video_id,
                    "ms_id": i,
                    "sample_id": sample_id,
                    "start_frame": 0,
                    "end_frame": max(0, frame_count - 1),
                    "clip_path": clip_path,
                }
                all_sequence_list.append(sequence_list)
                seq_lens.append(max(1, frame_count))
                labels.append(class_id)
                labels_dic[class_id] = labels_dic[class_id] + 1
            print(f"video {video_id} has {len(label_list)} gesture")
        print(labels_dic)
        print(f"{self.split} Length {len(all_sequence_list)}")
        return all_sequence_list, seq_lens, labels


    def _get_sequence_path(self, seq_id):
        """
        Get the path for the specified modality sequence.

        Parameters:
        - sequence_id (str): The ID of the sequence
        - modality (str): The type of modality (e.g., "RGB", "Depth")

        Returns:
        - str: The full path of the modality sequence
        """
        all_seq_name = self.sequence_list[seq_id]
        if self.modality == "RGB" and self.rgb_uses_mp4:
            return all_seq_name["clip_path"]
        if self.modality == SK_MAGA_MODALITY:
            return os.path.join(self.split_path, all_seq_name['video_id'])
        if self.modality == RGB_SK_MAGA_MODALITY:
            return (
                os.path.join(self.rgb_split_root, all_seq_name['video_id']),
                os.path.join(self.split_path, all_seq_name['video_id']),
            )
        if self.modality == 'RGBD':
            return os.path.join(self.rgb_split_root, all_seq_name['video_id']), os.path.join(self.depth_root, self.split, all_seq_name['video_id'])
        return os.path.join(self.rgb_split_root, all_seq_name['video_id'])

    def get_sequence_info(self, seq_id):
        """
        Get the length and label of the sequence.

        Parameters:
        - seq_id (str): The ID of the sequence

        Returns:
        - tuple: (length of the sequence, label of the sequence)
        """

        return self.sequence_list[seq_id], self.seq_lens[seq_id], self.labels[seq_id]

    def _get_frame_path(self, seq_path, frame_id, suffix='.jpg'):
        return os.path.join(seq_path, '{:06}'.format(frame_id) + suffix)  # frames start from 1

    def _get_frame(self, seq_path, frame_id, suffix='.jpg'):
        return self.image_loader(self._get_frame_path(seq_path, frame_id, suffix))

    def get_name(self):
        return 'IJCAI_MiGA_Track1'

    def get_frames(self, seq_id, frame_ids):
        """
        Get frames for the specified modality.

        Parameters:
            - seq_id (str): The ID of the sequence
            - frame_ids (list of int): List of frame IDs to retrieve
        Returns:
            - list: List of frame paths
        """

        seq_info, _, _ = self.get_sequence_info(seq_id)
        start_frame = seq_info['start_frame']
        info = {}

        if self.modality == RGB_SK_MAGA_MODALITY:
            rgb_frame_ids = frame_ids['rgb']
            skeleton_frame_ids = frame_ids['skeleton']
            rgb_seq_path, skeleton_seq_path = self._get_sequence_path(seq_id)
            rgb_frame_list = [self._get_frame(rgb_seq_path, f_id + start_frame) for f_id in rgb_frame_ids]
            skeleton_frame_list = self._load_sk_maga_skeleton(
                skeleton_seq_path, [f_id + start_frame for f_id in skeleton_frame_ids]
            )
            info['RGB'] = {'frames': rgb_frame_list, 'mask': None, 'seq_info': seq_info}
            info['Skeleton'] = {'frames': skeleton_frame_list, 'mask': None, 'seq_info': seq_info}
        elif self.modality == SK_MAGA_MODALITY:
            skeleton_seq_path = self._get_sequence_path(seq_id)
            skeleton_frame_list = self._load_sk_maga_skeleton(
                skeleton_seq_path, [f_id + start_frame for f_id in frame_ids]
            )
            info['Skeleton'] = {'frames': skeleton_frame_list, 'mask': None, 'seq_info': seq_info}
        elif self.modality == 'RGBD':
            rgb_seq_path, depth_seq_path = self._get_sequence_path(seq_id)
            rgb_frame_list = [self._get_frame(rgb_seq_path, f_id + start_frame) for f_id in frame_ids]
            depth_frame_list = [self._get_frame(depth_seq_path, f_id + start_frame) for f_id in frame_ids]
            info['RGB'] = {'frames': rgb_frame_list, 'mask': None, 'seq_info': seq_info}
            info['Depth'] = {'frames': depth_frame_list, 'mask': None, 'seq_info': seq_info}
        else:
            modal_seq_path = self._get_sequence_path(seq_id)
            if self.modality == "RGB" and self.rgb_uses_mp4:
                if os.path.isdir(modal_seq_path):
                    modal_frame_list = [self._get_frame(modal_seq_path, f_id) for f_id in frame_ids]
                else:
                    modal_frame_list = self._get_mp4_frames(modal_seq_path, frame_ids)
            else:
                modal_frame_list = [self._get_frame(modal_seq_path, f_id + start_frame) for f_id in frame_ids]
            if self.modality == 'RGB':
                info['RGB'] = {'frames': modal_frame_list, 'mask': None, 'seq_info': seq_info}
            elif self.modality == 'Depth':
                info['RGB'] = {'frames': modal_frame_list, 'mask': None, 'seq_info': seq_info}
        return info

    @staticmethod
    @lru_cache(maxsize=64)
    def _load_skeleton_csv(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        return np.asarray(rows[1:], dtype=np.float32)

    @staticmethod
    @lru_cache(maxsize=64)
    def _load_skeleton_npy(npy_path):
        return np.load(npy_path, mmap_mode='r')

    @staticmethod
    @lru_cache(maxsize=2048)
    def _get_mp4_frame_count(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            return 1
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(1, n)

    def _get_mp4_frames(self, mp4_path, frame_ids):
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open mp4: {mp4_path}")
        total = self._get_mp4_frame_count(mp4_path)
        if len(frame_ids) == 0:
            cap.release()
            return []

        # Avoid per-frame random seeks; decode sequentially after a single seek.
        clamped = [int(max(0, min(total - 1, f_id))) for f_id in frame_ids]
        order = np.argsort(np.asarray(clamped))
        sorted_ids = [clamped[i] for i in order]

        frames_sorted = [None] * len(sorted_ids)
        cur = sorted_ids[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur)

        for j, target in enumerate(sorted_ids):
            while cur < target:
                if not cap.grab():
                    break
                cur += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_sorted[j] = frame
            cur += 1

        frames = [None] * len(clamped)
        for pos, src_idx in enumerate(order):
            frames[src_idx] = frames_sorted[pos]
        cap.release()
        return frames

    def _get_local_cache_path(self, video_id):
        return os.path.join(self.split_path, self.skeleton_preprocess_dirname, f"{video_id}{self.skeleton_cache_suffix}")

    @staticmethod
    def _convert_csv_row_to_model24(frame_values):
        raw_points = torch.from_numpy(frame_values[1:].reshape(35, 3).astype(np.float32))
        canonical_22 = project_to_canonical_22(raw_points)
        canonical_24 = append_hand_centers(canonical_22)
        return reorder_canonical_24_to_model_24(canonical_24)

    def _load_sk_maga_skeleton(self, seq_path, frame_ids):
        video_id = os.path.basename(seq_path)
        npy_path = self._get_local_cache_path(video_id)
        if not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"Missing required skeleton cache file: {npy_path}. "
                "Please run prebuild_sk_cache.sh first."
            )
        skeleton_data = self._load_skeleton_npy(npy_path)
        frames = []
        for frame_id in frame_ids:
            if frame_id >= skeleton_data.shape[0]:
                frames.append(torch.zeros((24, 3), dtype=torch.float32))
                continue
            frames.append(torch.from_numpy(np.array(skeleton_data[frame_id], dtype=np.float32, copy=True)))
        return frames
