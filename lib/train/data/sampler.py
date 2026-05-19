import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
import math


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, dataset, num_frames, processing=no_processing, frame_sample_mode='mean'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.dataset = dataset
        self.num_frames = num_frames
        # self.samples_per_epoch = samples_per_epoch
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.dataset.get_dataset_len()

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, seq_id):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """

        # sample a sequence from the given dataset
        # seq_id = self.sample_seq_from_dataset(self.dataset)
        seq_name_info, seq_len, seq_label = self.dataset.get_sequence_info(seq_id)


        if isinstance(self.num_frames, dict):
            if self.frame_sample_mode == 'RANDOMLY':
                rgb_frame_ids = self.get_frame_ids_randomly(seq_len, self.num_frames['rgb'])
                skeleton_frame_ids = self.get_frame_ids_randomly(seq_len, self.num_frames['skeleton'])
            elif self.frame_sample_mode == 'UNIFORMLY':
                rgb_frame_ids = self.get_frame_ids_uniformly(seq_len, self.num_frames['rgb'])
                skeleton_frame_ids = self.get_frame_ids_uniformly(seq_len, self.num_frames['skeleton'])
            else:
                raise ValueError("Illegal frame sample mode")
            frame_ids = {'rgb': rgb_frame_ids, 'skeleton': skeleton_frame_ids}
        else:
            if self.frame_sample_mode == 'RANDOMLY':
                frame_ids = self.get_frame_ids_randomly(seq_len)
            elif self.frame_sample_mode == 'UNIFORMLY':
                frame_ids = self.get_frame_ids_uniformly(seq_len)
            else:
                raise ValueError("Illegal frame sample mode")

        # rgb_frames, depth_frams = self.dataset.get_frames(seq_id, frame_ids)
        all_modality_frames = self.dataset.get_frames(seq_id, frame_ids)
        modality_list = ['RGB', 'Depth', 'IR', 'Skeleton']
        data_dict = {}

        for modality in modality_list:
            if modality not in all_modality_frames:
                continue

            modality_frames = all_modality_frames[modality]['frames']
            data_dict[modality.lower() + '_frames'] = modality_frames
        data_dict['label'] = seq_label
        data_dict['dataset'] = self.dataset.get_name()
        data_dict['seq_info'] = seq_name_info
        if isinstance(frame_ids, dict):
            data_dict['rgb_frame_ids'] = frame_ids['rgb']
            data_dict['skeleton_frame_ids'] = frame_ids['skeleton']
        else:
            data_dict['frame_ids'] = frame_ids

        data = TensorDict(data_dict)

        data = self.processing(data)
        # for modality in modality_list:
        #     if modality not in all_modality_frames:
        #         continue
        #
        #     data['original_' + modality.lower() + '_frames'] = torch.Tensor(
        #         np.array(all_modality_frames[modality]['frames']))

        return data

    def get_frame_ids_uniformly(self, seq_len, sample_duration=None):
        """Sample frames from the input video."""
        sample_duration = self.num_frames if sample_duration is None else sample_duration
        frame_indices = []
        for i in range(sample_duration):
            start = int(seq_len * i / sample_duration)
            end = max(int(seq_len * i / sample_duration) + 1, int(seq_len * (i + 1) / sample_duration))
            possible_indices = list(range(start, end))

            if not possible_indices:
                chosen_index = start
            else:
                chosen_index = np.mean(possible_indices, keepdims=True, dtype=int).tolist()

            frame_indices = frame_indices + chosen_index

        return frame_indices

    def get_frame_ids_randomly(self, seq_len, sample_duration=None):
        """
        Get frame indices from a sequence using a specified interval-based method.

        Parameters:
        - seq_len (int): Total length of the sequence
        - sn (int): Number of frames to retrieve

        Returns:
        - list: List of retrieved frame indices
        """
        sample_duration = self.num_frames if sample_duration is None else sample_duration
        frame_indices = []
        for i in range(sample_duration):
            start = int(seq_len * i / sample_duration)
            end = max(int(seq_len * i / sample_duration) + 1, int(seq_len * (i + 1) / sample_duration))
            possible_indices = list(range(start, end))

            if not possible_indices:
                chosen_index = start
            else:
                chosen_index = random.choices(possible_indices, k=1)

            frame_indices = frame_indices + chosen_index

        return frame_indices


class GroupBalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, class_to_group, batch_size, quotas=(8, 8, 8), seed=42, drop_last=True):
        self.labels = [int(x) for x in labels]
        self.class_to_group = {int(k): int(v) for k, v in class_to_group.items()}
        self.batch_size = int(batch_size)
        self.quotas = [int(x) for x in quotas]
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if sum(self.quotas) != self.batch_size:
            raise ValueError("sum(quotas) must equal batch_size")
        self.group_to_indices = {0: [], 1: [], 2: []}
        for idx, label in enumerate(self.labels):
            gid = self.class_to_group[label]
            self.group_to_indices[gid].append(idx)
        for gid in range(3):
            if len(self.group_to_indices[gid]) == 0:
                raise ValueError(f"group {gid} has 0 samples")
        self.num_batches = len(self.labels) // self.batch_size if self.drop_last else math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        pools = {}
        ptr = {}
        for gid in range(3):
            arr = np.array(self.group_to_indices[gid], dtype=np.int64)
            rng.shuffle(arr)
            pools[gid] = arr
            ptr[gid] = 0

        for _ in range(self.num_batches):
            batch = []
            for gid in range(3):
                need = self.quotas[gid]
                if ptr[gid] + need > len(pools[gid]):
                    arr = np.array(self.group_to_indices[gid], dtype=np.int64)
                    rng.shuffle(arr)
                    pools[gid] = np.concatenate([pools[gid][ptr[gid]:], arr], axis=0) if ptr[gid] < len(pools[gid]) else arr
                    ptr[gid] = 0
                chosen = pools[gid][ptr[gid]:ptr[gid] + need].tolist()
                ptr[gid] += need
                batch.extend(chosen)
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
