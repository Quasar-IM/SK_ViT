import os
import torch
import json
import copy
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.data import sampler, opencv_loader, processing, LTRLoader, video_transformers as vt
from lib.train.dataset.ijcai_miga_track1 import IJCAI_MiGA_Track1
from lib.utils.misc import is_main_process


class ConcatVideoDataset:
    """Minimal concat wrapper compatible with TrackingSampler."""
    def __init__(self, datasets, name="concat_dataset"):
        self.datasets = datasets
        self.name = name
        self.offsets = []
        self.labels = []
        total = 0
        for ds in datasets:
            self.offsets.append(total)
            n = ds.get_dataset_len()
            total += n
            self.labels.extend(list(ds.labels))
        self.total_len = total

    def get_dataset_len(self):
        return self.total_len

    def _locate(self, seq_id):
        for i in range(len(self.datasets) - 1, -1, -1):
            if seq_id >= self.offsets[i]:
                return i, seq_id - self.offsets[i]
        raise IndexError(f"seq_id out of range: {seq_id}")

    def get_sequence_info(self, seq_id):
        ds_idx, local_idx = self._locate(seq_id)
        return self.datasets[ds_idx].get_sequence_info(local_idx)

    def get_frames(self, seq_id, frame_ids):
        ds_idx, local_idx = self._locate(seq_id)
        return self.datasets[ds_idx].get_frames(local_idx, frame_ids)

    def get_name(self):
        return self.name


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name: str, modality, settings, image_loader=opencv_loader,
                   pseudo_labels_file=None):
    assert name in [
        'iMiGUE_train',
        'iMiGUE_val',
        'iMiGUE_test',
        'iMiGUE_official_val',
        'iMiGUE_train_full',
        'iMiGUE_train_full_plus_official_val',
    ]
    split_ids_file = os.path.join(settings.env.workspace_dir, 'data', 'splits', 'imigue_holdout_9_1_seed20260501.json')
    kw = dict(pseudo_labels_file=pseudo_labels_file)
    if name == 'iMiGUE_train':
        dataset = IJCAI_MiGA_Track1(settings.env.ijcai_miga_track1_dir, split='train', modality=modality,
                                    image_loader=image_loader,
                                    split_ids_file=split_ids_file, split_ids_key='train_ids', **kw)
    elif name == 'iMiGUE_train_full':
        dataset = IJCAI_MiGA_Track1(settings.env.ijcai_miga_track1_dir, split='train', modality=modality,
                                    image_loader=image_loader, **kw)
    elif name == 'iMiGUE_val':
        dataset = IJCAI_MiGA_Track1(settings.env.ijcai_miga_track1_dir, split='train', modality=modality,
                                    image_loader=image_loader,
                                    split_ids_file=split_ids_file, split_ids_key='val_ids', **kw)
    elif name == 'iMiGUE_official_val':
        dataset = IJCAI_MiGA_Track1(settings.env.ijcai_miga_track1_dir, split='val', modality=modality,
                                    image_loader=image_loader, **kw)
    elif name == 'iMiGUE_test':
        dataset = IJCAI_MiGA_Track1(settings.env.ijcai_miga_track1_dir, split='test', modality=modality,
                                    image_loader=image_loader, **kw)
    elif name == 'iMiGUE_train_full_plus_official_val':
        train_full = IJCAI_MiGA_Track1(settings.env.ijcai_miga_track1_dir, split='train', modality=modality,
                                       image_loader=image_loader, **kw)
        val_kw = copy.deepcopy(kw)
        val_kw['pseudo_labels_file'] = None
        official_val = IJCAI_MiGA_Track1(settings.env.ijcai_miga_track1_dir, split='val', modality=modality,
                                         image_loader=image_loader, **val_kw)
        dataset = ConcatVideoDataset(
            datasets=[train_full, official_val],
            name='iMiGUE_train_full_plus_official_val'
        )
    return dataset


def build_dataloaders(cfg, settings):
    # Data transform
    transform_train = vt.Compose([vt.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE), interpolation='bilinear'),
                                  vt.RandomCrop(size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
                                  vt.ClipToTensor(),
                                  vt.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])])

    transform_val = vt.Compose([vt.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE), interpolation='bilinear'),
                                vt.CenterCrop(size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
                                vt.ClipToTensor(),
                                vt.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

    train_num_frames = cfg.DATA.SAMPLE_FRAMES
    val_num_frames = cfg.DATA.SAMPLE_FRAMES
    if cfg.DATA.MODALITY == 'RGB_SK_MAGA':
        train_num_frames = {
            'rgb': cfg.DATA.RGB_SAMPLE_FRAMES,
            'skeleton': cfg.DATA.SKELETON_SAMPLE_FRAMES,
        }
        val_num_frames = {
            'rgb': cfg.DATA.RGB_SAMPLE_FRAMES,
            'skeleton': cfg.DATA.SKELETON_SAMPLE_FRAMES,
        }

    data_processing_train = processing.Processing(mode='sequence', transform=transform_train)

    data_processing_val = processing.Processing(mode='sequence', transform=transform_val)

    # Train sampler and loader
    dataset_train = sampler.TrackingSampler(
        dataset=names2datasets(cfg.DATA.TRAIN.DATASET_NAME, cfg.DATA.MODALITY, settings, opencv_loader,
                               pseudo_labels_file=str(getattr(cfg.DATA.TRAIN, "PSEUDO_LABELS_FILE", ""))),
        num_frames=train_num_frames,
        processing=data_processing_train,
        frame_sample_mode=cfg.DATA.TRAIN.SAMPLER_MODE)
    expected_train_size = int(getattr(cfg.DATA.TRAIN, "EXPECTED_SIZE", 0))
    if expected_train_size > 0:
        actual_train_size = len(dataset_train.dataset)
        print(f"[PseudoTrain] expected train size={expected_train_size}, actual={actual_train_size}")
        if actual_train_size != expected_train_size:
            raise ValueError(f"Pseudo subset size mismatch: expected {expected_train_size}, got {actual_train_size}")

    use_group_balanced = bool(getattr(cfg.TRAIN, "GROUP_BALANCED_SAMPLER", False))
    if use_group_balanced:
        map_path = str(getattr(cfg.DATA, "GROUP_MAP_PATH", ""))
        with open(map_path, "r", encoding="utf-8") as f:
            group_meta = json.load(f)
        class_to_group = {int(k): int(v) for k, v in group_meta["class_to_group"].items()}
        quotas = list(getattr(cfg.TRAIN, "GROUP_BATCH_QUOTA", [8, 8, 8]))
        batch_sampler = sampler.GroupBalancedBatchSampler(
            labels=dataset_train.dataset.labels,
            class_to_group=class_to_group,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            quotas=quotas,
            seed=int(getattr(cfg.TRAIN, "GROUP_SAMPLER_SEED", 42)),
            drop_last=True,
        )
        loader_train = LTRLoader(
            'train', dataset_train, training=True, batch_sampler=batch_sampler,
            num_workers=cfg.TRAIN.NUM_WORKER, stack_dim=1
        )
    else:
        train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
        shuffle = False if settings.local_rank != -1 else True

        loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                                 num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(
        dataset=names2datasets(cfg.DATA.VAL.DATASET_NAME, cfg.DATA.MODALITY, settings, opencv_loader),
        num_frames=val_num_frames,
        processing=data_processing_val,
        frame_sample_mode=cfg.DATA.VAL.SAMPLER_MODE)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg, extra_param_groups=None):
    train_type = getattr(cfg.TRAIN.PROMPT, "TYPE", [])
    pretrained_lr_mult = float(getattr(cfg.TRAIN, "PRETRAINED_LR_MULT", 1.0))
    adapted_pretrained_lr_mult = float(getattr(cfg.TRAIN, "ADAPTED_PRETRAINED_LR_MULT", pretrained_lr_mult))

    if train_type:
        # print("Only training score_branch. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if
                        any([param in n for param in train_type]) and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if all([param not in n for param in train_type]):
                p.requires_grad = False
    else:
        exact_pretrained_names = set(getattr(net, "pretrained_exact_param_names", set()))
        adapted_pretrained_names = set(getattr(net, "pretrained_adapted_param_names", set()))
        pretrained_names = exact_pretrained_names | adapted_pretrained_names
        if pretrained_names and (pretrained_lr_mult != 1.0 or adapted_pretrained_lr_mult != 1.0):
            pretrained_params = []
            adapted_pretrained_params = []
            fresh_params = []
            for n, p in net.named_parameters():
                if not p.requires_grad:
                    continue
                if n in exact_pretrained_names:
                    pretrained_params.append(p)
                elif n in adapted_pretrained_names:
                    adapted_pretrained_params.append(p)
                else:
                    fresh_params.append(p)
            param_dicts = []
            if pretrained_params:
                param_dicts.append({"params": pretrained_params, "lr": cfg.TRAIN.LR * pretrained_lr_mult})
            if adapted_pretrained_params:
                param_dicts.append({"params": adapted_pretrained_params, "lr": cfg.TRAIN.LR * adapted_pretrained_lr_mult})
            if fresh_params:
                param_dicts.append({"params": fresh_params, "lr": cfg.TRAIN.LR})
        else:
            param_dicts = [{"params": [p for p in net.parameters() if p.requires_grad]}]

    if extra_param_groups:
        extra_param_ids = set()
        for group in extra_param_groups:
            for param in group.get('params', []):
                extra_param_ids.add(id(param))

        filtered_param_dicts = []
        for group in param_dicts:
            params = [p for p in group.get('params', []) if id(p) not in extra_param_ids]
            if not params:
                continue
            new_group = dict(group)
            new_group['params'] = params
            filtered_param_dicts.append(new_group)
        param_dicts = filtered_param_dicts

        base_lr = float(cfg.TRAIN.LR)
        for group in extra_param_groups:
            new_group = dict(group)
            lr_mult = float(new_group.pop('lr_mult', 1.0))
            new_group['lr'] = base_lr * lr_mult
            param_dicts.append(new_group)
    total_num = sum(p.numel() for n, p in net.named_parameters())
    trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
    if is_main_process():
        print("Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if p.requires_grad:
                print(n, p.numel())
    print('Total: ', total_num, 'Trainable: ', trainable_num)

    train_lr = float(cfg.TRAIN.LR)
    weight_decay = float(cfg.TRAIN.WEIGHT_DECAY)
    momentum = float(getattr(cfg.TRAIN, "MOMENTUM", 0.9))
    warmup_epochs = int(cfg.TRAIN.WARM_UP_EPOCH)
    total_epochs = int(cfg.TRAIN.EPOCH)
    initial_lr = float(getattr(cfg.TRAIN.SCHEDULER, "INITIAL_LR", train_lr))
    min_lr = float(getattr(cfg.TRAIN.SCHEDULER, "MIN_LR", 0.0))

    if cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(param_dicts, lr=train_lr, momentum=momentum,
                                    weight_decay=weight_decay)
    elif cfg.TRAIN.OPTIMIZER == 'ADAM':
        optimizer = torch.optim.Adam(param_dicts, lr=train_lr)
    elif cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=train_lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported Optimizer")

    if cfg.TRAIN.SCHEDULER.TYPE == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            total_epochs - warmup_epochs,
            eta_min=min_lr,
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == 'warmup_and_cosine':
        scheduler = WarmupAndCosineAnnealingScheduler(
            optimizer,
            initial_lr,
            train_lr,
            min_lr,
            total_epochs,
            warmup_epochs,
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                         gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, scheduler


class WarmupAndCosineAnnealingScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, initial_lr, lr_after_warmup, min_lr, total_epochs, warmup_epochs, last_epoch=-1,
                 verbose=False):

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        base_lr_ratios = [(base_lr / base_lrs[0]) for base_lr in base_lrs]

        self.warmup_epochs = warmup_epochs
        if warmup_epochs > 0:
            self.initial_lr = initial_lr
            self.lr_after_warmup = lr_after_warmup
            self.warmup_epochs = warmup_epochs
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                                      lr_lambda=self.linear_lr_lambda, verbose=verbose)
            self.warmup_scheduler.base_lrs = [base_lr_ratios[i] * initial_lr for i, base_lr in
                                              enumerate(base_lr_ratios)]
            self.warmup_scheduler._step_count = 0
            self.warmup_scheduler.last_epoch = -1

        eta_min_base = min_lr
        eta_mins = [eta_min_base * base_lr_ratio for base_lr_ratio in base_lr_ratios]
        self.annealing_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
            for eta_min in eta_mins]
        super().__init__(optimizer, last_epoch, verbose=verbose)

    def linear_lr_lambda(self, epoch):
        return 1 + (self.lr_after_warmup - self.initial_lr) * epoch / (self.warmup_epochs * self.initial_lr)

    def step(self):
        self.last_epoch += 1
        if self.warmup_epochs > 0 and self.last_epoch <= self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            if self.last_epoch != 0:
                values = []
                for idx, annealing_scheduler in enumerate(self.annealing_schedulers):
                    # annealing_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
                    annealing_scheduler.step()
                    values.append(annealing_scheduler.get_last_lr()[idx])
                for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                    param_group, lr = data
                    param_group['lr'] = lr
                    self.print_lr(self.verbose, i, lr)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
