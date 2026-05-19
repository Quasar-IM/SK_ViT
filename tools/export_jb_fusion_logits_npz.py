import argparse
import glob
import importlib
import os

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import _init_paths
from lib.models.sk_maga import build_sk_maga
from lib.train.base_functions import names2datasets, update_settings
from lib.train.data import LTRLoader, opencv_loader, processing, sampler
from lib.train.dataset.sk_maga_inputs import prepare_bone_inputs, prepare_joint_inputs
from lib.train.admin.output_paths import resolve_save_dir
from lib.train.run_training import init_seeds
import lib.train.admin.settings as ws_settings


def find_checkpoint(prj_dir, save_dir, script_name, config_name, checkpoint_epoch):
    checkpoint_dir = os.path.join(prj_dir, save_dir, f'checkpoints/train/{script_name}/{config_name}')
    checkpoint_pattern = os.path.join(checkpoint_dir, f'*_ep{checkpoint_epoch:04d}.pth.tar')
    checkpoint_list = sorted(glob.glob(checkpoint_pattern))
    if not checkpoint_list:
        raise FileNotFoundError(f'No checkpoint matched {checkpoint_pattern}')
    return checkpoint_list[-1]


def main(args):
    init_seeds(args.seed)
    resolved_save_dir = resolve_save_dir(args.save_dir, args.script_name)

    settings = ws_settings.Settings()
    settings.script_name = args.script_name
    settings.config_name = args.config_name
    settings.save_dir = resolved_save_dir
    settings.local_rank = args.local_rank

    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    settings.cfg_file = os.path.join(prj_dir, f'experiments/{args.script_name}/{args.config_name}.yaml')
    checkpoint_path = find_checkpoint(prj_dir, resolved_save_dir, args.script_name, args.config_name, args.checkpoint_epoch)

    if not os.path.exists(settings.cfg_file):
        raise ValueError(f"{settings.cfg_file} doesn't exist.")

    config_module = importlib.import_module(f"lib.config.{settings.script_name}.config")
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    update_settings(settings, cfg)

    identity_transform = lambda clip: clip
    data_processing = processing.Processing(mode='sequence', transform=identity_transform)
    dataset_eval = sampler.TrackingSampler(
        dataset=names2datasets(args.dataset_name, cfg.DATA.MODALITY, settings, opencv_loader),
        num_frames=cfg.DATA.SAMPLE_FRAMES,
        processing=data_processing,
        frame_sample_mode=cfg.DATA.VAL.SAMPLER_MODE,
    )
    eval_sampler = DistributedSampler(dataset_eval) if settings.local_rank != -1 else None
    loader_eval = LTRLoader(
        'eval', dataset_eval, training=False, batch_size=1,
        num_workers=cfg.TRAIN.NUM_WORKER, drop_last=False, stack_dim=1, sampler=eval_sampler,
        epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL,
    )

    net = build_sk_maga(cfg, training=False)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('net', ckpt)
    net.load_state_dict(state_dict, strict=True)
    net.cuda()
    net.eval()

    ids = []
    logits_all = []
    labels_all = []
    has_label = True
    smoke_max_steps = int(os.environ.get("SMOKE_MAX_STEPS", "0"))

    for step_i, data in tqdm(enumerate(loader_eval, 1), total=len(loader_eval)):
        if smoke_max_steps > 0 and step_i > smoke_max_steps:
            break
        data = data.to('cuda:0')
        coords = data["skeleton_frames"][0].view(-1, *data["skeleton_frames"].shape[2:]).float()
        joint_x, joint_index_t = prepare_joint_inputs(coords, cfg)
        bone_x, bone_index_t = prepare_bone_inputs(coords, cfg)
        out_dict = net(
            joint_x=joint_x,
            joint_index_t=joint_index_t,
            bone_x=bone_x,
            bone_index_t=bone_index_t,
        )
        logits = out_dict['logist']
        if isinstance(logits, list):
            logits = logits[0]
        ids.append(f"{data['seq_info']['video_id'][0]}_{data['seq_info']['ms_id'][0]}")
        logits_all.append(logits.detach().cpu().numpy()[0].astype(np.float32))
        if 'label' in data and (data['label'] >= 0).all():
            labels_all.append(int(data['label'].detach().cpu().numpy()[0]))
        else:
            has_label = False

    payload = {
        "ids": np.asarray(ids),
        "logits": np.stack(logits_all, axis=0).astype(np.float32),
    }
    if has_label and len(labels_all) == len(ids):
        payload["labels"] = np.asarray(labels_all, dtype=np.int64)

    output_path = os.path.abspath(args.output_npz)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **payload)
    print(f"saved: {output_path}")
    print(f"ids: {payload['ids'].shape}, logits: {payload['logits'].shape}")
    if "labels" in payload:
        print(f"labels: {payload['labels'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export JB-fusion logits npz on val/test.')
    parser.add_argument('--script_name', type=str, default='jb_fusion')
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='iMiGUE_val')
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--checkpoint_epoch', type=int, required=True)
    parser.add_argument('--output_npz', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())

