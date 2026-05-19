import csv
import importlib
import os
import argparse
import time
import zipfile
import re

import numpy as np
import torch
import visdom
from scipy.ndimage import binary_erosion
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from tqdm import tqdm

import _init_paths

# datasets related
from lib.models.sk_vit import build_sk_vit
from lib.train.admin import AverageMeter
from lib.train.admin.stats import topk_accuracy
from lib.train.base_functions import names2datasets, update_settings
from lib.train.data import sampler, opencv_loader, processing, LTRLoader, video_transformers as vt, functional
from lib.train.run_training import init_seeds
from lib.train.admin.output_paths import resolve_save_dir, get_output_area
import lib.train.admin.settings as ws_settings


def main(script_name, config_name, save_dir, local_rank, seed, checkpoint_epoch=None, save_logits_npz=False, no_zip=False):
    '''Set seed for different process'''
    init_seeds(seed)

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.save_dir = resolve_save_dir(save_dir, script_name)
    settings.local_rank = local_rank
    if checkpoint_epoch is None:
        total_epoch = re.search(r'ep(\d+)$', config_name)
        if total_epoch:
            total_epoch = int(total_epoch.group(1))
        else:
            raise RuntimeError("Epoch Error: config_name must end with 'epXXX', where XXX are digits.")
    else:
        total_epoch = checkpoint_epoch
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))
    checkpoint_path = os.path.join(prj_dir, settings.save_dir, 'checkpoints/train/{}'.format(script_name), config_name,
                                   'VisionTransformer_ep{:04}.pth.tar'.format(total_epoch))

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    transform_test = vt.Compose([vt.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE), interpolation='bilinear'),
                                 vt.CenterCrop(size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
                                 vt.ClipToTensor(),
                                 vt.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
    data_processing_test = processing.Processing(mode='sequence', transform=transform_test)
    test_num_frames = cfg.DATA.SAMPLE_FRAMES
    if cfg.DATA.MODALITY == 'RGB_SK_MAGA':
        test_num_frames = {
            'rgb': cfg.DATA.RGB_SAMPLE_FRAMES,
            'skeleton': cfg.DATA.SKELETON_SAMPLE_FRAMES,
        }
    # Validation samplers and loaders
    dataset_test = sampler.TrackingSampler(
        dataset=names2datasets('iMiGUE_test', cfg.DATA.MODALITY, settings, opencv_loader),
        num_frames=test_num_frames,
        processing=data_processing_test,
        frame_sample_mode=cfg.DATA.VAL.SAMPLER_MODE)
    test_sampler = DistributedSampler(dataset_test) if settings.local_rank != -1 else None
    loader_test = LTRLoader('test', dataset_test, training=False, batch_size=1,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=False, stack_dim=1, sampler=test_sampler,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    if settings.script_name == "sk_vit":
        net = build_sk_vit(cfg, training=False)
    # torch.load(checkpoint_path, map_location='cpu')['net']
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.cuda()
    net.eval()

    output = []
    output_ids = []
    output_logits = []
    smoke_max_steps = int(os.environ.get("SMOKE_MAX_STEPS", "0"))

    top1 = AverageMeter()
    class_acc = {}
    for i in range(cfg.DATA.NUM_CLASSES):
        class_acc[i] = {'CP': 0, 'TP': 0}
    for i, data in tqdm(enumerate(loader_test, 1)):
        if smoke_max_steps > 0 and i > smoke_max_steps:
            break
        data = data.to('cuda:0')
        prune_keep_ratio = cfg.MODEL.PRUNE_KEEP_RATIO

        all_images = {}
        for key in data.keys():
            if 'frames' not in key:
                continue
            all_images[key] = data[key][0].view(-1, *data[key].shape[2:])
        # out_dict, masks = grad_cam(all_images['rgb_frames'])  # Compute the Grad-CAM mask
        if cfg.DATA.MODALITY in ['RGB', 'RGB_SK_MAGA']:
            out_dict = net(x=all_images['rgb_frames'],
                           train_keep_ratio=prune_keep_ratio)

        if not isinstance(out_dict['logist'], list):
            out_dict['logist'] = [out_dict['logist']]
        prec1, _ = topk_accuracy(out_dict['logist'][0], data['label'], (1, 5))
        top1.update(prec1.item(), 1)

        if i % 100 == 0 or i == len(loader_test) - 1:
            print(f'[{i + 1}/{len(loader_test)}] Top1 accuracy: {top1.avg:}')

        pred_seq = '{}_{}'.format(data['seq_info']['video_id'][0], data['seq_info']['ms_id'][0])
        pred_label = torch.argmax(out_dict['logist'][0], dim=1).item()
        output_ids.append(pred_seq)
        output_logits.append(out_dict['logist'][0].detach().cpu().numpy()[0].astype(np.float32))
        class_acc[data['label'].item()]['CP'] += (data['label'].item() == pred_label)
        class_acc[data['label'].item()]['TP'] += 1

        if pred_label == 0:
            pred_label = 31
        else:
            pred_label = pred_label - 1
        output.append([pred_seq, pred_label])

    if len(output_logits) == 0:
        raise RuntimeError(
            "No test samples were loaded (output_logits is empty). "
            "Please check test dataset paths in lib/train/admin/local.py "
            "(imigue_rgb_test_root / imigue_sk_test_root) and the expected directory layout."
        )

    print(f'{config_name} Top1 accuracy: {top1.avg:}')
    for i in range(cfg.DATA.NUM_CLASSES):
        if class_acc[i]['TP'] != 0:
            print(f'Class {i:02d} Top1 accuracy: {class_acc[i]["CP"] / class_acc[i]["TP"]}')
        else:
            print(f'Class {i:02d} None')
    # save ./script_name/config_name/
    output_dir = os.path.join('./submission', get_output_area(script_name), script_name, config_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_basename = f'Submission_ep{total_epoch:04d}'
    output_filename = os.path.join(output_dir, f'{output_basename}.csv')
    output_file = open(output_filename, 'w')

    output_file.write('Id,Target\n')

    for inx in range(len(output)):
        output_file.write(output[inx][0] + "," + str(output[inx][1]) + "\n")
    output_file.close()

    if save_logits_npz:
        logits_npz_path = os.path.join(output_dir, f'{output_basename}_logits.npz')
        np.savez_compressed(
            logits_npz_path,
            ids=np.asarray(output_ids),
            logits=np.stack(output_logits, axis=0),
        )
        print(f'Saved logits npz: {logits_npz_path}')

    if not no_zip:
        zip_file_name = os.path.join(output_dir, f'{output_basename}.zip')
        with zipfile.ZipFile(zip_file_name, "w") as zip_file:
            zip_file.write(output_filename, os.path.basename(output_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Prediction using MF_ViT model')
    parser.add_argument('--script_name', type=str, default='sk_vit', help='Script name')
    parser.add_argument('--config_name', type=str, default='imigue_rgb_k400_ep30',
                        help='Config name')
    #  prune70_isogd_rgb_k400_ep30 isogd_rgb_k400_ep30
    parser.add_argument('--save_dir', type=str, default='./output', help='Directory to save outputs')
    parser.add_argument('--checkpoint_epoch', type=int, default=None, help='Checkpoint epoch to evaluate')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_logits_npz', action='store_true', help='Save test logits as compressed npz.')
    parser.add_argument('--no_zip', action='store_true', help='Only output csv (do not create zip).')

    args = parser.parse_args()

    main(
        args.script_name,
        args.config_name,
        args.save_dir,
        args.local_rank,
        args.seed,
        args.checkpoint_epoch,
        save_logits_npz=args.save_logits_npz,
        no_zip=args.no_zip,
    )
