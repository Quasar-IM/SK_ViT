import os
from timm.loss import LabelSmoothingCrossEntropy
from lib.train.trainers import LTRTrainer
from torch.nn.parallel import DistributedDataParallel as DDP

from .actors.sk_vit import SK_ViTActor
from .actors.sk_maga import SK_MAGAActor
from .actors.sk_maga_joint_bone_fusion import SK_MAGAJointBoneFusionActor
from .base_functions import *
import importlib
from lib.models.sk_vit.build_sk_vit import build_sk_vit
from lib.models.sk_maga.build_sk_maga import build_sk_maga



def run(settings):
    settings.description = 'Training script for SK_ViT and SK_MAGA classification models'

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
    settings.save_every_epoch = bool(getattr(cfg.TRAIN, "SAVE_EVERY_EPOCH", False))
    settings.save_epochs = list(getattr(cfg.TRAIN, "SAVE_EPOCHS", []))
    settings.save_epoch_range_start = int(getattr(cfg.TRAIN, "SAVE_EPOCH_RANGE_START", 0))

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name in ["sk_vit", "fusion"]:
        net = build_sk_vit(cfg)
    elif settings.script_name in ["sk_maga", "jb_fusion"]:
        net = build_sk_maga(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name in ["sk_vit", "fusion"]:
        cls_loss = LabelSmoothingCrossEntropy(cfg.TRAIN.SMOOTHING)
        objective = {'cls': cls_loss}
        loss_weight = {'cls': 1.0}
        actor = SK_ViTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name in ["sk_maga", "jb_fusion"]:
        cls_loss = LabelSmoothingCrossEntropy(cfg.TRAIN.SMOOTHING)
        objective = {'cls': cls_loss}
        loss_weight = {'cls': 1.0}
        if bool(getattr(cfg.MODEL, "JOINT_BONE_FEATURE_FUSION", False)):
            actor = SK_MAGAJointBoneFusionActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        else:
            actor = SK_MAGAActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    extra_param_groups = actor.get_optimizer_param_groups() if hasattr(actor, 'get_optimizer_param_groups') else None
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg, extra_param_groups=extra_param_groups)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    fail_safe = settings.local_rank == -1
    trainer.train(cfg.TRAIN.EPOCH, load_latest=getattr(cfg.TRAIN, "LOAD_LATEST", True), fail_safe=fail_safe)
