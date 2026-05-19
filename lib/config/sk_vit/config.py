from easydict import EasyDict as edict
import yaml


cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAINED = "checkpoint.pth"
cfg.MODEL.TYPE = "vit_base_patch16_224"
cfg.MODEL.PRUNE_LAYERS = []
cfg.MODEL.PRUNE_KEEP_RATIO = 1.0
cfg.MODEL.INIT_CHECKPOINT = ""

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.PRUNE_START_EPOCH = 500
cfg.TRAIN.PRUNE_WARM_EPOCH = 100
cfg.TRAIN.PROMPT = edict()
cfg.TRAIN.PROMPT.TYPE = []
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.1
cfg.TRAIN.EPOCH = 100
cfg.TRAIN.WARM_UP_EPOCH = 0
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.NUM_WORKER = 1
cfg.TRAIN.OPTIMIZER = "SGD"
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.LR_DROP_EPOCH = 100
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.SMOOTHING = 0.1
cfg.TRAIN.PRINT_INTERVAL = 1
cfg.TRAIN.VAL_EPOCH_INTERVAL = 1
cfg.TRAIN.SAVE_EVERY_EPOCH = False
cfg.TRAIN.SAVE_EPOCHS = []
cfg.TRAIN.SAVE_EPOCH_RANGE_START = 0
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.LOAD_LATEST = True
cfg.TRAIN.PRETRAINED_LR_MULT = 1.0
cfg.TRAIN.ADAPTED_PRETRAINED_LR_MULT = 1.0
cfg.TRAIN.CLS_LOSS_USES = "giou"

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.INITIAL_LR = 1e-8
cfg.TRAIN.SCHEDULER.MIN_LR = 1e-6

# DATA
cfg.DATA = edict()
cfg.DATA.MODALITY = "RGB"
cfg.DATA.SIZE = 256
cfg.DATA.CROP_SIZE = 224
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.NUM_CLASSES = 40
cfg.DATA.SAMPLE_FRAMES = 32
cfg.DATA.RGB_SAMPLE_FRAMES = 32
cfg.DATA.SKELETON_SAMPLE_FRAMES = 32
cfg.DATA.SKELETON_ORIG_SIZE = [1280, 720]

cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.SAMPLER_MODE = "mean"
cfg.DATA.TRAIN.DATASET_NAME = "iMiGUE_train"
cfg.DATA.TRAIN.SAMPLES_PER_CLASS = []

cfg.DATA.VAL = edict()
cfg.DATA.VAL.SAMPLER_MODE = "mean"
cfg.DATA.VAL.DATASET_NAME = "iMiGUE_val"


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is None:
            _update_config(cfg, exp_config)
        else:
            _update_config(base_cfg, exp_config)
