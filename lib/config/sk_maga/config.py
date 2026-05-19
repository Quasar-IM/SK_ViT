from easydict import EasyDict as edict
import yaml

from lib.train.dataset.sk_maga_schema import NUM_MODEL_JOINTS, SK_MAGA_MODALITY


cfg = edict()

cfg.MODEL = edict()
cfg.MODEL.TYPE = "skateformer_micro"
cfg.MODEL.PRETRAINED = ""
cfg.MODEL.IN_CHANS = 3
cfg.MODEL.NUM_POINTS = NUM_MODEL_JOINTS
cfg.MODEL.NUM_PEOPLE = 1
cfg.MODEL.DEPTHS = [2, 2, 2, 2]
cfg.MODEL.CHANNELS = [96, 192, 192, 192]
cfg.MODEL.EMBED_DIM = 96
cfg.MODEL.KERNEL_SIZE = 7
cfg.MODEL.NUM_HEADS = 32
cfg.MODEL.ATTN_DROP = 0.0
cfg.MODEL.HEAD_DROP = 0.0
cfg.MODEL.DROP = 0.0
cfg.MODEL.REL = True
cfg.MODEL.DROP_PATH = 0.2
cfg.MODEL.MLP_RATIO = 4.0
cfg.MODEL.TYPE_1_SIZE = [4, 6]
cfg.MODEL.TYPE_2_SIZE = [4, 4]
cfg.MODEL.TYPE_3_SIZE = [4, 6]
cfg.MODEL.TYPE_4_SIZE = [4, 4]
cfg.MODEL.GLOBAL_POOL = "avg"
cfg.MODEL.INDEX_T = True

# IE / IB
cfg.MODEL.USE_MEMORY_NECK = False
cfg.MODEL.MEMORY_FEATURE_DIM = 192
cfg.MODEL.MEMORY_NUM_CLASSES = 32
cfg.MODEL.MEMORY_SLOTS_PER_CLASS = 5
cfg.MODEL.MEMORY_TAU = 10.0
cfg.MODEL.MEMORY_READ_POS_WEIGHT = 0.0
cfg.MODEL.MEMORY_READ_NEG_WEIGHT = 1.0

# Joint+Bone fusion finetune
cfg.MODEL.JOINT_BONE_FEATURE_FUSION = False
cfg.MODEL.FUSION_JOINT_CONFIG = "imigue_sk_maga_ep30"
cfg.MODEL.FUSION_BONE_CONFIG = "imigue_sk_maga_bone_ep30"
cfg.MODEL.FUSION_JOINT_CHECKPOINT = ""
cfg.MODEL.FUSION_BONE_CHECKPOINT = ""
cfg.MODEL.FUSION_DROPOUT = 0.2
cfg.MODEL.JB_FUSION_PROJ_DIM = 256
cfg.MODEL.JB_BRANCH_DROP_PROB = 0.0

cfg.MODEL.INIT_CHECKPOINT = ""
cfg.MODEL.INIT_SKIP_CLASSIFIERS = False

cfg.TRAIN = edict()
cfg.TRAIN.PROMPT = edict()
cfg.TRAIN.PROMPT.TYPE = []
cfg.TRAIN.LOAD_LATEST = True
cfg.TRAIN.LR = 0.0005
cfg.TRAIN.WEIGHT_DECAY = 0.05
cfg.TRAIN.EPOCH = 30
cfg.TRAIN.WARM_UP_EPOCH = 5
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.BACKBONE_MULTIPLIER = 1.0
cfg.TRAIN.SMOOTHING = 0.1
cfg.TRAIN.PRINT_INTERVAL = 100
cfg.TRAIN.VAL_EPOCH_INTERVAL = 1
cfg.TRAIN.SAVE_EVERY_EPOCH = False
cfg.TRAIN.SAVE_EPOCHS = []
cfg.TRAIN.SAVE_EPOCH_RANGE_START = 0
cfg.TRAIN.GRAD_CLIP_NORM = 1.0
cfg.TRAIN.AMP = False
cfg.TRAIN.CLS_LOSS_USES = "cls"
cfg.TRAIN.PRETRAINED_LR_MULT = 0.1
cfg.TRAIN.ADAPTED_PRETRAINED_LR_MULT = 0.2

# IE/IB training
cfg.TRAIN.MEMORY_IB_BETA = 1.0
cfg.TRAIN.MEMORY_KL_WEIGHT = 1.0
cfg.TRAIN.MEMORY_DISTILL_TEMP = 1.0
cfg.TRAIN.MEMORY_CLUSTER_ONLY = False
cfg.TRAIN.MEMORY_DISABLE_Z_LOSS = False
cfg.TRAIN.MEMORY_PROTO_LOSS_WEIGHT = 0.2
cfg.TRAIN.MEMORY_RES_LOSS_WEIGHT = 0.1
cfg.TRAIN.MEMORY_DIV_LOSS_WEIGHT = 0.01
cfg.TRAIN.MEMORY_WARM_START_EPOCH = 10
cfg.TRAIN.MEMORY_WARM_END_EPOCH = 20
cfg.TRAIN.MEMORY_AUX_WARM_START_EPOCH = 10
cfg.TRAIN.MEMORY_AUX_WARM_END_EPOCH = 20
cfg.TRAIN.MEMORY_FORGET_SCHEDULE = "exp"
cfg.TRAIN.MEMORY_FORGET_DECAY = 0.1198
cfg.TRAIN.MEMORY_FORGET_MIN = 0.05
cfg.TRAIN.MEMORY_FORGET_STAGE1_END = 10
cfg.TRAIN.MEMORY_FORGET_STAGE1_VALUE = 0.8
cfg.TRAIN.MEMORY_FORGET_STAGE2_END = 20
cfg.TRAIN.MEMORY_FORGET_STAGE2_VALUE = 0.5
cfg.TRAIN.MEMORY_FORGET_DECAY_END = 30
cfg.TRAIN.MEMORY_FORGET_FINAL_VALUE = 0.01

# Fusion finetune strategy
cfg.TRAIN.FUSION_FREEZE_BACKBONES = True
cfg.TRAIN.FUSION_FREEZE_EPOCHS = 3
cfg.TRAIN.FUSION_BACKBONE_LR_MULT = 0.1
cfg.TRAIN.JB_FUSION_WEIGHT = 1.0
cfg.TRAIN.JB_JOINT_WEIGHT = 1.0
cfg.TRAIN.JB_BONE_WEIGHT = 1.0
cfg.TRAIN.JB_KL_JOINT_FROM_BONE_WEIGHT = 1.0
cfg.TRAIN.JB_KL_BONE_FROM_JOINT_WEIGHT = 1.0
cfg.TRAIN.JB_KD_TEMPERATURE = 2.0

cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "warmup_and_cosine"
cfg.TRAIN.SCHEDULER.INITIAL_LR = 1e-6
cfg.TRAIN.SCHEDULER.MIN_LR = 1e-5

cfg.DATA = edict()
cfg.DATA.MODALITY = SK_MAGA_MODALITY
cfg.DATA.SIZE = 256
cfg.DATA.CROP_SIZE = 256
cfg.DATA.MEAN = [0.0, 0.0, 0.0]
cfg.DATA.STD = [1.0, 1.0, 1.0]
cfg.DATA.NUM_CLASSES = 32
cfg.DATA.SKELETON_INPUT_TYPE = "joint"
cfg.DATA.SAMPLE_FRAMES = 32
cfg.DATA.RGB_SAMPLE_FRAMES = 16
cfg.DATA.SKELETON_SAMPLE_FRAMES = 32
cfg.DATA.SKELETON_ORIG_SIZE = [1280, 720]
cfg.DATA.SKELETON_CACHE_ROOT = "./data/sk_maga_cache"
cfg.DATA.SKELETON_CACHE_SUFFIX = "_sk_maga.npy"
cfg.DATA.JOINT_MASK_ENABLE = False
cfg.DATA.JOINT_MASK_MIN_RATIO = 0.1
cfg.DATA.JOINT_MASK_MAX_RATIO = 0.2
cfg.DATA.JOINT_MASK_ROOT_IDX = 0
cfg.DATA.JOINT_MASK_FILL = 0.0
cfg.DATA.JOINT_ROT_ENABLE = False
cfg.DATA.JOINT_ROT_MAX_DEG = 5.0
cfg.DATA.JOINT_JITTER_ENABLE = False
cfg.DATA.JOINT_JITTER_STD_MIN = 0.001
cfg.DATA.JOINT_JITTER_STD_MAX = 0.005
cfg.DATA.JOINT_JITTER_DIMS = 2

cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.SAMPLER_MODE = "RANDOMLY"
cfg.DATA.TRAIN.DATASET_NAME = "iMiGUE_train"
cfg.DATA.TRAIN.SAMPLES_PER_CLASS = []
cfg.DATA.TRAIN.PSEUDO_LABELS_FILE = ""
cfg.DATA.TRAIN.EXPECTED_SIZE = 0

cfg.DATA.VAL = edict()
cfg.DATA.VAL.SAMPLER_MODE = "UNIFORMLY"
cfg.DATA.VAL.DATASET_NAME = "iMiGUE_val"


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k not in base_cfg:
                raise ValueError(f"{k} not exist in config.py")
            if not isinstance(v, dict):
                base_cfg[k] = v
            else:
                _update_config(base_cfg[k], v)


def update_config_from_file(filename, base_cfg=None):
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
