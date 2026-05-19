import os
import copy
import importlib
from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.sk_maga.inbomem import InBoMemNeck
from lib.models.sk_maga.skateformer import SkateFormer


class JBConcatHead(nn.Module):
    def __init__(self, joint_dim, bone_dim, hidden_dim=256, dropout=0.2, num_classes=32, branch_drop_prob=0.0):
        super().__init__()
        self.branch_drop_prob = float(branch_drop_prob)
        self.head = nn.Sequential(
            nn.LayerNorm(joint_dim + bone_dim),
            nn.Dropout(dropout),
            nn.Linear(joint_dim + bone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, joint_feat, bone_feat):
        if self.training and self.branch_drop_prob > 0:
            batch_size = joint_feat.shape[0]
            drop_mask = torch.rand(batch_size, device=joint_feat.device) < self.branch_drop_prob
            drop_joint_mask = drop_mask & (torch.rand(batch_size, device=joint_feat.device) < 0.5)
            drop_bone_mask = drop_mask & (~drop_joint_mask)
            if drop_joint_mask.any():
                joint_feat = joint_feat.clone()
                joint_feat[drop_joint_mask] = 0.0
            if drop_bone_mask.any():
                bone_feat = bone_feat.clone()
                bone_feat[drop_bone_mask] = 0.0
        fused_feat = torch.cat([joint_feat, bone_feat], dim=1)
        return fused_feat, self.head(fused_feat)


class SKMagaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = SkateFormer(
            in_channels=cfg.MODEL.IN_CHANS,
            num_classes=cfg.DATA.NUM_CLASSES,
            embed_dim=cfg.MODEL.EMBED_DIM,
            depths=tuple(cfg.MODEL.DEPTHS),
            channels=tuple(cfg.MODEL.CHANNELS),
            num_people=cfg.MODEL.NUM_PEOPLE,
            num_frames=cfg.DATA.SAMPLE_FRAMES,
            num_points=cfg.MODEL.NUM_POINTS,
            kernel_size=cfg.MODEL.KERNEL_SIZE,
            num_heads=cfg.MODEL.NUM_HEADS,
            type_1_size=tuple(cfg.MODEL.TYPE_1_SIZE),
            type_2_size=tuple(cfg.MODEL.TYPE_2_SIZE),
            type_3_size=tuple(cfg.MODEL.TYPE_3_SIZE),
            type_4_size=tuple(cfg.MODEL.TYPE_4_SIZE),
            attn_drop=cfg.MODEL.ATTN_DROP,
            head_drop=cfg.MODEL.HEAD_DROP,
            drop=cfg.MODEL.DROP,
            rel=cfg.MODEL.REL,
            drop_path=cfg.MODEL.DROP_PATH,
            mlp_ratio=cfg.MODEL.MLP_RATIO,
            index_t=cfg.MODEL.INDEX_T,
            global_pool=cfg.MODEL.GLOBAL_POOL,
            use_hypergraph=False,
        )
        if not isinstance(self.backbone.head, nn.Linear):
            raise TypeError("Expected SkateFormer backbone.head to be nn.Linear before wrapping classifier")
        self.feat_dim = self.backbone.head.in_features
        self.classifier = nn.Linear(self.feat_dim, cfg.DATA.NUM_CLASSES)
        self.classifier.load_state_dict(self.backbone.head.state_dict())
        self.backbone.head = nn.Identity()
        self.use_memory_neck = bool(getattr(cfg.MODEL, "USE_MEMORY_NECK", False))
        self.memory_neck = None
        if self.use_memory_neck:
            memory_feature_dim = int(getattr(cfg.MODEL, "MEMORY_FEATURE_DIM", self.feat_dim))
            if memory_feature_dim != self.feat_dim:
                raise ValueError(
                    f"MEMORY_FEATURE_DIM ({memory_feature_dim}) must match backbone feat dim ({self.feat_dim})"
                )
            self.memory_neck = InBoMemNeck(
                feature_dim=memory_feature_dim,
                num_classes=int(getattr(cfg.MODEL, "MEMORY_NUM_CLASSES", cfg.DATA.NUM_CLASSES)),
                slots_per_class=int(getattr(cfg.MODEL, "MEMORY_SLOTS_PER_CLASS", 5)),
                tau=float(getattr(cfg.MODEL, "MEMORY_TAU", 10.0)),
                read_pos_weight=float(getattr(cfg.MODEL, "MEMORY_READ_POS_WEIGHT", 0.0)),
                read_neg_weight=float(getattr(cfg.MODEL, "MEMORY_READ_NEG_WEIGHT", 1.0)),
                forget_schedule=str(getattr(cfg.TRAIN, "MEMORY_FORGET_SCHEDULE", "exp")),
                forget_decay=float(getattr(cfg.TRAIN, "MEMORY_FORGET_DECAY", 0.1198)),
                forget_min=float(getattr(cfg.TRAIN, "MEMORY_FORGET_MIN", 0.05)),
                forget_stage1_end=int(getattr(cfg.TRAIN, "MEMORY_FORGET_STAGE1_END", 10)),
                forget_stage1_value=float(getattr(cfg.TRAIN, "MEMORY_FORGET_STAGE1_VALUE", 0.8)),
                forget_stage2_end=int(getattr(cfg.TRAIN, "MEMORY_FORGET_STAGE2_END", 20)),
                forget_stage2_value=float(getattr(cfg.TRAIN, "MEMORY_FORGET_STAGE2_VALUE", 0.5)),
                forget_decay_end=int(getattr(cfg.TRAIN, "MEMORY_FORGET_DECAY_END", 30)),
                forget_final_value=float(getattr(cfg.TRAIN, "MEMORY_FORGET_FINAL_VALUE", 0.01)),
            )
            self.memory_neck.classifier_v.load_state_dict(self.classifier.state_dict())
            self.memory_neck.classifier_z.load_state_dict(self.classifier.state_dict())

    def _encode_backbone(self, x, index_t):
        if index_t is None:
            raise ValueError("index_t is required for SK_MAGA forward")
        backbone = self.backbone
        batch_size, channels, _, _, _ = x.shape
        output = x.permute(0, 1, 2, 4, 3).contiguous().view(batch_size, channels, x.shape[2], -1)
        for layer in backbone.stem:
            output = layer(output)
        if backbone.index_t:
            te = torch.zeros(batch_size, x.shape[2], backbone.embed_dim, device=output.device)
            div_term = torch.exp(
                torch.arange(0, backbone.embed_dim, 2, dtype=torch.float, device=output.device)
                * -(torch.log(torch.tensor(10000.0, device=output.device)) / backbone.embed_dim)
            )
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            output = output + torch.einsum('b t c, c v -> b c t v', te, backbone.joint_person_embedding)
        else:
            output = output + backbone.joint_person_temporal_embedding
        stage_feats = []
        for stage_idx, stage in enumerate(backbone.stages):
            output = stage(output)
            stage_feats.append(output)
        return output, stage_feats

    def forward(self, x, index_t=None, return_stage_feats=False, y=None, epoch=0, **kwargs):
        output, stage_feats = self._encode_backbone(x, index_t)
        backbone = self.backbone
        feat = backbone.forward_head(output, pre_logits=True)
        logits = self.classifier(feat)
        out = {"feat": feat, "logist": logits}
        if self.memory_neck is not None:
            mem_out = self.memory_neck(feat, y=y, epoch=epoch)
            out.update(mem_out)
            out["logist"] = mem_out["logits_v"]
        if return_stage_feats:
            out["stage_feats"] = stage_feats
        return out


class SKJointBoneFeatureFusionModel(nn.Module):
    def __init__(self, cfg, preload_stream_ckpt=True):
        super().__init__()
        self.cfg = cfg
        self.preload_stream_ckpt = bool(preload_stream_ckpt)
        self.freeze_backbones = bool(getattr(cfg.TRAIN, "FUSION_FREEZE_BACKBONES", False))
        self.freeze_epochs = int(getattr(cfg.TRAIN, "FUSION_FREEZE_EPOCHS", 0))
        self.joint_model = self._build_stream_model(
            config_name=str(getattr(cfg.MODEL, "FUSION_JOINT_CONFIG", "imigue_sk_maga_ep30")),
            checkpoint_path=str(getattr(cfg.MODEL, "FUSION_JOINT_CHECKPOINT", "")),
        )
        self.bone_model = self._build_stream_model(
            config_name=str(getattr(cfg.MODEL, "FUSION_BONE_CONFIG", "imigue_sk_maga_bone_ep30")),
            checkpoint_path=str(getattr(cfg.MODEL, "FUSION_BONE_CHECKPOINT", "")),
        )
        self.joint_feat_dim = int(self.joint_model.feat_dim)
        self.bone_feat_dim = int(self.bone_model.feat_dim)
        self.num_classes = int(cfg.DATA.NUM_CLASSES)
        fusion_dropout = float(getattr(cfg.MODEL, "FUSION_DROPOUT", 0.2))
        proj_dim = int(getattr(cfg.MODEL, "JB_FUSION_PROJ_DIM", 256))
        self.fusion_head = JBConcatHead(
            self.joint_feat_dim,
            self.bone_feat_dim,
            hidden_dim=proj_dim,
            dropout=fusion_dropout,
            num_classes=self.num_classes,
            branch_drop_prob=float(getattr(cfg.MODEL, "JB_BRANCH_DROP_PROB", 0.0)),
        )

        self.backbones_trainable = True
        self.set_backbone_trainable(not self.freeze_backbones)

    def _build_stream_cfg(self, config_name):
        prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        cfg_file = os.path.join(prj_dir, f"experiments/sk_maga/{config_name}.yaml")
        if not os.path.exists(cfg_file):
            raise ValueError(f"{cfg_file} doesn't exist.")
        config_module = importlib.import_module("lib.config.sk_maga.config")
        stream_cfg = copy.deepcopy(config_module.cfg)
        config_module.update_config_from_file(cfg_file, base_cfg=stream_cfg)
        stream_cfg.MODEL.JOINT_BONE_FEATURE_FUSION = False
        stream_cfg.MODEL.USE_MEMORY_NECK = False
        return stream_cfg

    def _build_stream_model(self, config_name, checkpoint_path):
        stream_cfg = self._build_stream_cfg(config_name)
        model = SKMagaModel(stream_cfg)
        if self.preload_stream_ckpt:
            if not checkpoint_path:
                raise ValueError(f"Checkpoint path must be set for fusion stream {config_name}")
            ckpt_path = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
                checkpoint_path,
            )
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            checkpoint_model = checkpoint
            for model_key in "model|module|net|state_dict".split("|"):
                if isinstance(checkpoint, dict) and model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    break
            checkpoint_model = _strip_prefix_if_present(checkpoint_model, prefixes=("module.",))
            missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
            print(f"Loaded fusion stream checkpoint from {ckpt_path}")
            if missing:
                print(f"Missing stream keys: {missing}")
            if unexpected:
                print(f"Unexpected stream keys: {unexpected}")
        else:
            print(f"Skip stream preload for {config_name}; inference will rely on fusion checkpoint only.")
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbones_trainable:
            self.joint_model.eval()
            self.bone_model.eval()
        return self

    def set_backbone_trainable(self, trainable: bool):
        self.backbones_trainable = bool(trainable)
        modules = [self.joint_model, self.bone_model]
        for module in modules:
            module.train(trainable and self.training)
            for param in module.parameters():
                param.requires_grad_(trainable)

    def forward(self, joint_x, joint_index_t, bone_x, bone_index_t, **kwargs):
        epoch = int(kwargs.get("epoch", 0))
        should_train_backbones = (not self.freeze_backbones) or (epoch >= self.freeze_epochs)
        if should_train_backbones != self.backbones_trainable:
            self.set_backbone_trainable(should_train_backbones)
        ctx = nullcontext() if self.backbones_trainable else torch.no_grad()
        with ctx:
            joint_out = self.joint_model(x=joint_x, index_t=joint_index_t, return_stage_feats=False)
            bone_out = self.bone_model(x=bone_x, index_t=bone_index_t, return_stage_feats=False)
            joint_feat = joint_out["feat"]
            bone_feat = bone_out["feat"]
            joint_logits = joint_out["logist"]
            bone_logits = bone_out["logist"]
        fused_feat, logits = self.fusion_head(joint_feat, bone_feat)
        return {
            "feat_joint": joint_feat,
            "feat_bone": bone_feat,
            "logits_joint": joint_logits,
            "logits_bone": bone_logits,
            "feat": fused_feat,
            "logist": logits,
        }


def _strip_prefix_if_present(state_dict, prefixes):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                break
        cleaned[new_key] = value
    return cleaned


def _adapt_input_conv_weights(checkpoint_model, state_dict):
    for key, model_tensor in state_dict.items():
        ckpt_tensor = checkpoint_model.get(key)
        if ckpt_tensor is None:
            continue
        if ckpt_tensor.shape == model_tensor.shape:
            continue
        if ckpt_tensor.ndim != 4 or model_tensor.ndim != 4:
            continue
        if ckpt_tensor.shape[0] != model_tensor.shape[0]:
            continue
        if ckpt_tensor.shape[2:] != model_tensor.shape[2:]:
            continue
        if ckpt_tensor.shape[1] >= model_tensor.shape[1]:
            checkpoint_model[key] = ckpt_tensor[:, :model_tensor.shape[1], :, :]
            continue

        adapted = model_tensor.clone()
        adapted.zero_()
        adapted[:, :ckpt_tensor.shape[1], :, :] = ckpt_tensor
        checkpoint_model[key] = adapted
    return checkpoint_model


def _adapt_special_checkpoint_tensor(key, ckpt_tensor, model_tensor):
    if tuple(ckpt_tensor.shape) == tuple(model_tensor.shape):
        return ckpt_tensor

    if key.endswith("joint_person_embedding") and ckpt_tensor.ndim == 2 and model_tensor.ndim == 2:
        src_c, src_v = ckpt_tensor.shape
        dst_c, dst_v = model_tensor.shape
        if src_c == dst_c and src_v % dst_v == 0:
            factor = src_v // dst_v
            return ckpt_tensor.view(src_c, dst_v, factor).mean(dim=-1)

    if ".gconv" in key and ckpt_tensor.ndim == 3 and model_tensor.ndim == 3:
        src_h, src_v1, src_v2 = ckpt_tensor.shape
        dst_h, dst_v1, dst_v2 = model_tensor.shape
        if src_h == dst_h and src_v1 >= dst_v1 and src_v2 >= dst_v2:
            # Heuristic: keep the first-person/top-left subgraph when compressing
            # a larger pretrained graph (e.g. 48x48) into the current 24x24 graph.
            return ckpt_tensor[:, :dst_v1, :dst_v2]

    if key.endswith("relative_position_bias_table"):
        if ckpt_tensor.ndim == 2 and model_tensor.ndim == 2 and ckpt_tensor.shape[1] == model_tensor.shape[1]:
            src = ckpt_tensor.t().unsqueeze(0)
            resized = F.interpolate(src, size=model_tensor.shape[0], mode="linear", align_corners=False)
            return resized.squeeze(0).t()

        if ckpt_tensor.ndim == 4 and model_tensor.ndim == 4 and ckpt_tensor.shape[-1] == model_tensor.shape[-1]:
            src = ckpt_tensor.permute(3, 0, 1, 2).unsqueeze(0)
            resized = F.interpolate(
                src,
                size=model_tensor.shape[:3],
                mode="trilinear",
                align_corners=False,
            )
            return resized.squeeze(0).permute(1, 2, 3, 0)

    return ckpt_tensor


def _filter_pretrained_checkpoint(checkpoint_model, state_dict):
    skip_exact = {
        "classifier.weight",
        "classifier.bias",
        "head.weight",
        "head.bias",
    }
    skip_contains = (
        ".relative_position_index",
    )

    filtered = OrderedDict()
    exact_loaded_keys = set()
    adapted_loaded_keys = set()
    skipped = []
    for key, value in checkpoint_model.items():
        if key in skip_exact or any(token in key for token in skip_contains):
            skipped.append((key, "rule_skip"))
            continue

        model_key = key
        if model_key not in state_dict and ("backbone." + key) in state_dict:
            model_key = "backbone." + key

        if model_key not in state_dict:
            skipped.append((key, "missing_in_model"))
            continue

        model_tensor = state_dict[model_key]
        original_shape = tuple(getattr(value, "shape", ()))
        if hasattr(value, "shape"):
            value = _adapt_special_checkpoint_tensor(key, value, model_tensor)

        if not hasattr(value, "shape") or tuple(value.shape) != tuple(model_tensor.shape):
            skipped.append((key, f"shape_mismatch:{tuple(getattr(value, 'shape', []))}->{tuple(model_tensor.shape)}"))
            continue

        filtered[model_key] = value
        if tuple(value.shape) == original_shape and original_shape == tuple(model_tensor.shape):
            exact_loaded_keys.add(model_key)
        else:
            adapted_loaded_keys.add(model_key)
    return filtered, skipped, exact_loaded_keys, adapted_loaded_keys


def _filter_init_checkpoint(checkpoint_model, state_dict, skip_classifiers=False):
    skip_prefixes = ["module."]
    cleaned = OrderedDict()
    for key, value in checkpoint_model.items():
        new_key = key
        for prefix in skip_prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                break
        cleaned[new_key] = value

    filtered = OrderedDict()
    skipped = []
    for key, value in cleaned.items():
        if skip_classifiers and (
            key.startswith("classifier.") or
            key.startswith("memory_neck.classifier_v.") or
            key.startswith("memory_neck.classifier_z.")
        ):
            skipped.append((key, "rule_skip"))
            continue
        if key not in state_dict:
            skipped.append((key, "missing_in_model"))
            continue
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(state_dict[key].shape):
            skipped.append((key, f"shape_mismatch:{tuple(getattr(value, 'shape', []))}->{tuple(state_dict[key].shape)}"))
            continue
        filtered[key] = value
    return filtered, skipped


def build_sk_maga(cfg, training=True):
    model = (
        SKJointBoneFeatureFusionModel(cfg, preload_stream_ckpt=training)
        if bool(getattr(cfg.MODEL, "JOINT_BONE_FEATURE_FUSION", False))
        else SKMagaModel(cfg)
    )
    model.pretrained_loaded_param_names = set()
    model.pretrained_exact_param_names = set()
    model.pretrained_adapted_param_names = set()
    pretrained_name = getattr(cfg.MODEL, "PRETRAINED", "")
    if pretrained_name and training:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pretrained_root = os.path.join(current_dir, "../../../pretrained_models")
        ckpt_path = os.path.join(pretrained_root, pretrained_name)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        checkpoint_model = checkpoint
        for model_key in "model|module|net|state_dict".split("|"):
            if isinstance(checkpoint, dict) and model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                break
        checkpoint_model = _strip_prefix_if_present(
            checkpoint_model, prefixes=("module.", "backbone.")
        )
        state_dict = model.state_dict()
        checkpoint_model = _adapt_input_conv_weights(checkpoint_model, state_dict)
        if "backbone.head.weight" in checkpoint_model and "classifier.weight" not in checkpoint_model:
            checkpoint_model["classifier.weight"] = checkpoint_model.pop("backbone.head.weight")
        if "backbone.head.bias" in checkpoint_model and "classifier.bias" not in checkpoint_model:
            checkpoint_model["classifier.bias"] = checkpoint_model.pop("backbone.head.bias")
        if "head.weight" in checkpoint_model and "classifier.weight" not in checkpoint_model:
            checkpoint_model["classifier.weight"] = checkpoint_model.pop("head.weight")
        if "head.bias" in checkpoint_model and "classifier.bias" not in checkpoint_model:
            checkpoint_model["classifier.bias"] = checkpoint_model.pop("head.bias")
        filtered_checkpoint, skipped, exact_loaded_keys, adapted_loaded_keys = _filter_pretrained_checkpoint(
            checkpoint_model, state_dict
        )
        missing, unexpected = model.load_state_dict(filtered_checkpoint, strict=False)
        model.pretrained_loaded_param_names = set(filtered_checkpoint.keys())
        model.pretrained_exact_param_names = exact_loaded_keys
        model.pretrained_adapted_param_names = adapted_loaded_keys
        print(f"Loaded pretrained weights from {ckpt_path}")
        print(f"Loaded compatible keys: {len(filtered_checkpoint)}")
        print(f"Loaded exact-shape keys: {len(exact_loaded_keys)}")
        print(f"Loaded adapted-shape keys: {len(adapted_loaded_keys)}")
        print(f"Skipped pretrained keys: {len(skipped)}")
        if skipped:
            print(f"First skipped keys: {[item[0] for item in skipped[:20]]}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    init_checkpoint = getattr(cfg.MODEL, "INIT_CHECKPOINT", "")
    if init_checkpoint and training:
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        checkpoint_model = checkpoint
        for model_key in "model|module|net|state_dict".split("|"):
            if isinstance(checkpoint, dict) and model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                break
        checkpoint_model = _strip_prefix_if_present(
            checkpoint_model, prefixes=("module.",)
        )
        checkpoint_model = _adapt_input_conv_weights(checkpoint_model, model.state_dict())
        if getattr(cfg.MODEL, "INIT_SKIP_CLASSIFIERS", False):
            for key in list(checkpoint_model.keys()):
                if key.startswith("classifier") or key.startswith("memory_neck.classifier_"):
                    del checkpoint_model[key]
        missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Loaded init checkpoint from {init_checkpoint}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    return model
