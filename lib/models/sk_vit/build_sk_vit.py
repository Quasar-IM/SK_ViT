import os

import torch
from timm import create_model

from lib.models.sk_vit.sk_vit import load_state_dict


def _build_single_sk_vit(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "../../../pretrained_models")

    model = create_model(
        model_name=cfg.MODEL.TYPE,
        pretrained=False,
        num_classes=cfg.DATA.NUM_CLASSES,
        all_frames=cfg.DATA.SAMPLE_FRAMES,
        tubelet_size=2,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_checkpoint=False,
        use_mean_pooling=True,
        init_scale=0.001,
        prune_layers=getattr(cfg.MODEL, "PRUNE_LAYERS", []),
        prune_keep_ratio=getattr(cfg.MODEL, "PRUNE_KEEP_RATIO", 1.0),
    )

    if cfg.MODEL.PRETRAINED and training:
        ckpt_path = os.path.join(pretrained_path, cfg.MODEL.PRETRAINED)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        print("Load ckpt from %s" % ckpt_path)
        checkpoint_model = None
        for model_key in "model|module|net".split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        if "pos_embed" in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model["pos_embed"]
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            num_frames = model.patch_embed.num_frames
            orig_size = int(
                ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (num_frames // model.patch_embed.tubelet_size))
                ** 0.5
            )
            new_size = int((num_patches // (num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(
                    -1,
                    num_frames // model.patch_embed.tubelet_size,
                    orig_size,
                    orig_size,
                    embedding_size,
                )
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1,
                    num_frames // model.patch_embed.tubelet_size,
                    new_size,
                    new_size,
                    embedding_size,
                )
                pos_tokens = pos_tokens.flatten(1, 3)
                checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        load_state_dict(model, checkpoint_model, prefix="")

    init_checkpoint = getattr(cfg.MODEL, "INIT_CHECKPOINT", "")
    if init_checkpoint and training:
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        checkpoint_model = checkpoint
        for model_key in "model|module|net|state_dict".split("|"):
            if isinstance(checkpoint, dict) and model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                break
        if isinstance(checkpoint_model, dict):
            cleaned = {}
            for key, value in checkpoint_model.items():
                cleaned[key[7:] if key.startswith("module.") else key] = value
            checkpoint_model = cleaned
        missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
        model.pretrained_loaded_param_names = set(checkpoint_model.keys())
        model.pretrained_exact_param_names = set(checkpoint_model.keys())
        model.pretrained_adapted_param_names = set()
        print(f"Loaded init checkpoint from {init_checkpoint}")
        if missing:
            print(f"Missing init keys: {missing}")
        if unexpected:
            print(f"Unexpected init keys: {unexpected}")
    return model


def build_sk_vit(cfg, training=True):
    return _build_single_sk_vit(cfg, training=training)
