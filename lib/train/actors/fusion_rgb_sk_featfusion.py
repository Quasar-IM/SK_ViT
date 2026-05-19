import torch
import torch.nn.functional as F

from . import BaseActor
from ..admin.stats import topk_accuracy
from lib.utils.misc import reduce_tensor
from lib.train.dataset.sk_maga_inputs import prepare_joint_inputs
from lib.train.dataset.sk_maga_schema import NUM_MODEL_JOINTS


class FusionRGBSKFeatFusionActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg

    def _net(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def get_optimizer_param_groups(self):
        net = self._net()
        rgb_lr_mult = float(getattr(self.cfg.TRAIN, "FUSION_RGB_LR_MULT", 0.1))
        sk_lr_mult = float(getattr(self.cfg.TRAIN, "FUSION_SK_LR_MULT", 0.1))
        return [
            {"params": list(net.rgb_model.parameters()), "lr_mult": rgb_lr_mult},
            {"params": list(net.sk_model.parameters()), "lr_mult": sk_lr_mult},
        ]

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        all_images = {}
        for key in data.keys():
            if 'frames' not in key:
                continue
            all_images[key] = data[key][0].view(-1, *data[key].shape[2:])
        coords = all_images["skeleton_frames"].float()
        if coords.shape[1] != NUM_MODEL_JOINTS and coords.shape[2] != NUM_MODEL_JOINTS:
            raise ValueError(f"Expected {NUM_MODEL_JOINTS} joints in dim 1 or 2, got {tuple(coords.shape)}")
        skeleton_x, skeleton_index_t = prepare_joint_inputs(coords, self.cfg)
        return self.net(
            x=all_images["rgb_frames"],
            skeleton_x=skeleton_x,
            skeleton_index_t=skeleton_index_t,
            y=data["label"] if self.net.training else None,
            epoch=int(data.get("epoch", 0)),
        )

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_label = gt_dict["label"]
        fused_logits = pred_dict["logist"]
        rgb_logits = pred_dict["logits_rgb"]
        sk_logits = pred_dict["logits_sk"]
        fusion_strategy = str(getattr(self.cfg.MODEL, "FUSION_STRATEGY", "concat")).lower()
        fused_weight = float(getattr(self.cfg.TRAIN, "FUSION_LOSS_WEIGHT", 1.0))
        rgb_weight = float(getattr(self.cfg.TRAIN, "RGB_LOSS_WEIGHT", 1.0))
        sk_weight = float(getattr(self.cfg.TRAIN, "SK_LOSS_WEIGHT", 1.0))
        kl_rgb_weight = float(getattr(self.cfg.TRAIN, "KL_RGB_FROM_SK_WEIGHT", 1.0))
        kl_sk_weight = float(getattr(self.cfg.TRAIN, "KL_SK_FROM_RGB_WEIGHT", 1.0))
        kd_temp = float(getattr(self.cfg.TRAIN, "KD_TEMPERATURE", 2.0))

        fused_loss = self.objective["cls"](fused_logits, gt_label)
        rgb_loss = self.objective["cls"](rgb_logits, gt_label)
        sk_loss = self.objective["cls"](sk_logits, gt_label)
        weighted_fused_loss = fused_weight * fused_loss
        weighted_rgb_loss = rgb_weight * rgb_loss
        weighted_sk_loss = sk_weight * sk_loss
        weighted_kl_rgb = torch.zeros_like(weighted_rgb_loss)
        weighted_kl_sk = torch.zeros_like(weighted_rgb_loss)

        if fusion_strategy == "mutual_kl":
            kl_rgb = F.kl_div(
                F.log_softmax(rgb_logits / kd_temp, dim=1),
                F.softmax(sk_logits.detach() / kd_temp, dim=1),
                reduction="batchmean",
            ) * (kd_temp ** 2)
            kl_sk = F.kl_div(
                F.log_softmax(sk_logits / kd_temp, dim=1),
                F.softmax(rgb_logits.detach() / kd_temp, dim=1),
                reduction="batchmean",
            ) * (kd_temp ** 2)
            weighted_kl_rgb = kl_rgb_weight * kl_rgb
            weighted_kl_sk = kl_sk_weight * kl_sk
            loss = weighted_rgb_loss + weighted_sk_loss + weighted_kl_rgb + weighted_kl_sk
        else:
            loss = weighted_fused_loss + weighted_rgb_loss + weighted_sk_loss

        fused_top1, fused_top5 = topk_accuracy(fused_logits, gt_label, (1, 5))
        rgb_top1, rgb_top5 = topk_accuracy(rgb_logits, gt_label, (1, 5))
        sk_top1, sk_top5 = topk_accuracy(sk_logits, gt_label, (1, 5))
        fused_top1 = reduce_tensor(fused_top1)
        fused_top5 = reduce_tensor(fused_top5)
        rgb_top1 = reduce_tensor(rgb_top1)
        rgb_top5 = reduce_tensor(rgb_top5)
        sk_top1 = reduce_tensor(sk_top1)
        sk_top5 = reduce_tensor(sk_top5)

        status = {
            "Top1_fused": fused_top1.item(),
            "Top5_fused": fused_top5.item(),
            "Top1_rgb": rgb_top1.item(),
            "Top5_rgb": rgb_top5.item(),
            "Top1_sk": sk_top1.item(),
            "Top5_sk": sk_top5.item(),
            "Loss/fused": weighted_fused_loss.item(),
            "Loss/rgb": weighted_rgb_loss.item(),
            "Loss/sk": weighted_sk_loss.item(),
            "Loss/kl_rgb": weighted_kl_rgb.item(),
            "Loss/kl_sk": weighted_kl_sk.item(),
            "Loss/total": loss.item(),
        }
        return (loss, status) if return_status else loss
