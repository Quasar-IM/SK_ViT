import torch
import torch.nn.functional as F

from . import BaseActor
from ..admin.stats import topk_accuracy
from lib.utils.misc import reduce_tensor
from lib.train.dataset.sk_maga_inputs import prepare_joint_inputs, prepare_bone_inputs
from lib.train.dataset.sk_maga_schema import NUM_MODEL_JOINTS


class FusionRGBJBFusedSKActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg

    def _net(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def get_optimizer_param_groups(self):
        net = self._net()
        lr_mult = float(getattr(self.cfg.TRAIN, "FUSION_SK_LR_MULT", 0.1))
        return [
            {"params": list(net.rgb_model.parameters()), "lr_mult": lr_mult},
            {"params": list(net.sk_model.parameters()), "lr_mult": lr_mult},
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
        joint_x, joint_index_t = prepare_joint_inputs(coords, self.cfg)
        bone_x, bone_index_t = prepare_bone_inputs(coords, self.cfg)
        return self.net(
            x=all_images["rgb_frames"],
            joint_x=joint_x,
            joint_index_t=joint_index_t,
            bone_x=bone_x,
            bone_index_t=bone_index_t,
            y=data["label"] if self.net.training else None,
            epoch=int(data.get("epoch", 0)),
        )

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_label = gt_dict["label"]
        final_logits = pred_dict["logist"]
        rgb_logits = pred_dict["logits_rgb"]
        joint_logits = pred_dict["logits_joint"]
        bone_logits = pred_dict["logits_bone"]
        sk_fused_logits = pred_dict["logits_sk_fused"]

        fused_weight = float(getattr(self.cfg.TRAIN, "FUSION_LOSS_WEIGHT", 1.0))
        rgb_weight = float(getattr(self.cfg.TRAIN, "RGB_LOSS_WEIGHT", 1.0))
        joint_weight = float(getattr(self.cfg.TRAIN, "SK_JOINT_LOSS_WEIGHT", 0.2))
        bone_weight = float(getattr(self.cfg.TRAIN, "SK_BONE_LOSS_WEIGHT", 0.2))
        sk_fused_weight = float(getattr(self.cfg.TRAIN, "SK_FUSED_LOSS_WEIGHT", 0.2))
        kl_joint_weight = float(getattr(self.cfg.TRAIN, "SK_JOINT_FROM_BONE_KL_WEIGHT", 0.1))
        kl_bone_weight = float(getattr(self.cfg.TRAIN, "SK_BONE_FROM_JOINT_KL_WEIGHT", 0.1))
        kl_sk_from_rgb_weight = float(getattr(self.cfg.TRAIN, "SK_FUSED_FROM_RGB_KL_WEIGHT", 1.0))
        kd_temp = float(getattr(self.cfg.TRAIN, "KD_TEMPERATURE", 2.0))

        fused_loss = self.objective["cls"](final_logits, gt_label)
        rgb_loss = self.objective["cls"](rgb_logits, gt_label)
        joint_loss = self.objective["cls"](joint_logits, gt_label)
        bone_loss = self.objective["cls"](bone_logits, gt_label)
        sk_fused_loss = self.objective["cls"](sk_fused_logits, gt_label)

        weighted_fused = fused_weight * fused_loss
        weighted_rgb = rgb_weight * rgb_loss
        weighted_joint = joint_weight * joint_loss
        weighted_bone = bone_weight * bone_loss
        weighted_sk_fused = sk_fused_weight * sk_fused_loss

        kl_joint = F.kl_div(
            F.log_softmax(joint_logits / kd_temp, dim=1),
            F.softmax(bone_logits.detach() / kd_temp, dim=1),
            reduction="batchmean",
        ) * (kd_temp ** 2)
        kl_bone = F.kl_div(
            F.log_softmax(bone_logits / kd_temp, dim=1),
            F.softmax(joint_logits.detach() / kd_temp, dim=1),
            reduction="batchmean",
        ) * (kd_temp ** 2)
        kl_sk_from_rgb = F.kl_div(
            F.log_softmax(sk_fused_logits / kd_temp, dim=1),
            F.softmax(rgb_logits.detach() / kd_temp, dim=1),
            reduction="batchmean",
        ) * (kd_temp ** 2)

        weighted_kl_joint = kl_joint_weight * kl_joint
        weighted_kl_bone = kl_bone_weight * kl_bone
        weighted_kl_sk_from_rgb = kl_sk_from_rgb_weight * kl_sk_from_rgb

        loss = (
            weighted_fused
            + weighted_rgb
            + weighted_joint
            + weighted_bone
            + weighted_sk_fused
            + weighted_kl_joint
            + weighted_kl_bone
            + weighted_kl_sk_from_rgb
        )

        fused_top1, fused_top5 = topk_accuracy(final_logits, gt_label, (1, 5))
        rgb_top1, rgb_top5 = topk_accuracy(rgb_logits, gt_label, (1, 5))
        joint_top1, joint_top5 = topk_accuracy(joint_logits, gt_label, (1, 5))
        bone_top1, bone_top5 = topk_accuracy(bone_logits, gt_label, (1, 5))
        sk_fused_top1, sk_fused_top5 = topk_accuracy(sk_fused_logits, gt_label, (1, 5))

        fused_top1 = reduce_tensor(fused_top1)
        fused_top5 = reduce_tensor(fused_top5)
        rgb_top1 = reduce_tensor(rgb_top1)
        rgb_top5 = reduce_tensor(rgb_top5)
        joint_top1 = reduce_tensor(joint_top1)
        joint_top5 = reduce_tensor(joint_top5)
        bone_top1 = reduce_tensor(bone_top1)
        bone_top5 = reduce_tensor(bone_top5)
        sk_fused_top1 = reduce_tensor(sk_fused_top1)
        sk_fused_top5 = reduce_tensor(sk_fused_top5)

        status = {
            "Top1": fused_top1.item(),
            "Top5": fused_top5.item(),
            "Top1_rgb": rgb_top1.item(),
            "Top5_rgb": rgb_top5.item(),
            "Top1_joint": joint_top1.item(),
            "Top5_joint": joint_top5.item(),
            "Top1_bone": bone_top1.item(),
            "Top5_bone": bone_top5.item(),
            "Top1_sk_fused": sk_fused_top1.item(),
            "Top5_sk_fused": sk_fused_top5.item(),
            "Loss/fused": weighted_fused.item(),
            "Loss/rgb": weighted_rgb.item(),
            "Loss/joint": weighted_joint.item(),
            "Loss/bone": weighted_bone.item(),
            "Loss/sk_fused": weighted_sk_fused.item(),
            "Loss/kl_joint": weighted_kl_joint.item(),
            "Loss/kl_bone": weighted_kl_bone.item(),
            "Loss/kl_sk_from_rgb": weighted_kl_sk_from_rgb.item(),
            "Loss/total": loss.item(),
        }
        return (loss, status) if return_status else loss
