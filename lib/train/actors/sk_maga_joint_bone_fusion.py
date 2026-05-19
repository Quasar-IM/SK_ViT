import torch
import torch.nn.functional as F

from . import BaseActor
from ..admin.stats import topk_accuracy
from lib.utils.misc import reduce_tensor
from lib.train.dataset.sk_maga_inputs import prepare_joint_inputs, prepare_bone_inputs
from lib.train.dataset.sk_maga_schema import NUM_MODEL_JOINTS


class SK_MAGAJointBoneFusionActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def _net(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def get_optimizer_param_groups(self):
        net = self._net()
        lr_mult = float(getattr(self.cfg.TRAIN, "FUSION_BACKBONE_LR_MULT", 0.1))
        params = []
        modules = [net.joint_model, net.bone_model]
        for module in modules:
            params.extend(list(module.parameters()))
        return [{"params": params, "lr_mult": lr_mult}]

    def forward_pass(self, data):
        coords = data["skeleton_frames"][0].view(-1, *data["skeleton_frames"].shape[2:]).float()
        if coords.shape[1] != NUM_MODEL_JOINTS and coords.shape[2] != NUM_MODEL_JOINTS:
            raise ValueError(f"Expected {NUM_MODEL_JOINTS} joints in dim 1 or 2, got {tuple(coords.shape)}")
        joint_x, joint_index_t = prepare_joint_inputs(coords, self.cfg)
        bone_x, bone_index_t = prepare_bone_inputs(coords, self.cfg)
        return self.net(
            joint_x=joint_x,
            joint_index_t=joint_index_t,
            bone_x=bone_x,
            bone_index_t=bone_index_t,
            y=data["label"] if self.net.training else None,
            epoch=int(data.get("epoch", 0)),
        )

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_label = gt_dict["label"]
        fused_logits = pred_dict["logist"]
        fused_loss = self.objective["cls"](fused_logits, gt_label)
        joint_logits = pred_dict["logits_joint"]
        bone_logits = pred_dict["logits_bone"]
        joint_loss = self.objective["cls"](joint_logits, gt_label)
        bone_loss = self.objective["cls"](bone_logits, gt_label)
        fusion_weight = float(getattr(self.cfg.TRAIN, "JB_FUSION_WEIGHT", 1.0))
        joint_weight = float(getattr(self.cfg.TRAIN, "JB_JOINT_WEIGHT", 1.0))
        bone_weight = float(getattr(self.cfg.TRAIN, "JB_BONE_WEIGHT", 1.0))
        kl_joint_weight = float(getattr(self.cfg.TRAIN, "JB_KL_JOINT_FROM_BONE_WEIGHT", 1.0))
        kl_bone_weight = float(getattr(self.cfg.TRAIN, "JB_KL_BONE_FROM_JOINT_WEIGHT", 1.0))
        kd_temp = float(getattr(self.cfg.TRAIN, "JB_KD_TEMPERATURE", 2.0))

        weighted_fused_loss = fusion_weight * fused_loss
        weighted_joint_loss = joint_weight * joint_loss
        weighted_bone_loss = bone_weight * bone_loss
        weighted_kl_joint = torch.zeros_like(weighted_fused_loss)
        weighted_kl_bone = torch.zeros_like(weighted_fused_loss)

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
        weighted_kl_joint = kl_joint_weight * kl_joint
        weighted_kl_bone = kl_bone_weight * kl_bone

        loss = weighted_fused_loss + weighted_joint_loss + weighted_bone_loss + weighted_kl_joint + weighted_kl_bone

        prec1, prec5 = topk_accuracy(fused_logits, gt_label, (1, 5))
        joint_prec1, _ = topk_accuracy(joint_logits, gt_label, (1, 5))
        bone_prec1, _ = topk_accuracy(bone_logits, gt_label, (1, 5))
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)
        joint_prec1 = reduce_tensor(joint_prec1)
        bone_prec1 = reduce_tensor(bone_prec1)
        status = {
            "Top1": prec1.item(),
            "Top5": prec5.item(),
            "Loss/fused": weighted_fused_loss.item(),
            "Loss/joint": weighted_joint_loss.item(),
            "Loss/bone": weighted_bone_loss.item(),
            "Loss/kl_joint": weighted_kl_joint.item(),
            "Loss/kl_bone": weighted_kl_bone.item(),
            "Top1_joint": joint_prec1.item(),
            "Top1_bone": bone_prec1.item(),
            "Loss/total": loss.item(),
        }
        return (loss, status) if return_status else loss
