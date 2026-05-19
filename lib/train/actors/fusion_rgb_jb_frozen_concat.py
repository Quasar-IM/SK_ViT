from . import BaseActor
from ..admin.stats import topk_accuracy
from lib.utils.misc import reduce_tensor
from lib.train.dataset.sk_maga_inputs import prepare_joint_inputs, prepare_bone_inputs
from lib.train.dataset.sk_maga_schema import NUM_MODEL_JOINTS
import torch


class FusionRGBJBFrozenConcatActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg

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
            epoch=int(data.get("epoch", 0)),
        )

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_label = gt_dict["label"]
        logits = pred_dict["logist"]
        rgb_logits = pred_dict["logits_rgb"]
        sk_logits = pred_dict["logits_sk"]
        loss_fused = self.objective["cls"](logits, gt_label)

        alpha = float(getattr(self.cfg.MODEL, "RGB_JB_LATE_ALPHA", 0.377069))
        beta = float(getattr(self.cfg.MODEL, "RGB_JB_LATE_BETA", 0.622931))
        rgb_temp = float(getattr(self.cfg.MODEL, "RGB_JB_RGB_LOGIT_TEMP", 1.3))
        late_weight = float(getattr(self.cfg.TRAIN, "RGB_JB_LATE_LOSS_WEIGHT", 1.0))
        late_logits = alpha * sk_logits + beta * (rgb_logits / rgb_temp)
        loss_late = self.objective["cls"](late_logits, gt_label)

        loss = loss_fused
        top1, top5 = topk_accuracy(logits, gt_label, (1, 5))
        top1 = reduce_tensor(top1)
        top5 = reduce_tensor(top5)
        status = {
            "Top1": top1.item(),
            "Top5": top5.item(),
            "Loss/fused": loss_fused.item(),
            "Loss/late": (late_weight * loss_late).item(),
            "Loss/total": loss.item(),
        }
        return (loss, status) if return_status else loss
