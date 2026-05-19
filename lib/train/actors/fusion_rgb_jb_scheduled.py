from . import BaseActor
from ..admin.stats import topk_accuracy
from lib.utils.misc import reduce_tensor
from lib.train.dataset.sk_maga_inputs import prepare_joint_inputs, prepare_bone_inputs
from lib.train.dataset.sk_maga_schema import NUM_MODEL_JOINTS
import torch
import torch.nn.functional as F


class FusionRGBJBScheduledActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg
        self._epoch = 0
        self._ogm_state = {
            "conf_rgb": 0.0,
            "conf_sk": 0.0,
            "scale_rgb": 1.0,
            "scale_sk": 1.0,
        }

    def _net(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def get_optimizer_param_groups(self):
        net = self._net()
        lr_mult = float(getattr(self.cfg.TRAIN, "RGB_JB_BACKBONE_LR_MULT", 0.1))
        return [
            {"params": list(net.rgb_model.parameters()), "lr_mult": lr_mult},
            {"params": list(net.sk_model.parameters()), "lr_mult": lr_mult},
        ]

    def __call__(self, data):
        self._epoch = int(data.get("epoch", 0))
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def post_backward(self):
        self._net().apply_grad_schedule(self._epoch)
        if not bool(getattr(self.cfg.TRAIN, "RGB_JB_USE_OGM_GRAD_MOD", False)):
            return
        scale_rgb = float(self._ogm_state["scale_rgb"])
        scale_sk = float(self._ogm_state["scale_sk"])
        net = self._net()
        for p in net.rgb_model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale_rgb)
        for p in net.sk_model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale_sk)

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
            epoch=self._epoch,
        )

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_label = gt_dict["label"]
        fused_logits = pred_dict["logist"]
        rgb_logits = pred_dict["logits_rgb"]
        joint_logits = pred_dict["logits_joint"]
        bone_logits = pred_dict["logits_bone"]
        sk_logits = pred_dict["logits_sk"]
        teacher_logits = pred_dict["logits_teacher"]
        variant = str(getattr(self.cfg.MODEL, "RGB_JB_SCHEDULED_VARIANT", "teacher_concat")).lower()

        rgb_w = float(getattr(self.cfg.TRAIN, "RGB_JB_RGB_CE_WEIGHT", 0.2))
        joint_w = float(getattr(self.cfg.TRAIN, "RGB_JB_JOINT_CE_WEIGHT", 0.2))
        bone_w = float(getattr(self.cfg.TRAIN, "RGB_JB_BONE_CE_WEIGHT", 0.2))
        sk_w = float(getattr(self.cfg.TRAIN, "RGB_JB_SK_CE_WEIGHT", 0.5))
        kl_w = float(getattr(self.cfg.TRAIN, "RGB_JB_TEACHER_KL_WEIGHT", 0.5))
        T = float(getattr(self.cfg.MODEL, "RGB_JB_DISTILL_T", 2.0))
        shared_w = float(getattr(self.cfg.TRAIN, "RGB_JB_SHARED_LOSS_WEIGHT", 0.1))
        orth_w = float(getattr(self.cfg.TRAIN, "RGB_JB_ORTH_LOSS_WEIGHT", 0.05))
        contrastive_w = float(getattr(self.cfg.TRAIN, "RGB_JB_CONTRASTIVE_WEIGHT", 0.1))
        contrastive_temp = float(getattr(self.cfg.MODEL, "RGB_JB_CONTRASTIVE_TEMP", 0.1))

        loss_fused = self.objective["cls"](fused_logits, gt_label)
        loss_rgb = self.objective["cls"](rgb_logits, gt_label)
        loss_joint = self.objective["cls"](joint_logits, gt_label)
        loss_bone = self.objective["cls"](bone_logits, gt_label)
        loss_sk = self.objective["cls"](sk_logits, gt_label)
        loss_late = self.objective["cls"](teacher_logits, gt_label)
        if variant in {"teacher_concat", "teacher_delta"}:
            loss_kl = F.kl_div(
                F.log_softmax(fused_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean",
            ) * (T ** 2)
        else:
            loss_kl = fused_logits.new_zeros(())

        loss_shared = fused_logits.new_zeros(())
        loss_orth = fused_logits.new_zeros(())
        loss_contrastive = fused_logits.new_zeros(())
        if variant == "decouple":
            shared_rgb = pred_dict["shared_rgb"]
            shared_sk = pred_dict["shared_sk"]
            specific_rgb = pred_dict["specific_rgb"]
            specific_sk = pred_dict["specific_sk"]
            loss_shared = F.mse_loss(shared_rgb, shared_sk)
            rgb_dot = (shared_rgb * specific_rgb).sum(dim=1)
            sk_dot = (shared_sk * specific_sk).sum(dim=1)
            loss_orth = (rgb_dot.pow(2).mean() + sk_dot.pow(2).mean()) * 0.5
        elif variant == "contrastive":
            contrast_rgb = pred_dict["contrast_rgb"]
            contrast_sk = pred_dict["contrast_sk"]
            logits_contrast = contrast_rgb @ contrast_sk.t() / contrastive_temp
            targets = torch.arange(logits_contrast.shape[0], device=logits_contrast.device)
            loss_i2s = F.cross_entropy(logits_contrast, targets)
            loss_s2i = F.cross_entropy(logits_contrast.t(), targets)
            loss_contrastive = 0.5 * (loss_i2s + loss_s2i)

        if variant == "contrastive":
            total = (
                rgb_w * loss_rgb
                + joint_w * loss_joint
                + bone_w * loss_bone
                + sk_w * loss_sk
                + contrastive_w * loss_contrastive
            )
        else:
            total = (
                loss_fused
                + rgb_w * loss_rgb
                + joint_w * loss_joint
                + bone_w * loss_bone
                + kl_w * loss_kl
                + shared_w * loss_shared
                + orth_w * loss_orth
            )

        if bool(getattr(self.cfg.TRAIN, "RGB_JB_USE_OGM_GRAD_MOD", False)):
            with torch.no_grad():
                rgb_prob = F.softmax(rgb_logits, dim=1).gather(1, gt_label.view(-1, 1)).mean()
                sk_prob = F.softmax(sk_logits, dim=1).gather(1, gt_label.view(-1, 1)).mean()
                eps = 1e-6
                min_rgb_scale = float(getattr(self.cfg.TRAIN, "RGB_JB_OGM_MIN_RGB_SCALE", 0.3))
                max_sk_scale = float(getattr(self.cfg.TRAIN, "RGB_JB_OGM_MAX_SK_SCALE", 1.0))
                if rgb_prob.item() > sk_prob.item():
                    scale_rgb = max(min_rgb_scale, (sk_prob / (rgb_prob + eps)).item())
                    scale_sk = 1.0
                else:
                    scale_rgb = 1.0
                    scale_sk = min(max_sk_scale, (rgb_prob / (sk_prob + eps)).item())
                self._ogm_state = {
                    "conf_rgb": rgb_prob.item(),
                    "conf_sk": sk_prob.item(),
                    "scale_rgb": scale_rgb,
                    "scale_sk": scale_sk,
                }
        else:
            self._ogm_state = {
                "conf_rgb": 0.0,
                "conf_sk": 0.0,
                "scale_rgb": 1.0,
                "scale_sk": 1.0,
            }

        top1, top5 = topk_accuracy(fused_logits, gt_label, (1, 5))
        top1_rgb, top5_rgb = topk_accuracy(rgb_logits, gt_label, (1, 5))
        top1_joint, top5_joint = topk_accuracy(joint_logits, gt_label, (1, 5))
        top1_bone, top5_bone = topk_accuracy(bone_logits, gt_label, (1, 5))
        top1 = reduce_tensor(top1)
        top5 = reduce_tensor(top5)
        top1_rgb = reduce_tensor(top1_rgb)
        top5_rgb = reduce_tensor(top5_rgb)
        top1_joint = reduce_tensor(top1_joint)
        top5_joint = reduce_tensor(top5_joint)
        top1_bone = reduce_tensor(top1_bone)
        top5_bone = reduce_tensor(top5_bone)
        top1_sk, top5_sk = topk_accuracy(sk_logits, gt_label, (1, 5))
        top1_sk = reduce_tensor(top1_sk)
        top5_sk = reduce_tensor(top5_sk)

        status = {
            "Top1": top1.item(),
            "Top5": top5.item(),
            "Top1_rgb": top1_rgb.item(),
            "Top5_rgb": top5_rgb.item(),
            "Top1_joint": top1_joint.item(),
            "Top5_joint": top5_joint.item(),
            "Top1_bone": top1_bone.item(),
            "Top5_bone": top5_bone.item(),
            "Top1_sk": top1_sk.item(),
            "Top5_sk": top5_sk.item(),
            "Loss/fused": loss_fused.item(),
            "Loss/rgb": (rgb_w * loss_rgb).item(),
            "Loss/joint": (joint_w * loss_joint).item(),
            "Loss/bone": (bone_w * loss_bone).item(),
            "Loss/sk": (sk_w * loss_sk).item(),
            "Loss/late": loss_late.item(),
            "Loss/kl": (kl_w * loss_kl).item(),
            "Loss/shared": (shared_w * loss_shared).item(),
            "Loss/orth": (orth_w * loss_orth).item(),
            "Loss/contrastive": (contrastive_w * loss_contrastive).item(),
            "Loss/total": total.item(),
            "Temp/rgb": float(pred_dict["rgb_temp"].item()) if "rgb_temp" in pred_dict else float(getattr(self.cfg.MODEL, "RGB_JB_RGB_LOGIT_TEMP", 1.3)),
            "Conf/rgb": self._ogm_state["conf_rgb"],
            "Conf/sk": self._ogm_state["conf_sk"],
            "GradScale/rgb": self._ogm_state["scale_rgb"],
            "GradScale/sk": self._ogm_state["scale_sk"],
        }
        return (total, status) if return_status else total
