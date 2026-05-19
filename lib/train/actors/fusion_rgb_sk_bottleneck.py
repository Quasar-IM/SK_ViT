from . import BaseActor
from ..admin.stats import topk_accuracy
from lib.utils.misc import reduce_tensor
from lib.train.dataset.sk_maga_inputs import prepare_joint_inputs
from lib.train.dataset.sk_maga_schema import NUM_MODEL_JOINTS
import torch


class FusionRGBSKBottleneckActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg
        self._freeze_state = None

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
        self._apply_phase_freeze(int(data.get("epoch", 0)))
        out_dict = self.forward_pass(data)
        self._last_pred_dict = out_dict
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
        rgb_logits = pred_dict["logits_rgb"]
        sk_logits = pred_dict["logits_sk"]

        fused_weight = float(getattr(self.cfg.TRAIN, "BOTTLENECK_FUSED_LOSS_WEIGHT", 0.0))
        rgb_weight = float(getattr(self.cfg.TRAIN, "BOTTLENECK_RGB_LOSS_WEIGHT", 0.5))
        sk_weight = float(getattr(self.cfg.TRAIN, "BOTTLENECK_SK_LOSS_WEIGHT", 1.5))
        late_temp_rgb = float(getattr(self.cfg.TRAIN, "BOTTLENECK_LATE_RGB_TEMP", 1.3))
        late_temp_sk = float(getattr(self.cfg.TRAIN, "BOTTLENECK_LATE_SK_TEMP", 1.0))
        late_alpha = float(getattr(self.cfg.TRAIN, "BOTTLENECK_LATE_ALPHA", 0.5))
        late_beta = float(getattr(self.cfg.TRAIN, "BOTTLENECK_LATE_BETA", 0.5))

        rgb_loss = self.objective["cls"](rgb_logits, gt_label)
        sk_loss = self.objective["cls"](sk_logits, gt_label)
        weighted_rgb_loss = rgb_weight * rgb_loss
        weighted_sk_loss = sk_weight * sk_loss
        weighted_fused_loss = torch.tensor(0.0, device=rgb_logits.device)
        if fused_weight > 0.0 and "logist" in pred_dict:
            fused_loss = self.objective["cls"](pred_dict["logist"], gt_label)
            weighted_fused_loss = fused_weight * fused_loss
        loss = weighted_fused_loss + weighted_rgb_loss + weighted_sk_loss

        p_late = late_alpha * torch.softmax(sk_logits / late_temp_sk, dim=1) + \
                 late_beta * torch.softmax(rgb_logits / late_temp_rgb, dim=1)
        late_logits_eval = torch.log(p_late.clamp_min(1e-8))
        loss_late_eval = self.objective["cls"](late_logits_eval, gt_label)
        fused_top1, fused_top5 = topk_accuracy(late_logits_eval, gt_label, (1, 5))
        rgb_top1, rgb_top5 = topk_accuracy(rgb_logits, gt_label, (1, 5))
        sk_top1, sk_top5 = topk_accuracy(sk_logits, gt_label, (1, 5))
        fused_top1 = reduce_tensor(fused_top1)
        fused_top5 = reduce_tensor(fused_top5)
        rgb_top1 = reduce_tensor(rgb_top1)
        rgb_top5 = reduce_tensor(rgb_top5)
        sk_top1 = reduce_tensor(sk_top1)
        sk_top5 = reduce_tensor(sk_top5)

        status = {
            "Top1": fused_top1.item(),
            "Top5": fused_top5.item(),
            "Top1_rgb": rgb_top1.item(),
            "Top5_rgb": rgb_top5.item(),
            "Top1_sk": sk_top1.item(),
            "Top5_sk": sk_top5.item(),
            "Loss/fused": weighted_fused_loss.item(),
            "Loss/late_eval": loss_late_eval.item(),
            "Loss/rgb": weighted_rgb_loss.item(),
            "Loss/sk": weighted_sk_loss.item(),
            "Loss/total": loss.item(),
        }
        return (loss, status) if return_status else loss

    def _apply_phase_freeze(self, epoch):
        phase1_end = int(getattr(self.cfg.TRAIN, "BOTTLENECK_PHASE1_END", 3))
        phase2_end = int(getattr(self.cfg.TRAIN, "BOTTLENECK_PHASE2_END", 8))
        if epoch <= phase1_end:
            phase = "freeze_all"
        elif epoch <= phase2_end:
            phase = "partial"
        else:
            phase = "all_open"
        if phase == self._freeze_state:
            return
        self._freeze_state = phase

        net = self._net()
        if phase == "freeze_all":
            for p in net.rgb_model.parameters():
                p.requires_grad = False
            for p in net.sk_model.parameters():
                p.requires_grad = False
            return

        if phase == "all_open":
            for p in net.rgb_model.parameters():
                p.requires_grad = True
            for p in net.sk_model.parameters():
                p.requires_grad = True
            return

        # partial
        for p in net.rgb_model.parameters():
            p.requires_grad = False
        for p in net.sk_model.parameters():
            p.requires_grad = False

        for n, p in net.rgb_model.named_parameters():
            if any(k in n for k in ["blocks.8", "blocks.9", "blocks.10", "blocks.11", "norm", "head", "fc_dropout"]):
                p.requires_grad = True
        for n, p in net.sk_model.named_parameters():
            if any(k in n for k in ["stages.2", "stages.3", "head", "norm", "fc_norm", "classifier"]):
                p.requires_grad = True

    def collect_grad_stats(self):
        if not bool(getattr(self.cfg.TRAIN, "LOG_BOTTLENECK_GRAD_NORM", False)):
            return {}

        net = self._net()
        return {
            "Grad/rgb": self._grad_l2_norm(net.rgb_model.parameters()),
            "Grad/sk": self._grad_l2_norm(net.sk_model.parameters()),
            "Grad/fused": self._grad_l2_norm(net.fused_head.parameters()),
            "Grad/bottleneck": self._grad_l2_norm(
                list(net.rgb_token_proj.parameters()) +
                [net.bottleneck_tokens] +
                list(net.bottleneck_layers.parameters()) +
                list(net.rgb_head.parameters()) +
                list(net.sk_head.parameters())
            ),
        }

    def post_backward(self):
        scale = float(getattr(self.cfg.TRAIN, "BOTTLENECK_RGB_BACKWARD_SCALE", 1.0))
        if abs(scale - 1.0) < 1e-8:
            return
        net = self._net()
        for p in net.rgb_model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    @staticmethod
    def _grad_l2_norm(params):
        sq_sum = 0.0
        has_grad = False
        for p in params:
            if p is None or p.grad is None:
                continue
            grad = p.grad.detach()
            sq_sum += float(torch.sum(grad * grad).item())
            has_grad = True
        return sq_sum ** 0.5 if has_grad else 0.0
