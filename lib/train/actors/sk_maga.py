import math

import torch
import torch.nn.functional as F

from . import BaseActor
from ..admin.stats import topk_accuracy
from lib.utils.misc import reduce_tensor
from lib.train.dataset.sk_maga_schema import NUM_MODEL_JOINTS
from lib.train.dataset.sk_maga_inputs import prepare_sk_maga_inputs


def adjust_keep_rate(epoch, warmup_epochs, total_epochs, iters_per_epoch, base_keep_rate=0.5, max_keep_rate=1,
                     iters=-1):
    if epoch < warmup_epochs:
        return 1
    if epoch >= total_epochs:
        return base_keep_rate
    if iters == -1:
        iters = epoch * iters_per_epoch
    total_iters = iters_per_epoch * (total_epochs - warmup_epochs)
    iters = iters - iters_per_epoch * warmup_epochs
    return base_keep_rate + (max_keep_rate - base_keep_rate) * (math.cos(iters / total_iters * math.pi) + 1) * 0.5


class SK_MAGAActor(BaseActor):
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

    def get_ib_weight(self, epoch):
        warm_start = int(getattr(self.cfg.TRAIN, "MEMORY_WARM_START_EPOCH", 10))
        warm_end = int(getattr(self.cfg.TRAIN, "MEMORY_WARM_END_EPOCH", 20))
        if epoch < warm_start:
            return 0.0
        if epoch >= warm_end:
            return 1.0
        progress = (float(epoch) - warm_start) / float(max(1, warm_end - warm_start))
        return 0.5 - 0.5 * math.cos(math.pi * progress)

    def get_aux_weight(self, epoch):
        warm_start = int(getattr(self.cfg.TRAIN, "MEMORY_AUX_WARM_START_EPOCH", getattr(self.cfg.TRAIN, "MEMORY_WARM_START_EPOCH", 10)))
        warm_end = int(getattr(self.cfg.TRAIN, "MEMORY_AUX_WARM_END_EPOCH", getattr(self.cfg.TRAIN, "MEMORY_WARM_END_EPOCH", 20)))
        if epoch < warm_start:
            return 0.0
        if epoch >= warm_end:
            return 1.0
        progress = (float(epoch) - warm_start) / float(max(1, warm_end - warm_start))
        return 0.5 - 0.5 * math.cos(math.pi * progress)

    def forward_pass(self, data):
        prune_keep_ratio = None
        if getattr(self.cfg.MODEL, "PRUNE_LAYERS", []):
            prune_keep_ratio = adjust_keep_rate(
                data["epoch"],
                warmup_epochs=self.cfg.TRAIN.PRUNE_START_EPOCH,
                total_epochs=self.cfg.TRAIN.PRUNE_START_EPOCH + self.cfg.TRAIN.PRUNE_WARM_EPOCH,
                iters_per_epoch=1,
                base_keep_rate=getattr(self.cfg.MODEL, "PRUNE_KEEP_RATIO", 1.0),
            )

        coords = data["skeleton_frames"][0].view(-1, *data["skeleton_frames"].shape[2:]).float()
        skeleton, index_t = self.prepare_skeleton_inputs(coords)
        is_training = self.net.training
        return self.net(
            x=skeleton,
            index_t=index_t,
            train_keep_ratio=prune_keep_ratio,
            y=data["label"] if is_training else None,
            epoch=int(data.get("epoch", 0)),
        )

    def prepare_skeleton_inputs(self, coords, apply_train_aug=None):
        if coords.shape[1] != NUM_MODEL_JOINTS and coords.shape[2] != NUM_MODEL_JOINTS:
            raise ValueError(f"Expected {NUM_MODEL_JOINTS} skeleton joints in dim 1 or 2, got {tuple(coords.shape)}")
        if apply_train_aug is None:
            apply_train_aug = self.net.training
        return prepare_sk_maga_inputs(coords, self.cfg, apply_train_aug=bool(apply_train_aug))

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_label = gt_dict["label"]
        loss, status = self.calc_loss(pred_dict, gt_label, epoch=int(gt_dict.get("epoch", 0)), gt_dict=gt_dict)
        return (loss, status) if return_status else loss

    def calc_loss(self, pred_dict, gt_label, epoch=0, show_name="", gt_dict=None):
        logits_v = pred_dict.get("logits_v", pred_dict["logist"])
        cls_loss_v = self.objective["cls"](logits_v, gt_label)
        prec1, prec5 = topk_accuracy(logits_v, gt_label, (1, 5))
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)
        cluster_only = bool(getattr(self.cfg.TRAIN, "MEMORY_CLUSTER_ONLY", False))
        loss = cls_loss_v.new_zeros(())
        if not cluster_only:
            loss = loss + self.loss_weight["cls"] * cls_loss_v
        status = {
            f"{show_name}Top1": prec1.item(),
            f"{show_name}Top5": prec5.item(),
            f"{show_name}Loss/cls_v": (0.0 if cluster_only else cls_loss_v.item()),
        }

        aux_weight = self.get_aux_weight(epoch)

        disable_z_loss = bool(getattr(self.cfg.TRAIN, "MEMORY_DISABLE_Z_LOSS", False))

        if (not cluster_only) and (not disable_z_loss) and "logits_z" in pred_dict:
            logits_z = pred_dict["logits_z"]
            ce_z = self.objective["cls"](logits_z, gt_label)
            prec1_z, _ = topk_accuracy(logits_z, gt_label, (1, 5))
            prec1_z = reduce_tensor(prec1_z)
            distill_temp = float(getattr(self.cfg.TRAIN, "MEMORY_DISTILL_TEMP", 1.0))
            kl_weight = float(getattr(self.cfg.TRAIN, "MEMORY_KL_WEIGHT", 1.0))
            ib_beta = float(getattr(self.cfg.TRAIN, "MEMORY_IB_BETA", 10.0))
            ib_weight = self.get_ib_weight(epoch)
            log_prob_v = F.log_softmax(logits_v / distill_temp, dim=1)
            prob_z = F.softmax(logits_z / distill_temp, dim=1)
            kl_loss = F.kl_div(log_prob_v, prob_z, reduction='batchmean') * (distill_temp ** 2)
            ib_loss = ce_z + kl_weight * kl_loss
            loss = loss + ib_beta * ib_weight * ib_loss
            if not self.net.training:
                status[f"{show_name}Top1_z"] = prec1_z.item()
            status.update({
                f"{show_name}Loss/cls_z": (ib_beta * ib_weight * ce_z).item(),
                f"{show_name}Loss/kl": (ib_beta * ib_weight * kl_weight * kl_loss).item(),
                f"{show_name}Loss/ib": (ib_beta * ib_weight * ib_loss).item(),
            })

        extra_losses = []
        if "loss_proto" in pred_dict:
            proto_weight = float(getattr(self.cfg.TRAIN, "MEMORY_PROTO_LOSS_WEIGHT", 0.2))
            loss_proto = pred_dict["loss_proto"]
            weighted = aux_weight * proto_weight * loss_proto
            loss = loss + weighted
            extra_losses.append((f"{show_name}Loss/proto", weighted.item()))
        if "loss_res" in pred_dict:
            res_weight = float(getattr(self.cfg.TRAIN, "MEMORY_RES_LOSS_WEIGHT", 0.1))
            loss_res = pred_dict["loss_res"]
            weighted = aux_weight * res_weight * loss_res
            loss = loss + weighted
            extra_losses.append((f"{show_name}Loss/res", weighted.item()))
        if "loss_div" in pred_dict:
            div_weight = float(getattr(self.cfg.TRAIN, "MEMORY_DIV_LOSS_WEIGHT", 0.01))
            loss_div = pred_dict["loss_div"]
            weighted = aux_weight * div_weight * loss_div
            loss = loss + weighted
            extra_losses.append((f"{show_name}Loss/div", weighted.item()))

        for k, v in extra_losses:
            status[k] = v
        status[f"{show_name}Loss/total"] = loss.item()
        return loss, status
