import math

from . import BaseActor
from lib.utils.misc import reduce_tensor
from ..admin.stats import topk_accuracy


def adjust_keep_rate(epoch, warmup_epochs, total_epochs, iters_per_epoch, base_keep_rate=0.5, max_keep_rate=1.0):
    if epoch < warmup_epochs:
        return 1.0
    if epoch >= total_epochs:
        return base_keep_rate
    total_iters = iters_per_epoch * (total_epochs - warmup_epochs)
    cur_iters = epoch * iters_per_epoch - iters_per_epoch * warmup_epochs
    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) * (math.cos(cur_iters / total_iters * math.pi) + 1) * 0.5
    return keep_rate


class SK_ViTActor(BaseActor):
    """Pure RGB SK_ViT actor (ViViT + prune only)."""

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
        prune_keep_ratio = None
        if getattr(self.cfg.MODEL, "PRUNE_LAYERS", []):
            prune_start_epoch = int(self.cfg.TRAIN.PRUNE_START_EPOCH)
            prune_warm_epoch = int(self.cfg.TRAIN.PRUNE_WARM_EPOCH)
            base_keep_rate = float(self.cfg.MODEL.PRUNE_KEEP_RATIO)
            prune_keep_ratio = adjust_keep_rate(
                epoch=int(data["epoch"]),
                warmup_epochs=prune_start_epoch,
                total_epochs=prune_start_epoch + prune_warm_epoch,
                iters_per_epoch=1,
                base_keep_rate=base_keep_rate,
            )

        rgb_frames = data["rgb_frames"][0].view(-1, *data["rgb_frames"].shape[2:])
        out_dict = self.net(
            x=rgb_frames,
            train_keep_ratio=prune_keep_ratio,
            y=data["label"] if self.net.training else None,
            epoch=int(data.get("epoch", 0)),
        )
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_label = gt_dict["label"]
        loss, status = self.calc_loss(pred_dict, gt_label)
        if return_status:
            return loss, status
        return loss

    def calc_loss(self, pred_dict, gt_label, show_name=""):
        logits_v = pred_dict.get("logits_v", pred_dict["logist"])
        cls_loss_v = self.objective["cls"](logits_v, gt_label)
        prec1, prec5 = topk_accuracy(logits_v, gt_label, (1, 5))
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)

        loss = self.loss_weight["cls"] * cls_loss_v
        status = {
            f"{show_name}Top1": prec1.item(),
            f"{show_name}Top5": prec5.item(),
            f"{show_name}Loss/cls_v": cls_loss_v.item(),
            f"{show_name}Loss/cls_rgb": cls_loss_v.item(),
            f"{show_name}Acc/rgb": prec1.item(),
            f"{show_name}Loss/total": loss.item(),
        }
        return loss, status
