import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InBoMemNeck(nn.Module):
    def __init__(self, feature_dim, num_classes, slots_per_class=5, tau=10.0,
                 forget_decay=0.1198, forget_min=0.05,
                 read_pos_weight=0.0, read_neg_weight=1.0,
                 forget_schedule='exp',
                 forget_exp_start=0.95, forget_exp_end_epoch=10,
                 forget_stage1_end=10, forget_stage1_value=0.8,
                 forget_stage2_end=20, forget_stage2_value=0.5,
                 forget_decay_end=30, forget_final_value=0.01):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.slots_per_class = slots_per_class
        self.tau = tau
        self.forget_decay = forget_decay
        self.forget_min = forget_min
        self.read_pos_weight = float(read_pos_weight)
        self.read_neg_weight = float(read_neg_weight)
        self.forget_schedule = forget_schedule
        self.forget_exp_start = float(forget_exp_start)
        self.forget_exp_end_epoch = int(forget_exp_end_epoch)
        self.forget_stage1_end = int(forget_stage1_end)
        self.forget_stage1_value = float(forget_stage1_value)
        self.forget_stage2_end = int(forget_stage2_end)
        self.forget_stage2_value = float(forget_stage2_value)
        self.forget_decay_end = int(forget_decay_end)
        self.forget_final_value = float(forget_final_value)

        self.memory_bank = nn.Parameter(
            torch.empty(num_classes, slots_per_class, feature_dim), requires_grad=False
        )
        nn.init.normal_(self.memory_bank, std=0.02)
        self.classifier_v = nn.Linear(feature_dim, num_classes)
        self.classifier_z = nn.Linear(feature_dim, num_classes)
        self.register_buffer('smooth_kernel', torch.tensor([0.1, 0.8, 0.1], dtype=torch.float32))

    @staticmethod
    def _l2_normalize(x):
        return F.normalize(x, p=2, dim=-1, eps=1e-6)

    def get_forget_rate(self, epoch):
        if self.forget_schedule == 'exp_fixed':
            epoch = float(epoch)
            if epoch < self.forget_exp_end_epoch:
                progress = epoch / float(max(1, self.forget_exp_end_epoch))
                ratio = self.forget_min / max(self.forget_exp_start, 1e-8)
                return self.forget_exp_start * (ratio ** progress)
            return self.forget_min
        if self.forget_schedule == 'staged':
            epoch = float(epoch)
            if epoch < self.forget_stage1_end:
                return self.forget_stage1_value
            if epoch < self.forget_stage2_end:
                return self.forget_stage2_value
            if epoch < self.forget_decay_end:
                progress = (epoch - self.forget_stage2_end) / float(max(1, self.forget_decay_end - self.forget_stage2_end))
                ratio = self.forget_final_value / max(self.forget_stage2_value, 1e-8)
                return self.forget_stage2_value * (ratio ** progress)
            return self.forget_final_value
        rate = math.exp(-self.forget_decay * float(epoch))
        return max(self.forget_min, rate)

    def _smooth_scores(self, score):
        # score: [slots]
        kernel = self.smooth_kernel.to(device=score.device, dtype=score.dtype).view(1, 1, -1)
        padded = F.pad(score.view(1, 1, -1), (1, 1), mode='replicate')
        smoothed = F.conv1d(padded, kernel)
        return smoothed.view(-1)

    def write_memory(self, v, y, epoch):
        if v.ndim != 2:
            raise ValueError("Expected v with shape [B, feature_dim]")
        with torch.no_grad():
            v_det = v.detach()
            forget_rate = self.get_forget_rate(epoch)
            for feat, label in zip(v_det, y):
                class_idx = int(label.item())
                slots = self.memory_bank[class_idx]
                slots.mul_(1.0 - forget_rate)
                cossim = F.cosine_similarity(feat.unsqueeze(0), slots, dim=-1)
                phi_write = 1.0 + cossim
                w_o = torch.softmax(phi_write, dim=0)
                slots.add_(w_o.unsqueeze(-1) * feat.unsqueeze(0))
                slots.copy_(self._l2_normalize(slots))

    def read_memory(self, v, target_y):
        if v.ndim != 2:
            raise ValueError("Expected v with shape [B, feature_dim]")
        outputs = []
        total_read_weight = max(self.read_pos_weight + self.read_neg_weight, 1e-8)
        pos_coeff = self.read_pos_weight / total_read_weight
        neg_coeff = self.read_neg_weight / total_read_weight
        for feat, label in zip(v, target_y):
            class_idx = int(label.item())
            # Read from an isolated snapshot so training-time writeback does not
            # modify tensors that are still needed by the current backward graph.
            slots = self.memory_bank[class_idx].detach().clone()
            cossim = F.cosine_similarity(feat.unsqueeze(0), slots, dim=-1)
            phi_read_pos = 1.0 + cossim
            phi_read_neg = 1.0 - cossim
            w_t_pos = self._smooth_scores(torch.softmax(phi_read_pos, dim=0))
            w_t_neg = self._smooth_scores(torch.softmax(phi_read_neg, dim=0))
            w_r_pos = torch.softmax(w_t_pos * self.tau, dim=0)
            w_r_neg = torch.softmax(w_t_neg * self.tau, dim=0)
            z_pos = torch.sum(w_r_pos.unsqueeze(-1) * slots, dim=0)
            z_neg = torch.sum(w_r_neg.unsqueeze(-1) * slots, dim=0)
            z_i = pos_coeff * z_pos + neg_coeff * z_neg
            z_i = self._l2_normalize(z_i)
            outputs.append(z_i)
        return torch.stack(outputs, dim=0)

    def forward(self, v, y=None, epoch=0):
        logits_v = self.classifier_v(v)
        if self.training:
            if y is None:
                raise ValueError("Training InBoMemNeck requires ground-truth y")
            target_y = y
        else:
            target_y = logits_v.argmax(dim=1)

        z = self.read_memory(v, target_y)
        logits_z = self.classifier_z(z)

        if self.training:
            self.write_memory(v.detach(), y, epoch)

        return {
            'feat_v': v,
            'feat_z': z,
            'logits_v': logits_v,
            'logits_z': logits_z,
        }


class OnePlusNMemNeck(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_classes,
        slots_per_class=3,
        tau=10.0,
        tau_res=10.0,
        eta=1.0,
        forget_start=0.9,
        forget_decay=0.1198,
        forget_min=0.05,
        warmup_epochs=5,
        confidence_threshold=0.0,
        confidence_threshold_mid=0.5,
        confidence_threshold_high=0.6,
        confidence_stage1_end=10,
        confidence_stage2_end=15,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.slots_per_class = int(slots_per_class)
        self.tau = float(tau)
        self.tau_res = float(tau_res)
        self.eta = float(eta)
        self.forget_start = float(forget_start)
        self.forget_decay = float(forget_decay)
        self.forget_min = float(forget_min)
        self.warmup_epochs = int(warmup_epochs)
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_threshold_mid = float(confidence_threshold_mid)
        self.confidence_threshold_high = float(confidence_threshold_high)
        self.confidence_stage1_end = int(confidence_stage1_end)
        self.confidence_stage2_end = int(confidence_stage2_end)

        self.classifier_v = nn.Linear(self.feature_dim, self.num_classes)
        self.classifier_z = nn.Linear(self.feature_dim, self.num_classes)
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.mem_norm = nn.LayerNorm(self.feature_dim)
        self.z_norm = nn.LayerNorm(self.feature_dim)

        prototypes = torch.empty(self.num_classes, self.feature_dim, dtype=torch.float32)
        residuals = torch.empty(self.num_classes, self.slots_per_class, self.feature_dim, dtype=torch.float32)
        nn.init.normal_(prototypes, std=0.02)
        nn.init.normal_(residuals, std=0.02)
        self.register_buffer("prototypes", self._l2_normalize(prototypes))
        self.register_buffer("residual_bank", self._l2_normalize(residuals))
        self.register_buffer("prototype_initialized", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer(
            "residual_initialized",
            torch.zeros(self.num_classes, self.slots_per_class, dtype=torch.bool),
        )

    @staticmethod
    def _l2_normalize(x):
        return F.normalize(x, p=2, dim=-1, eps=1e-6)

    def get_update_rate(self, epoch):
        if float(epoch) < float(self.warmup_epochs):
            return 0.0
        effective_epoch = max(0.0, float(epoch) - float(self.warmup_epochs))
        rate = self.forget_start * math.exp(-self.forget_decay * effective_epoch)
        return max(self.forget_min, rate)

    def get_confidence_threshold(self, epoch):
        epoch = int(epoch)
        if epoch < self.confidence_stage1_end:
            return self.confidence_threshold
        if epoch < self.confidence_stage2_end:
            return self.confidence_threshold_mid
        return self.confidence_threshold_high

    def _update_memory(self, q, logits_v, y, epoch):
        if epoch < self.warmup_epochs:
            return
        with torch.no_grad():
            q = self._l2_normalize(q.detach())
            update_rate = self.get_update_rate(epoch)
            conf_threshold = self.get_confidence_threshold(epoch)
            probs_v = torch.softmax(logits_v.detach(), dim=1)
            pred_v = probs_v.argmax(dim=1)
            conf_v = probs_v.max(dim=1).values
            valid = (pred_v == y) & (conf_v >= conf_threshold)
            if not bool(valid.any()):
                return

            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
            for cls in y[valid].unique():
                cls_int = int(cls.item())
                cls_mask = y[valid] == cls
                q_mean = q[valid][cls_mask].mean(dim=0)
                q_mean = self._l2_normalize(q_mean)
                if not bool(self.prototype_initialized[cls_int].item()):
                    self.prototypes[cls_int].copy_(q_mean)
                    self.prototype_initialized[cls_int] = True
                else:
                    updated = (1.0 - update_rate) * self.prototypes[cls_int] + update_rate * q_mean
                    self.prototypes[cls_int].copy_(self._l2_normalize(updated))

            for idx in valid_idx.tolist():
                cls_int = int(y[idx].item())
                p_y = self.prototypes[cls_int]
                residual = self._l2_normalize(q[idx] - p_y)
                uninit = torch.nonzero(~self.residual_initialized[cls_int], as_tuple=False).flatten()
                if uninit.numel() > 0:
                    slot = int(uninit[0].item())
                    self.residual_bank[cls_int, slot].copy_(residual)
                    self.residual_initialized[cls_int, slot] = True
                else:
                    sims = torch.matmul(self.residual_bank[cls_int], residual)
                    slot = int(sims.argmax().item())
                    updated = (1.0 - update_rate) * self.residual_bank[cls_int, slot] + update_rate * residual
                    self.residual_bank[cls_int, slot].copy_(self._l2_normalize(updated))

    def forward(self, v, y=None, epoch=0):
        logits_v = self.classifier_v(v)
        out = {
            "feat_v": v,
            "logits_v": logits_v,
        }

        if not self.training:
            return out

        if y is None:
            raise ValueError("Training OnePlusNMemNeck requires ground-truth y")

        v_mem = self._l2_normalize(self.mem_norm(v))
        q = self._l2_normalize(self.proj(v))
        P_norm = self._l2_normalize(self.prototypes)
        R_norm = self._l2_normalize(self.residual_bank)

        proto_logits = torch.matmul(q, P_norm.t()) / self.tau
        loss_proto = F.cross_entropy(proto_logits, y)

        p_y = P_norm[y]
        res = self._l2_normalize(q - p_y)
        R_y = R_norm[y]
        sim_res = torch.einsum("bd,bnd->bn", res, R_y) / self.tau_res
        assign = sim_res.detach().argmax(dim=1)
        loss_res = F.cross_entropy(sim_res, assign)

        gram = torch.matmul(R_norm, R_norm.transpose(-1, -2))
        eye = torch.eye(self.slots_per_class, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        off_diag = gram * (1.0 - eye)
        loss_div = (off_diag ** 2).mean()

        r_mem = self._l2_normalize(v_mem - p_y)
        read_scores = torch.einsum("bd,bnd->bn", r_mem, R_y) / self.tau_res
        read_attn = torch.softmax(read_scores, dim=1)
        read_res = torch.einsum("bn,bnd->bd", read_attn, R_y)
        z = self.z_norm(p_y + self.eta * read_res)
        z = self._l2_normalize(z)
        logits_z = self.classifier_z(z)

        self._update_memory(q=q, logits_v=logits_v, y=y, epoch=epoch)

        out.update(
            {
                "feat_z": z,
                "logits_z": logits_z,
                "loss_proto": loss_proto,
                "loss_res": loss_res,
                "loss_div": loss_div,
            }
        )
        return out
