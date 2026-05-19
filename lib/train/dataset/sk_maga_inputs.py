import torch
import math


BONE_PARENT = torch.tensor([
    -1, 0, 1, 1, 2, 3,
    19, 20, 20, 20, 20, 20,
    22, 23, 23, 23, 23, 23,
    0, 18, 6, 0, 21, 12,
], dtype=torch.long)

BONE_REORDER = torch.tensor([
    0, 1, 2, 3, 4, 5,
    20, 7, 8, 9, 10, 11,
    23, 13, 14, 15, 16, 17,
    18, 19, 6, 21, 22, 12,
], dtype=torch.long)


def _compute_valid_mask(coords):
    xy = coords[..., :2]
    score = coords[..., 2]
    return ((score > 0) & ((xy[..., 0] != 0) | (xy[..., 1] != 0))).float()


def _ensure_joint_first(coords):
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords with shape [B,V,T,3] or [B,T,V,3], got {tuple(coords.shape)}")
    if coords.shape[1] == 24:
        return coords
    if coords.shape[2] == 24:
        return coords.permute(0, 2, 1, 3).contiguous()
    raise ValueError(f"Could not locate 24-joint dimension in coords shape {tuple(coords.shape)}")


def _apply_joint_masking(coords, cfg):
    if not bool(getattr(cfg.DATA, "JOINT_MASK_ENABLE", False)):
        return coords
    min_ratio = float(getattr(cfg.DATA, "JOINT_MASK_MIN_RATIO", 0.1))
    max_ratio = float(getattr(cfg.DATA, "JOINT_MASK_MAX_RATIO", 0.2))
    root_idx = int(getattr(cfg.DATA, "JOINT_MASK_ROOT_IDX", 0))
    mask_fill = float(getattr(cfg.DATA, "JOINT_MASK_FILL", 0.0))
    if max_ratio <= 0:
        return coords

    coords = coords.clone()
    bsz, num_joints, _, _ = coords.shape
    candidate = [i for i in range(num_joints) if i != root_idx]
    if not candidate:
        return coords
    device = coords.device
    candidate_idx = torch.tensor(candidate, device=device, dtype=torch.long)
    num_candidate = candidate_idx.numel()

    for b in range(bsz):
        ratio = float(torch.empty(1, device=device).uniform_(min_ratio, max_ratio).item())
        num_drop = int(round(num_candidate * ratio))
        if num_drop <= 0:
            continue
        perm = torch.randperm(num_candidate, device=device)[:num_drop]
        drop_idx = candidate_idx[perm]
        coords[b, drop_idx, :, :] = mask_fill
    return coords


def _apply_tiny_rotation(coords, cfg):
    if not bool(getattr(cfg.DATA, "JOINT_ROT_ENABLE", False)):
        return coords
    max_deg = float(getattr(cfg.DATA, "JOINT_ROT_MAX_DEG", 5.0))
    if max_deg <= 0:
        return coords

    coords = coords.clone()
    bsz, _, tlen, _ = coords.shape
    device = coords.device
    dtype = coords.dtype
    valid = _compute_valid_mask(coords)

    for b in range(bsz):
        angle_deg = float(torch.empty(1, device=device).uniform_(-max_deg, max_deg).item())
        angle = angle_deg * math.pi / 180.0
        cos_a = torch.tensor(math.cos(angle), device=device, dtype=dtype)
        sin_a = torch.tensor(math.sin(angle), device=device, dtype=dtype)
        rot = torch.stack([
            torch.stack([cos_a, -sin_a]),
            torch.stack([sin_a, cos_a]),
        ])  # [2,2]
        for t in range(tlen):
            mask_t = valid[b, :, t] > 0
            if mask_t.sum().item() < 1:
                continue
            pts = coords[b, mask_t, t, :2]  # [N,2]
            center = pts.mean(dim=0, keepdim=True)  # [1,2]
            pts_rot = (pts - center) @ rot.t() + center
            coords[b, mask_t, t, :2] = pts_rot
    return coords


def _apply_micro_gaussian_jitter(coords, cfg):
    if not bool(getattr(cfg.DATA, "JOINT_JITTER_ENABLE", False)):
        return coords
    std_min = float(getattr(cfg.DATA, "JOINT_JITTER_STD_MIN", 0.001))
    std_max = float(getattr(cfg.DATA, "JOINT_JITTER_STD_MAX", 0.005))
    if std_max <= 0:
        return coords

    coords = coords.clone()
    valid = _compute_valid_mask(coords).unsqueeze(-1)
    dims = int(getattr(cfg.DATA, "JOINT_JITTER_DIMS", 2))
    dims = 2 if dims <= 2 else 3

    std = float(torch.empty(1, device=coords.device).uniform_(std_min, std_max).item())
    noise = torch.randn_like(coords[..., :dims]) * std
    coords[..., :dims] = coords[..., :dims] + noise * valid

    if dims >= 3:
        coords[..., 2] = coords[..., 2].clamp_(0.0, 1.0)
    return coords


def prepare_joint_inputs(coords, cfg, apply_train_aug=False):
    coords = _ensure_joint_first(coords)
    if apply_train_aug:
        coords = _apply_joint_masking(coords, cfg)
        coords = _apply_tiny_rotation(coords, cfg)
        coords = _apply_micro_gaussian_jitter(coords, cfg)
    orig_w, orig_h = cfg.DATA.SKELETON_ORIG_SIZE
    xy = coords[..., :2].clone()
    score = coords[..., 2].clone()
    valid = _compute_valid_mask(coords)

    xy[..., 0] = 2.0 * xy[..., 0] / float(orig_w) - 1.0
    xy[..., 1] = 2.0 * xy[..., 1] / float(orig_h) - 1.0
    xy = xy * valid.unsqueeze(-1)
    score = score * valid

    skeleton = torch.cat((xy, score.unsqueeze(-1)), dim=-1)
    skeleton = skeleton.permute(0, 3, 2, 1).contiguous().unsqueeze(-1)
    index_t = torch.linspace(-1.0, 1.0, steps=skeleton.shape[2], device=skeleton.device)
    index_t = index_t.unsqueeze(0).expand(skeleton.shape[0], -1)
    return skeleton, index_t


def prepare_bone_inputs(coords, cfg):
    coords = _ensure_joint_first(coords)
    orig_w, orig_h = cfg.DATA.SKELETON_ORIG_SIZE
    xy = coords[..., :2].clone()
    score = coords[..., 2].clone()
    valid = _compute_valid_mask(coords)

    bone_xy = torch.zeros_like(xy)
    bone_score = torch.zeros_like(score)

    root_idx = 0
    bone_score[..., root_idx] = score[..., root_idx] * valid[..., root_idx]

    parent = BONE_PARENT.to(coords.device)
    for joint_idx in range(1, coords.shape[1]):
        parent_idx = int(parent[joint_idx].item())
        parent_valid = valid[:, parent_idx, :]
        joint_valid = valid[:, joint_idx, :]
        bone_valid = joint_valid * parent_valid
        bone_xy[:, joint_idx, :, :] = (xy[:, joint_idx, :, :] - xy[:, parent_idx, :, :]) * bone_valid.unsqueeze(-1)
        bone_score[:, joint_idx, :] = torch.minimum(score[:, joint_idx, :], score[:, parent_idx, :]) * bone_valid

    bone_xy[..., 0] = 2.0 * bone_xy[..., 0] / float(orig_w)
    bone_xy[..., 1] = 2.0 * bone_xy[..., 1] / float(orig_h)

    reorder = BONE_REORDER.to(coords.device)
    bone_xy = bone_xy.index_select(dim=1, index=reorder)
    bone_score = bone_score.index_select(dim=1, index=reorder)

    skeleton = torch.cat((bone_xy, bone_score.unsqueeze(-1)), dim=-1)
    skeleton = skeleton.permute(0, 3, 2, 1).contiguous().unsqueeze(-1)
    index_t = torch.linspace(-1.0, 1.0, steps=skeleton.shape[2], device=skeleton.device)
    index_t = index_t.unsqueeze(0).expand(skeleton.shape[0], -1)
    return skeleton, index_t


def prepare_sk_maga_inputs(coords, cfg, apply_train_aug=False):
    input_type = str(getattr(cfg.DATA, "SKELETON_INPUT_TYPE", "joint")).lower()
    if input_type == "joint":
        return prepare_joint_inputs(coords, cfg, apply_train_aug=apply_train_aug)
    if input_type == "bone":
        return prepare_bone_inputs(coords, cfg)
    raise ValueError(f"Unsupported SKELETON_INPUT_TYPE: {input_type}")
