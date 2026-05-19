import torch
from typing import Callable, Tuple


def get_merge_func(
        metric: torch.Tensor,
        bg_kept_number: int
) -> Tuple[Callable, Callable]:
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        nu_bg_metric, u_bg_metric = metric[:, bg_kept_number:], metric[:, :bg_kept_number]
        merge_number = nu_bg_metric.shape[1]

        score = nu_bg_metric @ u_bg_metric.transpose(-1, -2)
        node_max, node_idx = score.max(dim=-1)
        dst_idx = node_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src = x[:, bg_kept_number:]
        dst = x[:, :bg_kept_number]
        n, _, c = src.shape
        dst = dst.scatter_reduce(dim=-2, index=dst_idx.expand(n, merge_number, c), src=src, reduce=mode)

        return dst

    return merge, node_max