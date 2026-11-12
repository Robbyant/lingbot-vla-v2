"""Scheduling helpers for the per-layer FSDP2 units of the joint VLM / action-expert stack.

fsdp_no_reshard_after_backward: keep each layer's unsharded parameters through the whole backward, so the backward of
   its q/k/v call (the second segment of the layer's backward) does not all-gather them again. The caller must call
   after_backward() after every loss.backward(), before clip_grad_norm_ / optimizer.step(): it reshards the layers.
"""
from typing import List

import torch
from torch.distributed.fsdp import FSDPModule


class LayerCommScheduler:
    def __init__(self, vlm_layers: List[torch.nn.Module], expert_layers: List[torch.nn.Module],
                 no_reshard_after_backward: bool = False):
        if len(vlm_layers) != len(expert_layers):
            raise ValueError(f"{len(vlm_layers)} VLM layers but {len(expert_layers)} expert layers")
        for m in list(vlm_layers) + list(expert_layers):
            if not isinstance(m, FSDPModule):
                raise ValueError(f"{type(m).__name__} is not an FSDP2 module")
        self.vlm, self.exp = list(vlm_layers), list(expert_layers)
        self.n = len(self.vlm)
        self.no_reshard = bool(no_reshard_after_backward)

    # ---- one-time configuration, right after build_parallelize_model
    def configure(self) -> None:
        if self.no_reshard:
            for m in self.vlm + self.exp:
                m.set_reshard_after_backward(False, recurse=False)

    # ---- right after loss.backward(), before clip_grad_norm_ / optimizer.step()
    def after_backward(self) -> None:
        if self.no_reshard:
            for m in self.vlm + self.exp:
                m.reshard()
