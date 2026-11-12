"""Scheduling helpers for the per-layer FSDP2 units of the joint VLM / action-expert stack.

fsdp_no_reshard_after_backward: keep each layer's unsharded parameters through the whole backward, so the backward of
   its q/k/v call (the second segment of the layer's backward) does not all-gather them again. The caller must call
   after_backward() after every loss.backward(), before clip_grad_norm_ / optimizer.step(): it reshards the layers.
fsdp_forward_prefetch_depth k: before the forward of every micro-batch (from global step 2 on), all-gather the first k
   layers of both towers; inside the layer loop FSDP keeps a lookahead of k via set_modules_to_forward_prefetch, so the
   all-gather chain runs about k layers ahead of the compute. The caller decides when this prologue runs
   (fsdp_prologue_under): right before the model forward ('vit'), after prefix_units (the root and any other units the
   forward calls before the layers, in call order), so the layers' all-gathers run under the ViT / prefix compute and
   none of the prefix units queues behind them (the NCCL communicator runs collectives in issue order); or at the start
   of the micro-batch ('teacher'), so its all-gathers overlap the online teacher models.
"""
from typing import List, Optional

import torch
from torch.distributed.fsdp import FSDPModule


class LayerCommScheduler:
    def __init__(self, vlm_layers: List[torch.nn.Module], expert_layers: List[torch.nn.Module],
                 prefetch_depth: int = 0, no_reshard_after_backward: bool = False,
                 prefix_units: Optional[List[torch.nn.Module]] = None):
        if len(vlm_layers) != len(expert_layers):
            raise ValueError(f"{len(vlm_layers)} VLM layers but {len(expert_layers)} expert layers")
        for m in list(vlm_layers) + list(expert_layers) + list(prefix_units or []):
            if not isinstance(m, FSDPModule):
                raise ValueError(f"{type(m).__name__} is not an FSDP2 module")
        self.vlm, self.exp = list(vlm_layers), list(expert_layers)
        self.n = len(self.vlm)
        self.depth = int(prefetch_depth)
        self.no_reshard = bool(no_reshard_after_backward)
        self.prefix = list(prefix_units or [])
        if not 0 <= self.depth < self.n:
            raise ValueError(f"fsdp_forward_prefetch_depth must be in [0, {self.n - 1}] for {self.n} layers, "
                             f"got {self.depth}")

    # ---- one-time configuration, right after build_parallelize_model
    def configure(self) -> None:
        if self.no_reshard:
            for m in self.vlm + self.exp:
                m.set_reshard_after_backward(False, recurse=False)
        if self.depth > 0:
            for i in range(self.n - self.depth):
                # issued in layer i's pre-forward (its first call, compute_kqv=True), on FSDP's own all-gather streams;
                # appended to any forward-prefetch list build_parallelize_model may have set; all-gathers of units the
                # prologue already unsharded are no-ops
                ahead = [self.vlm[i + self.depth], self.exp[i + self.depth]]
                current = [s._modules[0] for s in self.vlm[i]._get_fsdp_state()._states_to_forward_prefetch]
                new = [m for m in ahead if not any(m is c for c in current)]
                self.vlm[i].set_modules_to_forward_prefetch(current + new)

    # ---- prefetch prologue, at the start of every forward (micro-batch) from global step 2 on. Step 1 is skipped so
    #      that it runs exactly as without prefetch while FSDP2 initializes lazily; FSDPModule.unshard would also work
    #      before the lazy init (e.g. right after a resume).
    def prologue(self) -> None:
        if self.depth == 0:
            return
        for m in self.prefix:
            m.unshard(async_op=True)
        for i in range(self.depth):
            self.vlm[i].unshard(async_op=True)
            self.exp[i].unshard(async_op=True)

    # ---- right after loss.backward(), before clip_grad_norm_ / optimizer.step()
    def after_backward(self) -> None:
        if self.no_reshard:
            for m in self.vlm + self.exp:
                m.reshard()
