"""Scheduling helpers for the per-layer FSDP2 units of the joint VLM / action-expert stack.

fsdp_no_reshard_after_backward: keep each layer's unsharded parameters through the whole backward, so the backward of
   its q/k/v call (the second segment of the layer's backward) does not all-gather them again. The caller must call
   after_backward() after every loss.backward(), before clip_grad_norm_ / optimizer.step(): it drains the
   reduce-scatter queue (when ReduceScatterInflight is installed) and reshards the layers.
fsdp_forward_prefetch_depth k: before the forward of every micro-batch (from global step 2 on), all-gather the first k
   layers of both towers; inside the layer loop FSDP keeps a lookahead of k via set_modules_to_forward_prefetch, so the
   all-gather chain runs about k layers ahead of the compute. The caller decides when this prologue runs
   (fsdp_prologue_under): right before the model forward ('vit'), after prefix_units (the root and any other units the
   forward calls before the layers, in call order), so the layers' all-gathers run under the ViT / prefix compute and
   none of the prefix units queues behind them (the NCCL communicator runs collectives in issue order); or at the start
   of the micro-batch ('teacher'), so its all-gathers overlap the online teacher models.
fsdp_rs_inflight_depth: let up to that many of FSDP2's backward reduce-scatters be in flight instead of one
   (ReduceScatterInflight below); after_backward() drains the queue.
"""
import collections
import inspect
from typing import List, Optional

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _pg


class ReduceScatterInflight:
    """Let up to `depth` FSDP2 reduce-scatters be in flight instead of one.

    FSDP2's FSDPParamGroup.post_backward makes the compute stream wait, before issuing a reduce-scatter, for
    (a) the previous reduce-scatter (any group) so that its input buffer can be freed, and
    (b) this group's previous post-reduce event.
    Both are stream-ordering safety waits, not data dependencies: the RS inputs are only read on the RS stream, and
    the per-group gradient accumulation happens on the RS stream in issue order.  We take the two objects out of
    FSDP's hands before the original post_backward runs (so it finds nothing to wait for), keep them in a FIFO, and
    only when more than `depth - 1` are pending do we make the compute stream wait for the oldest and drop it
    (freeing its input).  `drain()` must run after loss.backward() and before anything reads gradients.
    The FIFO counts post_backward calls that had something to move, including calls that do not reduce (FSDP2's
    end-of-backward calls for units whose hook did not run); at depth 1 every wait FSDP2 makes still happens, possibly
    earlier.
    """

    def __init__(self, depth: int):
        if depth < 1:
            raise ValueError(f"reduce-scatter depth must be >= 1, got {depth}")
        self.depth = int(depth)
        # [(ReduceScatterState | None, post_reduce_event | None)]
        self.pending: collections.deque = collections.deque()
        self.max_pending = 0

    @staticmethod
    def check(groups) -> None:
        """Raise unless the deferred waits are safe: torch 2.8's post_backward, and for every FSDPParamGroup no HSDP
        all-reduce and no all-reduce hook (so the post-reduce runs on the reduce-scatter stream), reduce-scatter
        buffers from the caching allocator and no CPU offload."""
        if not torch.__version__.startswith("2.8."):
            raise RuntimeError(f"fsdp_rs_inflight_depth > 1 defers waits inside FSDP2's FSDPParamGroup.post_backward, "
                               f"which was validated against torch 2.8 only; this is torch {torch.__version__}")
        src = inspect.getsource(_pg.FSDPParamGroup.post_backward)
        if "reduce_scatter_state" not in src or "_wait_for_post_backward" not in src:
            raise RuntimeError("FSDP2's FSDPParamGroup.post_backward does not match torch 2.8's "
                               "(or is wrapped already): "
                               "cannot install the reduce-scatter queue")
        for g in groups:
            unsupported = [what for what, bad in (
                ("HSDP", g._is_hsdp),
                ("an all-reduce hook", g._all_reduce_hook is not None),
                ("reduce-scatter buffers from the process group", g.allocate_memory_from_process_group),
                ("CPU offload", any(getattr(p, "offload_to_cpu", False) for p in g.fsdp_params)),
            ) if bad]
            if unsupported:
                raise ValueError(f"fsdp_rs_inflight_depth > 1 does not support {' / '.join(unsupported)} "
                                 f"(FSDP2 unit of {type(g.modules[0]).__name__})")

    def install(self, groups) -> None:
        """Check `groups` (every FSDP2 parameter group in the process: the patch is class-wide), then wrap
        FSDPParamGroup.post_backward. Once per process."""
        if getattr(_pg.FSDPParamGroup, "_lingbot_rs_inflight_installed", False):
            raise RuntimeError("ReduceScatterInflight installed twice")
        self.check(groups)
        orig = _pg.FSDPParamGroup.post_backward
        inflight = self

        def post_backward(group, *unused):
            ctx = group.comm_ctx
            prev_state, ctx.reduce_scatter_state = ctx.reduce_scatter_state, None     # (a)
            prev_evt, group._post_reduce_event = group._post_reduce_event, None       # (b)
            if prev_state is not None or prev_evt is not None:
                inflight.pending.append((prev_state, prev_evt))
                inflight.max_pending = max(inflight.max_pending, len(inflight.pending))
            while len(inflight.pending) > inflight.depth - 1:
                inflight._wait_oldest()
            return orig(group, *unused)

        _pg.FSDPParamGroup.post_backward = post_backward
        _pg.FSDPParamGroup._lingbot_rs_inflight_installed = True

    def _wait_oldest(self) -> None:
        state, evt = self.pending.popleft()
        cur = torch.cuda.current_stream()
        if state is not None and state.event is not None:
            cur.wait_event(state.event)
        if evt is not None:
            cur.wait_event(evt)
        # dropping `state` here frees the reduce-scatter input, ordered after the RS on the compute stream

    def drain(self) -> None:
        """After loss.backward(), before clip / optimizer: wait for everything still pending."""
        while self.pending:
            self._wait_oldest()


class LayerCommScheduler:
    def __init__(self, vlm_layers: List[torch.nn.Module], expert_layers: List[torch.nn.Module],
                 prefetch_depth: int = 0, no_reshard_after_backward: bool = False,
                 prefix_units: Optional[List[torch.nn.Module]] = None, rs_inflight_depth: int = 1):
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
        self.rs_inflight_depth = int(rs_inflight_depth)
        if not 0 <= self.depth < self.n:
            raise ValueError(f"fsdp_forward_prefetch_depth must be in [0, {self.n - 1}] for {self.n} layers, "
                             f"got {self.depth}")
        if self.rs_inflight_depth < 1:
            raise ValueError(f"fsdp_rs_inflight_depth must be >= 1, got {self.rs_inflight_depth}")
        self.rs_inflight: Optional[ReduceScatterInflight] = None

    # ---- one-time configuration, right after build_parallelize_model
    def configure(self, all_groups=()) -> None:
        """all_groups: every FSDPParamGroup in the process, checked before the class-wide post_backward wrapper is
        installed (only used when rs_inflight_depth > 1)."""
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
        if self.rs_inflight_depth > 1:
            groups = [m._get_fsdp_state()._fsdp_param_group for m in self.vlm + self.exp]
            seen = {id(g) for g in groups}
            groups += [g for g in all_groups if id(g) not in seen]
            self.rs_inflight = ReduceScatterInflight(self.rs_inflight_depth)
            self.rs_inflight.install(groups)

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
        if self.rs_inflight is not None:
            self.rs_inflight.drain()
        if self.no_reshard:
            for m in self.vlm + self.exp:
                m.reshard()
