"""Fused split of the joint flex-attention output into the VLM and action-expert streams.

With enable_fused_qkv_prep, flex attention returns the joint output O [B, Hq, L, D] (L = Lv + Le, no padding), and the
decoder layer of each stream consumes its own part of it. Without this op, flex_attention_prepared transposes and
reshapes O to [B, L, Hq*D] (one copy of all tokens), the VLM layer casts all L tokens to bf16 and slices [:, 0:Lv], and
the expert layer slices [:, Lv:L] (the output_atten branch of the two decoder layers). In backward each slice becomes a
full-size zero tensor plus a copy, the two are added, and the cast and the transpose are undone.

joint_attn_split(O, Lv, Le) does the same with one Triton kernel per direction:
    forward : att_vlm [B, Lv, Hq*D] in vlm_dtype (bf16 by default, round to nearest even) and
              att_exp [B, Le, Hq*D] in exp_dtype (O's dtype by default)
    backward: dO [B, Hq, L_pad, D] in O's dtype; rows >= Lv + Le (padding, if any) are zero
Both directions are copies plus at most one rounding and are bit-identical to the eager chain
(joint_attn_split_reference), including infinities and the sign of zero; NaN stays NaN. The layers then get their
own tensor with start = 0 and end = L_i, so their cast and slice are no-ops.

On CUDA the Triton kernels run; on CPU an equivalent torch implementation runs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


try:  # Triton is optional so that the CPU path works without it
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

_BLOCK_T = 16  # tokens per program
_NUM_WARPS = 8
_DTYPES = (torch.float32, torch.bfloat16)


if triton is not None:
    from .joint_qkv_prep import _f32_to_bf16_bits_rtne

    @triton.jit
    def _split_fwd(
        o_ptr, v_ptr, e_ptr,
        Lv, L, H,
        s_ob, s_oh, s_ot,  # O [B, H, L_pad, D] strides (d stride 1)
        s_vb, s_vt, s_eb, s_et,  # outputs [B, L_i, H*D] (d stride 1): batch and token strides, head offset h*D
        VLM_BF16: tl.constexpr, EXP_BF16: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, D)
        in_l = t < L
        is_v = t < Lv
        mv = in_l & is_v
        me = in_l & (t >= Lv)
        x = tl.load(
            o_ptr + b * s_ob + h * s_oh + t[:, None] * s_ot + d[None, :], mask=in_l[:, None], other=0.0
        ).to(tl.float32)
        tv = tl.where(is_v, t, 0)
        te = tl.where(is_v, 0, t - Lv)
        offv = b * s_vb + tv[:, None] * s_vt + h * D + d[None, :]
        offe = b * s_eb + te[:, None] * s_et + h * D + d[None, :]
        if VLM_BF16:
            tl.store(v_ptr.to(tl.pointer_type(tl.int16)) + offv, _f32_to_bf16_bits_rtne(x), mask=mv[:, None])
        else:
            tl.store(v_ptr + offv, x, mask=mv[:, None])
        if EXP_BF16:
            tl.store(e_ptr.to(tl.pointer_type(tl.int16)) + offe, _f32_to_bf16_bits_rtne(x), mask=me[:, None])
        else:
            tl.store(e_ptr + offe, x, mask=me[:, None])

    @triton.jit
    def _split_bwd(
        gv_ptr, ge_ptr, go_ptr,
        Lv, L, Lpad, H,
        s_vb, s_vt, s_eb, s_et,
        s_ob, s_oh, s_ot,
        OUT_BF16: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, D)
        in_pad = t < Lpad
        is_v = t < Lv
        in_e = (t >= Lv) & (t < L)
        tv = tl.where(is_v, t, 0)
        te = tl.where(in_e, t - Lv, 0)
        gv = tl.load(
            gv_ptr + b * s_vb + tv[:, None] * s_vt + h * D + d[None, :], mask=is_v[:, None], other=0.0
        ).to(tl.float32)
        ge = tl.load(
            ge_ptr + b * s_eb + te[:, None] * s_et + h * D + d[None, :], mask=in_e[:, None], other=0.0
        ).to(tl.float32)
        # the masked loads give +0.0, so this is gv on VLM rows, ge on expert rows and 0 on padding rows; adding
        # (instead of selecting) also turns -0.0 into +0.0, as autograd's sum of the two slice gradients does
        g = gv + ge
        offo = b * s_ob + h * s_oh + t[:, None] * s_ot + d[None, :]
        if OUT_BF16:
            tl.store(go_ptr.to(tl.pointer_type(tl.int16)) + offo, _f32_to_bf16_bits_rtne(g), mask=in_pad[:, None])
        else:
            tl.store(go_ptr + offo, g, mask=in_pad[:, None])


def _fwd_torch(o, Lv, Le, vlm_dtype, exp_dtype):
    B, H, _, D = o.shape
    att = o[:, :, :Lv + Le].transpose(1, 2).reshape(B, Lv + Le, H * D)
    return att[:, :Lv].to(vlm_dtype).contiguous(), att[:, Lv:].to(exp_dtype).contiguous()


def _bwd_torch(gv, ge, Lv, Le, Lp, H, D, dtype):
    B = gv.shape[0]
    g = torch.zeros(B, Lp, H, D, dtype=dtype, device=gv.device)
    g[:, :Lv] += gv.to(dtype).view(B, Lv, H, D)  # += like autograd's sum of the slice gradients (-0.0 -> +0.0)
    g[:, Lv:Lv + Le] += ge.to(dtype).view(B, Le, H, D)
    return g.transpose(1, 2).contiguous()


def _fwd_triton(o, Lv, Le, vlm_dtype, exp_dtype):
    B, H, _, D = o.shape
    L = Lv + Le
    v = torch.empty(B, Lv, H * D, dtype=vlm_dtype, device=o.device)
    e = torch.empty(B, Le, H * D, dtype=exp_dtype, device=o.device)
    _split_fwd[(B * H, triton.cdiv(L, _BLOCK_T))](
        o, v, e, Lv, L, H,
        o.stride(0), o.stride(1), o.stride(2), v.stride(0), v.stride(1), e.stride(0), e.stride(1),
        VLM_BF16=(vlm_dtype == torch.bfloat16), EXP_BF16=(exp_dtype == torch.bfloat16), D=D, BLOCK_T=_BLOCK_T,
        num_warps=_NUM_WARPS,
    )
    return v, e


def _bwd_triton(gv, ge, Lv, Le, Lp, H, D, dtype):
    B = gv.shape[0]
    go = torch.empty(B, H, Lp, D, dtype=dtype, device=gv.device)
    _split_bwd[(B * H, triton.cdiv(Lp, _BLOCK_T))](
        gv, ge, go, Lv, Lv + Le, Lp, H,
        gv.stride(0), gv.stride(1), ge.stride(0), ge.stride(1), go.stride(0), go.stride(1), go.stride(2),
        OUT_BF16=(dtype == torch.bfloat16), D=D, BLOCK_T=_BLOCK_T, num_warps=_NUM_WARPS,
    )
    return go


def _use_triton(t: torch.Tensor) -> bool:
    return triton is not None and t.is_cuda


class _JointAttnSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, Lv, Le, vlm_dtype, exp_dtype):
        assert o.dim() == 4 and o.stride(3) == 1 and Lv + Le <= o.shape[2], (o.shape, o.stride(), Lv, Le)
        assert o.dtype in _DTYPES and vlm_dtype in _DTYPES and exp_dtype in _DTYPES, (o.dtype, vlm_dtype, exp_dtype)
        ctx.Lv, ctx.Le, ctx.shape, ctx.dtype = Lv, Le, tuple(o.shape), o.dtype
        return (_fwd_triton if _use_triton(o) else _fwd_torch)(o, Lv, Le, vlm_dtype, exp_dtype)

    @staticmethod
    def backward(ctx, gv, ge):
        B, H, Lp, D = ctx.shape
        if gv is None:
            gv = torch.zeros(B, ctx.Lv, H * D, dtype=ctx.dtype, device=ge.device)
        if ge is None:
            ge = torch.zeros(B, ctx.Le, H * D, dtype=ctx.dtype, device=gv.device)
        gv, ge = gv.contiguous(), ge.contiguous()
        go = (_bwd_triton if _use_triton(gv) else _bwd_torch)(gv, ge, ctx.Lv, ctx.Le, Lp, H, D, ctx.dtype)
        return go, None, None, None, None


@torch.compiler.disable
def joint_attn_split(
    o: torch.Tensor,
    Lv: int,
    Le: int,
    vlm_dtype: torch.dtype = torch.bfloat16,
    exp_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """See module docstring. o: flex output [B, Hq, L_pad, D] with L_pad >= Lv + Le (fp32 or bf16). Returns
    att_vlm [B, Lv, Hq*D] in vlm_dtype and att_exp [B, Le, Hq*D] in exp_dtype (default: o's dtype)."""
    return _JointAttnSplit.apply(o, Lv, Le, vlm_dtype, o.dtype if exp_dtype is None else exp_dtype)


# --------------------------------------------------------------------------------------------------------------
# Reference: the eager chain (flex_attention_prepared's transpose + reshape, the layers' cast and slice)
# --------------------------------------------------------------------------------------------------------------
def joint_attn_split_reference(
    o: torch.Tensor,
    Lv: int,
    Le: int,
    vlm_dtype: torch.dtype = torch.bfloat16,
    exp_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, _, D = o.shape
    L = Lv + Le
    att = o[:, :, :L, :].transpose(1, 2).contiguous().reshape(B, L, H * D)
    return att.to(vlm_dtype)[:, 0:Lv], att.to(o.dtype if exp_dtype is None else exp_dtype)[:, Lv:L]
