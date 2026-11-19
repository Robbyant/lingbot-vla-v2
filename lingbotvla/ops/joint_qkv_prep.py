"""Fused input preparation for the joint (VLM + action-expert) flex attention.

Per layer the unfused path runs, for each of q/k/v (QwenvlWithExpertV2Model.forward: .float(), cat, apply_mrope;
flex_attention_with_block_mask: transpose + F.pad):

    x = torch.cat([x_vlm.float(), x_exp], dim=1)                 # [B, Lv+Le, H, D]  fp32
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)  # RoPE (q, k only)
    x = F.pad(x.transpose(1, 2), (0, 0, 0, L_pad - L))            # [B, H, L_pad, D]

i.e. 3-4 full passes over 150-200 MB tensors per layer (micro batch 32, RoboTwin recipe).  `joint_qkv_prep` does the
same with ONE kernel per tensor: it reads the VLM stream (bf16) and the expert stream (fp32) once, applies RoPE in fp32
and writes the flex layout [B, H, L, D] directly, without concat and without padding (torch 2.8 flex supports
L % 128 != 0).  The backward is one kernel per tensor: inverse rotation (RoPE is orthogonal, so
d x = g*cos - rotate_half(g*sin)), split back into the two streams, VLM part cast back to its dtype.  Only cos/sin are
saved for backward.

    Q, K, V = joint_qkv_prep(q_vlm, k_vlm, v_vlm, q_exp, k_exp, v_exp, cos, sin)
        q_vlm/k_vlm/v_vlm : [B, Lv, Hq|Hkv, D]  (bf16 in training; already through q_norm/k_norm)
        q_exp/k_exp/v_exp : [B, Le, Hq|Hkv, D]  (fp32)
        cos, sin          : [B, Lv+Le, D]       fp32, from Qwen3VLTextRotaryEmbedding (both halves identical)
        Q                 : [B, Hq, L, D]  contiguous, dtype = out_dtype (default fp32); K, V: [B, Hkv, L, D]

On CUDA the Triton kernels are used; on CPU an equivalent torch implementation of the same math runs.
RoPE follows HF exactly: rotate_half(x) = cat(-x2, x1);  y1 = x1*c - x2*s,  y2 = x2*c + x1*s  (c, s = first half).
The kernels are compiled with enable_fp_fusion=False: with Triton's default mul+add -> FMA contraction ~24% of the
rotated q/k elements differ from eager in the last fp32 bit, and ~2e-5 of the bf16 VLM grads land on the neighbouring
bf16 value. Without it, forward and backward are bit-identical to the eager chain (and the speed is the same).
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

_BLOCK_T = 16    # tokens per program
_NUM_WARPS = 4


if triton is not None:

    @triton.jit
    def _joint_prep_fwd(
        xv_ptr, xe_ptr, cos_ptr, sin_ptr, out_ptr,
        Lv, L, H,
        s_vb, s_vt, s_vh,
        s_eb, s_et, s_eh,
        s_cb, s_ct,
        s_ob, s_oh, s_ot,
        ROPE: tl.constexpr, HALF: tl.constexpr, BLOCK_T: tl.constexpr, OUT_BF16: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, HALF)
        in_l = t < L
        is_v = t < Lv
        mv = in_l & is_v
        me = in_l & (t >= Lv)
        tv = tl.where(is_v, t, 0)
        te = tl.where(is_v, 0, t - Lv)
        offv = b * s_vb + tv[:, None] * s_vt + h * s_vh + d[None, :]
        offe = b * s_eb + te[:, None] * s_et + h * s_eh + d[None, :]
        xv1 = tl.load(xv_ptr + offv, mask=mv[:, None], other=0.0).to(tl.float32)
        xv2 = tl.load(xv_ptr + offv + HALF, mask=mv[:, None], other=0.0).to(tl.float32)
        xe1 = tl.load(xe_ptr + offe, mask=me[:, None], other=0.0).to(tl.float32)
        xe2 = tl.load(xe_ptr + offe + HALF, mask=me[:, None], other=0.0).to(tl.float32)
        x1 = tl.where(is_v[:, None], xv1, xe1)
        x2 = tl.where(is_v[:, None], xv2, xe2)
        if ROPE:
            offc = b * s_cb + t[:, None] * s_ct + d[None, :]
            c = tl.load(cos_ptr + offc, mask=in_l[:, None], other=0.0).to(tl.float32)
            s = tl.load(sin_ptr + offc, mask=in_l[:, None], other=0.0).to(tl.float32)
            y1 = x1 * c - x2 * s
            y2 = x2 * c + x1 * s
        else:
            y1 = x1
            y2 = x2
        offo = b * s_ob + h * s_oh + t[:, None] * s_ot + d[None, :]
        if OUT_BF16:
            # explicit RTNE so that the bf16 output is identical to torch's .to(bfloat16) on every backend
            oi = out_ptr.to(tl.pointer_type(tl.int16))
            tl.store(oi + offo, _f32_to_bf16_bits_rtne(y1), mask=in_l[:, None])
            tl.store(oi + offo + HALF, _f32_to_bf16_bits_rtne(y2), mask=in_l[:, None])
        else:
            tl.store(out_ptr + offo, y1, mask=in_l[:, None])
            tl.store(out_ptr + offo + HALF, y2, mask=in_l[:, None])

    @triton.jit
    def _f32_to_bf16_bits_rtne(x):
        # round-to-nearest-even fp32 -> bf16, returned as int16 bit pattern (same result as torch's .to(bfloat16);
        # makes the VLM-side gradient bit-identical to eager on every backend, including TRITON_INTERPRET=1)
        u = x.to(tl.int32, bitcast=True)
        lsb = (u >> 16) & 1
        r = (u + 0x7FFF + lsb) >> 16
        # NaN stays NaN (0x7FFF, which CUDA's cvt.rn.bf16.f32, i.e. eager .to(bfloat16) on the GPU, gives); the carry
        # above would turn it into +-0 or +-inf. Integer test: Triton compiles float compares assuming no NaNs.
        r = tl.where((u & 0x7FFFFFFF) > 0x7F800000, 0x7FFF, r)
        return r.to(tl.int16)

    @triton.jit
    def _joint_prep_bwd(
        g_ptr, cos_ptr, sin_ptr, dv_ptr, de_ptr,
        Lv, L, H,
        s_gb, s_gh, s_gt,
        s_cb, s_ct,
        s_vb, s_vt, s_vh,
        s_eb, s_et, s_eh,
        ROPE: tl.constexpr, HALF: tl.constexpr, BLOCK_T: tl.constexpr, VLM_BF16: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, HALF)
        in_l = t < L
        is_v = t < Lv
        mv = in_l & is_v
        me = in_l & (t >= Lv)
        tv = tl.where(is_v, t, 0)
        te = tl.where(is_v, 0, t - Lv)
        offg = b * s_gb + h * s_gh + t[:, None] * s_gt + d[None, :]
        g1 = tl.load(g_ptr + offg, mask=in_l[:, None], other=0.0).to(tl.float32)
        g2 = tl.load(g_ptr + offg + HALF, mask=in_l[:, None], other=0.0).to(tl.float32)
        if ROPE:
            offc = b * s_cb + t[:, None] * s_ct + d[None, :]
            c = tl.load(cos_ptr + offc, mask=in_l[:, None], other=0.0).to(tl.float32)
            s = tl.load(sin_ptr + offc, mask=in_l[:, None], other=0.0).to(tl.float32)
            dx1 = g1 * c + g2 * s
            dx2 = g2 * c - g1 * s
        else:
            dx1 = g1
            dx2 = g2
        offv = b * s_vb + tv[:, None] * s_vt + h * s_vh + d[None, :]
        offe = b * s_eb + te[:, None] * s_et + h * s_eh + d[None, :]
        if VLM_BF16:
            dv_i16 = dv_ptr.to(tl.pointer_type(tl.int16))
            tl.store(dv_i16 + offv, _f32_to_bf16_bits_rtne(dx1), mask=mv[:, None])
            tl.store(dv_i16 + offv + HALF, _f32_to_bf16_bits_rtne(dx2), mask=mv[:, None])
        else:
            tl.store(dv_ptr + offv, dx1, mask=mv[:, None])
            tl.store(dv_ptr + offv + HALF, dx2, mask=mv[:, None])
        tl.store(de_ptr + offe, dx1, mask=me[:, None])
        tl.store(de_ptr + offe + HALF, dx2, mask=me[:, None])


def _check(xv: torch.Tensor, xe: torch.Tensor, cos: Optional[torch.Tensor], sin: Optional[torch.Tensor]) -> None:
    assert xv.dim() == 4 and xe.dim() == 4, (xv.shape, xe.shape)
    B, Lv, H, D = xv.shape
    assert xe.shape[0] == B and xe.shape[2] == H and xe.shape[3] == D, (xv.shape, xe.shape)
    assert D % 2 == 0 and xv.stride(3) == 1 and xe.stride(3) == 1, "last dim must be contiguous"
    if cos is not None:
        L = Lv + xe.shape[1]
        assert cos.shape == (B, L, D) and sin.shape == (B, L, D), (cos.shape, sin.shape, (B, L, D))
        assert cos.stride(2) == 1 and sin.stride(2) == 1


# --------------------------------------------------------------------------------------------------------------
# torch implementation of exactly the kernel math (CPU path / reference for the Triton path)
# --------------------------------------------------------------------------------------------------------------
def _fwd_torch(xv, xe, cos, sin, rope: bool, out_dtype) -> torch.Tensor:
    cdt = torch.float64 if xe.dtype == torch.float64 else torch.float32
    x = torch.cat([xv.to(cdt), xe.to(cdt)], dim=1)  # [B, L, H, D]
    if rope:
        half = x.shape[-1] // 2
        c = cos[..., :half].to(cdt).unsqueeze(2)
        s = sin[..., :half].to(cdt).unsqueeze(2)
        x1, x2 = x[..., :half], x[..., half:]
        x = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return x.transpose(1, 2).contiguous().to(out_dtype)


def _bwd_torch(g, cos, sin, Lv: int, rope: bool, dv_dtype, de_dtype):
    cdt = torch.float64 if g.dtype == torch.float64 else torch.float32
    g = g.to(cdt).transpose(1, 2)  # [B, L, H, D]
    if rope:
        half = g.shape[-1] // 2
        c = cos[..., :half].to(cdt).unsqueeze(2)
        s = sin[..., :half].to(cdt).unsqueeze(2)
        g1, g2 = g[..., :half], g[..., half:]
        g = torch.cat([g1 * c + g2 * s, g2 * c - g1 * s], dim=-1)
    return g[:, :Lv].to(dv_dtype).contiguous(), g[:, Lv:].to(de_dtype).contiguous()


# --------------------------------------------------------------------------------------------------------------
# Triton launchers
# --------------------------------------------------------------------------------------------------------------
def _fwd_triton(xv, xe, cos, sin, rope: bool, out_dtype) -> torch.Tensor:
    B, Lv, H, D = xv.shape
    L = Lv + xe.shape[1]
    out = torch.empty(B, H, L, D, dtype=out_dtype, device=xv.device)
    c = cos if rope else out
    s = sin if rope else out
    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_T"]))  # noqa: E731
    _joint_prep_fwd[grid](
        xv, xe, c, s, out, Lv, L, H,
        xv.stride(0), xv.stride(1), xv.stride(2),
        xe.stride(0), xe.stride(1), xe.stride(2),
        c.stride(0) if rope else 0, c.stride(1) if rope else 0,
        out.stride(0), out.stride(1), out.stride(2),
        ROPE=rope, HALF=D // 2, BLOCK_T=_BLOCK_T, OUT_BF16=(out_dtype == torch.bfloat16), num_warps=_NUM_WARPS,
        enable_fp_fusion=False,  # no FMA contraction: bit-identical to eager (see module docstring)
    )
    return out


def _bwd_triton(g, cos, sin, Lv: int, rope: bool, dv_dtype, de_dtype):
    if g.stride(3) != 1:
        g = g.contiguous()
    B, H, L, D = g.shape
    Le = L - Lv
    dv = torch.empty(B, Lv, H, D, dtype=dv_dtype, device=g.device)
    de = torch.empty(B, Le, H, D, dtype=de_dtype, device=g.device)
    c = cos if rope else g
    s = sin if rope else g
    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_T"]))  # noqa: E731
    _joint_prep_bwd[grid](
        g, c, s, dv, de, Lv, L, H,
        g.stride(0), g.stride(1), g.stride(2),
        c.stride(0) if rope else 0, c.stride(1) if rope else 0,
        dv.stride(0), dv.stride(1), dv.stride(2),
        de.stride(0), de.stride(1), de.stride(2),
        ROPE=rope, HALF=D // 2, BLOCK_T=_BLOCK_T, VLM_BF16=(dv_dtype == torch.bfloat16), num_warps=_NUM_WARPS,
        enable_fp_fusion=False,  # no FMA contraction: bit-identical to eager (see module docstring)
    )
    return dv, de


def _use_triton(t: torch.Tensor) -> bool:
    return triton is not None and t.is_cuda


def _fwd(xv, xe, cos, sin, rope, out_dtype):
    _check(xv, xe, cos if rope else None, sin if rope else None)
    return (_fwd_triton if _use_triton(xv) else _fwd_torch)(xv, xe, cos, sin, rope, out_dtype)


def _bwd(g, cos, sin, Lv, rope, dv_dtype, de_dtype):
    return (_bwd_triton if _use_triton(g) else _bwd_torch)(g, cos, sin, Lv, rope, dv_dtype, de_dtype)


class _JointQKVPrep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_v, k_v, v_v, q_e, k_e, v_e, cos, sin, out_dtype):
        ctx.save_for_backward(cos, sin)
        ctx.Lv = q_v.shape[1]
        ctx.dtypes = (q_v.dtype, q_e.dtype)
        Q = _fwd(q_v, q_e, cos, sin, True, out_dtype)
        K = _fwd(k_v, k_e, cos, sin, True, out_dtype)
        V = _fwd(v_v, v_e, None, None, False, out_dtype)
        return Q, K, V

    @staticmethod
    def backward(ctx, dQ, dK, dV):
        cos, sin = ctx.saved_tensors
        dv_dtype, de_dtype = ctx.dtypes
        dq_v, dq_e = _bwd(dQ, cos, sin, ctx.Lv, True, dv_dtype, de_dtype)
        dk_v, dk_e = _bwd(dK, cos, sin, ctx.Lv, True, dv_dtype, de_dtype)
        dv_v, dv_e = _bwd(dV, None, None, ctx.Lv, False, dv_dtype, de_dtype)
        return dq_v, dk_v, dv_v, dq_e, dk_e, dv_e, None, None, None


@torch.compiler.disable
def joint_qkv_prep(
    q_vlm: torch.Tensor, k_vlm: torch.Tensor, v_vlm: torch.Tensor,
    q_exp: torch.Tensor, k_exp: torch.Tensor, v_exp: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """See module docstring. Returns Q [B, Hq, L, D], K [B, Hkv, L, D], V [B, Hkv, L, D] in `out_dtype`."""
    return _JointQKVPrep.apply(q_vlm, k_vlm, v_vlm, q_exp, k_exp, v_exp, cos, sin, out_dtype)


# --------------------------------------------------------------------------------------------------------------
# Reference: the eager semantics (HF apply_rotary_pos_emb + cat + transpose [+ pad]), for numerical comparison
# --------------------------------------------------------------------------------------------------------------
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def joint_qkv_prep_reference(q_vlm, k_vlm, v_vlm, q_exp, k_exp, v_exp, cos, sin, pad_to: Optional[int] = None):
    cdt = torch.float64 if q_exp.dtype == torch.float64 else torch.float32
    q = torch.cat([q_vlm.to(cdt), q_exp.to(cdt)], dim=1)
    k = torch.cat([k_vlm.to(cdt), k_exp.to(cdt)], dim=1)
    v = torch.cat([v_vlm.to(cdt), v_exp.to(cdt)], dim=1)
    c = cos.to(cdt).unsqueeze(2)
    s = sin.to(cdt).unsqueeze(2)
    q = (q * c) + (_rotate_half(q) * s)
    k = (k * c) + (_rotate_half(k) * s)

    def lay(x):
        x = x.transpose(1, 2)
        if pad_to is not None and pad_to > x.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[2]), value=0.0)
        return x.contiguous()

    return lay(q), lay(k), lay(v)
