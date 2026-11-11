"""Fused SwiGLU activation  out = silu(gate) * up  for the Qwen3-VL text MLP (one kernel forward, one backward).

HF `Qwen3VLTextMLP.forward` is `down_proj(act_fn(gate_proj(x)) * up_proj(x))`; inside the eager decoder layer (with
activation checkpointing the layers run eagerly) this runs as two kernels forward (silu, mul) and three backward
(mul-backward x2, silu_backward) over [B*L, 9728] bf16 tensors (178 MB each at micro batch 32 with the RoboTwin
recipe): 890 MB of traffic forward, 1602 MB backward, per layer.  This module does the same with one kernel each
(534 MB / 890 MB).

Numerics ("exact" mode, default): the eager chain rounds silu(gate) to bf16 before the multiply, and in backward
rounds d_silu = dout*up and d_up = dout*silu to bf16 before silu_backward.  The kernels reproduce every one of
these roundings (RTNE), use IEEE division and libdevice expf like ATen's silu kernels, so the result is
bit-identical to eager (checked against the eager chain on GPU).  `exact=False` skips the intermediate roundings
(slightly more accurate, like Liger's swiglu) and is not bit-identical.

    from lingbotvla.ops.swiglu import swiglu, apply_fused_swiglu
    y = swiglu(gate, up)                 # same shape/dtype as gate
    n = apply_fused_swiglu(model)        # patch every Qwen3VLTextMLP.forward; returns number of modules patched

On CPU an equivalent torch implementation of the same math runs.
"""
from __future__ import annotations

import os
import types
from typing import Tuple

import torch


try:
    import triton
    import triton.language as tl
    try:  # precise expf (libdevice __nv_expf), same as ATen's ::exp in silu kernels
        from triton.language.extra import libdevice as _libdevice
    except Exception:  # pragma: no cover
        from triton.language.extra.cuda import libdevice as _libdevice
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _libdevice = None

_BLOCK = 1024    # elements per program
_NUM_WARPS = 8


if triton is not None:

    @triton.jit
    def _exp_libdevice(x):
        return _libdevice.exp(x)      # precise expf (__nv_expf), same as ATen's ::exp in the silu kernels

    @triton.jit
    def _exp_builtin(x):
        return tl.exp(x)              # CPU interpreter has no libdevice; numpy exp is exact there

    _EXP = _exp_builtin if os.environ.get("TRITON_INTERPRET") == "1" else _exp_libdevice

    @triton.jit
    def _rtne_bf16_bits(x):
        u = x.to(tl.int32, bitcast=True)
        lsb = (u >> 16) & 1
        r = (u + 0x7FFF + lsb) >> 16
        # NaN stays NaN (0x7FFF, which CUDA's cvt.rn.bf16.f32, i.e. eager .to(bfloat16) on the GPU, gives); the carry
        # above would turn it into +-0 or +-inf. Integer test: Triton compiles float compares assuming no NaNs.
        r = tl.where((u & 0x7FFFFFFF) > 0x7F800000, 0x7FFF, r)
        return r.to(tl.int16)

    @triton.jit
    def _round_bf16(x):
        # fp32 -> bf16 (RTNE) -> fp32, as a value
        return (_rtne_bf16_bits(x).to(tl.int32) << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def _silu_f32(g):
        # ATen: x / (1 + exp(-x)) in fp32, IEEE division
        return tl.div_rn(g, 1.0 + _EXP(-g))

    @triton.jit
    def _swiglu_fwd(g_ptr, u_ptr, o_ptr, N, EXACT: tl.constexpr, OUT_BF16: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        g = tl.load(g_ptr + i, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + i, mask=m, other=0.0).to(tl.float32)
        s = _silu_f32(g)
        if EXACT:
            s = _round_bf16(s)
        y = s * u
        if OUT_BF16:
            tl.store(o_ptr.to(tl.pointer_type(tl.int16)) + i, _rtne_bf16_bits(y), mask=m)
        else:
            tl.store(o_ptr + i, y, mask=m)

    @triton.jit
    def _swiglu_bwd(
        do_ptr, g_ptr, u_ptr, dg_ptr, du_ptr, N,
        EXACT: tl.constexpr, OUT_BF16: tl.constexpr, BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        do = tl.load(do_ptr + i, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + i, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + i, mask=m, other=0.0).to(tl.float32)
        sig = tl.div_rn(1.0, 1.0 + _EXP(-g))
        s = _silu_f32(g)
        if EXACT:
            s = _round_bf16(s)
        du = do * s                      # mul backward: grad_other = grad * self
        ds = do * u                      # mul backward: grad_self  = grad * other
        if EXACT:
            ds = _round_bf16(ds)
        dg = (ds * sig) * (1.0 + g * (1.0 - sig))   # ATen silu_backward: dy * s * (1 + x * (1 - s))
        if OUT_BF16:
            tl.store(dg_ptr.to(tl.pointer_type(tl.int16)) + i, _rtne_bf16_bits(dg), mask=m)
            tl.store(du_ptr.to(tl.pointer_type(tl.int16)) + i, _rtne_bf16_bits(du), mask=m)
        else:
            tl.store(dg_ptr + i, dg, mask=m)
            tl.store(du_ptr + i, du, mask=m)


# ---------------------------------------------------------------- torch implementation of the same math (CPU path)
def _rb(x: torch.Tensor, bf16: bool) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32) if bf16 else x


def _fwd_torch(g, u, exact):
    bf = g.dtype == torch.bfloat16
    gf, uf = g.float(), u.float()
    s = gf / (1.0 + torch.exp(-gf))
    if exact:
        s = _rb(s, bf)
    return (s * uf).to(g.dtype)


def _bwd_torch(do, g, u, exact):
    bf = g.dtype == torch.bfloat16
    dof, gf, uf = do.float(), g.float(), u.float()
    sig = 1.0 / (1.0 + torch.exp(-gf))
    s = gf / (1.0 + torch.exp(-gf))
    if exact:
        s = _rb(s, bf)
    du = dof * s
    ds = dof * uf
    if exact:
        ds = _rb(ds, bf)
    dg = (ds * sig) * (1.0 + gf * (1.0 - sig))
    return dg.to(g.dtype), du.to(u.dtype)


# ---------------------------------------------------------------- launchers
def _fwd_triton(g, u, exact):
    o = torch.empty_like(g)
    N = g.numel()
    grid = (triton.cdiv(N, _BLOCK),)
    bf = g.dtype == torch.bfloat16
    _swiglu_fwd[grid](g, u, o, N, EXACT=(exact and bf), OUT_BF16=bf, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    return o


def _bwd_triton(do, g, u, exact):
    dg, du = torch.empty_like(g), torch.empty_like(u)
    N = g.numel()
    grid = (triton.cdiv(N, _BLOCK),)
    bf = g.dtype == torch.bfloat16
    _swiglu_bwd[grid](do, g, u, dg, du, N, EXACT=(exact and bf), OUT_BF16=bf, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    return dg, du


def _use_triton(t: torch.Tensor) -> bool:
    return triton is not None and t.is_cuda


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u, exact):
        assert g.shape == u.shape and g.dtype == u.dtype, (g.shape, u.shape, g.dtype, u.dtype)
        g = g.contiguous()
        u = u.contiguous()
        ctx.save_for_backward(g, u)
        ctx.exact = exact
        return (_fwd_triton if _use_triton(g) else _fwd_torch)(g, u, exact)

    @staticmethod
    def backward(ctx, do):
        g, u = ctx.saved_tensors
        do = do.contiguous()
        dg, du = (_bwd_triton if _use_triton(g) else _bwd_torch)(do, g, u, ctx.exact)
        return dg, du, None


@torch.compiler.disable
def swiglu(gate: torch.Tensor, up: torch.Tensor, exact: bool = True) -> torch.Tensor:
    """silu(gate) * up in one kernel (forward and backward). exact=True reproduces eager bf16 rounding."""
    return _SwiGLU.apply(gate, up, exact)


def _fused_mlp_forward(self, x):
    return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def apply_fused_swiglu(model: torch.nn.Module, class_names: Tuple[str, ...] = ("Qwen3VLTextMLP",)) -> int:
    """Replace `forward` of every MLP module whose class name is in `class_names` (default: the Qwen3-VL text MLP)
    with the fused version. Call before FSDP wrapping / torch.compile. Returns the number of modules patched."""
    n = 0
    for mod in model.modules():
        if type(mod).__name__ in class_names:
            act = getattr(getattr(mod, "config", None), "hidden_act", "silu")
            assert act == "silu", f"fused swiglu expects hidden_act='silu', got {act!r}"
            mod.forward = types.MethodType(_fused_mlp_forward, mod)
            n += 1
    return n


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The eager computation, as a reference for numerical comparison."""
    return torch.nn.functional.silu(gate) * up
