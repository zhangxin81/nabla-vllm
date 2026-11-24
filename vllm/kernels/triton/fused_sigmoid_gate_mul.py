# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused sigmoid-gate-multiply Triton kernel: `out = x * sigmoid(gate)`."""

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_sigmoid_gate_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    scale_ptr,
    hidden,
    x_stride_t,
    gate_stride_t,
    out_stride_t,
    BLOCK: tl.constexpr,
    QUANT_FP8: tl.constexpr,
    BROADCAST_GATE: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden

    if launch_pdl:
        # PDL: this grid may start before the preceding kernel has finished;
        # wait for it before the first load.
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + row * x_stride_t + offs, mask=mask, other=0.0)
    if BROADCAST_GATE:
        g = tl.load(gate_ptr + row * gate_stride_t).to(tl.float32)
        # Preserve the eager BF16 sigmoid output before multiplying.
        sigmoid = tl.math.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-g))
        sigmoid = sigmoid.to(gate_ptr.dtype.element_ty).to(tl.float32)
    else:
        g = tl.load(gate_ptr + row * gate_stride_t + offs, mask=mask, other=0.0)
        sigmoid = tl.sigmoid(g.to(tl.float32))
    out = x.to(tl.float32) * sigmoid
    if QUANT_FP8:
        # Static FP8 from the rounded product, like scaled_fp8_quant.
        out = out.to(x_ptr.dtype.element_ty).to(tl.float32)
        out = tl.clamp(out * tl.math.div_rn(1.0, tl.load(scale_ptr)), -448.0, 448.0)
    tl.store(
        out_ptr + row * out_stride_t + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_sigmoid_gate_mul(
    x: torch.Tensor, gate: torch.Tensor, scale: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute `x * sigmoid(gate)` in a single Triton kernel.

    Falls back to the eager `x * torch.sigmoid(gate)` sequence unless both
    inputs are same-shape, same-dtype, contiguous 2D CUDA tensors. Hopper
    also supports a BF16 `[num_tokens, 1]` gate for up to 128 rows and 4096
    output columns, preserving eager sigmoid rounding. Compilation and
    batch-invariant execution keep the eager broadcast path.

    Args:
        x: (num_tokens, hidden) tensor, e.g. attention output.
        gate: (num_tokens, hidden) or (num_tokens, 1) pre-sigmoid gate.
        scale: optional static per-tensor FP8 scale. When given, the result is
            quantized to float8_e4m3fn; ineligible inputs take the eager
            sequence followed by scaled_fp8_quant.

    Returns:
        A fresh (num_tokens, hidden) tensor with x's dtype (or FP8) and device.

    """
    broadcast_gate = (
        not torch.compiler.is_compiling()
        and x.ndim == gate.ndim == 2
        and gate.shape == (x.shape[0], 1)
        and x.shape != gate.shape
        and 0 < x.shape[0] <= 128
        and x.shape[1] <= 4096
        and x.dtype == gate.dtype == torch.bfloat16
        and x.device == gate.device
        and x.is_cuda
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(90, x.device.index)
        and scale is None
        and not envs.VLLM_BATCH_INVARIANT
    )
    if (
        x.ndim != 2
        or (x.shape != gate.shape and not broadcast_gate)
        or x.dtype != gate.dtype
        or not x.is_contiguous()
        or not gate.is_contiguous()
        or not x.is_cuda
        # The FP8 output is hardcoded to e4m3fn; ROCm GEMMs take fnuz.
        or (scale is not None and not current_platform.is_cuda())
    ):
        out = x * torch.sigmoid(gate)
        if scale is None:
            return out
        # Strided gate view (num_heads == 1 per TP rank): quantize the eager
        # result, matching the unfused chain.
        from vllm import _custom_ops as ops

        return ops.scaled_fp8_quant(out, scale, group_shape=(-1, -1))[0]

    num_tokens, hidden = x.shape
    out = torch.empty_like(x, dtype=x.dtype if scale is None else torch.float8_e4m3fn)
    if num_tokens == 0 or hidden == 0:
        return out

    if hidden <= 1024:
        block = triton.next_power_of_2(hidden)
        num_warps = 4
    elif hidden <= 4096:
        block = 1024
        num_warps = 4
    else:
        block = 2048
        num_warps = 8

    grid = (num_tokens, triton.cdiv(hidden, block))
    _fused_sigmoid_gate_mul_kernel[grid](
        x,
        gate,
        out,
        scale if scale is not None else x,
        hidden,
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        BLOCK=block,
        QUANT_FP8=scale is not None,
        BROADCAST_GATE=broadcast_gate,
        num_warps=num_warps,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out
