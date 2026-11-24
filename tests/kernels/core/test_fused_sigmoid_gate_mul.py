# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch

from vllm.kernels.triton.fused_sigmoid_gate_mul import fused_sigmoid_gate_mul
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# Odd token counts, non-power-of-2 and large hidden, plus an empty batch.
SHAPES = [(1, 512), (3, 1792), (17, 4096), (128, 8192), (1023, 2048), (0, 1024)]
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_sigmoid_gate_mul Triton kernel requires a CUDA/ROCm device",
)


_TOLS = {
    torch.float32: (1e-6, 1e-5),
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_fused_sigmoid_gate_mul_correctness(shape: tuple[int, int], dtype: torch.dtype):
    """The fused kernel must match `x * torch.sigmoid(gate)` and must not
    write in place."""
    set_random_seed(13)
    num_tokens, hidden = shape
    x = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")
    # Scale the gate to exercise both tails of the sigmoid.
    gate = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda") * 4

    result = fused_sigmoid_gate_mul(x, gate)
    reference = x * torch.sigmoid(gate)

    if result.numel() > 0:
        assert result.data_ptr() != x.data_ptr()
    atol, rtol = _TOLS[dtype]
    torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)


@requires_cuda
@torch.inference_mode()
def test_fused_sigmoid_gate_mul_offset_row_view():
    """A row-offset (contiguous but misaligned-pointer) view must compile and
    match eager -- this exercises the unaligned pointer specialization."""
    set_random_seed(13)
    base_x = torch.randn(65, 1023, dtype=torch.bfloat16, device="cuda")
    base_g = torch.randn(65, 1023, dtype=torch.bfloat16, device="cuda")
    x, gate = base_x[1:], base_g[1:]

    result = fused_sigmoid_gate_mul(x, gate)
    reference = x * torch.sigmoid(gate)

    torch.testing.assert_close(result, reference, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_fused_sigmoid_gate_mul_fallback_noncontiguous():
    """Non-CUDA or non-contiguous inputs take the eager fallback (CPU-safe)."""
    x = torch.randn(64, 256, dtype=torch.float32).t()
    gate = torch.randn(64, 256, dtype=torch.float32).t()
    assert not x.is_contiguous()

    result = fused_sigmoid_gate_mul(x, gate)
    expected = x * torch.sigmoid(gate)

    torch.testing.assert_close(result, expected, atol=0.0, rtol=0.0)


def _fp8_codes(q: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude e4m3 bytes -> ordered ints (+0 == -0)."""
    b = q.view(torch.uint8).int()
    return torch.where(b >= 128, -(b & 0x7F), b)


@requires_cuda
@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="FP8 (e4m3fn) output requires NVIDIA SM89+",
)
# o_proj input widths at TP1 (6144) and TP2 (3072) of a 24x256-head model.
@pytest.mark.parametrize("shape", [(1, 6144), (37, 6144), (5, 3072), (300, 4096)])
@torch.inference_mode()
def test_fused_sigmoid_gate_mul_fp8_matches_chain(shape: tuple[int, int]):
    """The FP8 output (fused o_proj input quant) equals the bf16 kernel
    followed by static scaled_fp8_quant."""
    from vllm import _custom_ops as ops

    g = torch.Generator(device="cuda").manual_seed(shape[0])
    x = torch.randn(shape, generator=g, dtype=torch.bfloat16, device="cuda") * 8
    gate = torch.randn(shape, generator=g, dtype=torch.bfloat16, device="cuda") * 4
    scale = torch.tensor([0.02], device="cuda")

    ref_q, _ = ops.scaled_fp8_quant(
        fused_sigmoid_gate_mul(x, gate), scale, group_shape=(-1, -1)
    )
    q = fused_sigmoid_gate_mul(x, gate, scale)

    assert q.dtype == torch.float8_e4m3fn
    assert ref_q.float().abs().max().item() == 448.0
    # Arch-dependent compilation flips occasional bf16 rounding boundaries;
    # codes must be within one FP8 step.
    dist = (_fp8_codes(q) - _fp8_codes(ref_q)).abs()
    assert dist.max().item() <= 1
    assert (dist > 0).float().mean().item() <= 1e-4


@requires_cuda
@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="FP8 (e4m3fn) output requires NVIDIA SM89+",
)
@torch.inference_mode()
def test_fused_sigmoid_gate_mul_fp8_strided_gate_falls_back():
    """A strided gate view (num_heads == 1 per TP rank keeps the packed qkv
    row stride through reshape) cannot run the fused kernel; the FP8 path
    must quantize the eager result instead of asserting."""
    from vllm import _custom_ops as ops

    x = torch.randn(5, 128, dtype=torch.bfloat16, device="cuda")
    packed = torch.randn(5, 256, dtype=torch.bfloat16, device="cuda")
    gate = packed[:, :128].view(5, 1, 128).reshape(5, 128)
    assert not gate.is_contiguous()
    scale = torch.tensor([0.02], device="cuda")

    ref_q, _ = ops.scaled_fp8_quant(
        x * torch.sigmoid(gate), scale, group_shape=(-1, -1)
    )
    q = fused_sigmoid_gate_mul(x, gate, scale)

    assert q.dtype == torch.float8_e4m3fn
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))


@requires_cuda
@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90),
    reason="BF16 broadcast fusion targets Hopper",
)
@torch.inference_mode()
def test_broadcast_gate_preserves_every_bf16_sigmoid_rounding():
    """Exhaust all BF16 logits, including subnormals, infinities and NaNs."""
    gate = torch.arange(65536, device="cuda").to(torch.uint16)
    gate = gate.view(torch.bfloat16).reshape(-1, 1)
    values = torch.tensor(
        [
            1,
            -1,
            0,
            float("inf"),
            float("-inf"),
            2**-133,
            2**-126,
            torch.finfo(torch.bfloat16).max,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = values.expand(65536, -1).contiguous()
    actual = torch.cat(
        [
            fused_sigmoid_gate_mul(x[i : i + 128], gate[i : i + 128])
            for i in range(0, 65536, 128)
        ]
    )
    expected = x * torch.sigmoid(gate)
    valid = ~torch.isnan(expected)
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(
        actual.view(torch.int16)[valid], expected.view(torch.int16)[valid]
    )


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 17, 128, 129])
@torch.inference_mode()
def test_broadcast_gate_graph_replays_changed_input(tokens: int):
    """Preserve BF16 products and input ownership on offset views and replay."""
    set_random_seed(43)
    x = torch.randn(tokens + 1, 2560, device="cuda", dtype=torch.bfloat16)[1:]
    gate = torch.randn(tokens + 1, 1, device="cuda", dtype=torch.bfloat16)[1:]
    original = x.clone()
    fused_sigmoid_gate_mul(x, gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_sigmoid_gate_mul(x, gate)
    gate.mul_(7)
    graph.replay()
    assert torch.equal(actual, x * torch.sigmoid(gate))
    assert torch.equal(x, original)


@requires_cuda
@torch.inference_mode()
def test_broadcast_gate_compilation_keeps_eager_sequence():
    """Full-graph tracing must avoid device queries and keep ATen gate ops."""
    set_random_seed(43)
    x = torch.randn(3, 2560, device="cuda", dtype=torch.bfloat16)
    gate = torch.tensor(
        [-3.984375, -2.984375, 1.125], device="cuda", dtype=torch.bfloat16
    ).reshape(3, 1)
    compiled = torch.compile(fused_sigmoid_gate_mul, fullgraph=True, backend="eager")
    assert torch.equal(compiled(x, gate), x * torch.sigmoid(gate))


@requires_cuda
@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90),
    reason="Qwen scalar gate selection targets Hopper",
)
@pytest.mark.parametrize("tokens, expected_calls", [(8, 1), (9, 0)])
@torch.inference_mode()
def test_qwen_shared_gate_keeps_larger_batches_eager(
    tokens: int, expected_calls: int, dist_init, tmp_path: Path
):
    """Use the original chain above eight rows to avoid decode regressions."""
    set_random_seed(43)
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        gate = ReplicatedLinear(2560, 1, bias=False)
        mlp = Qwen3MoeMLP(2560, 640, "silu", expert_gate=gate, disable_tp=True)
    for parameter in mlp.parameters():
        torch.nn.init.normal_(parameter, std=0.01)
    for layer in (mlp.gate_up_proj, mlp.down_proj, gate):
        layer.quant_method.process_weights_after_loading(layer)
    x = torch.randn(tokens, 2560, device="cuda", dtype=torch.bfloat16)
    mlp(x)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        out = mlp(x)
        torch.accelerator.synchronize()
    path = tmp_path / "gate.json"
    profile.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    calls = sum(
        event.get("cat") == "kernel"
        and "_fused_sigmoid_gate_mul_kernel" in event.get("name", "")
        for event in events
    )
    assert calls == expected_calls
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()
