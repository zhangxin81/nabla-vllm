# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import vllm.third_party.flash_linear_attention.ops.fused_gdn_decode_post_conv_mtp as mtp_ops  # noqa: E501
from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _qwen_gdn_warmup_config,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_gdn_decode_post_conv_mtp_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
    _warm_state_gather_scatter_kernels,
)
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units


def _cuda_gdn_config() -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    device = torch.device("cuda")
    conv_state = torch.empty(
        (8, conv_dim, conv_kernel_size - 1),
        dtype=torch.bfloat16,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(
        device, config, max_num_tokens=16, x_dtype=config.conv_dtype
    )
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    _warm_state_gather_scatter_kernels(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA cache is required")
def test_gdn_warmup_copies_offloaded_parameters_to_the_cache_device(
    tmp_path, dist_init
):
    """Warmup must use device parameters when the real layer owns CPU weights."""
    from transformers import Qwen4ExpConfig

    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    hf_config = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config={
            "hidden_size": 32,
            "vocab_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "output_gate_type": "sigmoid",
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "ple_layer_ids": [],
        },
    )
    hf_config.save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), skip_tokenizer_init=True),
        additional_config={"gdn_prefill_backend": "triton"},
    )
    with set_current_vllm_config(config):
        layer = QwenGatedDeltaNetAttention(
            hf_config.text_config, config, prefix="model.layers.0.linear_attn"
        )
        with torch.no_grad():
            layer.A_log.fill_(-0.5)
            layer.dt_bias.fill_(0.25)
        conv = torch.zeros((2, 96, 3), dtype=torch.bfloat16, device="cuda")
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2).contiguous()
        state = torch.zeros((2, 2, 16, 16), dtype=torch.float32, device="cuda")
        layer.kv_cache = (conv, state)
        assert layer.A_log.device.type == layer.dt_bias.device.type == "cpu"
        warmup = _qwen_gdn_warmup_config(
            config.compilation_config.static_forward_context
        )
    assert warmup is not None
    assert warmup.a_log.device == warmup.dt_bias.device == conv.device
    torch.testing.assert_close(warmup.a_log.cpu(), layer.A_log, rtol=0, atol=0)
    torch.testing.assert_close(warmup.dt_bias.cpu(), layer.dt_bias, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only Triton kernel")
@pytest.mark.parametrize("max_num_reqs", [1, 1024])
def test_qwen_gdn_fused_mtp_warmup_covers_reachable_tile_sizes(
    max_num_reqs: int,
) -> None:
    """The MTP decode warmup compiles every V tile size the wrapper can pick
    for up to max_num_reqs requests."""
    k = v = 128
    config = replace(
        _cuda_gdn_config(),
        k=k,
        v=v,
        state_stride_token=2 * v * k,
        fused_spec_decode=True,
    )
    device = torch.device("cuda")
    kernel = mtp_ops.fused_gdn_decode_post_conv_mtp_triton
    with patch.object(
        mtp_ops, "fused_gdn_decode_post_conv_mtp_triton", wraps=kernel
    ) as wrapped:
        _warm_fused_gdn_decode_post_conv_mtp_kernel(
            device,
            config,
            torch.bfloat16,
            num_spec_tokens=3,
            max_num_reqs=max_num_reqs,
        )
    torch.accelerator.synchronize(device)

    num_cus = num_compute_units(device.index)
    tiles = set()
    for call in wrapped.call_args_list:
        num_reqs = call.kwargs["cu_seqlens"].numel() - 1
        assert num_reqs <= max_num_reqs
        heads = num_reqs * config.hv
        tiles.add(32 if heads < num_cus else 64 if heads <= 4 * num_cus else v)
    assert tiles == ({32} if max_num_reqs == 1 else {32, 64, v})
