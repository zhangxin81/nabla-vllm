# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


logger = init_logger(__name__)


class VerifyAndUpdateConfig:
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        return

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        return


def _set_deepseek_indexer_logits_dtype(vllm_config: "VllmConfig") -> None:
    """Default DeepSeek indexer logits to BF16 where the MQA logits kernels
    also reduce in BF16 (SM100) and to FP32 elsewhere."""
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

    attention_config = vllm_config.attention_config
    attention_config.indexer_logits_dtype = (
        attention_config.resolve_indexer_logits_dtype(
            "bfloat16" if mqa_logits_use_mx_scales() else "float32"
        )
    )


def _indexer_sparse_logits_supported(vllm_config: "VllmConfig") -> bool:
    """Whether to enable candidate-only indexer scoring by default."""
    hf_config = vllm_config.model_config.hf_text_config
    block = getattr(hf_config, "candidate_block_size", 0)
    topk_blocks = getattr(hf_config, "candidate_topk_blocks", 0)
    parallel = vllm_config.parallel_config
    if (
        block <= 0
        or topk_blocks <= 0
        or not current_platform.is_cuda()
        or parallel.decode_context_parallel_size > 1
        or parallel.prefill_context_parallel_size > 1
    ):
        return False
    if current_platform.is_device_capability_family(90):
        return True
    topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
    if not current_platform.is_device_capability_family(100) or topk_backend not in (
        "auto",
        "deep_select",
    ):
        return False
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        check_deep_select_layout,
        has_deep_select,
        pick_sparse_block_kv,
    )
    from vllm.utils.deep_gemm import has_deep_gemm_sparse_mqa

    if not (has_deep_gemm_sparse_mqa() and has_deep_select()):
        return False
    try:
        pick_sparse_block_kv(block)
        check_deep_select_layout(
            topk_blocks * block, getattr(hf_config, "index_topk", 0)
        )
    except ValueError:
        return False
    return True


class DeepseekV4ForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        # indexer_sparse_logits=None (default): enable the sparse-logits
        # indexer wherever it is supported; True / False are kept as given.
        attention_config = vllm_config.attention_config
        if attention_config.indexer_sparse_logits is None:
            attention_config.indexer_sparse_logits = _indexer_sparse_logits_supported(
                vllm_config
            )
        _set_deepseek_indexer_logits_dtype(vllm_config)

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        if quant_config is not None and quant_config.get("quant_method") == "fp8":
            model_type = getattr(model_config.hf_config, "model_type", None)
            if model_type in ("deepseek_v4", "deepseek_v41"):
                model_config.hf_config.quantization_config["quant_method"] = (
                    "deepseek_v4_fp8"
                )

        hf_text_quant_config = getattr(
            model_config.hf_text_config, "quantization_config", None
        )
        if (
            hf_text_quant_config is not None
            and hf_text_quant_config.get("quant_method") == "fp8"
        ):
            model_type = getattr(model_config.hf_text_config, "model_type", None)
            if model_type in ("deepseek_v4", "deepseek_v41"):
                model_config.hf_text_config.quantization_config["quant_method"] = (
                    "deepseek_v4_fp8"
                )


class HybridAttentionMambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Perform early validation and setup for hybrid attention/mamba models.

        Block size alignment with mamba page sizes is handled later by
        Platform.update_block_size_for_backend(), which runs after model
        layers are constructed and the attention backend is known.

        Args:
            vllm_config: vLLM Config

        """
        # Enable FULL_AND_PIECEWISE by default
        MambaModelConfig.verify_and_update_config(vllm_config)


class MambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Enable FULL_AND_PIECEWISE cuda graph mode by default (required
        to get good performance for mamba layers in V1).

        Args:
            vllm_config: vLLM Config

        """
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        if cache_config.enable_prefix_caching:
            if cache_config.mamba_cache_mode == "none":
                cache_config.mamba_cache_mode = "align"
                logger.info(
                    "Mamba cache mode is set to '%s' for %s by default "
                    "when prefix caching is enabled",
                    cache_config.mamba_cache_mode,
                    model_config.architecture,
                )
            # By default, mamba block size will be set to max_model_len (see
            # below). When enabling prefix caching, we align mamba block size
            # to the block size as the basic granularity for prefix caching.
            if cache_config.mamba_block_size is None:
                cache_config.mamba_block_size = cache_config.block_size
        else:
            if cache_config.mamba_cache_mode != "none":
                cache_config.mamba_cache_mode = "none"
                logger.warning(
                    "Mamba cache mode is set to 'none' when prefix caching is disabled"
                )
            if cache_config.mamba_block_size is None:
                cache_config.mamba_block_size = model_config.max_model_len


class Qwen3_5ForConditionalGenerationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Update mamba_ssm_cache_dtype for Qwen3.5 models when set to 'auto'
        (or not explicitly set), to the value specified in the HF config's
        mamba_ssm_dtype field. Warn if the user explicitly overrides it to a
        different value.
        """
        cache_config = vllm_config.cache_config
        hf_text_config = vllm_config.model_config.hf_text_config
        mamba_ssm_dtype = getattr(hf_text_config, "mamba_ssm_dtype", None)
        if cache_config.mamba_ssm_cache_dtype == "auto":
            if mamba_ssm_dtype is not None:
                cache_config.mamba_ssm_cache_dtype = mamba_ssm_dtype
        elif (
            mamba_ssm_dtype is not None
            and cache_config.mamba_ssm_cache_dtype != mamba_ssm_dtype
        ):
            logger.warning(
                "Qwen3.5 model specifies mamba_ssm_dtype='%s' in its config, "
                "but --mamba-ssm-cache-dtype='%s' was passed. "
                "Using the user-specified value.",
                mamba_ssm_dtype,
                cache_config.mamba_ssm_cache_dtype,
            )


def _strip_qwen4_exp_mrope(model_config: "ModelConfig") -> None:
    configs = {
        id(config): config
        for config in (
            getattr(model_config, "hf_config", None),
            model_config.hf_text_config,
        )
        if config is not None
    }
    for config in configs.values():
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_parameters.pop("mrope_section", None)
            rope_parameters.pop("mrope_interleaved", None)


class Qwen4ExpForConditionalGenerationConfig(Qwen3_5ForConditionalGenerationConfig):
    """Apply the Qwen3.5 hybrid-cache contract to Qwen4Exp."""

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:

        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        text_config = vllm_config.model_config.hf_text_config
        if text_config.hc_count <= 1:
            raise ValueError("Qwen4Exp requires hc_count > 1")
        parallel_config = vllm_config.parallel_config
        # MoE SP also enables HC SP without the explicit HC flag.
        if (
            current_platform.is_cuda_alike()
            and parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Qwen4Exp MoE SP requires PP=1")

        uses_ple_or_qsa = bool(text_config.ple_layer_ids) or (
            getattr(text_config, "indexer_n_heads", None) is not None
        )
        if uses_ple_or_qsa and (
            parallel_config.enable_dbo or parallel_config.ubatch_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp PLE/QSA does not support dual-batch overlap or microbatching"
            )
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is not None and multimodal_config.language_model_only:
            _strip_qwen4_exp_mrope(vllm_config.model_config)
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.method not in {
            "mtp",
            "ngram",
            "ngram_gpu",
        }:
            raise NotImplementedError(
                "Qwen4Exp speculative decoding supports only its native MTP "
                "checkpoint and linear n-gram proposers"
            )


class Qwen4ExpForCausalLMConfig(Qwen4ExpForConditionalGenerationConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)

        _strip_qwen4_exp_mrope(vllm_config.model_config)


class Qwen4ExpMTPConfig(Qwen4ExpForConditionalGenerationConfig):
    """Preserve MRoPE for a VL target and use 1D RoPE for a text target."""

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        if hasattr(vllm_config.model_config.hf_config, "vision_config"):
            return
        _strip_qwen4_exp_mrope(vllm_config.model_config)


MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {
    "DeepseekV41ForCausalLM": DeepseekV4ForCausalLMConfig,
    "Qwen4ExpForCausalLM": Qwen4ExpForCausalLMConfig,
    "Qwen4ExpForConditionalGeneration": Qwen4ExpForConditionalGenerationConfig,
    "Qwen4ExpMTP": Qwen4ExpMTPConfig,
}
