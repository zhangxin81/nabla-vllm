# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek CUDA entry points, independent of shared helper imports."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common.vl_model import DeepseekV4ForConditionalGeneration
    from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP
    from .quant_config import DeepseekV4FP8Config


def __getattr__(name: str) -> Any:
    if name == "DSparkDeepseekV4ForCausalLM":
        from .nvidia.dspark import DSparkDeepseekV4ForCausalLM

        return DSparkDeepseekV4ForCausalLM
    if name == "DeepSeekV4MTP":
        from .nvidia.mtp import DeepSeekV4MTP

        return DeepSeekV4MTP
    if name == "DeepseekV4FP8Config":
        from .quant_config import DeepseekV4FP8Config

        return DeepseekV4FP8Config
    if name == "DeepseekV4ForCausalLM":
        from .nvidia.model import DeepseekV4ForCausalLM

        return DeepseekV4ForCausalLM
    if name == "DeepseekV4ForConditionalGeneration":
        from .common.vl_model import DeepseekV4ForConditionalGeneration

        return DeepseekV4ForConditionalGeneration
    raise AttributeError(name)


__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForConditionalGeneration",
]
