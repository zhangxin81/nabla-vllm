# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 CUDA entry points, loaded independently of shared helpers."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from .nvidia.vl_model import DeepseekV41ForCausalLM
    from .quant_config import DeepseekV4FP8Config


def __getattr__(name: str) -> Any:
    if name == "DSparkDeepseekV4ForCausalLM":
        from .nvidia.dspark import DSparkDeepseekV4ForCausalLM

        return DSparkDeepseekV4ForCausalLM
    if name == "DeepseekV41ForCausalLM":
        from .nvidia.vl_model import DeepseekV41ForCausalLM

        return DeepseekV41ForCausalLM
    if name == "DeepseekV4FP8Config":
        from .quant_config import DeepseekV4FP8Config

        return DeepseekV4FP8Config
    raise AttributeError(name)


__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]
