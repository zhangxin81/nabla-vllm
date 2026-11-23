# Quantization

Nabla selects quantized implementations for its two target checkpoints on
NVIDIA CUDA. A format name alone does not establish target-model support:
checkpoint metadata, activation/KV precision, GPU architecture, MoE backend
and parallel layout all affect the path. Preserve the precision recorded in
the [baseline](../../benchmarking/README.md) when comparing performance.

!!! tip
    To get started with quantization, see [LLM Compressor](llm_compressor/README.md), a library for optimizing models for deployment with vLLM that supports FP8, INT8, INT4, and other quantization formats.

    For the names vLLM accepts for methods and schemes, see [Methods and Schemes](formats.md). For the kernel backends of quantized linear layers, see [Kernel Backends](kernels.md).

The following inherited guides describe quantization formats and tools.
They are references, not a tested Nabla format matrix:

- [AutoAWQ](auto_awq.md)
- [BitsAndBytes](bnb.md)
- [GPTQModel](gptqmodel.md)
- [LLM Compressor](llm_compressor/README.md)
    - [FP8 W8A8](llm_compressor/fp8.md)
    - [INT4 W4A16](llm_compressor/int4.md)
    - [INT8 W4A8](llm_compressor/int8_w4a8.md)
    - [INT8 W8A8](llm_compressor/int8_w8a8.md)
- [NVIDIA Model Optimizer](modelopt.md)
- [Online Quantization](online.md)
- [Quantized KV Cache](quantized_kvcache.md)
- [TorchAO](torchao.md)
- [FP8 ViT Encoder Attention](fp8_vit_attn.md)

## Selecting Linear Backends

To choose the kernel backend of quantized linear layers, for all of them with
`--linear-backend` or per quantization type with `linear_backend_per_quant`,
see [Kernel Backends](kernels.md#selecting-a-backend).

## Hadamard Input Transforms (QuIP)

Compressed-tensors NVFP4 checkpoints with a QuIP-style Hadamard input transform (`transform_config` with `type: hadamard` and `head_dim` of 16, 32, 64, or 128) run on the Humming backend by default on compute capability 10.0 and newer. The Hadamard is fused with input quantization, and the activation dtype declared by the checkpoint is used, with no fallback to 16-bit activations.

Other transforms, older GPUs, and an explicit `--linear-backend` other than `auto` or `humming` use the generic transform path, which applies the transform separately before the selected kernel.

!!! note
    The `fp_quant` quantization method and the QuTLASS backend have been removed. Use a compressed-tensors checkpoint instead.

!!! note
    Marlin kernels no longer quantize activations to INT8 or FP8, and the `VLLM_MARLIN_INPUT_DTYPE` environment variable has been removed (setting it is an error). Marlin always runs with 16-bit activations. To run GPTQ, AWQ, or compressed-tensors weight-only checkpoints with 8-bit activations (W4A8), use the `humming` quantization method and set the activation dtype with `VLLM_HUMMING_INPUT_QUANT_CONFIG`, for example:

    ```bash
    VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8"}' \
      vllm serve <model> --quantization humming
    ```

    Use `{"dtype":"float8e4m3"}` for FP8 activations. Checkpoints that declare 8-bit activations, such as compressed-tensors W4A8, select a kernel that supports them automatically.

## Activation Quantization Fallbacks

A quantized checkpoint (or `--quantization` / `--quantization-config`) requests both a weight and an activation scheme, for example FP8 W8A8 or NVFP4 W4A4. When the kernels that run the requested activation quantization are unavailable for the device or deployment, vLLM by default falls back to a weight-only kernel (Marlin or Humming) that keeps the quantized weights but runs activations in 16-bit, so W8A8 runs as W8A16 and W4A4 as W4A16. This changes the execution scheme and must be evaluated for the target checkpoint; it is not the scheme the checkpoint was calibrated and benchmarked with, and it is logged once as a warning:

```text
MarlinExperts runs QuantKey(...) activations as unquantized. Set VLLM_STRICT_QUANT_SCHEME=1 to forbid this fallback.
```

Set `VLLM_STRICT_QUANT_SCHEME=1` (default `0`) to forbid these fallbacks. Kernel selection then skips kernels that would not run the requested activation quantization and continues down the priority list, so it can still land on a slower kernel that does (for example NVFP4 emulation). If no kernel is left, or an explicitly requested `--moe-backend` / `--linear-backend` cannot run the scheme, startup fails with the reason. Humming input-schema fallbacks are forbidden the same way.

Strict mode only covers dropping or changing a requested activation quantization at kernel selection. It does not cover backends that add activation quantization to a weight-only request, or per-shape and per-batch-size fallbacks inside a selected kernel.

## NVIDIA kernel capability

Use the generated [kernel backend reference](kernels.md) and the target
checkpoint's measured configuration. Retained H20/H100/SM100 choices have
different dtype, shape and architecture requirements. CPU/ROCm/XPU paths
and their upstream compatibility tables do not apply to Nabla inference.

## Out-of-Tree Quantization Plugins

vLLM supports registering custom, out-of-tree quantization methods using the `@register_quantization_config` decorator. This allows you to implement and use your own quantization schemes without modifying the vLLM codebase.

### Registering a Custom Quantization Method

To register a custom quantization method, create a class that inherits from `QuantizationConfig` and decorate it with `@register_quantization_config`. The `get_quant_method` dispatches to the appropriate quantize method based on the layer type:

```python
import torch
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fused_moe import FusedMoE

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    """Custom quantization config."""

    def get_name(self) -> str:
        return "my_quant"

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Minimum GPU compute capability, -1 for no restriction
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        # Config files to search for in model directory
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MyQuantConfig":
        # Create config from model's quantization config
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Dispatch based on layer type
        # NOTE: you only need to implement methods you care about
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod()
        elif isinstance(layer, FusedMoE):
            return MyQuantMoEMethod(layer.moe_config)
        return None
```

### Required QuantizationConfig Methods

Your custom `QuantizationConfig` subclass must implement these abstract methods:

| Method | Description |
| ------ | ----------- |
| `get_name()` | Returns the name of the quantization method |
| `get_supported_act_dtypes()` | Returns list of supported activation dtypes (e.g., `torch.float16`) |
| `get_min_capability()` | Returns minimum GPU compute capability (e.g., 80 for Ampere, -1 for no restriction) |
| `get_config_filenames()` | Returns list of config filenames to search for in model directory |
| `from_config(config)` | Class method to create config from model's quantization config dict |
| `get_quant_method(layer, prefix)` | Returns the quantization method for a given layer, or `None` to skip |

### Implementing a Quantized Linear Method

For linear layers, return a `QuantizeMethodBase` subclass from `get_quant_method`. You can extend `UnquantizedLinearMethod` as a starting point:

```python
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

class MyQuantLinearMethod(UnquantizedLinearMethod):
    """Custom quantization method for linear layers."""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        # Create quantized weights for the layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Apply custom quantization logic here
        ...
```

### Implementing a Quantized MoE Method

For Mixture of Experts (MoE) models, return a `FusedMoEMethodBase` subclass from `get_quant_method`. You can use `UnquantizedFusedMoEMethod` to skip MoE quantization:

```python
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

class MyQuantMoEMethod(FusedMoEMethodBase):
    """Custom quantization method for MoE layers."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Create quantized weights for the MoE layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        router: "FusedMoERouter",
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Apply MoE computation with quantized weights
        ...

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # Return the MoE quantization configuration
        ...
```

See existing implementations like `Fp8MoEMethod` in `vllm/model_executor/layers/quantization/fp8.py` for reference.

### Using the Plugin

Once registered, you can use your custom quantization method with vLLM:

```python
# Register your quantization method (import the module containing your config)
import my_quant_plugin

from vllm import LLM

# Use the custom quantization method
llm = LLM(model="your-model", quantization="my_quant")
```

For more information on the plugin system, see the [Plugin System documentation](../../design/plugin_system.md).
