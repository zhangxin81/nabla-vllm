# Target models

nabla-vllm specializes the native NVIDIA implementations of
Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash. It retains H20, H100 and B200
kernel selection. The CPU, ROCm, XPU, TPU and Zen CPU inference platforms
and unrelated native model registrations are removed.

| Model | Registered native architecture | Cache resources |
| --- | --- | --- |
| Qwen3.8-Flash-Next | `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration` | QSA/indexer pages and GDN/PLE state in the upstream hybrid BlockPool layout |
| DeepSeek-V4.1-Flash | `DeepseekV41ForCausalLM` | Sparse MLA/indexer pages and compressor state |

The conditional-generation architecture remains because target checkpoints
use it; the serving baseline uses `--language-model-only`. Multimodal
inference is not covered by that baseline. Registered target-compatible
DFlash, DFlash2, DSpine, DSpark and MTP classes remain available, but the
baseline disables speculative decoding. Their presence does not establish
speculative correctness or performance on every hardware/parallel layout.

The [current cache policy](../design/specialized_hybrid_cache.md) restores
upstream hybrid grouping, BlockPool allocation, checkpoint behavior and page
alignment/padding. The earlier independent cache ownership and byte arena
were reverted; they are historical records, not current model resources.
[NIXL configuration](../features/nixl_connector_usage.md) describes P/D
transfer and same-host CUDA IPC. The
[current H100 cache/connector comparison](../benchmarks/nabla-vllm/h100-cache-rollback-20261122.md) records source
revisions, precision, kernel choices and input/output shapes separately for
each checkpoint. H100 results do not validate H20 or B200.

```bash
nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --language-model-only --disable-extensible-kv-cache
```

Use the model-specific flags from the baseline for reproducible P/D runs.
Metadata-only development installs use `VLLM_TARGET_DEVICE=empty` and have
no inference backend. Python imports and environment variables retain the
`vllm` namespace; the distribution and primary executable are `nabla-vllm`.

The broader [upstream model catalog](https://docs.vllm.ai/en/latest/models/supported_models/)
describes upstream vLLM, not this runtime's registration scope.
