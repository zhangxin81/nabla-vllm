# Nabla examples and inherited reference scripts

Use the two [target model families](../models/supported_models.md) and the
NVIDIA installation described in [Quickstart](../getting_started/quickstart.md).
Many inherited scripts still demonstrate upstream vLLM APIs with smaller
Llama/OPT/Qwen3 or pooling/ASR checkpoints; those defaults are not registered
target models in nabla-vllm.

| Start here | Purpose |
| --- | --- |
| [Offline generation](../../examples/basic/offline_inference/README.md) | Pass an existing target checkpoint to a script that accepts model arguments. |
| [P/D examples](../../examples/disaggregated/disaggregated_serving/README.md) | Understand the proxy flow; use the recorded NIXL target-model configurations for deployment. |
| [Target-model serving matrix](../../benchmarks/specialized/README.md) | Measure fixed source, model fingerprints, throughput, TTFT and TPOT. |
| [Observability](../../examples/observability/dashboards/README.md) | Reuse the retained metric names while checking which metrics your configuration emits. |

The remaining `generate`, `pooling`, `speech_to_text`, `features`, `rl`, `ray_serving`
and integration directories are source/reference examples. Adapt model and
hardware assumptions before running them; they do not expand Nabla's native
registration scope or establish performance on untested combinations.
