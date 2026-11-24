# Using nabla-vllm

Start with a [source-matched CUDA installation](../getting_started/installation/README.md)
and a supported [target checkpoint](../models/supported_models.md). The primary
use is text generation on Qwen3.8-Flash-Next or DeepSeek-V4.1-Flash.

- [Offline inference](../serving/offline_inference.md) uses the retained
  `vllm.LLM` Python API.
- [Online serving](../serving/online_serving/README.md) uses the compatible
  HTTP frontend and the checkpoint's tokenizer/chat template.
- [NIXL P/D](../features/nixl_connector_usage.md) separates prefill and decode
  with matching cache geometry and protocol versions.
- [Optimization](../configuration/optimization.md) covers Graph settings and
  target-model execution choices.
- [Performance records](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md)
  state the baseline, input/output shape and output differences for each result.

Retained interfaces for pooling, speech, multimodal input, training integration
or optional frontends are not a general model/hardware support promise. Use the
scope and validation notes on each page before adopting them for a target model.
