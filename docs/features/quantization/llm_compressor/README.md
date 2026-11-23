# LLM Compressor reference

[LLM Compressor](https://github.com/vllm-project/llm-compressor) is an upstream
checkpoint preparation tool. The format guides in this directory explain
compressed-tensors metadata used by retained quantization implementations;
they do not certify every format or calibration recipe for Nabla's targets.

For Qwen3.8-Flash-Next or DeepSeek-V4.1-Flash, first confirm the native
architecture, checkpoint precision and NVIDIA kernel path. Requantization
requires model evaluation and a paired performance comparison. Memory savings
alone do not establish faster serving or equivalent output quality.

- [Nabla quantization scope](../README.md)
- [Target models](../../../models/supported_models.md)
- [Fixed-source performance records](../../../benchmarking/README.md)
- [Upstream LLM Compressor examples](https://github.com/vllm-project/llm-compressor/tree/main/examples)
