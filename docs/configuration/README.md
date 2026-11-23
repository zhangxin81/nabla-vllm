# Configuration Options

This section configures nabla-vllm's two target models on NVIDIA CUDA, including
P/D geometry, checkpoint precision and multidimensional operator dispatch.
The retained `vllm` argument/environment names are compatibility interfaces;
they do not imply that every upstream model or hardware backend is registered.
See [target scope](../models/supported_models.md) and the
[fixed-source H20 record](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md).

There are three main levels of configuration, from highest priority to lowest priority:

- [Request parameters](../serving/online_serving/openai_compatible_server.md#completions-api) and [input arguments](../api/README.md#inference-parameters)
- [Engine arguments](./engine_args.md)
- [Environment variables](./env_vars.md)

For performance settings, see [Optimization and Tuning](./optimization.md),
including [Qwen3.8-Flash-Next HC sequence parallelism](./optimization.md#qwen38-flash-next-hc-sequence-parallelism).
