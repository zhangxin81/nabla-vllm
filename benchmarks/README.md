# Nabla performance benchmarks

This directory contains the inherited vLLM benchmark clients and the retained
target-model helpers used to evaluate nabla-vllm. The scope is Qwen3.8-Flash-Next
and DeepSeek-V4.1-Flash on NVIDIA CUDA, with P/D and operator-level work recorded
separately.

- [Target-model serving matrix](specialized/README.md) records paired sources,
  model fingerprints, request shapes, completion counts, throughput, TTFT and TPOT.
- `kernels/` measures individual operators; a kernel gain is not an end-to-end gain.
- `attention_benchmarks/` and `vllm bench forward` measure selected batch shapes.
- `multi_turn/` and the prefix-cache tools need explicit hit/reuse conditions.

Start from [Nabla's benchmark records](../docs/benchmarks/nabla-vllm/h20-input-host-planning-20261123.md)
and [benchmark CLI](../docs/benchmarking/cli.md). Keep baseline/implementation SHAs,
native artifacts, precision, hardware and seeds fixed; preserve failures and
output differences alongside latency. Historical upstream scripts for other
model tasks are references, not evidence of Nabla support.
