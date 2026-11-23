# V1 engine in nabla-vllm

Nabla retains the vLLM V1 scheduler, worker, sampler and serving interfaces for
its two [target models](../models/supported_models.md) on NVIDIA CUDA.
Hybrid cache grouping, BlockPool allocation and recurrent checkpoint policy
follow the [current cache implementation](../design/specialized_hybrid_cache.md).
P/D transfer and [operator buckets](../design/operator_kernel_buckets.md) have
separate configuration and verification requirements.

The following V0 comparisons explain inherited API behavior. They do not
establish near-zero CPU overhead, zero-configuration deployment or support for
upstream's broader hardware/model catalog. See the
[performance records](../benchmarking/README.md) for measured source versions.

## Differences from V0

This section lists some differences in behavior between V0 and V1.

### Chunked Prefill

Chunked prefill is enabled by default whenever possible, unlike in V0 where it was conditionally enabled based on model characteristics.

### CUDA Graphs

CUDA graph capture takes up more memory in V1 than in V0.

### Semantic Changes to Logprobs

#### Logprobs Calculation

By default, logprobs in V1 are now returned immediately once computed from the model’s raw output (i.e.
before applying any logits post-processing such as temperature scaling or penalty
adjustments). As a result, the returned logprobs do not reflect the final adjusted
probabilities used during sampling.

You can adjust this behavior by setting the `--logprobs-mode` flag.
Four modes are supported: `raw_logprobs` (default), `processed_logprobs`, `raw_logits`, `processed_logits`.
Raw means the values before applying any logit processors, like bad words.
Processed means the values after applying all processors, including temperature and top_k/top_p.

#### Prompt Logprobs with Prefix Caching

While V1 supports passing prompt logprobs with prefix caching enabled, it no longer caches the logprobs.
For a request requiring prompt logprobs, the engine will ignore the prefix cache and recompute the prefill of full prompt to generate the logprobs.

## Scheduling and retained features

The scheduler budgets tokens per request for prefill and decode and retains
FCFS/priority policies via `--scheduling-policy`. Prefix caching, chunked
prefill, CUDA graphs and speculative verification interact with the target's
attention and recurrent state layout. Use the [feature index](../features/README.md)
and the model-specific measured configuration to determine applicable paths.

CPU, ROCm, XPU and TPU platforms and unrelated native model registrations
have been removed. Pooling, ASR, generic Mamba and third-party hardware plugins
in upstream V1 are not Nabla's support matrix.

### Removed Features

As part of the major architectural rework in vLLM V1, several legacy features have been removed.

#### Sampling features

- **best_of**: This feature has been removed due to limited usage. See details at [RFC #13361](https://github.com/vllm-project/vllm/issues/13361).
- **Per-Request Logits Processors**: In V0, users could pass custom
  processing functions to adjust logits on a per-request basis. In vLLM V1, this
  feature has been removed. Instead, we now support **global logits processors**
  which are set at startup time, see [RFC #17799](https://github.com/vllm-project/vllm/issues/17799).

#### KV Cache features

- **GPU <> CPU KV Cache Swapping**: with the new simplified core architecture, vLLM V1 no longer requires KV cache swapping
to handle request preemptions.

#### Structured Output features

- **Request-level Structured Output Backend**: Removed; alternative backends (outlines, guidance) with fallbacks are supported now.
