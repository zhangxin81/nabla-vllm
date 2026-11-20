# CUDA specialization for Qwen3.8 and DeepSeek V4.1

This fork targets Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash on NVIDIA CUDA
GPUs, including H20, H100 and B200. The initial upstream baseline is
`898a7c68c2d1e869ebde2ce25ffbfc1e253bf0b4`.

Imported upstream contributions and their validation scope are recorded in the
[downstream PR integration record](bytevllm_pr_integration.md).

## Cache ownership

Prefix identity, physical allocation and checkpoint placement are separate
decisions. A prefix hash identifies committed input history. A pool owns
the physical resource used to resume that history. A checkpoint policy
decides which positions have reusable recurrent or compressor state.

The first refactor introduces an engine-owned `PrefixCacheIndex`. Each
physical `BlockPool` binds a view of this index. Pools retain independent
local IDs and free lists. The index's internal IDs are never sent to GPU
block tables or connectors. Thus token page 1 and state slot 1 can be
indexed together without naming the same resource.

Lookups continue to use the chained prefix hash and cache group ID, including
existing salt, multimodal and lookahead inputs. Duplicate prefix entries
retain their physical owners. Resetting one pool removes only its entries;
it leaves another pool's pinned and cached resources intact. Cross-pool
touches and frees update the actual owner's references and free list.

The hybrid coordinator creates the shared index. The existing HiSparse host
pool also binds it, exercising multiple physical owners through the existing
allocation path. This refactor preserves current GPU cache geometry and
checkpoint behavior.

## Allocation migration

The next allocation change uses a common byte budget with resource-specific
allocation units:

| Resource | Allocation unit | Retention constraint |
| --- | --- | --- |
| Attention KV | Token page | Committed prefix coverage or a live attention window |
| QSA/indexer cache | Its native page and compression unit | Index and KV coverage must agree |
| Qwen GDN/PLE state | State slot | Exact committed checkpoint and speculative recovery state |
| DeepSeek sparse/compressor state | Its native state or page | Compression, shared-cache and replay boundaries |
| Draft cache | Draft model's native unit | Draft window and acceptance semantics |

The budget must price actual backing bytes, including alignment, padding,
live-request state and in-flight transfer/copy references. It must neither
treat a state slot as a token page nor reserve a fixed KV/GDN fraction without
a capacity policy. Admission checks must succeed for every resource family
before any request allocation is committed.

Physical allocation, per-pool admission, worker zero/copy geometry and NIXL
region counts must migrate together. Prefix checkpoints must remain pinned
until every reader and transfer fence has completed. CUDA Graph buffers and
views need stable addresses throughout replay.

The independent state allocator and byte-budget arena are not yet wired to
the model runner in this first refactor. Removing attention/state page-size
coupling requires that migration, plus model and transfer validation.

## SGLang references

The design comparison uses SGLang commit
`f913bf9aa1b6f9d428b3d521b72c42ed1d00292e`:

- [Mamba slot allocator](https://github.com/sgl-project/sglang/blob/f913bf9aa1b6f9d428b3d521b72c42ed1d00292e/python/sglang/srt/mem_cache/allocator/mamba.py)
  allocates state slots independently of token KV indices.
- [Unified state allocator](https://github.com/sgl-project/sglang/blob/f913bf9aa1b6f9d428b3d521b72c42ed1d00292e/python/sglang/srt/mem_cache/unified_memory_pool.py)
  separates virtual state IDs, physical slots and byte-limited admission.
- [Unified tree's Mamba component](https://github.com/sgl-project/sglang/blob/f913bf9aa1b6f9d428b3d521b72c42ed1d00292e/python/sglang/srt/mem_cache/unified_cache/components/mamba.py)
  manages checkpoint positions, prefix coverage and state retention.

These mechanisms inform the refactor; SGLang's runtime and Python modules
are not added as dependencies.

Relevant upstream vLLM proposals are
[#46251](https://github.com/vllm-project/vllm/pull/46251), which separates
block granularity from allocation size, and
[#64611](https://github.com/vllm-project/vllm/pull/64611), which introduces a
separate draft pool. Their allocator, fragmentation, worker and connector
changes must be evaluated together before reuse.

## Platform support and validation

CUDA platform selection and NVIDIA kernel architecture selection remain.
CPU, ROCm, XPU, TPU and Zen CPU platform implementations are removed.
Non-CUDA build targets are rejected. Additional unreachable backend-specific
operators and build recipes will be pruned after tracing both target models'
dependency closure.

For host-only cache and scheduler tests, `VLLM_TARGET_DEVICE=empty` selects
a metadata platform with host tensors. It has no inference backend.

```bash
uv venv --python 3.12
VLLM_TARGET_DEVICE=empty uv pip install --no-build-isolation --no-deps -e .
VLLM_TARGET_DEVICE=empty .venv/bin/python -m pytest \
  --confcutdir=tests/v1/core \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py \
  tests/v1/core/test_deferred_block_free.py -q
```

The new tests cover pool-local ID collisions, checkpoint/page lookup,
cross-owner reference accounting, duplicate owners and selective reset.
GPU validation still needs both models' prefill/decode, draft acceptance,
prefix reuse, cancellation/preemption and PD continuation. No throughput,
latency or memory-capacity improvement is claimed by the index refactor.
