# CUDA specialization for Qwen3.8 and DeepSeek V4.1

This fork targets Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash on NVIDIA CUDA
GPUs, including H20, H100 and B200. The current upstream baseline is
`8e45c458f619dbabff879c0f53c58fd9f296e983`. The initial specialization used
`898a7c68c2d1e869ebde2ce25ffbfc1e253bf0b4`.

Imported upstream contributions and their validation scope are recorded in the
[downstream PR integration record](nabla_vllm_pr_integration.md).

## Cache ownership

Prefix identity, physical allocation and checkpoint placement are separate
decisions. A prefix hash identifies committed input history. A pool owns
the physical resource used to resume that history. A checkpoint policy
decides which positions have reusable recurrent or compressor state.

The coordinator owns a `PrefixCacheIndex`. Each
physical `BlockPool` binds a view of this index. Pools retain independent
local IDs and free lists. The index's internal IDs are never sent to GPU
block tables or connectors. Thus token page 1 and state slot 1 can be
indexed together without naming the same resource.

On NVIDIA SM90 and newer, the retained packed GDN decode, attention sigmoid
gate, gated norm and Triton MoE kernels use programmatic dependent launch (PDL). Each
kernel waits for its producer before reading inputs, and releases dependent
grids at normal grid completion. This changes launch scheduling without
changing the arithmetic or introducing a new user setting.

Lookups continue to use the chained prefix hash and cache group ID, including
existing salt, multimodal and lookahead inputs. Duplicate prefix entries
retain their physical owners. Resetting one pool removes only its entries;
it leaves another pool's pinned and cached resources intact. Cross-pool
touches and frees update the actual owner's references and free list.

The HiSparse host pool also binds the shared index. Its host allocations keep
their own budget. Device pools use a common `CacheMemoryArena` with an exact
byte ledger and pool-specific strides.

## Independent allocations over stable views

Target models use the common byte budget with resource-specific allocation
units:

| Resource | Allocation unit | Retention constraint |
| --- | --- | --- |
| Attention KV | Token page | Committed prefix coverage or a live attention window |
| QSA/indexer cache | Its native page and compression unit | Index and KV coverage must agree |
| Qwen GDN/PLE state | State slot | Exact committed checkpoint and speculative recovery state |
| DeepSeek sparse/compressor state | Its native state or page | Compression, shared-cache and replay boundaries |
| Draft cache | Draft model's native unit | Draft window and acceptance semantics |

Every device pool has a native stride and a local ID space. The planner creates
stable tensor views over one backing allocation. A pool-local slot owns the
byte interval `[id * stride, (id + 1) * stride)`. The null region reserves the
largest stride; its contents are cleared during warmup. Live references reserve
bytes, while unreferenced cached slots remain reclaimable by any resource pool.

Admission first prices all groups in bytes, then plans their aligned intervals
from largest stride to smallest. Hits are reserved before planning new slots.
Every group consumes the same plan even when its allocation order differs.
Unsuccessful admission commits no allocations. The plan is discarded at the
end of allocation, including exceptional exits.

Fresh slots of the same stride use a soft address cursor to reduce interleaved
alignment holes. Initial request needs determine cursor starting positions,
without reserving a quota for any family. If that position is occupied,
allocation falls back to available ranges and cached-slot LRU order. Cursors
reset when the arena has no live allocations, including transfer references.

Each queried stride caches its exact count of aligned free slots. Reserving a
range adjusts the count for that range's split; releasing it adjusts the merged
ranges. Repeated pool-capacity queries therefore do not scan every free interval.
Admission snapshots omit these query caches and only plan byte ownership.
Steps needing no new slots and no prefix hits keep an empty reservation plan
without copying the arena. These paths preserve the same alignment, borrowing
and prefix-reference rules.

A write through one pool invalidates overlapping cached entries in other pools.
The shared prefix index maintains the cached-slot membership used by this
invalidation; it does not scan every virtual token slot when the index is sparse.

Worker initialization and copy-on-write use byte ranges. A GDN slot is copied
as its native byte range, without expanding it into token-page IDs. Retained
copy and transfer references keep these intervals live until their fences
complete. CUDA Graphs retain the same backing and view addresses.

NIXL protocol 19 advertises each region's capacity and stride. Imported IDs
cover only the actual loaded prefix; a synchronous import excludes slots
reserved for the following decode. GDN receives the state at the prefix's
logical end, with speculative scratch columns excluded by position.
Registration generations are checked before a synchronous handshake installs
new handles, so a first load cannot invalidate its own connection and run
state preparation before the transfer has been submitted.

Checkpoint retention lives in `prefix_checkpoint_policy.py`. It selects
committed token positions from retention intervals and reachable replay
boundaries. The state allocator supplies physical slots for those positions.
Explicit extensible-cache mode is rejected for the stable arena.

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
Non-CUDA build targets are rejected. Unreachable backend-specific operators,
model files, linear-kernel dispatch, build recipes and precompiled wheel routes
have also been pruned against the two target models' dependency closure.
Host-side tokenization, Engram offload and metadata tests remain.

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
