# bytevllm downstream PR integration

This integration preserves the CUDA specialization and shared prefix index on
upstream baseline `898a7c68c2d1e869ebde2ce25ffbfc1e253bf0b4`. Source PRs are pinned
below so later upstream updates can be compared without importing an old branch's
unrelated history. It does not merge or modify the upstream PRs.

## Source revisions

| Upstream PR | Source head | Downstream behavior |
| --- | --- | --- |
| [#67468](https://github.com/vllm-project/vllm/pull/67468) | `293757b4e919d12e750591ad9bba15983ae2be54` | Native small-head BF16 FlashMLA dispatch on the measured H20/H100 configurations; other cutoffs retained. |
| [#43594](https://github.com/vllm-project/vllm/pull/43594) | `fadc6e752e92ab10603a7de1d2f68479c7fea866` | Batched native context KV writes for the DFlash family, with per-layer mappings and backend fallback. |
| [#65136](https://github.com/vllm-project/vllm/pull/65136) | `d99c7cb54b86c11c051fc825941e0b6924e7c6a6` | Development-mode prefix reset through every configured toy-proxy backend, with authentication and partial failure reporting. |
| [#65639](https://github.com/vllm-project/vllm/pull/65639) | `a7ae623ae938e61d5385d7349581dcc12c3b0e39` | Opt-in same-host CUDA IPC pulls with stream dependencies and descriptor alias coalescing. |
| [#41115](https://github.com/vllm-project/vllm/pull/41115) | `094be7739a29924f0fb297c06528070d50aeea44` | Shape-aware per-token FP8 launch sizes and direct-op argument validation. Shared-memory row caching remains disabled. |
| [#42301](https://github.com/vllm-project/vllm/pull/42301) | `f328e7925aec7899314ee1f0f38df8dc885ae5dd` | Already covered by upstream [#64425](https://github.com/vllm-project/vllm/pull/64425), commit `914d61fc538a84e8487cb16fe8241a70f6f4bf9e`, present in this baseline. |

The GDN overlap implementation from #64425 is retained. Qwen3.5's NVIDIA caller
enables it outside `torch.compile`; the layer restricts overlap to compatible
cuBLAS projections, CUDA graph capture, and at most 256 tokens. LoRA and
ubatching retain serial execution. Reapplying #42301's older unconditional
stream overlap would remove these restrictions.

Closed PR [#39902](https://github.com/vllm-project/vllm/pull/39902) is superseded by
merged [#39904](https://github.com/vllm-project/vllm/pull/39904). Its changes are
already in the baseline.

## Adaptation to the current tree

NIXL descriptor handles remain owned by the current per-PP-stage structures.
Lease-expiry checks, source lifetime tracking, and M-RoPE initialization are
preserved. Protocol version **18** includes the optional IPC allocation metadata;
version 17 was already used by upstream lease metadata. P and D must run the same
revision. IPC remains disabled unless both ends explicitly opt in; see the
[NIXL usage guide](../features/nixl_connector_usage.md).

FP8 changes are applied to `csrc/quantization/w8a8/fp8/common.cu`, the current
location of the stable-ABI implementation. Other static and dynamic per-tensor
vectorized kernels are retained. Per-token launch sizes are 128 threads up to
hidden size 2048, 512 threads for hidden sizes at least 16384 on SM90+, and 256
otherwise. The direct op validates the scales' CUDA device, float32 dtype and
capacity, the output device, and a scalar float32 CUDA scale bound when supplied.

The DFlash documentation keeps the current DSpine instructions alongside the
context-write description. The proxy reset documentation keeps the current DCP
configuration rules.

## Validation scope

Host validation uses the existing uv-managed Python environment and
`VLLM_TARGET_DEVICE=empty`. This mode has no inference backend.

```bash
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=empty .venv/bin/python -m pytest -q \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py \
  tests/v1/core/test_deferred_block_free.py

HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=empty .venv/bin/python -m pytest -q \
  -m cpu_test \
  tests/v1/kv_connector/unit/test_nixl_connector.py \
  tests/v1/kv_connector/unit/test_nixl_desc_geometry.py \
  -k 'local_fastpath or local_ipc'

HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=empty .venv/bin/python -m pytest -q \
  -m cpu_test tests/v1/kv_connector/unit/test_nixl_proxy.py \
  -k 'not real_backend'
```

The cache/prefix/deferred-free suites passed: **530 passed in 89.32s**. IPC
geometry and scheduler-selection tests passed: **6 passed, 583 deselected in
2.34s**. Peer replacement and lease lifecycle checks passed: **35 passed,
187 deselected in 6.43s**. PP receiving-role rejection tests passed after
installing the missing dependency: **8 passed, 214 deselected in 7.72s**.
Proxy HTTP tests passed: **10 passed, 2 deselected in 2.39s**.

The full NIXL CPU suites are not a passing validation of this environment.
Failures include hybrid-cache construction and host NUMA discovery. The
259 failing test IDs exactly match the failures on the pre-integration commit
`36e545952313efa58e2109ab6b1abe9f6b8668ba`. The initial candidate run also had
seven collection/setup errors from a missing test dependency. These are not a
full-suite pass, and these patches do not resolve the baseline failures.
Native-server reset coverage requires an inference-enabled installation and
model weights.

This integration has not been CUDA-compiled or GPU-tested. Original PR benchmark
results describe their original revisions and environments; they do not establish
correctness or speedups for the combined downstream tree. Before deployment,
validate FP8 outputs and strided layouts, FlashMLA graph replay, DFlash context
writes and acceptance, and IPC output equivalence, concurrency, cancellation,
preemption, peer restart, block reuse, and transfer lifetime on the actual P/D
configuration. H20/H100/B200 coverage and both target models remain necessary.
