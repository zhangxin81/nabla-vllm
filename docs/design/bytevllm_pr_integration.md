# bytevllm downstream PR integration

The current integration uses upstream baseline
`5a21c6c853bff525dbaa4cc06596e3c374599f85`. The original specialization used
`898a7c68c2d1e869ebde2ce25ffbfc1e253bf0b4`. Source PRs are pinned
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
| [#65094](https://github.com/vllm-project/vllm/pull/65094) | `5c64c17f32d0cdce57120a7852472b3272d6846d` | Forward UCX backend parameters and synchronize agent access in both pull and push modes; retain independent region capacities. |
| [#64921](https://github.com/vllm-project/vllm/pull/64921) | `644f2acb4ce773746838c9dc85027657d62f39ca` | Keep DeepSeek V4.1 decoder replay disabled when the selected cut cannot be replayed. |
| [#67740](https://github.com/vllm-project/vllm/pull/67740) | `819fbdf5645e1c4f139bf81b741548bc31c22779` | SM100 QSA FP8 prefill with contiguous DeepGEMM logits, native compressed key addressing and bounded scratch planning. The shared metadata builder installs the plan. |
| [#68856](https://github.com/vllm-project/vllm/pull/68856) | `4c1896978a9173f4a9918eedf2ca7be345c7bc0c` | Preserve the existing SM90 router tuning and add an H100 CI job for its correctness suite. |
| [#64846](https://github.com/vllm-project/vllm/pull/64846) | `c17f9ce00a52fa7b5ce89ff25b02e6bb30722b51` | PDL launches for packed GDN decode, sigmoid gate, gated norm and Triton MoE. Each waits before its first read; no early dependent trigger. Gated norm is moved into the shared GDN package and connected to Qwen4Exp outside compilation. |

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
preserved. Protocol version **19** carries independent region capacities and
strides in addition to the IPC allocation metadata introduced in version 18.
P and D must run the same revision. IPC remains disabled unless both ends
explicitly opt in; see the
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

The results below describe the original integration. The new native arena and
combined PR adaptations are being validated separately on H100; they do not
inherit a passing result from these historical runs.

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

## Current combined-tree validation

The independent cache arena passed **475 passed in 183.14s (0:03:03)** on the
Linux H100 worker. The transfer-ratio regression subset passed **10 passed,
222 deselected, 17 warnings in 45.10s**. These precede the latest PDL addition.
The current host admission regression command was:

```bash
VLLM_TARGET_DEVICE=empty HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  --confcutdir=tests/v1/core tests/v1/core/test_prefix_caching.py -q \
  -k 'state_slots_and_token or overlapping_allocation or arena_admission or admission_plan_survives or live_state_references or byte_ranges_reject or native_cache_manager or arena_reserves or native_gdn_import'
```

Result: **12 passed, 238 deselected in 12.91s**. Both the loaded GDN state and
the first writable decode state participate in one admission plan.

After the PDL addition and shared gated-norm migration, the existing packed
decode, gated-norm and sigmoid-gate suites plus native byte-copy/zeroing tests
passed on H100: **63 passed, 83 deselected in 32.03s**. The 83 deselections are
the unrelated attention utilities in the same worker file; no failing case was
removed or skipped. The first run exposed 17 imports of the deleted Qwen3.5
norm module. Their imports now exercise the shared GDN implementation, retaining
the original exact and FP8 assertions.

The stage17 Qwen H100 4P4D IPC diagnostic cancelled 12 requests, drained the
queues, served three follow-up requests and reset the prefix caches. The
strict direct-versus-P/D diagnostic had **12 token mismatches in 30 comparisons**.
Upstream `2aa47529f02bb34cf869c35c4345000fa2934c78` reproduced all 12 at exactly
the same comparison labels with prefix caching disabled. Comparing candidate
and upstream on the same paths and shapes produced identical token sequences
for every saved length case: 63, 64, 65, 127, 128, 129, 512, 4096 and 16384
input tokens, including 129/256 and 4096/256 input/output cases. The short
cases also had identical chosen-token logprobs; long-case maximum deltas were
0.00020–0.01840. The original strict diagnostic remains recorded as failed;
the matched-path comparison does not weaken it or establish batch invariance.
No throughput or release-readiness claim follows from these tests.

Real connector construction with an external NIXL backend recorder passed
**6 passed, 226 deselected, 17 warnings in 42.80s**:

```bash
PYTHONPATH=<candidate-source> HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  <candidate-source>/tests/v1/kv_connector/unit/test_nixl_connector.py -q \
  -k 'ucx_backend_params or agent_is_synchronized'
```

## Community screening boundaries

The November 20 screening also considered the following proposals. Keeping a
proposal out of this integration does not assert that it has no benefit.

| PRs | Current disposition |
| --- | --- |
| #68636 | The change targets the older DeepSeek V4 attention specs. V4.1 already constrains SWA stride by the interleaved record sizes and indexer stride by its 256-byte requirement; do not substitute the older 576-byte constant. |
| #64670 | GDN parameter offload warmup fix identified; separate regression validation is needed before integration. |
| #67593 and #67203 | Chunked WRITE uses `NixlPushConnector`, including a stacked protocol and scheduler change. The requested baseline uses `NixlConnector` pull; this needs a separate transfer validation and performance arm. |
| #46251 and #64611 | Alternative cache allocation designs; the independent native-stride arena already changes their ownership and worker assumptions. Importing them together would install competing allocation designs. |
| #58863, #67590, #64870, #64884, #65061, #65064, #65069 | RecoverSSM and batch-invariance stacks require a complete compatible dependency chain and model validation. They are not enabled by the greedy serving baseline. |
| #68291, #68366 | Mega gate / specialized decode proposals require architecture and parallel-layout validation; a PR title is insufficient evidence for the H100 TP4/EP4 baseline. |
| #68569 | Processed-logprob batch-invariance fix identified; the current diagnostic uses the default numerical mode, so this does not establish a fix for its divergence. |
| ROCm, XPU, CPU and unrelated model proposals | Outside the NVIDIA target-model dependency closure. |

Original PR benchmark results describe their original revisions and
environments; they do not establish speedups for the combined downstream tree.
The current combined-tree checks above supersede the original validation
scope only for the paths they exercise. Before deployment,
validate FP8 outputs and strided layouts, FlashMLA graph replay, DFlash context
writes and acceptance, and IPC output equivalence, concurrency, cancellation,
preemption, peer restart, block reuse, and transfer lifetime on the actual P/D
configuration. H20/H100/B200 coverage and both target models remain necessary.
