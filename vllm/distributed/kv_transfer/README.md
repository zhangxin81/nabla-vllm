# Nabla P/D cache transfer

Nabla retains the vLLM V1 connector interface for separate prefill and decode
instances of Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash on NVIDIA CUDA.
The target path uses `NixlConnector`, with ordinary NIXL or explicit same-host
CUDA IPC. Follow the
[NIXL configuration guide](../../../docs/features/nixl_connector_usage.md)
for roles, enabling IPC, geometry and lifetime requirements.

P and D must agree on model, precision, cache groups and connector protocol.
Attention/indexer cache and the target's recurrent/compressor state are handled
by the selected connector path. The
[current cache policy](../../../docs/design/specialized_hybrid_cache.md) uses
upstream hybrid BlockPool and checkpoint management; the earlier independent
KV/state allocator was reverted.

For evidence, use the
[fixed-source H100 comparison](../../../docs/benchmarks/nabla-vllm/h100-cache-rollback-20261122.md).
The [inherited demo scripts](../../../examples/disaggregated) cover several
connectors and unrelated default models; they are not a target deployment
matrix. The older FIFO-pipe/lookup-buffer abstraction is not the V1 NIXL path.
