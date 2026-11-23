# Disaggregated Serving

Nabla's target-model P/D path uses NixlConnector with ordinary NIXL or
explicit same-host CUDA IPC. Start with the
[NIXL configuration](../../../docs/features/nixl_connector_usage.md) and
[measured H100 topology](../../../docs/benchmarks/nabla-vllm/h100-cache-rollback-20261122.md).
The demo scripts below are inherited connector/proxy examples. Match source,
model, precision and cache geometry on P and D; unrelated default models and
other connector demos do not establish target-model correctness.

## Files

- `disagg_proxy_demo.py` - Demonstrates XpYd (X prefill instances, Y decode instances).
- `kv_events.sh` - Demonstrates KV cache event publishing.
- `mooncake_connector` - A proxy demo for MooncakeConnector.
