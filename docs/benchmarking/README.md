# Benchmarking nabla-vllm

Measure the retained target-model runtime on NVIDIA CUDA using named source
revisions and identical model/hardware conditions. Components, offline forward
passes, complete HTTP services and P/D transfers answer different questions.

- [CLI tools](cli.md) provide `nabla-vllm bench` / compatible `vllm bench` entrypoints.
- [Target-model P/D matrix](../../benchmarks/specialized/README.md) preserves source,
  checkpoint, input/output, concurrency and per-request evidence.
- [H100 cache/connector comparison](../benchmarks/nabla-vllm/h100-cache-rollback-20261122.md)
  and [H20 input/host planning](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md)
  have distinct baselines and validation limits.
- [Parameter sweeps](sweeps.md) can select workload-specific settings after the
  target configuration is correct.

Inherited dashboard and benchmark scripts do not establish that a continuous
Nabla GPU CI run completed. Report actual completed runs, failures, variation
and output correctness; no general speedup follows from code presence alone.
