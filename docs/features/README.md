# Nabla inference features

The feature scope is the retained NVIDIA execution path for
[Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash](../models/supported_models.md).
Compatibility depends on the checkpoint, backend, hardware and parallel layout;
upstream's broad feature-by-feature table does not validate these combinations.

| Area | Nabla behavior and evidence |
| --- | --- |
| Operator dispatch | [Unified multidimensional buckets](../design/operator_kernel_buckets.md) retain per-op thresholds and Prefill/Decode/Mixed/Verify identities. |
| Input and CPU planning | [PLE staging, workspace/Humming plans and attention metadata reuse](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md) have separately scoped component and serving results. |
| CUDA Graphs | [SM90 defaults](../configuration/optimization.md#nabla-sm90-graph-coverage-and-host-planning) separate prefill and decode grids; explicit settings take precedence and padding can change output. |
| Prefix and recurrent cache | [Current cache policy](../design/specialized_hybrid_cache.md) uses upstream hybrid BlockPool/checkpoint management; the earlier independent ownership design was reverted. |
| P/D transfer | [NixlConnector](nixl_connector_usage.md) retains ordinary NIXL and explicit same-host CUDA IPC; matching configuration and lifetime checks are required. |
| Engram/PLE storage | Target tables can be CPU-offloaded with the recorded configurations; [storage choices](../configuration/optimization.md) are constrained by capacity and address lifetime. |
| Quantization | [Checkpoint formats and kernel selection](quantization/README.md) are model- and hardware-specific. |
| Speculative decoding | [Retained target-compatible drafts/MTP](speculative_decoding/README.md) need their own output, acceptance and performance validation. |

The H20/H100 records distinguish components, complete services, actual output
checks and unvalidated combinations. Presence of LoRA, pooling, ASR, EPD or
other inherited API/configuration code is not evidence that unrelated models
or non-CUDA platforms are supported here.
