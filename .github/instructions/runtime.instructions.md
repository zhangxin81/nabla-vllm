---
applyTo: "vllm/models/**,vllm/v1/**,vllm/distributed/kv_transfer/**"
---

# Runtime changes

Review changes against the supported target models, CUDA execution paths,
cache lifetime and P/D continuation. Reuse nearby behavioral tests.
Model-affecting changes need output or accuracy validation; cache and
serving changes need a real runtime reproducer. Report unavailable GPU
coverage explicitly rather than interpreting lightweight CI as runtime validation.
