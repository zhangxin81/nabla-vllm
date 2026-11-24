---
name: review-performance-evidence
description: Use when a Nabla issue, PR or document claims a performance improvement or regression; assess whether the measurements support the claim.
---

# Review Nabla performance evidence

Identify the baseline and candidate source SHAs, model/checkpoint identity,
GPU and parallelism, Torch/CUDA dependencies, precision and kernel selection.
Require full commands and matched request shapes, sampling and cache settings.
Read the public benchmark reports and raw artifacts referenced by the change.

Classify the measurement as a kernel, model-forward or full-serving result.
Compare only equivalent workloads. Check repetitions, variation, failed
requests, latency, regressions and correctness or quality checks. Do not
convert a microbenchmark gain into a serving claim, or token completion into
model-quality evidence.

If evidence is missing, name the exact measurement needed. Reuse the existing
kernel-microbenchmark, model-microbenchmark or PR-checklist skill to design it.
Do not launch GPU work, obtain checkpoints or widen credentials just to review
an artifact. Report established findings separately from untested hypotheses.
