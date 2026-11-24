---
applyTo: "csrc/**,vllm/**/nvidia/**,vllm/**/kernels/**"
---

# CUDA changes

Read the applicable directory's `AGENTS.md` and the kernel-writing or
microbenchmark skill before changing a kernel. Preserve target-model and
architecture coverage; a launch or syntax check is not GPU correctness.
Include matched baseline and candidate measurements for performance claims.
