# CPU inference is outside the Nabla runtime

nabla-vllm retains NVIDIA CUDA inference for its two target models. CPU, Zen,
ROCm, XPU and TPU platform implementations were removed from this downstream.
Use the [CUDA installation guide](gpu.md).

CPU execution still has roles in scheduling, metadata, offloaded Engram tables
and metadata-only tests with `VLLM_TARGET_DEVICE=empty`. Those uses do not
provide a CPU model runner.

The `cpu.*.inc.md` files retain upstream installation material for reference.
Current upstream CPU support is documented by
[vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/);
it should not be attributed to nabla-vllm.
