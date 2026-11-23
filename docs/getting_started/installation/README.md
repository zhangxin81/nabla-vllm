# Installing nabla-vllm

nabla-vllm is built from this repository for the two
[target models](../../models/supported_models.md) on NVIDIA CUDA. CPU, ROCm,
XPU, TPU and other upstream hardware-plugin installation routes are outside
this runtime's inference scope. CPU-offloaded weights or CPU metadata tests
do not add a CPU inference backend.

Use the [CUDA build guide](gpu.md) for source installation and version-matched
binary reuse. The installed distribution and primary executable are
`nabla-vllm`; imports and the compatible `vllm` executable retain their names.

## Choose a reproducible build

- Record the Nabla commit, Python/PyTorch/CUDA versions, GPU model and checkpoint.
- Build the retained CUDA/C++ sources when native inputs differ.
- Reuse an existing binary only when its native build inputs match your source.
- Keep weights and kernel caches persistent; changing commits can invalidate
  compile artifacts even when the model path is unchanged.

Upstream PyPI wheels, nightly indices and `vllm/vllm-openai` images describe
upstream vLLM builds. They are not automatically artifacts of this repository.
For containers, [build from this checkout](../../deployment/docker.md).

The [history migration guide](../../design/nabla_history.md) explains why old
checkout ancestry changed. Source links and measured versions remain pinned in
the [performance records](../../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md).
