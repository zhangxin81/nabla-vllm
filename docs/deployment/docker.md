# Container deployment for nabla-vllm

Build the NVIDIA image from this checkout so it contains Nabla's target-model
closure, operator buckets and retained P/D implementation. Upstream
`vllm/vllm-openai`, CPU/ROCm/XPU images and nightly tags are upstream artifacts,
not this repository's builds.

## Build image from source

The retained Docker target name is `vllm-openai`; the image tag is chosen locally:

```bash
docker build --target vllm-openai -t nabla-vllm:local -f docker/Dockerfile .
```

The Dockerfile pins its Python, CUDA and native build dependencies. Initialize
submodules, record the source revision and verify the build on the intended GPU.
Python/native extension and optional Rust frontend artifact reuse require
matching source/build inputs.

## Run a target model

Mount an existing checkpoint directory and persistent model/kernel caches.
For a four-GPU Qwen example, replace the placeholder model path:

```bash
docker run --rm --gpus all --ipc=host \
  -v /models:/models:ro -v nabla-cache:/root/.cache/vllm \
  -p 8000:8000 nabla-vllm:local /models/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
  --engram-config '{"cpu_offload":true}' --max-model-len 8192
```

Use [model-specific measured settings](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md)
for precision and KV budgets. Persisting cache directories avoids repeated
downloads/compilation where the cache identity is still valid; changed source,
Torch/CUDA, hardware or configuration can require new artifacts.

## P/D deployment

Use matching images and protocol versions on P and D. Follow
[NIXL configuration](../features/nixl_connector_usage.md) for roles, cache
geometry, ordinary NIXL and explicit same-host IPC. The
[H100 matrix](../benchmarks/nabla-vllm/h100-cache-rollback-20261122.md)
records one tested topology; it does not certify every container or GPU layout.

The Dockerfile also retains `vllm-openai-nonroot` and other inherited build
targets. Inspect their actual user/cache permissions before adopting a target.
They are build mechanisms, not a published Nabla image support matrix.
