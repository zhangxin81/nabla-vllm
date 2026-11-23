# NVIDIA CUDA installation

Build nabla-vllm from its own checkout for Qwen3.8-Flash-Next and
DeepSeek-V4.1-Flash. H20/H100 have recorded validation; retained SM100 paths and
other NVIDIA configurations still need model-specific deployment checks.

## Requirements

- Linux, Python 3.11–3.14 and `uv`.
- An NVIDIA driver, CUDA toolkit, compiler and PyTorch build that are compatible
  with each other and with the selected kernels.
- The build dependencies pinned in `pyproject.toml` and
  `requirements/build/cuda.txt`.
- Initialized source submodules and enough build, GPU and host memory.

The H20 record used Torch 2.14.0+cu130; the Docker build has its own pinned
toolchain. Use the versions recorded for your build rather than mixing a
runtime wheel with unrelated CUDA headers or shared libraries.

## Set up using Python

```bash
git clone --recurse-submodules https://github.com/zhangxin81/nabla-vllm.git
cd nabla-vllm
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=auto
uv pip install -e . --no-build-isolation
```

Select the PyTorch backend matching your intended CUDA environment if `auto`
does not describe it. Run scripts with `.venv/bin/python` and inspect the
installed command with `.venv/bin/nabla-vllm --help`.

### Pre-built wheels {#pre-built-wheels}

For Python-only development, an already verified, matching native artifact can
be reused. An explicit wheel location may be passed through the retained
`VLLM_PRECOMPILED_WHEEL_LOCATION` mechanism. Check the native input hashes,
Torch/CUDA versions and architecture first. Upstream wheel discovery based on
an upstream merge-base is not a substitute for this check after history
compaction.

### Build wheel from source

Use this checkout's build dependencies and source to produce a wheel:

```bash
uv pip install build
.venv/bin/python -m build --wheel --no-isolation
```

For CUDA/C++ development, follow the
[incremental build workflow](../../contributing/incremental_build.md).

## Set up using Docker

Build a local image from the retained `docker/Dockerfile`; see
[container deployment](../../deployment/docker.md). Published upstream image
names should not be relabeled as Nabla images.

## Supported features

Hardware capability, checkpoint quantization and parallel layout determine the
selected implementation. Use the [target-model page](../../models/supported_models.md),
[feature scope](../../features/README.md) and fixed-source performance records.
Archived CPU/ROCm/XPU/Apple installation snippets are upstream references,
not installation paths for this runtime.

## Troubleshooting

Check `nvcc --version`, the driver, PyTorch CUDA version and selected GPU
architecture against the native build inputs. If a reused wheel fails to load,
rebuild the native sources with matching dependencies rather than selecting an
unrelated upstream nightly. Capture the source commit, full install command
and error in a [Nabla installation issue](https://github.com/zhangxin81/nabla-vllm/issues).
