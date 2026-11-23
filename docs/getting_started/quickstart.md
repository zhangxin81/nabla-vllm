# Quickstart

This guide runs nabla-vllm with a local Qwen3.8-Flash-Next or DeepSeek-V4.1-Flash
checkpoint on NVIDIA CUDA. It is not the upstream vLLM model/hardware catalog.
See [target-model scope](../models/supported_models.md) before choosing weights.

## Prerequisites

- Linux and NVIDIA CUDA, with a compatible driver, toolkit and PyTorch build.
- Python 3.11–3.14; use `uv` and the repository's `.venv`.
- Enough GPU/host memory for the selected checkpoint and parallel layout.
  The recorded target-model runs use TP4/EP4 and CPU-offloaded Engram tables.
- An existing local checkpoint. Reuse persistent weights or an approved storage
  source rather than downloading them for every run.

## Installation

Install [this repository](installation/README.md). Installing the upstream
`vllm` package or using a generic upstream image does not reproduce Nabla's
specialized source and operator buckets. The distribution and primary command
are `nabla-vllm`; `vllm` remains the Python import and a compatible CLI alias.

## Offline Batched Inference

Replace the placeholder with your Qwen checkpoint directory. DeepSeek requires
its own tokenizer and quantization settings; use the recorded model-specific
[benchmark configurations](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md).

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/models/Qwen3.8-Flash-Next-FP8",
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    language_model_only=True,
    engram_config={"cpu_offload": True},
    max_model_len=8192,
)
outputs = llm.generate(["Explain continuous batching."],
                       SamplingParams(temperature=0, max_tokens=32))
for output in outputs:
    print(output.outputs[0].text)
```

Save as a local script and run it with `.venv/bin/python`. This text-only example
does not validate every checkpoint precision, speculative method or topology.

## Online Serving

Start the same checkpoint through the retained HTTP frontend:

```bash
.venv/bin/nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 \
  --host 127.0.0.1 --served-model-name nabla-model \
  --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
  --engram-config '{"cpu_offload":true}' --max-model-len 8192
```

The server model name is used in requests; it is distinct from the checkpoint
path. A Completions request uses `model` and `prompt`:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nabla-model","prompt":"Explain continuous batching.","max_tokens":32,"temperature":0}'
```

Chat Completions uses `messages` and the checkpoint's chat template. See
[online serving](../serving/online_serving/README.md) for protocol details and
authentication limitations.

## Performance and P/D

Use [optimization settings](../configuration/optimization.md) with the intended
input/output sizes and precision. Wider CUDA Graph coverage can change padding,
kernel selection and floating-point output; full-model `torch.compile` was not
enabled by the H20 optimization record.

For separate prefill and decode instances, follow
[NIXL configuration](../features/nixl_connector_usage.md). The same-host IPC path
is explicit and has additional geometry and lifecycle requirements. Keep both
sides on matching source and protocol versions, and measure TTFT, TPOT,
throughput and output correctness together.
