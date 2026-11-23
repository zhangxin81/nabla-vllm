# Structured Outputs

Use a running Nabla target-model server configured through the
[quickstart](../../../docs/getting_started/quickstart.md). Constraint backend,
reasoning parser and checkpoint combinations require their own output checks.

This script demonstrates various structured output capabilities of vLLM's OpenAI-compatible server.
It can run individual constraint type or all of them.
It supports both streaming responses and concurrent non-streaming requests.

Start a supported target-model server with its explicit deployment settings.

```bash
nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
  --engram-config '{"cpu_offload":true}'
```

For reasoning-enabled requests, select the parser matching the target
checkpoint and validate it with the chosen constraint backend.

Install the client dependencies into the repository environment with
`uv pip install openai pydantic`, then run the retained client script below.

See [feature docs](../../../docs/features/structured_outputs.md) for more information.

!!! tip
    If vLLM is running remotely, then set `OPENAI_BASE_URL=<remote_url>` before running the script.

## Usage

Run all constraints, non-streaming:

```bash
.venv/bin/python examples/features/structured_outputs/structured_outputs_client.py
```

Run all constraints, streaming:

```bash
.venv/bin/python examples/features/structured_outputs/structured_outputs_client.py --stream
```

Run certain constraints, for example `structural_tag` and `regex`, streaming:

```bash
.venv/bin/python examples/features/structured_outputs/structured_outputs_client.py \
    --constraint structural_tag regex \
    --stream
```

Run all constraints, with reasoning models and streaming:

```bash
.venv/bin/python examples/features/structured_outputs/structured_outputs_client.py --reasoning --stream
```
