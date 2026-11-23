# Target-model offline inference

The retained `vllm.LLM` Python interface runs the target model without a
separate HTTP server. Install this checkout with `uv`, reuse existing weights,
and use `.venv/bin/python`.

## Generate or chat with explicit model settings

The `generate.py` and `chat.py` scripts accept engine arguments and sampling
parameters. Their inherited defaults select an unrelated model, so pass the
local target checkpoint and its deployment settings explicitly:

```bash
.venv/bin/python examples/basic/offline_inference/generate.py \
  --model /models/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
  --engram-config '{"cpu_offload":true}' --max-model-len 8192 \
  --temperature 0 --max-tokens 32
```

For chat, use `chat.py` with the same engine settings and the checkpoint's chat
template. Inspect arguments with `--help`. DeepSeek requires its own precision
and tokenizer settings from the
[measured configuration](../../../docs/benchmarking/README.md).

`--generation-config auto` loads checkpoint sampling defaults;
`--generation-config vllm` selects runtime defaults. Record this setting in
comparisons because it can change sampling and output limits.

## Inherited examples

`basic.py` uses a fixed unrelated model; `classify.py`, `embed.py` and `score.py`
require pooling models. They remain upstream API references and are outside
Nabla's native generation scope. Generic GGUF or CPU-offload examples do not
validate these target checkpoints. See the
[quickstart](../../../docs/getting_started/quickstart.md) for the primary path.
