# LLM Smoke Test

> **Upstream reference in Nabla.** This material is retained for the upstream
> component or API it describes. Models and hardware outside
> [Nabla's target scope](https://github.com/zhangxin81/nabla-vllm/blob/main/docs/models/supported_models.md)
> are not enabled by retaining this example. Use a matching Nabla checkout,
> local target weights and NVIDIA CUDA settings before adapting it.

Start headless `vllm`:

```bash
source ../vllm/.venv/bin/activate
HF_HUB_OFFLINE=1 \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_CPU_KVCACHE_SPACE=2 \
VLLM_HOST_IP=127.0.0.1 \
VLLM_LOOPBACK_IP=127.0.0.1 \
python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1 \
  --max-model-len 512 \
  --dtype float16
```

Run the Rust smoke test through the `vllm-llm` generate interface:

```bash
cargo run -p vllm-llm --example external_engine_smoke -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1
```

IMPORTANT: You must restart `vllm` each time you run the smoke test, as the vLLM engine cannot manage frontend closures and subsequent reconnects. In other words, do not reuse existing `vllm` instances, if any.
