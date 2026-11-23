# Disaggregated Prefill V1

> **Upstream reference in Nabla.** This material is retained for the upstream
> component or API it describes. Models and hardware outside
> [Nabla's target scope](https://github.com/zhangxin81/nabla-vllm/blob/main/docs/models/supported_models.md)
> are not enabled by retaining this example. Use a matching Nabla checkout,
> local target weights and NVIDIA CUDA settings before adapting it.

This example contains scripts that demonstrate disaggregated prefill in the offline setting of vLLM.

## Files

- `run.sh` - A helper script that will run `prefill_example.py` and `decode_example.py` sequentially.
    - Make sure you are in the `examples/disaggregated/example_connector` directory before running `run.sh`.
- `prefill_example.py` - A script which performs prefill only, saving the KV state to the `local_storage` directory and the prompts to `output.txt`.
- `decode_example.py` - A script which performs decode only, loading the KV state from the `local_storage` directory and the prompts from `output.txt`.
