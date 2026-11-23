# nabla-vllm CLI Guide

The primary command is `nabla-vllm`; `vllm` remains a compatible alias.
Use a local target checkpoint on NVIDIA CUDA and include its parallelism,
quantization and Engram settings from the [quickstart](../getting_started/quickstart.md).
The examples below show command syntax; they do not validate every retained
frontend feature or speculative method. Start with:

```bash
nabla-vllm --help
```

Detailed CLI reference pages are generated from this checkout during the full
docs build. In a source-only checkout, use `nabla-vllm <subcommand> --help=all`
for the current argument list.

Available Commands:

```bash
nabla-vllm {chat,complete,serve,launch,bench,collect-env,download-kernels,run-batch,preload}
```

## serve

Starts the vLLM OpenAI Compatible API server.

Start with a model:

```bash
nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
  --engram-config '{"cpu_offload":true}'
```

Specify the port:

```bash
nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
  --engram-config '{"cpu_offload":true}' --port 8100
```

Serve over a Unix domain socket:

```bash
nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
  --engram-config '{"cpu_offload":true}' --uds /tmp/vllm.sock
```

Check with --help for more options:

```bash
# To list all flags
nabla-vllm serve --help=all

# To view an argument group
nabla-vllm serve --help=ModelConfig

# To view a single argument
nabla-vllm serve --help=max-num-seqs

# To search by keyword or flag name
nabla-vllm serve --help=max
```

!!! tip "Human-readable integer arguments"
    Many integer arguments accept human-readable suffixes for convenience. For example:

    - `1k` = 1,000 (decimal kilo)
    - `1K` = 1,024 (binary kibibyte)
    - `1m` = 1,000,000 (decimal mega)
    - `1M` = 1,048,576 (binary mebibyte)
    - `1g` / `1G` = 1 billion / 1 gibibyte
    - `1t` / `1T` = 1 trillion / 1 tebibyte
    
    Decimal suffixes (`k`, `m`, `g`, `t`) also accept floating point: `25.6k` = 25,600.
    Binary suffixes (`K`, `M`, `G`, `T`) require integers: `32K` = 32,768.
    
    Supported arguments include: `--max-model-len`, `--max-num-batched-tokens`, `--max-num-scheduled-tokens`, `--kv-cache-memory-bytes`, `--safetensors-prefetch-block-size`.

See [vllm serve](./serve.md) for the full reference of all available arguments.

## launch

Launch individual vLLM components.

```bash
# Launch the rendering server component
nabla-vllm launch render /models/Qwen3.8-Flash-Next-FP8

# Inspect all available flags for the render component
nabla-vllm launch render --help=all
```

See [vllm launch render](./launch/render.md) for the current launch
component reference.

## chat

Generate chat completions via the running API server.

```bash
# Directly connect to localhost API without arguments
nabla-vllm chat

# Specify API url
nabla-vllm chat --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# Quick chat with a single prompt
nabla-vllm chat --quick "hi"

# Print TTFT and throughput statistics after each response
nabla-vllm chat --stats
```

See [vllm chat](./chat.md) for the full reference of all available arguments.

## complete

Generate text completions based on the given prompt via the running API server.

```bash
# Directly connect to localhost API without arguments
nabla-vllm complete

# Specify API url
nabla-vllm complete --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# Quick complete with a single prompt
nabla-vllm complete --quick "The future of AI is"

# Print TTFT and throughput statistics after each response
nabla-vllm complete --stats
```

See [vllm complete](./complete.md) for the full reference of all available arguments.

## bench

Run benchmark tests for latency online serving throughput and offline inference throughput.

To use benchmark commands, please install with extra dependencies using `uv pip install -e ".[bench]"`.

Available Commands:

```bash
nabla-vllm bench {forward, latency, serve, throughput}
```

### forward

Benchmark the model forward of exact batches: each case lists its requests as `[N*]Q[@K][c][:D]` (query length Q, KV length K including the query, D draft tokens). A prefill's prefix is a prefix-cache hit unless `c` marks a chunked prefill; they differ only on models with SWA bounded replay, which recompute the last window of a hit. Each request's prefix length varies by ±50% by default (`--kv-len-jitter`), identically on every DP rank. Each step (forward and target sampling) is captured as a one-shot CUDA graph; CPU overhead isn't measured. Requests use the model's default sampling parameters from `generation_config.json` (`--generation-config`, `--override-generation-config`), or temperature 1.0 without top-p/top-k if it sets none. Use `--exclude-sampler` to drop target sampling and rejection sampling from the timing. Use `torchrun`, one process per GPU, for DP/TP/EP/PCP/PP. CUDA only for now. Add `--profile-dir DIR` to write a Perfetto trace and a per-kernel GPU time summary of each case; `.venv/bin/python -m vllm.benchmarks.forward.profile A.kernels.json [B.kernels.json]` prints a summary or compares two. Under Nsight Systems, `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat --cuda-graph-trace=node` records just those replays, one report per case.

```bash
nabla-vllm bench forward \
    --model /models/Qwen3.8-Flash-Next-FP8 \
    --case decode=32*1@4096 \
    --case mix=2048,32*1@4096
```

For hybrid attention/recurrent models, the benchmark reserves the recurrent
state slots for all configured speculative tokens. Every replay restores these
states and the attention blocks it can overwrite, including packed KV layouts.
The snapshots consume extra GPU memory, but their restoration is outside the
measured forward interval.

PCP uses `--prefill-context-parallel-size` with the model runner's supported
MLA models. Cases describe the global batch per DP rank, before PCP splits it;
`num_local_tokens`, `num_tokens_after_padding`, and `batch_order` in each rank's
JSON result describe the local partition (a request can appear twice).
Latency is the maximum across all ranks. For example, with four GPUs:

```bash
torchrun --nproc-per-node=4 -m vllm.entrypoints.cli.main bench forward \
    --model MODEL --tensor-parallel-size 2 --prefill-context-parallel-size 2 \
    --case prefill=8192 --case mix=2048,8*1@4096
```

With `--pipeline-parallel-size`, the benchmark streams identical microbatches
through captured stages. `median_us` measures each microbatch from its first
stage's forward start to its last stage's forward completion, including PP
transfers while other microbatches are in flight. Pipeline fill and drain are
excluded. This is latency under steady-state load, not the completion interval.
Per-rank JSON includes stage timelines and the estimated clock synchronization
uncertainty in microseconds.

```bash
torchrun --nproc-per-node=4 -m vllm.entrypoints.cli.main bench forward \
    --model MODEL --pipeline-parallel-size 2 --tensor-parallel-size 2 \
    --case prefill8x1k=8*1024 --case prefill1x8k=8192
```

With `--speculative-config`, timing includes the target forward, rejection
sampling of verify rows, and the MTP, EAGLE, DSpark, or DFlash proposal,
including the drafter's token sampling. Use `--exclude-drafter` to drop the
proposal from the timing. The drafter assumes all draft tokens are accepted and
uses a fixed bonus token, independent of the rejection sampler's result. All
paths use CUDA graphs and restore the batch's state before each replay.
Synthetic MoE routing fixtures (`--routing balanced` or `--routing zipf`) apply
to the target model; the drafter uses its model routing, so use real weights
when measuring a MoE drafter. For the Zipf distribution and its limitations,
see [Model Forward Benchmark](../benchmarking/cli.md#model-forward-benchmark).

```bash
nabla-vllm bench forward --model /models/Qwen3.8-Flash-Next-FP8 \
    --speculative-config '{"model":"/models/target-compatible-draft","method":"dflash","num_speculative_tokens":3}' \
    --case verify=16*4@4096:3
```

See [vllm bench forward](./bench/forward.md) for the full reference of all available arguments.

### latency

Benchmark the latency of a single batch of requests.

```bash
nabla-vllm bench latency \
    --model /models/Qwen3.8-Flash-Next-FP8 \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

See [vllm bench latency](./bench/latency.md) for the full reference of all available arguments.

### serve

Benchmark the online serving throughput.

```bash
nabla-vllm bench serve \
    --model /models/Qwen3.8-Flash-Next-FP8 \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

See [vllm bench serve](./bench/serve.md) for the full reference of all available arguments.

### throughput

Benchmark offline inference throughput.

```bash
nabla-vllm bench throughput \
    --model /models/Qwen3.8-Flash-Next-FP8 \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

See [vllm bench throughput](./bench/throughput.md) for the full reference of all available arguments.

## collect-env

Start collecting environment information.

```bash
nabla-vllm collect-env
```

## download-kernels

Install the precompiled kernels that vLLM's dependencies would otherwise download
or compile at startup. Run it after installing or upgrading vLLM. It currently
installs FlashInfer's kernels and does nothing when FlashInfer is not installed.

```bash
nabla-vllm download-kernels

# Print the install commands without running them
nabla-vllm download-kernels --dry-run
```

See [vllm download-kernels](./download-kernels.md) for the full reference of all available arguments.

## run-batch

Run batch prompts and write results to file.

Running with a local file:

```bash
nabla-vllm run-batch \
    -i examples/features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model /models/Qwen3.8-Flash-Next-FP8
```

Using remote file:

```bash
nabla-vllm run-batch \
    -i https://raw.githubusercontent.com/zhangxin81/nabla-vllm/main/examples/features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model /models/Qwen3.8-Flash-Next-FP8
```

Batch request `model` fields must match the served model name. The inherited
embedding examples require pooling model registrations, which are outside
Nabla's native target scope.

See [vllm run-batch](./run-batch.md) for the full reference of all available arguments.

## preload

Launch weight cache daemons (one per GPU) that hold the post-quantized,
TP-sharded weights in GPU memory and serve CUDA IPC handles to vLLM engines
over a Unix domain socket. Restarting engines then map the weights via
zero-copy IPC instead of reloading from disk, enabling fast engine restarts.

```bash
# Launch one daemon per GPU
nabla-vllm preload --model /models/Qwen3.8-Flash-Next-FP8 --tensor-parallel-size 4

# Engines then load from the daemons
nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 --tensor-parallel-size 4 \
    --load-format ipc_cache
```

The daemon accepts the standard engine arguments (model, dtype, quantization,
tensor-parallel-size, ...) plus `--weight-cache-socket-dir` to override the
directory holding the per-GPU Unix sockets, and `--weight-cache-master-port` /
`--weight-cache-draft-master-port` to pin the daemon rendezvous ports for
multi-node and speculative-decoding setups. Tensor, expert and data
parallelism are supported; pipeline parallelism is rejected at launch.

See [Preload](../features/preload.md) for how it works, cache modes, and
limitations, and [vllm preload](./preload.md) for the full reference of all
available arguments.

## More Help

For detailed options of any subcommand, use:

```bash
nabla-vllm <subcommand> --help
```
