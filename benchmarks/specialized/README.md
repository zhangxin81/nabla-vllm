# Target-model serving matrix

`serve_matrix.py` runs against a healthy, already configured P/D proxy. It
uses the installed vLLM benchmark CLI, preserves detailed request results and
refuses to mix different experiment manifests in one result directory.

The intended topology is eight H100s: GPUs 0–3 for prefill and GPUs 4–7 for
decode, with TP=4, EP=4, DP=1 and PP=1 on each side. Both sides must use the
same source revision and `NixlConnector` protocol. For the same-host IPC arm,
both sides set `kv_connector_extra_config.enable_local_fastpath=true`.

Start the servers with the recorded model-specific settings and close the
prefix cache for the main matrix. Set `PYTHONPATH` to one fixed benchmark
client source for all arms. The manifest should record server/client source
SHAs, model fingerprints, dependency versions, hardware, sampling and server
arguments. At minimum the runner consumes this model mapping:

```json
{
  "models": {
    "qwen": {"weight_path": "/models/qwen"},
    "deepseek": {
      "weight_path": "/models/deepseek",
      "tokenizer_mode": "deepseek_v41"
    },
    "qwen_fp8_additional": {"weight_path": "/models/qwen-fp8-additional"}
  }
}
```

The keys must match `--served-model-name`. Use a new output root whenever
sources, checkpoints or other experiment conditions change:

```bash
.venv/bin/python benchmarks/specialized/serve_matrix.py \
  --manifest experiment.json --output-root results/new-experiment \
  --model qwen --arm upstream_nixl --round 1
```

Repeat for the candidate NIXL and candidate IPC arms and all three model
checkpoints. Round 1 has 39 cases. Rounds 2 and 3 select concurrency 1, 8 and
the smallest concurrency reaching 95% of the upstream round-1 peak for each
input/output shape. The selected points are shared across arms.

```bash
.venv/bin/python benchmarks/specialized/serve_matrix.py \
  --manifest experiment.json --output-root results/new-experiment \
  --model qwen --arm candidate_ipc --round 2
```

`--dry-run` prints commands without sending requests or writing results.
Each measured case uses one warmup, `max(8, 2 * concurrency)` requests,
greedy sampling, ignore-EOS and fixed length/seed parameters. Report actual
request lengths, completion counts, token throughput, TTFT, TPOT, ITL and
end-to-end latency. Low-sample p95/p99 values describe these runs and do not
estimate production SLAs. Shared-prefix workloads are separate from this
matrix and must report cache-hit metrics and the actual reusable boundary.
