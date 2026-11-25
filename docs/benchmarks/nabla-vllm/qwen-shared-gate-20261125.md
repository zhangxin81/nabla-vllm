# Qwen 门控融合的 H100 服务验证

Qwen3.8-Flash-Next-FP8 在 H100、TP4/EP4、1024/128 请求下，C1/C8 的四个配对形状
吞吐提高 **1.07%～1.97%**，TPOT 降低 **1.07%～2.04%**。比较基线是
`84de281406`，已经包含 H100 MoE 激活与 UE8M0 量化融合。本报告只描述共享专家门控的增量。

候选保留原 `expert_gate(x)`、GEMM、权重加载和 hooks，只把 BF16 sigmoid 与乘法融合。
Qwen 模型在 1～8 个 padded rows 时调用新路径；大于 8 行保留原链路。
共享 helper 的 scalar specialization 还要求 Hopper、contiguous BF16、输出宽度不超过 4096，
并保持 sigmoid 的 BF16 舍入。编译、batch-invariant 模式、其他 dtype 和不符合条件的输入走 fallback。
原有 same-shape attention gate 和 FP8 量化链路保持原算法。

方向参考 [SGLang #26727](https://github.com/sgl-project/sglang/pull/26727) 和
[vLLM #43190](https://github.com/vllm-project/vllm/pull/43190)，没有重复提出上游 PR。
该上游 PR 的 GEMM 全融合版本在独立评估中重复出现额外错题，未采用；本实现保留原 GEMM。

## 测量范围与结果

测试源码为 `6be3a044ca96c152e5afe7e74b153e0efc9fcb48`，基线为
`84de2814060397e90aa137a75473199ee1a12563`。
当前主干 `3b2ef7e6bc` 相对该基线只增加文档和 CI，其 runtime 和 native 输入一致。
提交到主干前的 rebase 也没有改变已测试的 kernel、模型 caller、测试或 benchmark 文件。

本轮使用同一台 8×H100 80GB 主机中的 GPU 4–7，单个 TP4/EP4 服务，
FP8 main KV、BF16 indexer KV、chunk=4096、每 rank KV 预算 16 GiB。
每组独占整机并等待连续空闲五分钟，顺序为 A1/B1/B2/A2。
并发 C=1/8/32/64/128、seeds=43/47；每点计时请求数为 `max(8,2*C)`，
warmup 为 `max(16,2*C)`。共 **40 点、3776 个计时请求、483328 个输出 tokens**。
四组使用相同权重、客户端、启动参数和九项冻结依赖，Torch 为 `2.14.0+cu130`。

下面按形状比较两次 A 的均值与两次 B 的均值；TTFT/TPOT 负数表示降低。
重复波动为同组两次测量之差除以该组均值。

| 并发 | seed | 吞吐变化 | TTFT 变化 | TPOT 变化 | 吞吐重复波动 A / B |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 43 | +1.970% | -1.546% | -2.043% | 0.476% / 0.371% |
| 1 | 47 | +1.072% | +0.166% | -1.073% | 0.603% / 0.350% |
| 8 | 43 | +1.231% | -1.555% | -1.151% | 0.420% / 0.398% |
| 8 | 47 | +1.900% | -2.114% | -1.717% | 0.205% / 0.384% |
| 32 | 43 | +2.135% | +1.443% | -1.384% | 4.441% / 0.330% |
| 32 | 47 | +2.275% | -2.872% | -2.500% | 0.695% / 1.635% |
| 64 | 43 | +1.258% | -3.077% | -1.336% | 3.932% / 3.741% |
| 64 | 47 | -0.012% | -1.965% | -0.886% | 1.609% / 3.469% |
| 128 | 43 | +1.624% | -0.923% | -1.829% | 1.677% / 2.239% |
| 128 | 47 | +1.725% | -0.120% | -1.829% | 2.773% / 3.706% |

C1/C8 的收益均超过各自 A/B 重复波动。较高并发的部分变化被重复波动覆盖，
例如 C32/seed43 的基线吞吐重复波动为 4.441%，高于配对提升 2.135%；
C64/seed47 的吞吐变化为 -0.012%，也远小于重复波动。本报告不据此宣称高并发的稳定收益。

四组基础探针均为 **67/67**，长上下文均为 **8/8**，固定 GSM8K 子集均为 **62/64**。
逐条检查输入、预期答案和输出，没有候选两次都新增、基线两次都正确的失败。
这组子集支持本轮回归检查，不是完整模型准确率或统计显著性结论。

所有计时请求的输入输出长度、完成数、错误列表和有限指标均已校验；
`output_throughput = total_output_tokens / duration`。
profiler 只在计时结束后启动，四个 GPU rank 的基线融合调用均为 0，
候选均为 720，确认 profile 中的代码路径发生改变。

[40 点原始指标](qwen-shared-gate-20261125.csv)与
[逐请求延迟、质量记录和 profile 校验信息](qwen-shared-gate-20261125.json.gz)保留了全部形状与回退点。

## 正确性与复现

完整 GPU suite：**33 passed in 19.22s**。
包含 scalar gate 的所有 65536 种 BF16 编码、不同乘数、改变输入的 CUDA graph，
以及真实 Qwen MLP 的 8/9-row 分派边界。原有 same-shape 和 FP8 测试也运行。

```bash
PYTHONPATH=<candidate-source> CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python -m pytest tests/kernels/core/test_fused_sigmoid_gate_mul.py -q
```

完整 `ReplicatedLinear → sigmoid → multiply` wrapper 使用 CUPTI 13.5.0、CUDA graph、cold L2：
M1 从 10.464 μs 降至 6.528 μs（1.603 倍），M8 从 11.392 μs 降至 7.552 μs（1.508 倍）。
七个形状逐位一致；大于 8 行使用相同 fallback，其微小计时变化不计为算子收益。

```bash
PYTHONPATH=<candidate-source> CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python benchmarks/kernels/benchmark_shared_expert_gate.py \
  --output-json gate-wrapper.json
```

服务启动参数如下。A/B 只更换源码及 profile 输出目录；需要同一份 Qwen 权重和匹配的 native 构建。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=<source> OMP_NUM_THREADS=1 \
VLLM_USE_RUST_FRONTEND=0 VLLM_PLUGINS= VLLM_SSM_CONV_STATE_LAYOUT=DS \
VLLM_DEEP_GEMM_WARMUP=skip VLLM_WORKER_MULTIPROC_METHOD=spawn \
  taskset -c 52-102,156-206 .venv/bin/python -m vllm.entrypoints.cli.main serve <checkpoint> \
  --host 127.0.0.1 --port 8307 --served-model-name qwen \
  --tensor-parallel-size 4 --enable-expert-parallel --max-model-len 131072 \
  --max-num-batched-tokens 4096 --max-num-seqs 128 --gpu-memory-utilization 0.95 \
  --engram-config '{"cpu_offload":true}' --language-model-only \
  --disable-prefix-caching --disable-extensible-kv-cache --no-async-scheduling \
  --renderer-num-workers 4 --moe-backend triton --kv-cache-dtype fp8 \
  --attention-config '{"indexer_kv_dtype":"bf16"}' --kv-cache-memory-bytes 17179869184 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"<profile>","torch_profiler_with_stack":false,"torch_profiler_record_shapes":false}'
```

客户端示例为 C1/seed43；其他形状沿用上面的请求数与 warmup 规则。

```bash
.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm --model qwen --tokenizer <checkpoint> \
  --base-url http://127.0.0.1:8307 --endpoint /v1/completions --dataset-name random \
  --random-input-len 1024 --random-output-len 128 --random-range-ratio 0 \
  --random-prefix-len 0 --ignore-eos --temperature 0 --request-rate inf \
  --max-concurrency 1 --num-prompts 8 --num-warmups 16 --seed 43 --disable-tqdm \
  --save-result --save-detailed --result-dir <results> --result-filename c1-s43.json
```

本轮是 BF16 indexer 的优化筛选。v0.2 的历史版本复测采用 FP8 indexer、4P4D 和 420 点，
仍需独立执行；本报告不作为该版本复测结果。代码、测试和数据整理使用了 AI 辅助。
