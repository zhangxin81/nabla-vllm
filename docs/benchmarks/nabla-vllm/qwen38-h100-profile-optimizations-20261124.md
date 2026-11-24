# Qwen3.8 的 H100 优化验收

本轮保留一项已验证的改动：融合 MoE 的 BF16 SiLU-and-mul 与 UE8M0 FP8
量化。在 8×H100、4P4D、两侧 TP4／EP4、NixlConnector CUDA IPC 下，54 点
服务对照的 18 个配对吞吐比中位数为 **1.019829**，TTFT 比为 **0.969314**，
TPOT 比为 **0.985455**。这是相对本轮未修改的 nabla-vllm 基线的增量，
不能直接当作相对原始 vLLM 的增幅，也不能与历史表中的增幅相加。

本轮基线为 `fffdea13c889d0d64947c05ad49cb9150c2ec316`。
它已回退统一算子分桶；此前 [算子 profile](qwen38-h100-operator-profile-20261124.md)
和旧候选的数据对应另一份源码。融合候选已移到新主干并重新执行全部服务对照。
测试时的 runtime 文件 SHA256、native 库身份和原始数值记录见
[验证数据](qwen38-h100-actquant-20261124.json)。

| 项目 | 验证与取舍 |
| --- | --- |
| 1：Allreduce 分派、同步与 staging | 完整 TP4 wrapper 对照未支持更换分派；扩大 Prefill Graph 的候选未通过数值验收，已回退 |
| 2：小 M 投影与 HC | 多轮代码和服务对照未形成稳定收益，已回退 |
| 3：专家 GEMM 配置寻优 | 用户驳回，未实施 |
| 4：MoE 激活与量化 | 通过严格算子、真实输入及完整 P/D 检查，保留受限 H100 融合 |
| 5：H100 QSA Prefill | 107 条配置筛选、22 个 GPU 用例和 72 点服务对照完成；目标尾部负载未获益，已回退 |

## 融合减少一次 launch 和中间读写

原路径在 GEMM1 后先写回 BF16 SiLU-and-mul 的结果，再读取它做 group-128
UE8M0 FP8 量化。新 kernel 在寄存器里完成两步，保留 SiLU 与乘法后的两次
BF16 舍入，输出 row-major FP8 字节和 FP32 scales。NaN、负零、absmax 和
UE8M0 的幂次取整规则与被替代的 native 路径逐字节对照。

启用条件限定为 H100、640 维专家中间宽、BF16 输入、e4m3fn、`[128,128]`
block 量化和 UE8M0。sorted TMA、clamp、显式 a2 scale、LoRA、hook、量化
emulation 及其他宽度保留原路径。GEMM 配置、路由、combine、缓存与 Connector
均未修改。行数不超过 64 时采用 BM1／W4，65～640 为 BM4／W4，更大为 BM4／W1。

| 路由行数 | 原激活＋量化，冷 L2 | 融合，冷 L2 |
| ---: | ---: | ---: |
| 10 | 4.3904 μs | 1.9520 μs |
| 40 | 4.4128 μs | 2.1472 μs |
| 80 | 4.4800 μs | 2.1312 μs |
| 640 | 5.2192 μs | 2.8160 μs |
| 40960 | 88.3968 μs | 49.7888 μs |

计时使用 CUPTI 和 CUDA Graph，排除编译、输入复制与诊断。表格测量最终
runtime helper，未逐行选择其他配置；hot L2 结果同样保留在验证数据中。
原型的 372 条严格检查和 21 条配置过渡检查通过。跳过远端路由行的版本更慢，已弃用。

正式 GPU 测试覆盖 offset view、空行、行数配置边界、动态 Graph 输入、全部
65,536 种 BF16 gate 位模式，以及 M=1／8、EP map=1／4 的完整 MoE 对照。
EP map 用例在单 GPU 执行；实际八卡通信使用服务对照验证。

```bash
.venv/bin/python -m pytest tests/kernels/moe/test_block_fp8.py \
  -k unsorted_silu_ue8m0 -v
.venv/bin/python benchmarks/kernels/benchmark_silu_mul_ue8m0.py \
  --rows 10 40 80 640 40960 --output-json actquant-micro.json
```

```text
15 passed, 343 deselected, 1 warning in 8.51s
```

warning 来自零行用例捕获空 Graph。原 CUDA 参考需要对 offset view 显式 clone
以保证地址对齐，候选始终读取原 offset view，未放宽数值阈值。

## 服务对照保留全部轮次

固定同机 8×H100 80GB HBM3，4 卡 Prefill、4 卡 Decode，两侧 TP4／EP4。
使用 Qwen3.8-Flash-Next-FP8、FP8 主 KV 和 indexer KV、4096-token chunk，
关闭 prefix caching、extensible KV、async scheduling 和 speculative decoding。
每请求输出 128 token，greedy、ignore EOS。软件栈为 PyTorch 2.14.0+cu130、
Triton 3.8.0、Transformers 5.23.0、FlashInfer 0.7.4、NIXL 1.5.0。

按基线 B1、候选 C1、基线 B2 启动三组服务，每组六种负载、三个种子，合计
54 点、1,224 个计时请求、156,672 个输出 token。每点请求数为
`max(8, 2 * concurrency)`，种子为 43、44、45，第二种子的负载顺序反转。
先运行同形状预热；计时期间关闭 profiler、逐 step 审计和真实输入数值检查。
专用 Worker 只在加载与预热后记录源码、cache 几何和已编译的融合变体。

每种子计算 `C1 / sqrt(B1 * B2)`，下表再取三个比值的中位数。吞吐增加为改善，
延迟减少为改善。完整绝对值、p95／p99 与原始轮次见
[54 点 CSV](qwen38-h100-actquant-20261124.csv) 和 [JSON](qwen38-h100-actquant-20261124.json)。

| 输入／并发 | 吞吐变化 | TTFT 变化 | TPOT 变化 |
| --- | ---: | ---: | ---: |
| 128／8 | +1.58% | −2.54% | −1.45% |
| 2048／8 | +0.88% | −6.55% | −1.18% |
| 8192／1 | +1.64% | −1.74% | −1.59% |
| 8192／8 | +3.30% | −5.70% | −1.41% |
| 8192／32 | +2.88% | −3.21% | −1.46% |
| 32768／8 | +2.25% | −2.65% | −1.86% |

15／18 个配对点的 TPOT 下降；首轮三个点的 TPOT 增加约 1.6%～2.0%，
128／8 首轮吞吐下降 1.885%。两组基线自身的单点吞吐漂移最高为 7.703%。
因此保留这项窄路径优化，但不宣称每次测量都更快，也不将合成请求矩阵的中位数
当作生产流量的总收益。B1／C1／B2 能暴露漂移，不能完全消除服务时序与温度影响。

8,583 个普通 tracked 文件、10 个 symlink 已核对。364 个 native 输入匹配
分桶改造前源码，12 个库匹配其构建回执，未复用旧分桶版的 core／MoE 二进制。
三组服务均确认 Decode rank 0～3 的 CUDA IPC，候选八个 rank 在预热后均记录
三个融合变体。所有点的实际输入输出长度、请求数、错误数组、有限延迟与
`output_throughput = output_tokens / duration` 已检查。

## 真实输入检查与模型输出边界分开记录

独立服务在 Graph capture 结束后，对真实 GEMM1 输入的有效本地 EP 路由行，
比较融合与原激活＋量化的 FP8 字节和 scales，继续返回融合输出。
四个 Prefill rank 各检查 528 次，共 **2,112 次调用、18,675,360 个有效行**，
全部精确一致。该检查未进入性能计时。

128、1024、1025-token 的串行探针在候选和两组基线间匹配。长输入在两组
未修改基线间也出现 token 差异，本轮没有证明长输入整模型逐 token 一致性，
也没有执行公开任务准确率评估。算子精确一致是融合契约的证据，不能替代任务分数。
H20、B200 和 DeepSeek 的性能尚未在本轮验证，融合默认限定 H100 的目标形状。

## 用相同配置复测

在同一套依赖、同一权重与匹配 native 产物下分别检出基线和候选。以下命令提供
服务配置；`MODEL_PATH` 替换为本地权重目录。CPU 绑核按本机拓扑设置，两组保持一致。

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-FP8
export VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_USE_RUST_FRONTEND=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 VLLM_PLUGINS=
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_NIXL_SIDE_CHANNEL_PORT=5630 \
  .venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --served-model-name qwen-profile --host 127.0.0.1 --port 8310 \
  --tensor-parallel-size 4 --enable-expert-parallel --renderer-num-workers 4 \
  --max-model-len 131072 --max-num-batched-tokens 4096 --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 --engram-config '{"cpu_offload":true}' \
  --language-model-only --disable-prefix-caching --disable-extensible-kv-cache \
  --no-async-scheduling --moe-backend triton --kv-cache-dtype fp8 \
  --attention-config.indexer_kv_dtype fp8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"enable_local_fastpath":true}}'
```

另一个终端以相同参数启动 D：GPU 改为 `4,5,6,7`，端口改为 `8320`，
side-channel port 改为 `5631`，`kv_role` 改为 `kv_consumer`。随后启动 proxy。

```bash
.venv/bin/python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --host 127.0.0.1 --port 8392 --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 8310 --decoder-hosts 127.0.0.1 --decoder-ports 8320
.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm --model qwen-profile --tokenizer "$MODEL_PATH" \
  --base-url http://127.0.0.1:8392 --endpoint /v1/completions \
  --dataset-name random --random-input-len 8192 --random-output-len 128 \
  --random-range-ratio 0 --random-prefix-len 0 --ignore-eos --temperature 0 \
  --request-rate inf --max-concurrency 8 --num-prompts 16 --num-warmups 1 \
  --seed 43 --disable-tqdm --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 --save-result --save-detailed \
  --result-dir results --result-filename i8192-c8-r1.json
```

正式记录由启用相同启动期回执的专用 Worker 采集。以上默认 Worker 命令用于
复现执行配置，不会生成私人回执。对六种负载和三个种子执行相同客户端参数，
再按 B1／C1／B2 的公式复算；完整统计不能用一个点代替。

## 未保留的候选

20 MiB TP4 Allreduce 完整 wrapper 的中位耗时为 multimem 132.304 μs、
two-shot 149.904 μs、PyNCCL 142.176 μs，计时包含 staging 和输出复制，
每个样本取最慢 rank。该实验不足以支持更换主要负载的分派。

小 M 投影与 HC 最后一个 gate 候选通过 23 个 GPU 用例，但 45 个 P/D 点的
三组吞吐变化仅为 +0.098%、+0.203%、−0.138%，并发负载 TTFT 增加 2.34%。
这些实验对应分桶回退前的源码，只说明当时没有形成可接受的增量，不当作当前
主干的新性能结论。候选代码已回退，原始记录保留在实验资料中。

## QSA 的局部收益未通过服务验收

第 5 项已完成验证，候选回退。基线是包含已接受激活量化融合的
`7a2fbe18e2a259d27cf7a7349c7001604964dc83`；候选只把特定 H100 FP8 Prefill 的
2047／2048 行从四路 split-K 改为一路，减少中间 workspace 与 merge。
限制为 6 个 Q head、1 个 KV head、256 维 head、1600-token page、
2051 列逻辑选择索引。selection budget 2048、compress ratio 4、KV 内容、
缓存分配、GDN checkpoint、Connector 和 MoE GEMM 均未改动。
4096 行原本已经使用一路 split-K，因此大 chunk 没有新增优化。

52 条初始配置与 55 条中间行数筛选均保留在
[QSA 验证 JSON](qwen38-h100-qsa-screen-20261124.json)。中间行数收益不单调，短 context 的较小行数
切到一路反而更慢，因此只实施了 2047／2048 行候选。最终 runtime helper
独立导入基线与候选后计时，冷 L2 结果如下；基线列取前后测量的几何均值。

| Q 行数／context | 基线，μs | 候选，μs | 最大绝对差 |
| --- | ---: | ---: | ---: |
| 128／2048 | 46.373 | 46.246 | 0.00000000 |
| 2047／2048 | 309.097 | 310.829 | 0.00390625 |
| 2047／8192 | 515.934 | 479.509 | 0.00097656 |
| 2048／32768 | 524.704 | 486.646 | 0.00097656 |
| 4096／8192 | 956.072 | 955.880 | 0.00000000 |

2047／8192 与 2048／32768 的完整算子耗时下降约 7%，2047 行、2048-token
context 略慢。改变 split 数会改变浮点归约顺序，目标区域未达到 bit exact；
保留原有 `rtol=2e-2, atol=2e-2`，没有放宽阈值。GPU 测试覆盖实际跨块
5.337 GB 字节偏移、2046～2049 行、Prefill／Decode 和动态 Graph 输入、
有效 count 归零。warmup 与 runtime 使用相同分派条件，候选的 CI 配置显式选择这些用例。

```bash
.venv/bin/python -m pytest tests/models/qwen4_exp/test_qsa_reference.py \
  -k 'strided_fp8_prefill or sparse_paged_attention_correctness' -v
```

```text
22 passed, 116 deselected in 21.87s
```

以上是候选验证记录；回退后的主干保留既有测试，候选补丁与新增测试保存在
实验资料中。它们没有作为新 runtime 功能合入。

服务测试沿用本页的 8×H100、4P4D、TP4／EP4、NixlConnector CUDA IPC
和精度配置。增加预先登记的 6144／1、6144／8 两种输入，使 4096-token
chunk 后出现目标 2K 尾部。B1／C1／B2 每组 24 点，共 **72 点、1,440 请求、
184,320 输出 token**。每点三个种子的比较仍为 `C1 / sqrt(B1 * B2)`，
所有长度、请求数、错误数组与吞吐公式通过检查，三个服务组均确认四个 Decode
rank 的 IPC。计时阶段没有 profiler 或逐调用数值检查。

| 输入／并发 | 吞吐变化 | TTFT 变化 | TPOT 变化 |
| --- | ---: | ---: | ---: |
| 128／8 | +0.39% | -1.80% | -0.19% |
| 2048／8 | +0.59% | -0.77% | -0.58% |
| 8192／1 | +0.20% | -0.28% | +0.11% |
| 8192／8 | +0.15% | -1.08% | +0.07% |
| 8192／32 | +0.08% | -0.52% | -0.18% |
| 32768／8 | +1.27% | -1.93% | -0.03% |
| 6144／1 | -0.12% | +0.26% | -0.04% |
| 6144／8 | -0.38% | +1.27% | -0.04% |

24 个配对吞吐比的中位数为 **1.002878**，
TTFT 比为 **0.991410**，TPOT 比为
**0.998974**。其中 7／24 个点吞吐下降，
两组基线单点吞吐漂移最高为 3.572%。针对 2K 尾部的 6144-token 负载没有
显示收益，8 并发 TTFT 还增加约 1.27%。32K 输入的改善和不变路径的漂移同时
存在，这组数据不足以把整体变化归因于候选。按稳定服务收益的验收条件回退。

独立真实输入检查在性能测量后执行：四个 Prefill rank 共比较 **720 次**
QSA 调用，包含 **144 次**目标行数调用，全部通过原有数值阈值，最大
绝对差为 **0.00781250**。逐调用的 exact 标志与差值保留在 JSON 中。
这证明了该输入集的算子数值边界。整模型探针中，128、1024、1025、2049-token
输入的 16 个输出 token 匹配；4097-token 输入的两组基线匹配，候选与两组基线
均不匹配。另一些长输入的基线自身也不一致，具体逐组结果保留在 JSON 中。
因此长输入逐 token 一致性未通过，也没有任务准确率评估。
完整绝对值、全部轮次与 p95／p99 见
[72 点 CSV](qwen38-h100-qsa-screen-20261124.csv)。复测时在本页客户端命令中替换输入长度与并发，
覆盖原六种负载及两种 6144-token 负载，保持三个种子和 B1／C1／B2 顺序。

第 4 项公共微基准 CLI 也已在 H100 上复核：10 行冷 L2 的原路径为
4.384 μs，已接受融合为 1.9456 μs；runtime 文件指纹匹配已合入代码。
