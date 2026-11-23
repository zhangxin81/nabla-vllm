# H20 输入准备与 CPU 计划优化

2026-11-23，nabla-vllm 合入了 PLE 输入融合、IPC 地址索引、workspace 与
Humming 计划缓存、attention 公共元数据复用、关闭缓存时跳过 publication，
以及 SM90 目标模型的独立 CUDA Graph 默认值。本文件记录每项 feature 的实现、
性能证据和验证范围，作为后续复测的固定参照。

最明确的组件收益是 PLE 准备和 IPC 地址查找。正式组合在 H20 的 Qwen B128
服务测试中，整组耗时从 2.949／2.981 秒降到 1.854／1.826 秒，两轮均值减少
37.9%。**这两轮分别只有 47/128、32/128 个请求的输出文本相同，不能把该数字
称为精度等价的服务提升。** DeepSeek 的输出与 logprobs 也有明显差异。

## 源码和测量分开定位

| 记录 | 固定版本 | 范围 |
| --- | --- | --- |
| 首轮 cache publication 消融 | `2a48fc651e98562bef4f95e803712570031c438d`，另测本地分桶快照 | 历史组件与离线生成实验，不能代替新 main 的服务对照 |
| 扩展消融中的 current | `1f075a31586a80627d47c475b62784890dc02ccb` | 已包含统一算子分桶；PLE、workspace、Humming、Graph 原型分别开关 |
| 正式组合的基线 | `1f075a31586a80627d47c475b62784890dc02ccb` | 未加入本文件列出的新优化 |
| 正式实现 | [22dd6e0ddc][implementation] | 输入准备、CPU 计划和 Graph 策略改动及回归测试 |
| 合入后的文档版本 | [0c31551dfc][integration] | 仅比实现提交多一个 decode budget 措辞修正 |

下文中的组件计时和 cProfile 数字来自**原型消融**，没有在最终实现上逐项重新
计时。正式实现补充了缓存失效、不可变计划、Tensor 元数据隔离和指针类型校验。
最后的 HTTP 服务结果比较的是整个正式组合，不能把它拆成各 feature 的独立收益，
也不能把组件降幅相加。

这里的基线已经包含[统一算子分桶](../../design/operator_kernel_buckets.md)，
原生二进制在两边相同。本记录不代表统一分桶相对原始 vLLM 的增益；此前
[H100 缓存回退与 Connector 对照](h100-cache-rollback-20261122.md)使用另一套
源码、硬件和请求矩阵，应独立阅读。

## 已合入 feature 与已有证据

| Feature | 改动 | 已测结果 | 证据层级 |
| --- | --- | --- | --- |
| PLE 输入融合 | 一个 Triton kernel 生成 history context 和 request offsets | UVA、B32、容量 128：21.920 → 3.552 µs，减少 83.8% | 原型 GPU 输入准备组件 |
| IPC 注册地址索引 | 注册时排序；热路径 searchsorted 与批量 bounds check | 32 个区间、312,064 个 descriptor：669.659 → 12.049 ms，减少 98.2% | 合成 CPU 地址翻译；不含传输 |
| Workspace view 缓存 | locked 后按形状、dtype、ubatch／lane 复用 | Qwen B32 函数累计 6.656 → 2.267 ms，减少 65.9% | 原型 cProfile；未证明单项端到端收益 |
| Humming buffer plan 缓存 | 复用不可变 shape／dtype／packing 计划 | DeepSeek B32 原函数体累计 19.069 → 4.017 ms，减少 78.9% | 原型 cProfile；正式实现的命中路径不同 |
| Attention CPU query lengths 复用 | 同批次各 group 共享只读 diff，offset 替换时失效 | 没有独立消融计时；相关回归测试通过 | 功能与组合验证，不填写推算的提升比例 |
| 关闭缓存时跳过 publication | `enable_caching=False` 时不调用 coordinator cache_blocks | 历史 DeepSeek 局部 36.540 → 0.521 µs／32 请求，减少 98.6% | 原型 CPU 组件；没有稳定的生成收益 |
| SM90 独立 Graph 默认值 | 目标模型 prefill 上限 1024，decode token budget 512 | 单项原型 Qwen B32 总耗时 810.592 → 570.405 ms，减少 29.6%，输出不一致 | 原型离线整请求；正式组合另列 HTTP 结果 |

### PLE 融合减少 Graph 外的整数准备

原流程依次复制 query_start_loc、填充尾部、清空 context、转换索引 dtype、
计算历史位置、clamp、gather、where 和复制。新 kernel 直接写入两块固定地址
buffer，保留 GPU computed boundary、request mapping、EOS、inactive rows、
padding 和 64 位历史偏移；disk-offload staging 仍走原有流程。

实现见 [prepare_ple_inputs][ple] 和 [model_state 调用][ple-call]。

原型以容量 128、context 深度 2、history 长度 8192，比较相同输入上的完整准备
流程。使用 CUPTI CUDA Graph span、中位数、cold L2、5 次 dry run 和 100 次
计时，融合 kernel 的 BLOCK 为 128。以下是 GPU 时间，不是每请求时延。

| History | Active requests | 原流程 µs | 融合 µs | 时间减少 |
| --- | ---: | ---: | ---: | ---: |
| CPU UVA | 1 | 22.624 | 2.976 | 86.8% |
| CPU UVA | 32 | 21.920 | 3.552 | 83.8% |
| CPU UVA | 128 | 24.000 | 3.712 | 84.5% |
| HBM | 1 | 22.896 | 2.016 | 91.2% |
| HBM | 32 | 20.704 | 2.464 | 88.1% |
| HBM | 128 | 22.928 | 2.528 | 89.0% |

原型的 240 组完整 buffer 逐位比较通过，覆盖 UVA／HBM、context 深度 1／2／3、
容量 128／130／1024 和空批次到满容量。真实 Qwen 请求另比较了 284 个 rank-step，
context 和 query_start_loc 均无差异，这些校验中的 logprobs 全有限。
正式实现还通过 PLE staging、offload replay 和 rollback 的定向测试。

这项消融没有证明稳定的模型级增益。例如 current Qwen B1 离线整请求中位数
363.157 → 361.394 ms，约减少 0.49%；B32 与 B128 的小幅变化伴随输出差异。
HBM history 放置没有全局开启，它还需要单独评估容量和 staging 成本。

### IPC 地址计划只索引注册信息

原路径将 descriptor 转为 Python list，逐个遍历 allocation。新
[IpcAddressPlan][ipc]在注册时建立只读排序数组，在每次 pull 时使用 NumPy
searchsorted 和批量边界检查。重叠注册仍保留原 first-match 优先级；peer 清理
或重新注册会丢弃旧计划。expand、coalesce、stream／event、通知和 DMA 生命周期
均沿用现有路径。

以下使用合成 int64 地址、2048-byte descriptor、32 个互不重叠的注册区间。
地址翻译取 7 次中位数，完整 span 规划取 3 次中位数，各先 warmup 一次。
索引构造在翻译计时之外；完整规划仅包含 expand → translate → coalesce。

| Descriptor 数 | 翻译原流程 ms | 翻译索引 ms | 完整规划原流程 ms | 完整规划索引 ms |
| --- | ---: | ---: | ---: | ---: |
| 1,024 | 2.148 | 0.036 | 3.382 | 1.228 |
| 41,324 | 87.250 | 1.571 | 149.083 | 66.832 |
| 312,064 | 669.659 | 12.049 | 1,403.935 | 772.915 |

最大一组的翻译减少 98.2%，完整规划减少 44.9%，两者不能混用。原型 748 组
翻译结果和拒绝行为比较通过。正式实现补充 signed int64 bounds、uint64 精度、
不可变数组和注册 precedence 检查；实际双进程／双 GPU 链路完成了
export／import → translate → coalesce → swap_blocks_batch → event，
256 个 descriptor 合成 1 个 span，65,536 字节完全相等，三种越界输入被拒绝。

这些结果不包含完整 P/D 服务测量，不能据此发布传输或 TTFT 的 98.2% 提升。

### Workspace 和 Humming 缓存保留失效条件

[WorkspaceManager][workspace]只在 locked 状态复用 view，每个 workspace
最多缓存 64 个形状组合，保留 ubatch／lane 隔离和 retained-tail 重叠失效，
unlock／growth 时清空。缓存内部保存 detach view，命中时返回新的 detach Tensor，
避免调用者的 `unsqueeze_` 等操作修改缓存对象的 shape。

[Humming get_buffer_metas][humming]按 M、topk、activation、派生维度、expert
数量、dispatch capacity、GEMM 类型和 dtype 建 key，每个实例最多 64 项。
计划和 required_buffers 不可变，初始化配置时重置缓存，相关配置变化产生新 key。

| 消融原型 | 观测 | 原流程累计 ms | 缓存累计 ms |
| --- | --- | ---: | ---: |
| Current Qwen B32 workspace | 192 次调用 | 6.656 | 2.267 |
| Current DeepSeek B32 Humming | 原函数体执行次数 480 → 80 | 19.069 | 4.017 |

这是具体函数的 cProfile 累计时间，含 profiler 开销，不能相加为完整 CPU 时长
或除以 decode step 数。Humming 原型通过外部缓存绕过原函数体；正式实现仍进入
get_buffer_metas，命中后跳过 dtype map、packing 和字典构造。因此“480 → 80”
描述原型中的原函数体执行次数，不是正式方法的调用次数。

Workspace 的单项 current Qwen B32 离线请求中位数为 810.592 → 830.054 ms，
并未形成稳定改善。Humming 的单项 current DeepSeek B32 为
1143.730 → 1113.857 ms，减少约 2.6%，但各版本重复生成均不稳定。
函数级开销下降成立，单项精度等价的端到端收益尚未建立。

### Attention 只共享公共 query lengths

[CommonAttentionMetadata.query_lens_cpu][attention]把同一 batch 的 CPU query
length diff 复用给 GDN、Mamba 和 PLE builder。[跨 group 构建][attention-build]
传递这个只读结果；替换 query_start_loc_cpu 时失效，修改 diff 的 capture 路径
继续使用独立 Tensor。不同 state-slot、window 和 rollback 元数据不共享。

此 feature 没有独立的前后计时。原型曾定位到 current Qwen B32
build_attn_metadata 的 37 次调用累计 38.777 ms，这只是热点定位值；它没有证明
这项缓存能省去多少时间。已有 metadata、workspace 和真实服务检查覆盖新路径。

### 关闭 prefix caching 时省去无效 publication

[allocate_slots][cache-skip]在 `enable_caching=False` 时直接跳过可发布 token 数
计算和 coordinator.cache_blocks。开启缓存时仍执行原流程，不改 publication、
rollback 或 cache ownership。

历史 `2a48fc651e` 消融使用真实 coordinator 和 32 个 Request，重复 2000 次，
不开 cProfile：DeepSeek 局部 36.540 → 0.521 µs／32 请求，Qwen 为
34.770 → 0.524 µs／32 请求。对应离线生成没有稳定收益：Qwen 中位数
4211.6 → 4248.6 ms，DeepSeek 为 5373.0 → 5406.6 ms；DeepSeek 输出不一致。

本次正式 HTTP 测试开启了 prefix caching，这个 skip 不生效，不能将正式组合
的速度变化归给它。

### SM90 Graph 使用独立的 prefill 与 decode 网格

[平台策略][graph-policy]对 qwen4_exp／qwen4_exp_text、deepseek_v41／
deepseek_v41_text 的 SM90 提供 prefill 上限 1024、decode request 上限
min(max_num_seqs, 512) 和 decode token budget 512；resolver 再按调度配置约束。
SM100 保持原有 8192-token 默认值，显式 compilation 设置优先。更宽覆盖需要
额外 capture 内存和启动工作。

单项原型在同一加载的 current Qwen 上交错运行四次／臂：original 显式 prefill
cap512，graph1024 cap1024，并预先捕获较宽图，dispatch 切换在计时之外。

| Batch／input／output | 原流程中位数 ms | graph1024 中位数 ms | 时间减少 | 输出与原流程相同 |
| --- | ---: | ---: | ---: | --- |
| 1／128／64 | 363.157 | 362.889 | 0.07% | 是 |
| 32／128／32 | 810.592 | 570.405 | 29.6% | 否 |
| 128／128／16 | 2297.633 | 1194.342 | 48.0% | 否 |

原型 rank0 B32 的 NONE forward 从每次 4 次降到 0，B128 从 17 次降到 0，
对应更多 PIECEWISE／FULL dispatch。这支持“减少未捕获执行”的机制判断，
不证明所有 forward 都成为一张完整 Graph。

Padding 还会改变 kernel：Qwen HC down+SiLU 融合要求 padded M≤48，
33 个请求补到 40 可保留它，补到 64 则退出。单独捕获 33／40／64 的四次
离线消融，中位数分别为 1201.863／1212.379／1243.312 ms，输出均不稳定。
本轮保留这些阈值，没有通过删除硬件或 shape guard 强行启用不支持的实现。
默认值与覆盖示例见[SM90 推理配置](../../configuration/optimization.md#nabla-sm90-graph-coverage-and-host-planning)。

## 正式组合的 H20 HTTP 服务对照

两模型均使用单实例 4×H20、TP4／EP4、MP executor、async scheduling、
prefix caching、Engram CPU offload、seed43、max_model_len8192、
max_num_seqs128、max_num_batched_tokens1024 和 gpu_memory_utilization0.95。
Qwen 使用 Triton MoE、FP8 KV／indexer、16 GiB KV；DeepSeek 使用
deepseek_v41 tokenizer、MXFP8 linear Humming、1 GiB KV。
实际 prefill Graph cap 为基线 256、正式组合 1024，与前面的 cap512 单项消融不同。

环境为 Torch 2.14.0+cu130、Triton 3.8.0、FlashInfer 0.7.2.post1、
Transformers 5.20.2。两边使用相同原生二进制、现有权重与 OMP_NUM_THREADS=23，
未下载新权重、未锁 GPU 时钟，主机资源共享。运行前后分别校验了基线 8566、
实现 8567 个受版本控制文件的 SHA256。准备阶段曾误复制符号链接，导致一次
基线启动加载候选源码；该启动在计时前停止，后续独立目录的结果才纳入比较。

Compilation mode 为 NONE，Graph mode 为 FULL_AND_PIECEWISE。
**本轮没有启用全模型 torch.compile。** Humming input quant 的 fullgraph
测试通过，只证明该局部 helper 可编译。

使用 vllm bench serve，输入 128 tokens、输出 32 tokens，每个并发点两轮。
dataset seed 分别为 71、72，同轮两个版本使用同 seed；greedy、ignore EOS、
无限 request rate。每轮计时前 reset prefix cache。各 benchmark 有单请求
preflight，未补充每种并发的完整 warmup，部分形状仍在真实请求时触发 Triton JIT。
四版本共 24 组、1536 个计时请求，全部完成、零失败。

“整组耗时”包含该组全部请求：并发 1／32 每组 32 请求，并发 128 每组 128 请求。
时间降幅按两轮整组耗时的算术均值计算，负值表示变慢。各行负载不同，不能跨行
比较单请求时延。文本一致数按同 seed 的 saved generated_texts 列表位置比较，
并核对输入长度相同；它不衡量答案正确率。

| 模型 | 并发 | 基线两轮秒 | 正式组合两轮秒 | 均值时间减少 | 同轮生成文本相同数 |
| --- | ---: | --- | --- | ---: | --- |
| Qwen | 1 | 6.408／6.359 | 6.317／6.309 | 1.1% | 32/32、32/32 |
| Qwen | 32 | 1.292／0.890 | 0.634／0.615 | 42.7% | 7/32、2/32 |
| Qwen | 128 | 2.949／2.981 | 1.854／1.826 | 37.9% | 47/128、32/128 |
| DeepSeek | 1 | 9.349／9.363 | 9.403／9.382 | -0.4% | 3/32、4/32 |
| DeepSeek | 32 | 1.212／1.168 | 1.049／1.012 | 13.4% | 1/32、8/32 |
| DeepSeek | 128 | 3.848／3.813 | 3.036／2.995 | 21.3% | 15/128、22/128 |

下面保留每轮 throughput、平均 TTFT 和平均 TPOT，便于按相同轮次复算。
吞吐是输出 tok/s；TTFT 和 TPOT 单位为 ms。两轮均值的吞吐或延迟下降不代表
生产流量的总收益，也不能代替更充分的稳态复测。

<!-- h20-serving-samples:start -->

| 模型 | 版本 | 并发 | 轮次 | 完成／失败 | 整组秒 | 输出 tok/s | 平均 TTFT ms | 平均 TPOT ms |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Qwen | 基线 | 1 | 1 | 32/0 | 6.408198 | 159.795 | 35.413 | 5.310 |
| Qwen | 基线 | 1 | 2 | 32/0 | 6.359057 | 161.030 | 34.149 | 5.301 |
| Qwen | 正式组合 | 1 | 1 | 32/0 | 6.317261 | 162.096 | 33.800 | 5.272 |
| Qwen | 正式组合 | 1 | 2 | 32/0 | 6.308737 | 162.315 | 33.372 | 5.278 |
| Qwen | 基线 | 32 | 1 | 32/0 | 1.292426 | 792.308 | 735.103 | 17.264 |
| Qwen | 基线 | 32 | 2 | 32/0 | 0.889619 | 1151.054 | 372.923 | 15.954 |
| Qwen | 正式组合 | 32 | 1 | 32/0 | 0.634403 | 1614.117 | 221.954 | 12.570 |
| Qwen | 正式组合 | 32 | 2 | 32/0 | 0.614978 | 1665.100 | 203.671 | 12.538 |
| Qwen | 基线 | 128 | 1 | 128/0 | 2.948638 | 1389.116 | 1315.186 | 48.008 |
| Qwen | 基线 | 128 | 2 | 128/0 | 2.980888 | 1374.087 | 1344.290 | 48.285 |
| Qwen | 正式组合 | 128 | 1 | 128/0 | 1.854420 | 2208.777 | 743.950 | 31.113 |
| Qwen | 正式组合 | 128 | 2 | 128/0 | 1.826090 | 2243.044 | 726.830 | 30.880 |
| DeepSeek | 基线 | 1 | 1 | 32/0 | 9.348738 | 109.533 | 37.186 | 8.217 |
| DeepSeek | 基线 | 1 | 2 | 32/0 | 9.362989 | 109.367 | 37.430 | 8.224 |
| DeepSeek | 正式组合 | 1 | 1 | 32/0 | 9.402830 | 108.903 | 37.814 | 8.252 |
| DeepSeek | 正式组合 | 1 | 2 | 32/0 | 9.382356 | 109.141 | 37.192 | 8.252 |
| DeepSeek | 基线 | 32 | 1 | 32/0 | 1.212180 | 844.759 | 460.782 | 23.246 |
| DeepSeek | 基线 | 32 | 2 | 32/0 | 1.167811 | 876.854 | 439.546 | 22.615 |
| DeepSeek | 正式组合 | 32 | 1 | 32/0 | 1.048568 | 976.570 | 360.522 | 21.150 |
| DeepSeek | 正式组合 | 32 | 2 | 32/0 | 1.012167 | 1011.691 | 339.676 | 20.799 |
| DeepSeek | 基线 | 128 | 1 | 128/0 | 3.848366 | 1064.348 | 1667.191 | 64.143 |
| DeepSeek | 基线 | 128 | 2 | 128/0 | 3.812970 | 1074.228 | 1629.377 | 64.204 |
| DeepSeek | 正式组合 | 128 | 1 | 128/0 | 3.035805 | 1349.230 | 1195.976 | 53.184 |
| DeepSeek | 正式组合 | 128 | 2 | 128/0 | 2.995472 | 1367.397 | 1190.746 | 52.006 |

<!-- h20-serving-samples:end -->

Qwen B32 首轮存在明显延迟尖峰，不能将 42.7% 当作稳态估计。Qwen B128 两轮
速度变化一致，但输出文本不同；DeepSeek 两边均存在 Humming MXFP8 activation
降级为未量化的警告，性能范围限定在该实际配置。生成路径、Graph padding 和
kernel 选择都会影响这些数字，完整组合的结果不能归给某一个 CPU 缓存。

## 生成与精度冒烟保留差异

每个版本使用相同的 32 例串行 few-shot 算术 completion，生成上限 64 tokens，
按生成文本的首个整数计分。再对固定答案做 teacher-forced logprob 比较，
比较 echoed prompt 的所选 token logprobs，排除首个空值和末尾生成 token。
这只是小样本冒烟，未覆盖完整 GSM8K 或大并发模型准确率。

| 模型 | 基线／组合得分 | 32 例生成文本相同 | logprob 最大绝对差 | logprob 平均绝对差 |
| --- | --- | --- | ---: | ---: |
| Qwen | 14/32 → 14/32 | 32/32 | 0.000569 | 0.000006 |
| DeepSeek | 18/32 → 16/32 | 14/32 | 7.466086 | 0.256168 |

全部版本的所选 token logprobs 有限。Qwen 串行算术文本一致，不代表它在 B32／
B128 服务 benchmark 中输出一致。DeepSeek 的原型重复生成已观察到自身不稳定，
本轮仍有明显输出差异；无法仅凭这 32 例区分基线噪声与 Graph 策略影响。
**本记录没有建立 DeepSeek 精度等价或无回归结论。**

## 验证和复测方法

正式代码的四组定向测试汇总如下，各组存在重复覆盖，不能相加为互斥用例数。

| 范围 | pytest 原始汇总 |
| --- | --- |
| 真实配置、Humming、workspace 和新增 IPC | 36 passed, 4 warnings in 104.73s |
| PLE staging、metadata、offload replay、rollback | 22 passed, 4 warnings in 12.95s |
| Prefix caching CPU 文件 | 236 passed, 1 warning in 101.30s |
| Descriptor geometry、attention utilities、workspace、Humming fullgraph compile 文件 | 468 passed, 1 warning in 287.31s |

测试使用最终运行代码；最后的缓存变量名和 optional-list assert 修正只影响
静态类型检查。两项旧 Qwen4ExpKVCacheGrouping 导入测试在固定基线上也失败，
没有通过 skip、xfail、放宽容差或删除测试隐藏它们。代码 pre-commit 的对应旧
导入错误仍保留。合入时远端 pre-commit／pre-run-check 为 skipped，未宣称 CI 通过。

使用同一 uv-managed 环境检查相关文件，例如：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS='' \
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m pytest \
  tests/models/qwen4_exp/test_config.py \
  tests/models/qwen4_exp/test_ple.py::test_ple_input_staging_preserves_history_and_padding \
  tests/v1/kv_connector/unit/test_nixl_desc_geometry.py \
  tests/v1/worker/test_workspace.py \
  tests/v1/worker/test_attn_utils.py \
  tests/kernels/moe/test_moe.py::test_humming_buffer_plans_track_configuration \
  tests/quantization/test_humming_compile.py -q

HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=empty \
  .venv/bin/python -m pytest tests/v1/core/test_prefix_caching.py -m cpu_test -q
```

服务复测应从上述固定 commit 分别启动，使用相同本地权重、依赖和 benchmark
client 源码。下面展示 Qwen 的有效配置，模型路径由使用者填写；原实验默认
async scheduling 开启并回读确认。

```bash
export TASK_QWEN_MODEL=/path/to/Qwen3.8-Flash-Next-FP8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS='' \
VLLM_SERVER_DEV_MODE=1 VLLM_GPU_SYNC_CHECK=error OMP_NUM_THREADS=23 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  .venv/bin/python -m vllm.entrypoints.cli.main serve "$TASK_QWEN_MODEL" \
  --host 127.0.0.1 --port 18473 --served-model-name profile-model \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --distributed-executor-backend mp --max-model-len 8192 \
  --max-num-seqs 128 --max-num-batched-tokens 1024 --language-model-only \
  --engram-config '{"cpu_offload":true}' --enable-prefix-caching \
  --seed 43 --gpu-memory-utilization 0.95 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --moe-backend triton --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 17179869184 \
  --attention-config '{"indexer_kv_dtype":"fp8"}'
```

DeepSeek 换为相应模型路径，去掉 Qwen 的 `--moe-backend`、`--kv-cache-dtype`、
`--kv-cache-memory-bytes`、`--attention-config` 参数，改用
`--tokenizer-mode deepseek_v41 --kv-cache-memory-bytes 1073741824`
和 `--kernel-config '{"linear_backend_per_quant":{"mxfp8":"humming"}}'`。
原实验保留已有 CUDA 13 forward-compat 和 libstdc++ 环境，未更新共享 runtime。

每轮开始前，对该本地测试服务重置 prefix cache，再运行固定 benchmark。
这里示例为 Qwen B128 第一轮；第二轮改 seed72，其他并发点改并发与请求数。

```bash
curl --fail --request POST http://127.0.0.1:18473/reset_prefix_cache
.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend openai --base-url http://127.0.0.1:18473 \
  --model profile-model --tokenizer "$TASK_QWEN_MODEL" \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 128 --max-concurrency 128 --request-rate inf \
  --temperature 0 --ignore-eos --seed 71 --disable-tqdm \
  --save-result --save-detailed --result-dir results/h20-qwen \
  --result-filename bench-b128-r0.json
```

原始结果中的 completed、failed、input／output lengths、generated_texts、TTFT、
TPOT 和整组 duration 应一同保留。复测时需额外区分首次形状 JIT 与充分 warmup
后的稳态，并重新检查输出和模型准确率。本轮未测完整双实例 P/D，也没有 H100／
SM100 的新增硬件计时；未执行的兼容分支不能仅凭删除代码推断性能收益。

[implementation]: https://github.com/zhangxin81/nabla-vllm/commit/22dd6e0ddcc96dde7b95d61432035b83da8abc38
[integration]: https://github.com/zhangxin81/nabla-vllm/commit/0c31551dfc2f51f1b0dffd8091771c85089f0bc5
[ple]: https://github.com/zhangxin81/nabla-vllm/blob/0c31551dfc2f51f1b0dffd8091771c85089f0bc5/vllm/models/qwen4_exp/nvidia/ops/ple_inputs.py#L42
[ple-call]: https://github.com/zhangxin81/nabla-vllm/blob/0c31551dfc2f51f1b0dffd8091771c85089f0bc5/vllm/models/qwen4_exp/nvidia/model_state.py#L77
[ipc]: https://github.com/zhangxin81/nabla-vllm/blob/0c31551dfc2f51f1b0dffd8091771c85089f0bc5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/local_fastpath.py#L204
[workspace]: https://github.com/zhangxin81/nabla-vllm/blob/0c31551dfc2f51f1b0dffd8091771c85089f0bc5/vllm/v1/worker/workspace.py#L198
[humming]: https://github.com/zhangxin81/nabla-vllm/blob/0c31551dfc2f51f1b0dffd8091771c85089f0bc5/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py#L486
[attention]: https://github.com/zhangxin81/nabla-vllm/blob/0c31551dfc2f51f1b0dffd8091771c85089f0bc5/vllm/v1/attention/backend.py#L503
[attention-build]: https://github.com/zhangxin81/nabla-vllm/blob/0c31551dfc2f51f1b0dffd8091771c85089f0bc5/vllm/v1/worker/gpu/attn_utils.py#L561
[cache-skip]: https://github.com/zhangxin81/nabla-vllm/blob/0c31551dfc2f51f1b0dffd8091771c85089f0bc5/vllm/v1/core/kv_cache_manager.py#L633
[graph-policy]: https://github.com/zhangxin81/nabla-vllm/blob/0c31551dfc2f51f1b0dffd8091771c85089f0bc5/vllm/platforms/cuda.py#L640
