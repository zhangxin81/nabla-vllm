# Nabla 版本优化总览

从 base 到 v0.1，主要是集成目标模型所需的内核、传输和缓存改动；
从 v0.1 到 v0.2，主要是减少 MoE、共享专家门控和输入准备的中间步骤，复用 CPU 计划，
并恢复上游缓存管理。下面按实际发布源码和测试路径列出主要变化。

## 版本和整体性能

| 版本 | 固定源码 | 主要变化 | Qwen H100 普通 NIXL 整体结果 |
| --- | --- | --- | --- |
| base | 上游 vLLM `8e45c458f6` | 原始调度、混合缓存、内核和 NIXL；作为比较起点 | 基线 |
| [v0.1.0](https://github.com/zhangxin81/nabla-vllm/tree/v0.1.0) | 发布源码 `727494c888`；主矩阵运行时 `b1e3f20751` | 目标模型与 CUDA 收口，集成内核和 IPC，重构混合缓存 | 相对当时上游，历史组吞吐中位数 +2.32%，补充组 +2.23%；各 25 个形状、三轮 |
| [v0.2.0](https://github.com/zhangxin81/nabla-vllm/tree/v0.2.0) | `42812804b2` | 保留内核与传输适配，恢复上游缓存，新增 MoE/门控融合、输入准备和 CPU 计划优化、QSA 分片 | 相对同机 base 首轮，吞吐中位数 +4.84%，39/39 点提升；TTFT −4.70%，TPOT −5.11% |

v0.2 相对 v0.1 历史组的 25 个三轮配对形状，吞吐中位数提高 3.02%；补充组为 2.97%。
v0.1 来自不同日期和物理 GPU。这些版本对照和下面的单项实验使用不同基线与负载，收益不能相加。
[三版本曲线与逐点数据](../benchmarks/nabla-vllm/v02-qwen-release.md)

## base 到 v0.1 的主要改动

| 改动 | 通俗解释 | 来源与保留情况 | 与当前 Qwen 复测的关系 |
| --- | --- | --- | --- |
| 目标模型与 CUDA 收口 | 聚焦 Qwen3.8-Flash-Next、DeepSeek-V4.1-Flash，移除无关模型和硬件后端 | v0.1 起保留 | 工程范围调整，没有单独测得的吞吐增幅 |
| FlashMLA、GDN/门控和 FP8 内核适配 | 按目标 GPU、尺寸和数据类型选择内核，减少适用路径的启动或计算成本 | vLLM #67468、#64846、#41115 等，v0.2 保留；详见[集成记录](nabla_vllm_pr_integration.md) | BF16 小头 FlashMLA、per-token FP8 等各有条件，不能把所有已集成内核都算作此次收益来源 |
| 同机 CUDA IPC 与描述符合并 | P/D 在同一台机器时直接读取 GPU 内存，合并兼容的传输描述 | vLLM #65639，v0.2 保留并继续显式开启 | 此次只测普通 NIXL，CUDA IPC 关闭 |
| DFlash context KV 批量写入 | 推测解码时一次处理多层缓存写入，减少重复启动 | vLLM #43594，v0.2 保留 | 此次推测解码关闭 |
| 四个生成 renderer worker | 增加异步生成的 CPU 渲染并行度 | vLLM #69041，v0.2 保留 | base、v0.1、v0.2 对照均显式设为 4，不构成版本差异 |
| 独立前缀索引、GDN state slot、统一字节预算 | 分开管理 token KV 与 GDN 状态，统一准入和回收，独立决定检查点位置 | 参考 SGLang state pool/checkpoint 管理；v0.1 发布，之后回退 | v0.2 已恢复上游 BlockPool 与 checkpoint，旧缓存设计不计入 v0.2 优化 |

GDN 投影重叠的上游 #64425 已在 base 中，不能把它再次记为 Nabla 独有增量。
v0.1 的缓存设计和实验见[历史缓存设计](https://github.com/zhangxin81/nabla-vllm/blob/v0.1.0/docs/design/specialized_hybrid_cache.md)
与[历史服务报告](https://github.com/zhangxin81/nabla-vllm/blob/v0.1.0/docs/benchmarks/nabla-vllm/h100-4p4d-20261120.md)。

## v0.1 到 v0.2 的主要改动

| 优化 | 原来 → 现在 | 条件与证据 |
| --- | --- | --- |
| **MoE 激活与 FP8 量化融合** | 专家第一次 GEMM 后，原来先写回激活结果，再读回来量化；现在一个 kernel 完成两步，保留原 BF16 舍入 | H100、640 维专家中间宽、UE8M0 block FP8。相对已回退统一分桶的 Nabla 基线，54 点 4P4D/IPC 实验的配对吞吐增幅中位数为 **1.98%**；此次普通 NIXL 版本测试也采用该实现。[报告](../benchmarks/nabla-vllm/qwen38-h100-profile-optimizations-20261124.md) · [实现](../../vllm/model_executor/layers/fused_moe/experts/triton_moe.py) |
| **共享专家门控融合** | 保留原 gate GEMM，把其后的 sigmoid 和乘法合成一个 kernel，减少一次启动和中间读写 | Hopper BF16，模型只在 **1～8 个 padded rows** 时启用。相对已融合 MoE 的基线，独立单服务 1024/128、C1/C8 四个配对点吞吐提高 **1.07%～1.97%**；高并发不宣称稳定收益。[报告](../benchmarks/nabla-vllm/qwen-shared-gate-20261125.md) · [实现](../../vllm/kernels/triton/fused_sigmoid_gate_mul.py) |
| **PLE 输入准备融合** | 原来多次填充、转换、索引、拷贝；现在一次生成历史 context 和请求 offsets | 原型完整准备流程 **21.920 → 3.552 μs**；这是组件计时，正式实现与组合服务另有验证。[报告](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md) · [实现](../../vllm/models/qwen4_exp/nvidia/ops/ple_inputs.py) |
| **CPU 计划和元数据复用** | 复用 workspace view、Humming buffer plan、同批 attention query lengths，关闭缓存时跳过 publication | 减少重复构造；有组件或组合验证，未证明每项独立端到端收益。此次使用 Triton MoE，Humming plan 缓存不参与。[报告](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md) |
| **IPC 注册地址索引** | 原来逐描述符查注册区间；现在注册时排序，批量定位和检查边界 | 合成 CPU 地址翻译 **669.659 → 12.049 ms**，不含传输。当前普通 NIXL 测试未启用 IPC。[报告](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md) |
| **SM90 Graph 默认值** | 为目标模型设置独立的 prefill/decode Graph 网格，prefill 默认覆盖到 1024 tokens | 减少适用形状的启动开销；历史 H20 组合有输出差异，不将其计时降幅单独作为精度等价收益。[配置](../configuration/optimization.md#nabla-sm90-graph-coverage-and-host-planning) · [报告](../benchmarks/nabla-vllm/h20-input-host-planning-20261123.md) |
| **QSA prefill 按 query 行分片** | 原来各 TP rank 重复选取索引；现在分工处理 query 行，再交换最终索引 | SM90 TP4/TP8、BF16 indexer、实际 prefill 行数至少 4096、context 至少 65536。独立 H100 单服务 256K/8192 chunk 两个 seed 吞吐提高 **7.32%/7.52%**；4P4D/4096 chunk 未建立稳定收益。此次版本复测为 FP8 indexer，**未触发该路径**。[报告](../benchmarks/nabla-vllm/qwen-qsa-20261124.md) · [实现](../../vllm/models/qwen4_exp/nvidia/qsa_indexer.py) |

共享专家门控方向参考 SGLang #26727 与已有 vLLM #43190；本实现保留原 GEMM。
QSA 行分片参考 SGLang #42968，复用已有 vLLM #65606 并适配 Nabla 布局。
两项的来源、固定 head 与验证命令均在各自报告中。

## 已回退的方案

| 方案 | 最终取舍 |
| --- | --- |
| v0.1 独立缓存 arena、前缀索引和 checkpoint 策略 | 恢复上游缓存资源管理；[回退与对照](../benchmarks/nabla-vllm/h100-cache-rollback-20261122.md) |
| 统一算子分桶 | 已回退，各算子保留原有阈值与配置选择 |
| H100 FP8-indexer QSA 候选、小 M 投影/HC 候选、Allreduce 分派候选 | 未形成可接受服务收益，或数值验收未通过，未保留；[筛选记录](../benchmarks/nabla-vllm/qwen38-h100-profile-optimizations-20261124.md) |
| vLLM #65794 的增量 backend-scale/native FP8 候选 | 相对已融合 MoE 基线的 12 点筛选没有吞吐收益，未纳入 v0.2 |

当前 **+4.84%** 是 v0.2 整个运行时相对 base 的服务实测结果。
上面的单项实验解释实现和适用场景，尚未在同一套 39 点普通 NIXL 矩阵上逐项开关归因。
本文依据发布源码差异和已有验证报告整理，使用了 AI 辅助。
