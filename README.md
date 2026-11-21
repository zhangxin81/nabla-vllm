# nabla-vllm

nabla-vllm 从 vLLM 出发，为 **Qwen3.8-Flash-Next** 和
**DeepSeek-V4.1-Flash** 构建专用推理运行时。硬件范围限定为 NVIDIA CUDA，
保留 H20、H100、B200 等架构各自需要的执行路径。

我们保留目标模型所需的调度、并行、通信和内核能力，逐步删除无关后端和模型依赖。
每次裁剪或重构都要能说明它改变了什么，并用同配置的服务基线验证正确性、吞吐和延迟。

## 缓存的索引、分配与检查点分别管理

目标设计是：**统一管理前缀索引和内存预算，KV 按 token page 分配，
GDN 按 state slot 分配，checkpoint 的位置单独决定。**

前缀索引回答哪些已提交的输入历史可以复用；物理池负责资源分配和回收；
checkpoint 策略决定在哪些位置保留可续写的状态。
三者通过明确的引用和生命周期约束协作。

| 资源 | 分配单位 | 复用条件 |
| --- | --- | --- |
| Attention KV | Token page | 输入历史和有效注意力窗口一致 |
| Qwen GDN／PLE | State slot | 精确的已提交检查点，含必要的回滚状态 |
| QSA／indexer | 原生 page 和压缩单位 | 索引、KV 与输入覆盖范围一致 |
| DeepSeek sparse／compressor | 原生 page 或 state | 压缩边界、共享缓存与续写边界一致 |

公共预算按实际 backing bytes 计价，包含对齐、padding、活跃请求和传输中的引用。
请求必须同时满足各资源池的准入条件，才能提交分配。
异步传输完成前保留源检查点，CUDA Graph 回放期间保持缓冲区地址稳定。

这一方向深度参考 SGLang 的独立 Mamba slot allocator、统一 state pool 和
checkpoint 管理，但不引入 SGLang 运行时依赖。
源码指针和迁移约束见 [缓存设计](docs/design/specialized_hybrid_cache.md)。

## 当前完成度

| 工作 | 状态 |
| --- | --- |
| CUDA 平台收口 | 已删除 CPU、ROCm、XPU、TPU、Zen CPU 平台实现与 MoRIIO 的 ROCm P/D 实现；保留 NVIDIA 架构选择 |
| 公共前缀索引 | `PrefixCacheIndex` 已接入 coordinator 和物理 `BlockPool`，保留池内 ID 与 owner 引用 |
| 独立 state slot 和统一字节预算 | 已接入准入、分配、runner、清零、复制和 NIXL 续写；H100 主矩阵和前缀复用验证完成 |
| 模型与内核依赖裁剪 | 已删除无关模型和后端实现，保留两个模型及 draft 的依赖闭包 |
| 下游 PR 集成 | 已导入选定的 FlashMLA、DFlash、FP8、NIXL IPC 和 proxy 改动，见集成记录 |

Attention page 与 GDN state slot 已按各自的原生 stride 计价，
通过稳定的 tensor view 共用 backing allocation。H100 组合服务的吞吐结果见下文。
导入的每个 PR 也需要在组合后的代码、目标模型和硬件上重新验证。
具体来源和历史验证范围见 [PR 集成记录](docs/design/nabla_vllm_pr_integration.md)。

## 性能基线与迭代

最新实验为 **8× H100 80GB，4P4D，两侧 TP=4／EP=4**。
Prefill 使用 GPU 0～3，Decode 使用 GPU 4～7。
比较固定源码的上游 vLLM、nabla-vllm 的 NIXL 路径，以及显式开启同机 CUDA IPC 的 nabla-vllm。

测试分别记录以下三组运行：

- Qwen3.8-Flash-Next-FP8（历史权重）。
- DeepSeek-V4.1-Flash。
- Qwen3.8-Flash-Next-FP8（补充复测）。

两份 Qwen 权重的完整文件指纹相同，补充组是同一权重的独立复测。

[H100 基线记录](docs/benchmarks/nabla-vllm/h100-4p4d-20261120.md)
包含输入输出矩阵、吞吐、TTFT、TPOT、实际请求长度和验证边界。
每份基线固定源码 SHA、权重指纹、依赖、并行方式、内核选择和采样参数，
并保留可比较的 CSV、图表和复现命令。

主矩阵运行时固定为 `b1e3f20751`，修复后的前缀实验固定为 `0d42cd781e`。
上游和公共 benchmark client 固定为 `8e45c458f6`。
两侧均设置 4 个 renderer worker，主矩阵关闭 prefix caching 和 speculative decoding。
首轮扫描 39 个输入输出与并发组合，第二、三轮复测并发 1、8 和同一上游选出的饱和点。

三组运行的主矩阵已完成 801 个有效测量点、58,248 个请求。
共享前缀另完成 162 点、4,752 个计时请求及 9 组真实 Prefill 复用边界检查。
下表先对每个复测点计算三轮 output throughput 均值，再与同点上游均值相除，
最后取 25 个比值的中位数。各点等权，不能据此估计生产流量的总收益。

| 运行组 | nabla NIXL／上游 | nabla CUDA IPC／上游 |
| --- | --- | --- |
| Qwen3.8-Flash-Next-FP8，历史组 | 1.0232 | 1.0297 |
| DeepSeek-V4.1-Flash | 0.9979 | 1.0004 |
| Qwen3.8-Flash-Next-FP8，补充复测 | 1.0223 | 1.0257 |

Qwen 在这组负载上有小幅提升；DeepSeek 整体接近上游，仍有回退点。
Qwen 历史组的 65536／128、并发 8 IPC 吞吐三轮均值为 51.84 output tok/s，
上游为 53.00，低 2.2%；IPC 变异系数为 8.55%，首轮低 11.3% 的样本仍保留。
DeepSeek 的复测点最低均值比分别为 NIXL 0.9684、IPC 0.9690。
首轮 2048／1024、并发 128 的最低比值 0.9604／0.9471 未入选复测，仍按单轮证据记录。

首轮完整数据分别见 [Qwen CSV](docs/benchmarks/nabla-vllm/qwen-first-round.csv)、
[DeepSeek CSV](docs/benchmarks/nabla-vllm/deepseek-first-round.csv)
和 [补充 Qwen CSV](docs/benchmarks/nabla-vllm/qwen-additional-first-round.csv)。
完整 [801 点 CSV](docs/benchmarks/nabla-vllm/all-cases.csv)、
[75 个复测点的均值与波动](docs/benchmarks/nabla-vllm/all-repeat-points.csv)
和 [同点饱和吞吐及延迟](docs/benchmarks/nabla-vllm/matched-saturation.csv)
已完成核验。[共享前缀逐轮数据](docs/benchmarks/nabla-vllm/prefix-all-cases.csv)
与 [三轮汇总](docs/benchmarks/nabla-vllm/prefix-summary.csv) 在基线记录中单列。
默认 checkpoint 策略已修复换后缀时缺少 GDN 状态的问题，所有下游前缀组均重测。
源码版本、公开逐请求数据与复算脚本见
[v0.1.0 release](https://github.com/zhangxin81/nabla-vllm/releases/tag/v0.1.0)。

Qwen 两组的 NIXL 与 IPC 串行同路径 token 对照各 24 组均匹配，logprobs 有限。
DeepSeek 两条路径各有 18／24 组严格 token 对照不同，上游自身重复也会不同；
该模型的数值一致性仍未通过。这轮按性能测试范围核验请求完成数、长度和有限输出，
没有进行模型质量评估。
开启缓存、使用 unique 8448／128 请求时，DeepSeek 并发 32 的 NIXL 三轮均值比为 0.8408，
IPC 为 0.9639；这组缓存开启后的回退与关闭缓存的主矩阵分开记录，尚未解决。

合成性能负载使用固定 dataset seed、`temperature=0` 和 `ignore_eos`。
主矩阵关闭 prefix caching，共享前缀另测。
吞吐饱和点需要与延迟一起阅读，低样本量的 p95／p99 不作为 SLA 估计。
源代码版本不同、后端配置变化和未通过的检查都在记录中明确说明。

## 开发与运行

项目包名和主要 CLI 为 `nabla-vllm`，Python 导入路径为 `vllm`，保留 `vllm` CLI 别名。
使用 `uv` 管理环境和依赖：

```bash
git clone https://github.com/zhangxin81/nabla-vllm.git
cd nabla-vllm
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/lint.txt
pre-commit install
```

预编译安装用于准备环境。要使用本仓库的 CUDA 改动，按
[增量编译流程](docs/contributing/incremental_build.md) 编译并安装相应原生扩展，
确认 PyTorch、CUDA Toolkit 和 GPU 架构匹配。H100 基线的实际依赖和构建来源
以实验记录为准。

```bash
.venv/bin/python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install
```

部署参数可参考 [Qwen recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
和 [DeepSeek recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)，
然后在目标硬件上验证。H100 测试结果不代表 H20 或 B200 已通过相同验证。

## 上游与贡献

nabla-vllm 基于 [vLLM](https://github.com/vllm-project/vllm)，初始上游基线为
`898a7c68c2d1e869ebde2ce25ffbfc1e253bf0b4`。
沿用其 [Apache-2.0 许可](LICENSE)，并保留上游归属和贡献记录。
贡献时遵循 [AGENTS.md](AGENTS.md) 和现有领域指南；PR 应说明设计、重复工作检查、
实际测试与 AI 辅助范围。

研究中使用 vLLM／PagedAttention 时，请引用上游论文：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
