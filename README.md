# nabla-vllm

nabla-vllm 从 vLLM 出发，为 **Qwen3.8-Flash-Next** 和
**DeepSeek-V4.1-Flash** 构建专用推理运行时。硬件范围限定为 NVIDIA CUDA，
保留 H20、H100、B200 等架构各自需要的执行路径。

我们保留目标模型所需的调度、并行、通信和内核能力，逐步删除无关后端和模型依赖。
每次裁剪或重构都要能说明它改变了什么，并用同配置的服务基线验证正确性、吞吐和延迟。

目标模型的 PLE 输入准备、工作区 view、Humming buffer 计划和 IPC 注册地址查找
已减少重复构造；H20／H100 默认覆盖到 1024-token prefill Graph，显式配置优先。
默认值、缓存失效条件和 Graph padding 的数值边界见
[SM90 推理配置](docs/configuration/optimization.md#nabla-sm90-graph-coverage-and-host-planning)。
各项 feature 的组件收益、正式组合的 H20 服务结果和输出差异见
[输入准备与 CPU 计划优化记录](docs/benchmarks/nabla-vllm/h20-input-host-planning-20261123.md)。

[文档首页](docs/README.md)提供安装、模型、P/D 和性能入口。

v0.1 的 [Qwen3.8 H100 算子现状](docs/benchmarks/nabla-vllm/qwen38-h100-operator-profile-20261124.md)
记录了最新推理源码在 8×H100、4P4D、TP4／EP4、NixlConnector CUDA IPC 下的
Prefill、纯 Decode 和 D 接续。六组负载的算子占比、真实 batch、Graph 节点和
HC／通信重叠量均由 Nsight trace 复算；profiler 耗时与下方服务吞吐基线分开使用。

后续 [H100 优化验收](docs/benchmarks/nabla-vllm/qwen38-h100-profile-optimizations-20261124.md)
以回退统一分桶后的 `fffdea13c8` 为基线，保留 640 维 MoE 的 BF16 激活与
UE8M0 FP8 量化融合。8×H100、4P4D、TP4／EP4、NixlConnector CUDA IPC 的
54 点对照中，18 个配对吞吐比中位数为 **1.0198**，TTFT 比为 **0.9693**，
TPOT 比为 **0.9855**。六组负载的吞吐中位变化为 +0.88%～+3.30%，仍有单轮
退化和基线漂移；完整结果及数值边界见验收记录。这是相对此次 nabla 基线的增量，
不能与下方历史上游对照的收益相加。

## 为什么维护 nabla-vllm

性能优势是这个代码库的主要目标。我们聚焦两个目标模型、NVIDIA CUDA 和
P/D 分离，把适用的优化放进可运行、可复测的组合，再用固定源码的原始 vLLM
作对照。README 同时展示吞吐、延迟和回退点，性能优势以实测负载为准。

**这个代码库的一部分出发点，就是我们在 vLLM 提交的 PR 没能合入上游。**
nabla-vllm 承接这些优化，按目标模型和硬件适配、验证，并保留源 PR 与源码版本。
上游合入后会重新核对差异。PR 已集成代表功能可用，其端到端收益仍须单独测量。

## 已集成、尚未被上游合入的个人 PR

以下状态核对于 2026-11-23，作者均为 `zhangxin81`。源码版本和下游适配见
[PR 集成记录](docs/design/nabla_vllm_pr_integration.md)。

| PR | 上游状态 | 本仓库保留的能力 | 性能适用范围 |
| --- | --- | --- | --- |
| [#67468](https://github.com/vllm-project/vllm/pull/67468) | Open，未合入 | H20／H100 原生小头数 BF16 FlashMLA 分派 | 对应 MLA shape 和 dtype；不把它的原始测量推广到全部模型 |
| [#65639](https://github.com/vllm-project/vllm/pull/65639) | Open，未合入 | 同机 CUDA IPC pull、stream 依赖及别名复制合并 | 显式开启的同机 NixlConnector P/D；本次单列测量 |
| [#65136](https://github.com/vllm-project/vllm/pull/65136) | Open，未合入 | toy proxy 向所有后端重置前缀缓存，带鉴权和失败报告 | 支持可重复的缓存实验；不单独宣称吞吐收益 |
| [#43594](https://github.com/vllm-project/vllm/pull/43594) | Open，未合入 | DFlash／DFlash2 context KV 批量写入 | 推测解码路径；关闭推测解码的基线不衡量其收益 |
| [#41115](https://github.com/vllm-project/vllm/pull/41115) | Open，未合入 | 按 shape 选择 per-token FP8 launch size，校验直接 op 参数 | per-token FP8 操作；测试中的 Qwen 权重使用 block FP8 |

[#42301](https://github.com/vllm-project/vllm/pull/42301) 已关闭、未合入，但 GDN
投影重叠已由上游 #64425 覆盖，本仓库保留该受限实现。

[#39902](https://github.com/vllm-project/vllm/pull/39902) 已由合入的
[#39904](https://github.com/vllm-project/vllm/pull/39904) 替代。
这两项不计入当前相对原始 vLLM 的独有优势。

其他社区来源的优化也记录在集成文档中。

## 当前缓存实现

独立前缀索引、共享字节 arena、GDN state slot 独立分配和自定义 checkpoint
策略已回退。当前 KV／GDN 的分组、分配、前缀保留和清零／复制恢复到
原始 vLLM `8e45c458f6`，包含混合模型的块对齐和 GDN 页面 padding，
同时保留内核优化和 NixlConnector 的 CUDA IPC 路径。
回退后的源码与原始提交已完成同配置主矩阵测试，作为当前版本的性能对照。
历史重构及结果保留在 [v0.1.0](https://github.com/zhangxin81/nabla-vllm/releases/tag/v0.1.0)。

## 当前完成度

| 工作 | 状态 |
| --- | --- |
| CUDA 平台收口 | 已删除 CPU、ROCm、XPU、TPU、Zen CPU 平台实现与 MoRIIO 的 ROCm P/D 实现；保留 NVIDIA 架构选择 |
| KV／GDN 缓存管理 | 已恢复原始 vLLM 的共享 BlockPool 和 checkpoint 策略 |
| NixlConnector CUDA IPC | 保留同机 stream 依赖、源引用生命周期和加载后的续写 |
| 模型与内核依赖裁剪 | 已删除无关模型和后端实现，保留两个模型及 draft 的依赖闭包 |
| 下游 PR 集成 | 已导入选定的 FlashMLA、DFlash、FP8、NIXL IPC 和 proxy 改动，见集成记录 |

目标模型使用原始 vLLM 的缓存资源管理方式。每个已集成 PR 的验证范围见
[PR 集成记录](docs/design/nabla_vllm_pr_integration.md)，历史测试只适用于记录中的源码。

## 当前默认版本的性能对照

修正后的缓存回退版本与未经修改的 vLLM
`8e45c458f619dbabff879c0f53c58fd9f296e983`
进行 8×H100、4P4D、TP4／EP4 对照，固定相同的权重、依赖、输入输出、并发
和采样参数。每个配对点指同一个模型、输入长度、输出长度、并发与轮次。

| 对照组 | 源码 | P/D 传输路径 |
| --- | --- | --- |
| 原始 vLLM | `8e45c458f6` | 上游默认 NixlConnector |
| nabla 初建 connector | `f4a4ab4ccc` | 保留 #65639，关闭 CUDA IPC，使用普通 NIXL |
| nabla 初建 connector，CUDA IPC | `f4a4ab4ccc` | 保留 #65639，显式开启同机 CUDA IPC |
| nabla 默认 connector | `85a2d4572e` | NIXL 实现恢复为上游 `8e45c458f6` |

初建 connector 指本仓库集成的 #65639 路径及其安全修正。默认 connector
对照保留相同的缓存回退、模型和内核优化，独立回退 NIXL 实现。
两份 nabla 源码的差别限定为 connector 和对应的测试、说明。
主矩阵已完成 **696 个有效测量点、50,240 个计时请求**。
每个模型的四组实验均包含首轮 39 点，以及第二、三轮各 24 点。
下表先计算同点三轮均值，再取 24 个相对上游比值的中位数。
吞吐比越高越好，TTFT／TPOT 比越低越好；各点等权，不能代表生产流量的总收益。

| 模型 | 路径 | 吞吐比 | TTFT p50 比 | TPOT p50 比 | 低于上游点数 | 最低吞吐比 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next-FP8 | 初建普通 NIXL | 1.0179 | 0.9764 | 0.9881 | 1／24 | 0.9975 |
| Qwen3.8-Flash-Next-FP8 | 初建 CUDA IPC | 1.0235 | 0.9480 | 0.9920 | 0／24 | 1.0052 |
| Qwen3.8-Flash-Next-FP8 | 恢复默认 NIXL | 1.0168 | 0.9721 | 0.9889 | 0／24 | 1.0043 |
| DeepSeek-V4.1-Flash | 初建普通 NIXL | 1.0285 | 1.0041 | 0.9504 | 4／24 | 0.9962 |
| DeepSeek-V4.1-Flash | 初建 CUDA IPC | 1.0303 | 0.9903 | 0.9510 | 1／24 | 0.9868 |
| DeepSeek-V4.1-Flash | 恢复默认 NIXL | 1.0279 | 1.0044 | 0.9503 | 5／24 | 0.9563 |

在这组 H100 负载上，Qwen 吞吐比中位数提高约 1.7%～2.3%，DeepSeek 提高约
2.8%～3.0%。恢复默认 Connector 后仍有相近提升，不能把整个版本的收益归给 IPC。
DeepSeek 的普通 NIXL TTFT 比中位数略高于上游，TPOT 比中位数约低 5%。

Connector 直接对照同样使用每点三轮均值，结果如下。

| 模型 | 初建普通 NIXL／默认 | 初建 IPC／默认 | 初建 IPC／初建普通 NIXL |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next-FP8 | 1.0003 | 1.0023 | 1.0036 |
| DeepSeek-V4.1-Flash | 1.0002 | 1.0043 | 1.0037 |

普通 NIXL 的两份 Connector 在这个统计口径下接近持平；IPC 仍显式开启，
其增益取决于负载。DeepSeek `128／1024、并发 128` 的 IPC 三轮均值为
2529.56 output tok/s，上游为 2563.35，低 1.32%；相对默认 Connector 的
2620.69 低 3.48%。默认 Connector 的 `65536／128、并发 8` 比上游低 4.37%，
吞吐变异系数为 7.52%，该波动和全部原始轮次保留。

[696 点 CSV](docs/benchmarks/nabla-vllm/cache-rollback-all-cases.csv)、
[48 个复测点的均值与波动](docs/benchmarks/nabla-vllm/cache-rollback-all-repeat-points.csv)、
[同点饱和吞吐和延迟](docs/benchmarks/nabla-vllm/cache-rollback-matched-saturation.csv)
及 [Connector 直接对照](docs/benchmarks/nabla-vllm/cache-rollback-connector-pairs.csv)
保留完整数据。缓存开启的共享前缀另完成 144 点、4,224 个计时请求及 8 组边界检查，
全实验合计 **840 点、54,464 个计时请求**。

共享前缀的同点三轮均值，以下列出并发 32、8192 公共前缀＋256 独立后缀、
128 输出 tokens 的吞吐；每列都与上游同条件比较。

| 模型 | 上游 shared tok/s | 初建普通 NIXL | 初建 IPC | 默认 NIXL |
| --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next-FP8 | 1636.57 | 1665.07（+1.74%） | 1675.25（+2.36%） | 1675.26（+2.36%） |
| DeepSeek-V4.1-Flash | 1221.71 | 1256.97（+2.89%） | 1276.23（+4.46%） | 1250.85（+2.39%） |

unique 负载仍有小幅回退：DeepSeek 并发 8 的初建普通 NIXL 比上游低 0.97%。
默认 Connector 的 DeepSeek shared 并发 32 TTFT p50 比上游高 6.01%，
吞吐优势需要与延迟一起读。全部并发、模式和各轮波动见
[前缀逐轮 CSV](docs/benchmarks/nabla-vllm/cache-rollback-prefix-all-cases.csv) 与
[前缀三轮汇总](docs/benchmarks/nabla-vllm/cache-rollback-prefix-summary.csv)。
直接 Prefill 边界检查中，四组 Qwen 均命中 8000／8448 tokens，
四组 DeepSeek 均命中 8192／8448，当前使用原始 vLLM 的 checkpoint 策略。

31 次实际服务启动中，11 次 DeepSeek 退出使用了强制清理，发生在测量完成之后；
本轮未证明优雅退出通过。完整退出、源文件及数据指纹记录见基线报告。
[完整逐请求数据与复算脚本](https://github.com/zhangxin81/nabla-vllm/releases/download/v0.1.0/nabla-vllm-cache-rollback-h100-20261123-data.tar.gz)、
[SHA256 校验文件](https://github.com/zhangxin81/nabla-vllm/releases/download/v0.1.0/SHA256SUMS-cache-rollback-20261123.txt) 已作为补充资产发布；
历史 v0.1.0 的源码标签和原有数据保持不变。
Qwen 的三组下游串行 token 对照各 24 组均匹配；DeepSeek 分别有
19／24、18／24、21／24 组不同，严格数值一致性仍未通过。没有进行模型质量评估。
源码、矩阵、回归命令和证据见 [缓存回退与 Connector 对照](docs/benchmarks/nabla-vllm/h100-cache-rollback-20261122.md)。
下面的 v0.1.0 数据包含已回退的缓存重构，属于历史记录。

## v0.1.0 历史性能

历史实验为 **8× H100 80GB，4P4D，两侧 TP=4／EP=4**。
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
git clone --recurse-submodules https://github.com/zhangxin81/nabla-vllm.git
cd nabla-vllm
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=auto
uv pip install -e . --no-build-isolation
uv pip install -r requirements/lint.txt
pre-commit install
```

上述命令从本仓库构建。仅修改 Python 时，可以用
`VLLM_PRECOMPILED_WHEEL_LOCATION` 显式指定已经验证、与源码及 Torch／CUDA
匹配的原生 wheel。历史压缩后的上游 nightly discovery 不作为制品匹配依据。
CUDA 改动的迭代见[增量编译流程](docs/contributing/incremental_build.md)，
构建依赖和故障检查见[CUDA 安装](docs/getting_started/installation/gpu.md)。

```bash
.venv/bin/python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install
```

部署从[快速开始](docs/getting_started/quickstart.md)和本仓库的
[实测配置](docs/benchmarking/README.md)出发。上游 Recipes 作为参考，
H100 测试结果不代表 H20 或 B200 已通过相同验证。

## 上游与贡献

nabla-vllm 基于 [vLLM](https://github.com/vllm-project/vllm)，初始上游基线为
`898a7c68c2d1e869ebde2ce25ffbfc1e253bf0b4`。
main 已整理为单一初始源码快照，由 `zhangxin81` 维护。
旧 checkout 的迁移见[历史说明](docs/design/nabla_history.md)。
沿用其 [Apache-2.0 许可](LICENSE)，并保留上游来源和源码版权声明。
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
