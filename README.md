# bytevllm

bytevllm 从 vLLM 出发，为 **Qwen3.8-Flash-Next** 和
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
| CUDA 平台收口 | 已删除 CPU、ROCm、XPU、TPU、Zen CPU 平台实现；保留 NVIDIA 架构选择 |
| 公共前缀索引 | `PrefixCacheIndex` 已接入 coordinator 和物理 `BlockPool`，保留池内 ID 与 owner 引用 |
| 独立 state slot 和统一字节预算 | 已接入准入、分配、runner、清零和复制；NIXL 首步续写正在验证 |
| 模型与内核依赖裁剪 | 继续按两个目标模型的依赖闭包推进 |
| 下游 PR 集成 | 已导入选定的 FlashMLA、DFlash、FP8、NIXL IPC 和 proxy 改动，见集成记录 |

Attention page 与 GDN state slot 已按各自的原生 stride 计价，
通过稳定的 tensor view 共用 backing allocation。正式显存容量与吞吐结果仍待服务测试。
导入的每个 PR 也需要在组合后的代码、目标模型和硬件上重新验证。
具体来源和历史验证范围见 [PR 集成记录](docs/design/bytevllm_pr_integration.md)。

## 性能基线与迭代

最新实验为 **8× H100 80GB，4P4D，两侧 TP=4／EP=4**。
Prefill 使用 GPU 0～3，Decode 使用 GPU 4～7。
比较最新上游 vLLM、bytevllm 的 NIXL 路径，以及显式开启同机 CUDA IPC 的 bytevllm。

测试分别记录以下权重，避免混合不同 checkpoint 的结果：

- Qwen3.8-Flash-Next-FP8（历史权重）。
- DeepSeek-V4.1-Flash-3。
- Qwen3.8-Flash-Next-FP8（补充权重）。

[H100 基线记录](docs/benchmarks/bytevllm/h100-4p4d-20261120.md)
包含输入输出矩阵、吞吐、TTFT、TPOT、实际请求长度和验证边界。
每份基线固定源码 SHA、权重指纹、依赖、并行方式、内核选择和采样参数，
并保留可比较的 CSV、图表和复现命令。

合成性能负载使用固定 dataset seed、`temperature=0` 和 `ignore_eos`。
主矩阵关闭 prefix caching，共享前缀另测。
吞吐饱和点需要与延迟一起阅读，低样本量的 p95／p99 不作为 SLA 估计。
源代码版本不同、后端配置变化和未通过的检查都在记录中明确说明。

## 开发与运行

包名和 CLI 暂时仍为 `vllm`。使用 `uv` 管理环境和依赖：

```bash
git clone https://github.com/zhangxin81/bytevllm.git
cd bytevllm
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

bytevllm 基于 [vLLM](https://github.com/vllm-project/vllm)，初始上游基线为
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
