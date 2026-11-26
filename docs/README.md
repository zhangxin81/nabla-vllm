# nabla-vllm

nabla-vllm 是独立维护的 vLLM 下游，面向 **Qwen3.8-Flash-Next** 和
**DeepSeek-V4.1-Flash** 的 NVIDIA CUDA 推理。项目聚焦目标模型的执行路径、
P/D 分离和目标算子优化，按固定源码、硬件、精度和请求形状记录性能与正确性。

## 目标模型和执行范围

| 范围 | 当前定位 |
| --- | --- |
| 模型 | 两个目标模型及其保留的兼容 draft／MTP 架构 |
| 硬件 | NVIDIA CUDA；保留 H20、H100、B200 等架构的选择路径，实测范围逐项记录 |
| 缓存 | 上游混合 KV／GDN BlockPool 与 checkpoint 策略；旧独立 ownership 方案已回退 |
| P/D | NixlConnector 和显式开启的同机 CUDA IPC，要求匹配配置和缓存几何 |
| 算子 | 保留原有多维阈值和各算子的配置选择，优化范围以对应源码和实测为准 |
| 兼容入口 | 安装包与主命令为 nabla-vllm，Python imports、兼容命令和配置继续使用 vllm 名称 |

代码中保留某项接口或硬件分支，不等于全部部署组合已验证。多模态、推测解码、
不同量化格式和并行拓扑以各自的检查与实测为准。CPU／ROCm／XPU／TPU 推理平台
及无关的 native 模型注册已移除；上游的通用模型和硬件目录不代表本项目能力。

## 从目标模型配置开始

- [快速开始](getting_started/quickstart.md)：使用本仓库和已有目标模型权重。
- [CUDA 安装与构建](getting_started/installation/README.md)：核对源码与二进制版本。
- [模型范围](models/supported_models.md)：确认 checkpoint architecture 和验证范围。
- [离线推理](serving/offline_inference.md)与[在线服务](serving/online_serving/README.md)。
- [NIXL P/D 配置](features/nixl_connector_usage.md)：检查传输与缓存条件。

## 性能证据对应具体版本

[版本优化总览](design/nabla_version_optimizations.md)：base → v0.1 → v0.2 的主要改动、来源与收益。

[v0.2 Qwen H100 端到端复测](benchmarks/nabla-vllm/v02-qwen-release.md)
包含 109 点逐轮数据和版本对比。

[下游 PR 集成](design/nabla_vllm_pr_integration.md)记录源提交和实际适配。
[H100 对照](benchmarks/nabla-vllm/h100-cache-rollback-20261122.md)与
[H20 输入准备和 CPU 计划优化](benchmarks/nabla-vllm/h20-input-host-planning-20261123.md)
使用不同源码与负载，分别保留原始结果和输出差异。
组件、消融和完整服务收益不能相加，也不能把未建立精度等价的观测写成普遍优势。

[Qwen3.8 H100 算子现状](benchmarks/nabla-vllm/qwen38-h100-operator-profile-20261124.md)
记录 Prefill 与 Decode 的热点；
[后续优化验收](benchmarks/nabla-vllm/qwen38-h100-profile-optimizations-20261124.md)
记录回退统一分桶后主干上的 MoE 融合、54 点 P/D 对照及真实输入数值检查。

[Qwen 共享专家门控](benchmarks/nabla-vllm/qwen-shared-gate-20261125.md)
记录相对已融合 MoE 基线的 40 点服务对照、组内重复波动和固定质量检查。

## 来源和贡献

上游来源是 [vLLM](https://github.com/vllm-project/vllm)，沿用 Apache-2.0 许可和
源码版权。main 已整理为单一初始快照，旧 checkout 迁移见
[历史说明](design/nabla_history.md)。项目问题和建议提交到
[nabla-vllm GitHub](https://github.com/zhangxin81/nabla-vllm/issues)，
开发流程见[贡献指南](contributing/README.md)。
