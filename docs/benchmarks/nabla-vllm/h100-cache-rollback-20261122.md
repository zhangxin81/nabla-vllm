# H100 缓存回退与 Connector 对照

本实验比较缓存回退后的 nabla-vllm、初建 NixlConnector、上游默认
NixlConnector 和未经修改的 vLLM。测试正在进行，当前没有完整性能结论。
历史 v0.1.0 数据见 [原始基线记录](h100-4p4d-20261120.md)。

## 两份源码独立比较 Connector

| 对照组 | 固定源码 | 传输路径 |
| --- | --- | --- |
| 原始 vLLM | `8e45c458f619dbabff879c0f53c58fd9f296e983` | 上游默认 NixlConnector |
| nabla 初建 connector | `f4a4ab4ccc19c111ebc426e6739e2e4085a04eef` | 保留 #65639，关闭 CUDA IPC |
| nabla 初建 connector，CUDA IPC | `f4a4ab4ccc19c111ebc426e6739e2e4085a04eef` | 保留 #65639，显式开启 CUDA IPC |
| nabla 默认 connector | `85a2d4572e51511ef4ca73a8d623dc89cd596db3` | 恢复上游默认 NixlConnector |

两份 nabla 源码的缓存、模型、内核和调度代码相同。默认 connector 版本将
NIXL 实现文件逐个恢复为上游 `8e45c458f6`，删除 CUDA IPC 扩展及其专用测试。
初建 connector 指本仓库集成的
[#65639](https://github.com/vllm-project/vllm/pull/65639) 路径及其安全修正。
P 和 D 在每组实验中使用匹配的源码，初建 connector 协议版本为 20，
上游与默认 connector 版本为 17。

KV／GDN 的分组、分配、前缀保留和清零／复制已恢复到原始 vLLM。
回退包括混合模型块对齐和 GDN 页面 padding。
第一版 `8aaf0470ca` 遗漏了 Qwen 对齐路径，已停止其下游测试，保留失败与部分结果。
本报告的下游数据只使用修正后的源码。

## 配对点使用相同的请求条件

一个配对点包含模型、输入 token 数、输出 token 数、并发和轮次。
例如，Qwen3.8-Flash-Next-FP8 的 `8192／128、并发 8、第二轮`，
在四组实验中使用相同的 dataset seed 和采样参数。
先计算同点三轮均值，再比较吞吐、TTFT 和 TPOT；同时保留各轮波动。
各点等权的比值中位数不能解释为生产流量的总收益。

- 同机 8× NVIDIA H100 80GB HBM3，NVLink。
- P 使用 GPU 0～3，D 使用 GPU 4～7，两侧 TP=4、EP=4、DP=1、PP=1。
- 两侧 4 个 renderer worker，关闭 async scheduling、extensible KV cache 和 speculative decoding。
- 主矩阵关闭 prefix caching，固定 greedy、ignore EOS、dataset seed 和无穷请求速率。
- 每点 `max(8, 2 × concurrency)` 个计时请求，1 个 warmup。
- Qwen：`max_num_batched_tokens=4096`，显存利用率预算 0.90，FP8 KV。
- DeepSeek：`max_num_batched_tokens=1024`，显式 KV 预算 640 MiB，Humming MXFP8。
- 两个模型均开启 Engram CPU offload。

| 输入 tokens | 输出 tokens | 首轮并发 |
| --- | --- | --- |
| 128 | 128、1024 | 1、8、32、64、128 |
| 2048 | 128、1024 | 1、8、32、64、128 |
| 8192 | 128、1024 | 1、8、32、64、128 |
| 32768 | 128 | 1、8、32、64、128 |
| 65536 | 128 | 1、8 |
| 32768 | 1024 | 1、8 |

首轮每组每模型扫描 39 点。第二、三轮复测并发 1、8 和首轮上游吞吐
达到峰值 95% 的最小并发，四组共用上游选点。
共享前缀另测 `8192 + 256／128`，unique 与 shared、并发 1／8／32、三轮，
每组每模型 18 点。直接 Prefill 换后缀的边界检查单独记录实际复用 token 数。

## 权重与依赖固定

| 模型 | 权重文件集合 SHA256 |
| --- | --- |
| Qwen3.8-Flash-Next-FP8 | `9ffea2323a0e44acf7a013957adda7a2cff7dd684ff939e2b157602105b220b4` |
| DeepSeek-V4.1-Flash | `bb7ac4092747ef29ff339f5c21a9c36de2a188619b646783c3c71d803a52922a` |

之前两份 Qwen 权重的文件指纹相同，本次只测一份。
PyTorch 2.14.0／CUDA 13.0、Triton 3.8.0、Transformers 5.23.0、
FlashInfer 0.7.4、NIXL 1.5.0、Humming kernels 0.1.20、InstantTensor 0.2.3。
实验使用现有 `uv` 环境和 CUDA forward-compat 库。
C／CUDA／CMake 与依赖输入没有变化，复用原生产物时核对各文件 SHA256。
两份 nabla 源码的 8547／8546 个受版本控制文件已逐个核对。

原始 vLLM 的 Qwen 首轮 39 点和 24 组串行输出检查已在第一版回退实验中完成。
这份原始 vLLM 源码未修改，启动脚本、权重、依赖与请求参数核对一致后，
复用其未修改的原始数据，保留原 manifest 和逐文件哈希。
后续原始 vLLM 的复测轮次重新运行。
第一版不完整缓存回退的下游数据全部排除，历史文件保留。

## 缓存与传输回归

两份源码均运行以下目标缓存测试，使用无推理后端的 metadata 模式：

```bash
VLLM_TARGET_DEVICE=empty HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py \
  tests/v1/core/test_deferred_block_free.py -q
```

初建版本：`461 passed in 171.03s (0:02:51)`。
默认版本：`461 passed in 170.09s (0:02:50)`。

```bash
VLLM_TARGET_DEVICE=empty HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/v1/core/test_kv_cache_utils.py -q -k qwen_hybrid_alignment
```

初建版本：`2 passed, 114 deselected in 4.90s`。
默认版本：`2 passed, 114 deselected in 4.71s`。
该回归使用真实 Qwen 配置，在开启和关闭 prefix caching 时核对 GDN 与
attention 的物理页面大小。第一版未修正源码的同类检查为 `2 failed`。

```bash
VLLM_TARGET_DEVICE=cuda HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_nixl_connector.py \
  tests/v1/kv_connector/unit/test_nixl_desc_geometry.py -q -m cpu_test \
  -k 'local_fastpath or local_ipc or first_stream_ordered or real_model_layout_pull_copies_every_page or basic_interface or prompt_less_than_block_size or kv_transfer_handshake'
```

初建版本：`20 passed, 574 deselected in 17.57s`。
默认版本：`14 passed, 569 deselected in 15.74s`。
默认版本没有 CUDA IPC 扩展及其对应测试，两组实际选择的用例数量不同。

扩展六文件缓存测试在 metadata 模式下为
`46 failed, 474 passed in 204.12s (0:03:24)`。
44 项在加载离线环境中缺少的通用模型配置时失败，另外两项触发无 GPU 模式
自动关闭混合缓存管理的规则。后两项在实际 CUDA 平台重新运行，结果为
`2 passed, 114 deselected in 3.73s`。原失败日志保留，扩展套件仍记录为未通过。
同类 NIXL 几何测试在 metadata 模式下为
`6 failed, 14 passed, 574 deselected in 15.46s`；
同一选择在实际 CUDA 平台的通过结果如上，未修改断言或容差。

计时前检查实际请求完成数、长度和有限 logprobs。
Qwen 初建普通 NIXL 组的 24 组串行输出对照已与原始 vLLM 一致。
CUDA IPC 组必须在四个 Decode rank 的日志中实际出现复制记录，
控制组必须没有该记录。DeepSeek 的历史严格 token 对照未通过，本次仍单列检查，
不将性能测试解释为模型质量评估。

## Qwen 普通 NIXL 首轮

缓存回退后的 `f4a4ab4ccc` 与原始 `8e45c458f6` 已完成 39 个同点对照。
下表对每点计算 output throughput 比值，再取中位数，各点等权。
这是单轮结果，完整三轮复测及其波动仍待完成。

| 路径 | 对照点数 | 吞吐比中位数 | 低于上游的点数 | 最低吞吐比 |
| --- | --- | --- | --- | --- |
| nabla 初建普通 NIXL | 39 | 1.0228 | 1 | 0.9736 |

最低点为 `8192／128、并发 8`，本轮吞吐低约 2.6%。
逐点吞吐、TTFT、TPOT 见 [Qwen 普通 NIXL 首轮 CSV](cache-rollback-qwen-nixl-first-round.csv)。
这份数据使用已修正的缓存源码，历史 v0.1.0 数据单列保留。

## 完成后公布同点结果

完整矩阵、复测均值、Connector 直接比值、共享前缀和回退点仍待测试完成。
本页不会用历史数据或未完成的单轮样本填入当前版本的整体性能结论。
