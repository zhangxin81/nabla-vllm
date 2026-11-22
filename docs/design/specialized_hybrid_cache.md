# 当前 KV／GDN 缓存策略

独立 cache ownership 重构已回退。当前运行时的 BlockPool、KVCacheManager、
KVCacheCoordinator、KVCacheConfig 和缓存规划、清零／复制采用上游
`8e45c458f619dbabff879c0f53c58fd9f296e983` 的实现。
Qwen PLE／GDN 自定义合并、每组独立容量、共享字节 arena、独立前缀索引和
额外的 prefill chunk checkpoint 保留均已移除。

NixlConnector 保留同机 CUDA IPC 与 stream 依赖。普通 NIXL pull 使用上游的
导入 block 选择；同步 IPC 导入按已完成前缀截取 block，避免覆盖下一步的
可写 GDN 状态，并继续保留源引用直到复制和通知完成。P/D 两端必须使用
相同下游协议版本 20。

GDN 算子与 kernel 优化、投影重叠、FP8 和 FlashMLA 改动保留。
缓存策略回退不改变其算术实现。H20／H100／B200 执行路径继续保留。

独立分配方案和原始测量保留在
[v0.1.0 缓存设计](https://github.com/zhangxin81/nabla-vllm/blob/v0.1.0/docs/design/specialized_hybrid_cache.md)
及 [历史 H100 基线](../benchmarks/nabla-vllm/h100-4p4d-20261120.md)。
历史重构曾出现 DeepSeek cache-enabled unique 请求下的稳定 NIXL 回退；
当前源码的性能需要用回退后的完整组合重新测量。
