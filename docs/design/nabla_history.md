# Nabla 历史与迁移

2026-11-23，按维护者要求，main 的全部提交合为一个初始源码快照，
作者和提交者为 `zhangxin81`。已有目标模型代码、仓库介绍和性能文档保留，
文档整理记录及其入口删除。其他开发分支和 `v0.1.0` 标签保持各自历史。

上游来源是 [vLLM](https://github.com/vllm-project/vllm)。源码中的版权、
Apache-2.0 许可、第三方归属和 PR 来源保留；main 的快照提交者表示下游集成归属。
性能记录里的旧 SHA 继续标识当时测量的版本，不代表当前 main。
[nabla_commit_map.tsv](nabla_commit_map.tsv) 仅记录先前上游历史压缩时的映射，
供历史实验和旧开发分支使用，不是当前 main 的提交列表。

旧 clone 先保存本地改动，再从新 main 开始；需要保留自有提交时逐项 cherry-pick。
直接合并旧分支会带回旧历史，GitHub Contributors 也会重新统计其中作者。
没有本地改动时可重新 clone。上游仓库、其他远端及其他 checkout 未改写。
