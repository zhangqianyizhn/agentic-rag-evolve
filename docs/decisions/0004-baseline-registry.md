# ADR 0004：用线性不可变账本管理 baseline

## 状态

Accepted。

## 决策

baseline registry 不维护可变的 `current_version` 文件。generation 0 从一个明确的完整仓库 commit 初始化；之后每个 entry 只能引用前一个 entry 和以该 commit 为唯一父节点的 accepted materialization。当前 baseline 是完整账本校验后的唯一末端。

每个 entry 都创建 `refs/agentic-rag-evolve/baselines/<baseline-id>` 内部 Git ref，防止 detached materialized commit 被垃圾回收。该 ref 不属于 `refs/heads/`，不会创建用户分支或切换任何 checkout。

## 原因

- 追加式 entry 保留每轮演化的父 baseline、materialization 和 commit 谱系；
- 不依赖可变指针，避免中断时出现“当前版本已变但历史未写完”；
- 线性约束使两个并行 candidate 不能同时从同一父节点晋升；
- 内部 durable ref 让 registry 中的 commit 真正长期可达，而不是只有可能被 Git GC 的 SHA 文本。

## 与 HarnessFix 的差异

HarnessFix 通过整数 `current_base_version` 和版本目录推进。这里通过完整仓库 Git commit、不可变 entry 和内部 ref 推进；数据集结果与 outcome artifact 仍位于 Git 外部，不复制进 baseline 源码。
