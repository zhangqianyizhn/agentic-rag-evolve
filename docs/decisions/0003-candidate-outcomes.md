# ADR 0003：候选终态先记录，后物化

## 状态

Accepted。

## 决策

只有 promotion 级 validation gate 可以产生 candidate 终态。门禁通过写入 `accepted/`，门禁失败写入 `rejected/`；development gate 只是中间反馈，不能产生终态。

accepted 的含义是 `eligible_for_materialization`，而不是已经成为新 baseline。记录过程不提交代码、不创建分支、不移动 worktree，也不修改 baseline 指针。后续 materialization 步骤必须再次核对 candidate snapshot，才能形成可作为下一轮起点的 Git commit。

终态唯一键是 `(candidate_id, candidate_snapshot_sha256)`。这是因为 candidate ID 在编辑前生成，同一 plan/base 的不同修订可能共享 ID；同一源码快照则不能同时拥有两个终态。

## 原因

- 将“验证结论”和“改变开发基线”拆开，便于人工检查第一批自动修复；
- accepted/rejected 都保留，失败尝试可成为下一轮的负向经验；
- 完整上游 artifact 哈希链使错配、替换或事后修改可以在落库前被拒绝；
- 确定性 decision ID 和拒绝覆盖避免重复运行产生相互矛盾的记录。

## 与 HarnessFix 的差异

HarnessFix 在 pipeline 中通过版本目录和 `current_base_version` 直接推进下一轮。本项目先写不可变 outcome record，待 materialization 明确实现后才改变 baseline；这样更适合当前逐模块检查与尚未稳定的自动修改阶段。
