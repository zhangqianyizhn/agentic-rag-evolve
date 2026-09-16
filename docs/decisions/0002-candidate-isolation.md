# ADR 0002：使用 detached Git worktree 隔离 candidate

## 状态

Accepted。

## 决策

只有 `decision=proceed` 的 modification plan 可以从一个明确的 Git commit 创建 candidate。candidate 使用完整仓库的 detached worktree，不自动创建分支；candidate manifest 保存在 worktree 外部，并冻结 plan hash、base commit、allowed paths 和文件预算。

## 原因

- 完整 worktree 保留 runner、framework 和 target 的真实导入关系，可直接运行后续验证；
- Git commit 比目录复制更容易复现和核对基线，也不会把主工作区未提交文件带入 candidate；
- detached 状态避免每个被拒绝 candidate 都留下长期分支；只有通过验证门禁的 candidate 才在后续阶段决定如何晋升；
- Git 共享对象存储，通常比完整复制多个候选更节省空间。

## 与 HarnessFix 的差异

HarnessFix 常复制 task-agent 目录并用 allowed path audit 约束修改。AgenticRAGEvolve 的 runner 与 `systems/deepread/DeepRead` 位于同一仓库，完整 worktree 能保留可运行的组合，同时通过 plan 引用闭包把实际编辑范围限制在 target 文件。

## 安全约束

- candidate path 必须不存在且位于源仓库之外；
- manifest 不能位于 candidate 内；
- 不提供自动删除命令，避免误删 worktree 或用户数据；
- 创建后 HEAD 必须保持 base commit，修改作为未提交 diff 接受审计；
- symlink、超预算文件、allowed path 外修改、forbidden root 修改、Python 语法错误或 `git diff --check` 失败均阻止后续验证。
