# Accepted candidate materialization

`runner/materialize_candidate.py` 将 accepted outcome 对应的未提交 diff 固化为 detached Git commit。它只操作 outcome 指向的 candidate worktree，不创建分支、不修改主仓库 checkout，也不更新 baseline。

运行前会重新验证：

- outcome 为 `accepted / eligible_for_materialization`；
- 六类上游 artifact 的当前 SHA256 与 outcome 完全一致；
- outcome decision ID 与 validation gate 哈希可重新推导；
- candidate manifest、静态审计和 outcome 的 ID、base commit、路径及 snapshot 一致；
- worktree HEAD 仍是原 base commit，变化路径和文件内容仍等于 accepted snapshot。

提交过程只 stage 静态审计中的明确路径，然后核对 staged path 集合。commit 使用 base commit 时间加一秒、固定作者身份和确定性消息，通过 `git commit-tree` 创建；因此相同 base、tree 和 outcome 会得到相同 commit。最后 candidate worktree 被移动到这个 commit，并要求工作区完全 clean。

materialization 是幂等的。如果进程在 Git commit/reset 完成后、JSON 产物写入前中断，再次运行会核对父 commit、commit message、tree 内容对应的 accepted snapshot 与 clean 状态，并返回同一个 commit。

示例：

```bash
uv run python runner/materialize_candidate.py \
  --outcome artifacts/candidate-outcomes/accepted/outcome-<digest>.json \
  --output artifacts/materializations/outcome-<digest>.json
```

输出使用 outcome 哈希与 commit 推导稳定的 materialization ID，状态为 `materialized_detached`，并明确记录 `branch_created=false`、`baseline_updated=false`。下一阶段的 baseline registry 只能引用这一 materialization artifact，而不能直接引用可变 worktree。
