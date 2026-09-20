# M6 真实单题全流程 smoke test

## 范围

- task：`financebench_id_00517`
- corpus：历史 FinanceBench sample20 global store
- baseline：DeepRead `7fe3ba23...` 对应的当前独立 runtime
- 目标：验证真实 provider 调用、确定性产物、诊断路由、空 repair cohort 和可恢复终态；不用于报告数据集准确率。

## 结果

真实 baseline 调用成功，产生 33 个规范化事件且没有未归属事件或 compiler warning。确定性 token F1 为 `0.2313`、evidence recall 为 `0.0`，但真实 LLM judge 给出 `4/4`；生成答案正确列出了 Boeing FY2022 三个超过 20% 的业务分部。因此系统没有把低词面重合误当成 agent failure。

基于 judge 结果重建 diagnostic bundle 后，诊断路由为 `eligible=false, target=none`，诊断模型调用次数和 token 均为零。该 skipped audit 被显式归入 cohort exclusion，随后 hypothesis aggregation 和 modification planning 分别产生空 hypotheses 与空 plans。迭代以 `outcome=no_candidate` 结束，没有创建 candidate、没有运行验证门禁，也没有改变 baseline。

终态账本包含 10 个事件：9 个完成事件和 1 个失败尝试。失败来自首版 runbook 使用不存在的 `/usr/bin/test`；执行器没有推进阶段。新增版本化 runbook 改为 `/bin/test` 后，从 `baseline_run` 原地恢复并到达 terminal，证明失败不推进和跨进程恢复有效。

## 结论

当前已验证完整的合法短路路径：

```text
real baseline → real judge → trajectory → diagnostic bundle
→ diagnosis skipped → empty cohort → no hypotheses → no plan
→ no-candidate report → terminal
```

尚未由真实数据覆盖的是 proceeding 路径：recurring hypothesis → modifier → static/fixed-test audit → development/promotion validation → accepted/rejected。下一次 smoke test 应选择至少两个具有同一可修复机制的真实 bad cases，而不是通过降低门禁或伪造 diagnosis 强制产生 candidate。
