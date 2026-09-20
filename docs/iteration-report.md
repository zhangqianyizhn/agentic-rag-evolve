# Compact iteration report

`runner/build_iteration_report.py` 在一轮 candidate 已经达到持久终态后生成 `deepread-iteration-report-v1`。它是已有事实的只读投影，不重新运行模型、DeepRead、评测或 Git 操作，也不作为恢复执行的状态文件。

```bash
uv run python runner/build_iteration_report.py \
  --outcome <candidate-outcome.json> \
  --terminal-memory <repair-or-preservation-memory.json> \
  --cohort <hypothesis-cohort.json> \
  --hypotheses <improvement-hypotheses.json> \
  --regression-feedback <optional-regression-feedback.json> \
  --output <iteration-report.json>
```

终态要求是非对称的：

- rejected candidate 必须已经生成可重新验证的 repair memory，报告状态为 `rejected_retained`，baseline 明确保持不变；
- accepted candidate 必须已经确定性 materialize、登记为新 baseline，并生成 preservation memory，报告状态才是 `accepted_registered`；
- 仅通过 promotion gate 但尚未登记 baseline 的 accepted candidate 不算完成一轮，不能提前生成成功报告。

构建器重新计算 outcome、plan、manifest、static audit、fixed-test audit、validation suite/gate 和 terminal memory 的 SHA256，并复核 candidate/plan/snapshot、promotion gate 与 decision ID。accepted 路径继续通过 preservation memory 重新推导 materialization 和 baseline identity；rejected 路径通过 repair memory 重新推导失败投影。

## 紧凑字段

报告只保留后续观察和编排需要的摘要：

- diagnosis 输入、eligible、excluded 数量及 excluded route 统计；
- hypothesis 数量、maturity 分布和被 plan 使用的数量；
- 选定 plan 的源码路径、behavior delta 与风险等级；
- candidate snapshot、变化文件、固定测试数量；
- 每个 validation cohort 的平均变化、改善/回退数量、token ratio 和失败原因；
- outcome 决策与 accepted/rejected 的持久终态；
- 可选 regression feedback 中 development/sealed 回退数量。

不会复制 diagnosis 正文、trajectory、evaluation、源码、diff、模型消息、测试 stdout/stderr 或 plan markdown。development、holdout、cross-dataset 的 task ID 都不会进入报告，所有 cohort 只暴露计数；`large_payloads_embedded` 和 `final_test_accessed` 固定为 `false`。

## 与 HarnessFix 的差异

HarnessFix 的 SWE iteration report 直接嵌入 audit、diff summary、train/validation compare、plan markdown 和 next-iteration guidance，并被下一轮 aggregate step读取。这里将“可读汇总”和“执行状态”分开：report 只总结已经由独立 artifact 决定的事实，不复制大输入、不自行提出修复建议，也不以文件存在代表步骤完成。

下一阶段的 append-only round ledger 会引用 report SHA256 和各状态转换，用作恢复与调度依据；iteration report 本身仍保持不可变、可删除后重建的派生产物。
