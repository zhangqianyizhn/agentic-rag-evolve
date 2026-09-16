# M5 归并与计划

HarnessFix 的 `aggregate_results.py` 在一次模型调用中同时完成固定类别归并、优先级排序、改进计划和 modify spec。DeepRead 不预设 defect class 或 repair operator，因此先建立一个确定性的 cohort 边界，再在后续模块中进行开放式语义归并。

## Hypothesis cohort

`runner/build_hypothesis_cohort.py` 接收一个或多个已经生成的 `deepread-diagnosis-v1`：

- 只有 `status=diagnosed` 可以进入 `eligible_diagnoses`；
- `not_agent_failure` 路由到 `evaluation_review`；
- `insufficient_evidence` 路由到 `evidence_review`；
- 重复 task、未知 schema/status 和覆盖已有输出都会被拒绝；
- 每条输入保留 SHA-256，整个 cohort 具有与输入顺序无关的稳定 ID。

这一层不自行聚类，也不从文本猜测 repair operator。它的职责是防止数据/评测问题或证据不足样本污染 DeepRead 修改计划。

```bash
python runner/build_hypothesis_cohort.py \
  --diagnosis <diagnosis-1.json> \
  --diagnosis <diagnosis-2.json> \
  --output <hypothesis-cohort.json>
```

下一模块只读取 `eligible_diagnoses`，生成包含目标 cohort、共同机制、反证和可证伪验证条件的 improvement hypothesis；源码修改范围和 candidate 创建仍留在更后面的 planning/modify 阶段。
