# Repair and preservation memory

HarnessFix 的 harness memory 会把 accepted/rejected 历史、缺陷类型和 repair operator 组织成下一轮提示。DeepRead 演化框架不预设后两类枚举，而是分别保存“不要重复的失败”和“必须保持的已验证收益”，且所有内容均可由不可变 artifact 重新推导。

## 两层协议

`runner/build_repair_memory.py` 从一个 `rejected / not_eligible` outcome 生成不可变 `deepread-repair-memory-v2`：

- 上一次尝试的 plan、hypothesis、允许源码路径和 `required_behavior_delta`；
- 必须保持的行为与 non-goals；
- 失败 cohort、数据集角色、均值变化、token 比率和门禁原因；
- outcome 与 modification plan 的路径和 SHA256。

它不复制 trajectory、完整 diff、模型 transcript，也不推断固定 defect class 或 repair operator。attempt fingerprint 只由排序后的源码路径和规范化行为改动计算，因此 candidate ID 改变不会掩盖语义完全相同的重试。

反馈可见性由 cohort role 决定：development 回退题可以进入下一轮诊断和保护约束；holdout 与 cross-dataset 只公开回退数量和聚合指标，题目 ID 固定标记为 `sealed`。完整 outcome 仍保留原始审计事实，但 planning memory 不会把 sealed task 暴露给模型。

```bash
uv run python runner/build_repair_memory.py \
  --outcome artifacts/candidate-outcomes/rejected/outcome-<digest>.json \
  --memory-root artifacts/repair-memory
```

`runner/build_preservation_memory.py` 只在 accepted candidate 已经 materialize 并登记为新 baseline 后生成 `deepread-preservation-memory-v1`：

```bash
uv run python runner/build_preservation_memory.py \
  --outcome artifacts/candidate-outcomes/accepted/outcome-<digest>.json \
  --materialization <materialization.json> \
  --baseline-entry <baseline-entry.json> \
  --memory-root artifacts/repair-memory
```

它重新推导 materialization ID、baseline entry basis hash、baseline ID 和 durable ref，并保存已验证 behavior delta、`must_preserve`、non-goals、regression scenarios 及 cohort 级收益。任何 cohort 的具体 improved task ID 都不会进入 preservation memory。

`runner/build_planning_memory.py` 再针对当前 hypothesis set 构造 `deepread-planning-memory-context-v2`。选择规则固定为当前 hypothesis 受影响源码路径与历史 rejected/accepted attempt 路径的精确交集；两类 memory 每个 hypothesis 默认各最多 12 条。超过上限会报错并要求显式处理，不会自动摘要或静默丢弃。

```bash
uv run python runner/build_planning_memory.py \
  --memory-root artifacts/repair-memory \
  --cohort <hypothesis-cohort.json> \
  --hypotheses <improvement-hypotheses.json> \
  --output <planning-memory.json>
```

## Planner 门禁

planning context 交给模型前会重新验证：

- cohort 和 hypotheses 的精确 SHA256；
- 每个 repair/preservation memory 文件及其上游 artifact 的 SHA256；
- hypothesis 到源码路径的解析；
- 路径交集、memory ID 及 attempt/failure/constraint 投影。

修改计划可以采用不同策略，也可以 defer；但 validator 会拒绝以相同源码路径和相同 `required_behavior_delta` 再次 `proceed`。planner 同时必须保持匹配 accepted baseline 的已验证约束。这些规则不依赖固定 defect/operator 分类；完整 artifact 只作为按需追溯引用，planner 输入只携带紧凑投影。

```bash
uv run python runner/run_modification_planning.py \
  --cohort <hypothesis-cohort.json> \
  --hypotheses <improvement-hypotheses.json> \
  --memory-context <planning-memory.json> \
  --output <modification-plan.json>
```

本模块不自动触发下一轮、不修改 candidate，也不推进 baseline。单轮外层状态机仍由后续 orchestration 模块负责。
