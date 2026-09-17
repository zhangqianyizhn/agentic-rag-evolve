# Rejected candidate repair memory

HarnessFix 的 harness memory 会把历史缺陷类型和 repair operator 组织成下一轮提示。DeepRead 演化框架不预设这两类枚举，而是只保留 promotion gate 已经观察到、且可由不可变 artifact 复核的失败事实。

## 两层协议

`runner/build_repair_memory.py` 从一个 `rejected / not_eligible` outcome 生成不可变 `deepread-repair-memory-v1`：

- 上一次尝试的 plan、hypothesis、允许源码路径和 `required_behavior_delta`；
- 必须保持的行为与 non-goals；
- 失败 cohort、数据集角色、均值变化、回退题、token 比率和门禁原因；
- outcome 与 modification plan 的路径和 SHA256。

它不复制 trajectory、完整 diff、模型 transcript，也不推断固定 defect class 或 repair operator。attempt fingerprint 只由排序后的源码路径和规范化行为改动计算，因此 candidate ID 改变不会掩盖语义完全相同的重试。

```bash
uv run python runner/build_repair_memory.py \
  --outcome artifacts/candidate-outcomes/rejected/outcome-<digest>.json \
  --memory-root artifacts/repair-memory
```

`runner/build_planning_memory.py` 再针对当前 hypothesis set 构造 `deepread-planning-memory-context-v1`。选择规则固定为当前 hypothesis 受影响源码路径与历史 attempt 路径的精确交集；每个 hypothesis 默认最多 12 条。超过上限会报错并要求显式处理，不会自动摘要或静默丢弃。

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
- 每个 repair memory 文件的 SHA256；
- hypothesis 到源码路径的解析；
- 路径交集、memory ID 及 attempt/failure/constraint 投影。

修改计划可以采用不同策略，也可以 defer；但 validator 会拒绝以相同源码路径和相同 `required_behavior_delta` 再次 `proceed`。这是一条确定性防重复门禁，而不依赖模型是否理解历史提示。完整 memory artifact 只作为按需追溯引用，planner 输入只携带紧凑投影。

```bash
uv run python runner/run_modification_planning.py \
  --cohort <hypothesis-cohort.json> \
  --hypotheses <improvement-hypotheses.json> \
  --memory-context <planning-memory.json> \
  --output <modification-plan.json>
```

本模块不自动触发下一轮、不修改 candidate，也不推进 baseline。单轮外层状态机仍由后续 orchestration 模块负责。
