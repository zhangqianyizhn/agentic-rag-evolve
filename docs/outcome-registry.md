# Candidate outcome registry

`runner/record_candidate_outcome.py` 把 promotion gate 的结论写入追加式 registry：

```text
<registry-root>/
  accepted/outcome-<digest>.json
  rejected/outcome-<digest>.json
```

写入前会重新计算 candidate manifest、modification plan、静态审计、固定测试审计、validation suite 和 validation gate 的 SHA256，并核对它们记录的 candidate ID、plan ID、测试策略与源码 snapshot。任一 artifact 被替换或谱系不一致都会拒绝写入。

只有同时具备 development、holdout、cross-dataset cohort 的 promotion gate 能产生终态：

- `accepted / eligible_for_materialization`：全部 cohort 通过，可以进入后续 baseline 物化步骤；
- `rejected / not_eligible`：保留具体失败 cohort、检查项、回退题和成本摘要，供后续 failure memory 使用。

记录使用确定性 decision ID，不写运行时间等不稳定字段；同一 `(candidate_id, snapshot)` 已存在记录时拒绝覆盖。记录阶段的 `git_mutation_performed` 固定为 `false`。

示例：

```bash
uv run python runner/record_candidate_outcome.py \
  --registry-root artifacts/candidate-outcomes \
  --candidate-manifest <candidate-manifest.json> \
  --plan <modification-plan.json> \
  --candidate-audit <candidate-audit.json> \
  --candidate-test-audit <candidate-test-audit.json> \
  --validation-suite <promotion-suite.json> \
  --validation-gate <promotion-gate.json>
```

本模块不读取 final test，也不自动提交 accepted candidate。具体状态决策见 [ADR 0003](decisions/0003-candidate-outcomes.md)，后续固化协议见 [materialization](materialization.md)。
