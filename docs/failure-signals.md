# 确定性失败信号与 bad-case 报告

## 边界

`diagnostics/signals.py` 只根据已对齐的 prediction、judge 结果和 evidence ladder 生成可复现的现象标签，不推断 root cause，不选择修复算子，也不授权代码修改。诊断 agent 在 M4 中可以反驳或细化这些标签，但必须引用 trajectory、payload 或源码证据。

每个诊断 bundle 内含 `deepread-failure-signals-v1`：

- `triage`：用于队列分流的单一主类别；
- `signals`：可以并存的事实信号及其直接证据；
- `evidence_path`：corpus、candidate、read、answer 四层召回和最早缺失层；
- `confidence`：只表示规则结论的确定性，不表示根因正确率；
- `interpretation`：明确禁止将信号直接解释为根因或修复方案。

## Triage 类别

| 类别 | 确定条件 | 是否进入 bad-case 队列 |
|---|---|---:|
| `execution_failure` | prediction status 非 `ok` | 是 |
| `no_answer` | 成功运行但最终答案为空 | 是 |
| `incorrect_answer` | 成功 judge 得分 0–1 | 是 |
| `partial_answer` | 成功 judge 得分 2–3 | 是 |
| `evaluation_suspicious` | gold evidence 不在冻结 corpus、judge 失败，或 baseline Recall 与 canonical candidate coverage 冲突 | 是 |
| `needs_judgment` | 无可靠 judge，且没有更强的确定性结论 | 是 |
| `pass` | 成功 judge 得分 4 | 否 |

路径信号与 triage 正交。例如 `evidence_not_read` 说明 gold evidence 已进入候选但没有出现在 `read_section` 结果中；如果 judge 仍确认答案正确，该样本可以是 `pass`，路径信号依然保留供分析替代证据或潜在脆弱性。`gold_evidence_not_lexicalized_in_answer` 只表示词面覆盖，不代表答案错误。

`baseline_recall_disagreement` 比较 evaluator 的历史兼容 Recall 与诊断侧局部窗口 canonical coverage。它提示 HTML/Markdown 表达、retrieved-text 收集或 evaluator 对齐需要检查，不自动证明 evaluator 存在缺陷。

## 报告

一个或多个 bundle 可以汇总为紧凑报告：

```bash
python runner/build_bad_case_report.py \
  --bundle <bundle-or-directory> \
  --bundle <another-bundle-or-directory> \
  --output <empty-output-directory>
```

输出包括机器可读的 `summary.json`、只含 bad case 的 `bad_cases.json` 和人工快速检查用的 `bad_cases.md`。报告只保留问题、答案、指标、信号、四层摘要及每条 gold evidence 每层最多一个引用，不复制完整 trajectory 或 gold evidence。
