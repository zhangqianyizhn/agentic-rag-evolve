# FinanceBench 00517 真实 judge 与 triage 实验

## 目的

验证答案语义正确、历史兼容指标偏低且标注证据未被 `read_section` 读取时，确定性分诊是否会错误地把样本归为 DeepRead 答案失败。

实验使用冻结的 FinanceBench 00517 prediction、trajectory 和 sample20 global store。经用户明确授权，将问题、gold answer 和 generated answer 发送到 `.env` 配置的 Volcengine Ark endpoint；没有发送 gold evidence 或 API key。

## 结果

| 观测 | 结果 |
|---|---:|
| LLM judge | 4 / 4 |
| normalized accuracy | 1.0 |
| token F1 | 0.4033613445 |
| baseline Recall | 0.0 |
| corpus canonical coverage | 1.0 |
| candidate canonical coverage | 1.0 |
| read canonical coverage | 0.0 |
| answer canonical coverage | 0.0 |

Judge 认为生成答案完整覆盖 gold answer，且没有事实错误。Evidence ladder 定位到 gold evidence 位于 Boeing `doc_id=5`、`node_id=111`、paragraphs 10–11，并在第 5 轮 BM25 rank 1 中出现；agent 实际读取的是 node 193。Node 193 包含能够支持同一答案的分段数据，因此 `evidence_not_read` 不能单独推出答案错误。

最终 triage 为 `evaluation_suspicious`，并同时保留：

- `evidence_not_read`；
- `baseline_recall_disagreement`；
- `judged_correct`。

该结果符合预期：样本仍进入 bad-case 报告供评测/证据标注审查，但不能作为 `incorrect_answer` 直接驱动 DeepRead 修复。回归测试固定了“judge=4 不得掩盖 evaluator disagreement”的优先级。

本地实验产物：

- evaluation：`/tmp/agentic-rag-evolve-diagnostic-eval-judge-authorized-v2-20260909/evaluation.json`；
- diagnostic bundle：`/tmp/agentic-rag-evolve-diagnostic-signals-judge-20260909/bundle.json`；
- bad-case report：`/tmp/agentic-rag-evolve-badcase-report-judge-20260909/bad_cases.md`。
