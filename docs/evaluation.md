# DeepRead 评测模块

Evaluator 位于被进化系统之外，不导入 DeepRead 的 agent、索引或检索实现。它只读取数据集中的 gold label 和 runner 产生的 predictions，因此 gold answer/evidence 不会进入回答阶段。

## 当前指标

- `f1`：与历史 ruc-ov 一致的英文 token F1；多 gold answer 取最高值；
- `recall`：先尝试原文子串匹配，再对不少于 4 个有效 token 的 evidence 使用 0.8 token 覆盖率；
- `accuracy_0_4`：可选 LLM judge，采用历史 Generic 0–4 语义；
- `accuracy_normalized`：`accuracy_0_4 / 4`。

每条 evidence 额外记录 `exact_substring`、`soft_token_coverage`、`short_evidence_exact_required` 或 `not_matched`，以及 token coverage。后续诊断使用这些明细，而不是只读取聚合 Recall。

## 错误语义

预测失败、judge 未配置和 judge 调用/解析失败是三种不同状态。judge API 故障的 score 为 `null`，不会静默伪装为答案得 0 分；汇总同时报告 judge 成功、失败和跳过数量。

Judge 结果同时记录模型名与独立 input/output token，用于把回答成本和评测成本分开统计。

## 历史对齐

`runner/evaluate_deepread.py --historical <qa_eval_detailed_results.json>` 按 question 配对，输出当前与历史 F1、Recall、Accuracy 及 delta。当前阶段只做逐题事实对齐，不把单次 LLM judge 波动解释为系统改进。
