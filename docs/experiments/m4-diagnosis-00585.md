# M4 诊断验证：FinanceBench 00585

## 实验对象

- task：`financebench_id_00585`
- DeepRead 预测：FY2021 `14.7%`，FY2022 `-0.6%`
- gold：FY2022 `0.62%`，FY2021 `-14.76%`
- 诊断产物：`/tmp/agentic-rag-evolve-m4-00585-diagnosis-v14-20260916`
- 模型：`deepseek-v4-flash-ga-260731`

本实验复用既有 DeepRead、embedding、judge、trajectory 和 diagnostic bundle，不重跑前置阶段。

## 运行结果

诊断通过当时的 v1 validator，返回 `not_agent_failure`：7 轮模型调用、8 次按需读取，累计 129,569 input tokens、68,825 output tokens，其中 65,840 reasoning tokens。模型没有给出 DeepRead 最早干预节点，`affected_sources` 为空。

HTTP 层另行验证了共享账户的并发限流行为：同一 endpoint/model 的 95-token 请求成功，而完整诊断首轮曾收到 `AccountRateLimitExceeded`。诊断 runner 因而采用 provider 内有限重试，不重启 agent 轮次。

## 与人工诊断比较

| 维度 | 人工判断 | agent 判断 | 结论 |
| --- | --- | --- | --- |
| 最早节点 | 不应归因到 DeepRead 节点 | `earliest_intervention=null` | 一致 |
| 根因 | 文档 reconciliation table 与 gold 使用相反的符号约定 | gold/evaluator 与 corpus presentation 的 sign-convention mismatch | 一致 |
| 修改范围 | 不应据此修改 DeepRead；应先审查数据/评测约定 | `affected_sources=[]` | 一致 |
| 反证 | 若以迎合 gold 为目标，读取 income statement 并统一按带符号公式计算可得到 gold | 明确给出该反事实及 falsifier | 一致 |

内容归因通过人工比较。这个样本不应进入 DeepRead repair cohort；应进入 evaluation/data review，或者至少在更多样本证明存在稳定 agent 行为缺陷前保持隔离。

## 证据坐标审计发现

人工复核 payload 字符范围后发现，当时 validator 仍有缺口：

- reconciliation table anchor 使用 `[2280, 3240)`，但其声称的 `Income tax expense/(benefit)` 行从约 3325 才开始；
- income statement anchor 使用 `[2000, 4200)`，能覆盖 pre-tax loss，但 tax amount 行从约 4195 开始，范围没有完整覆盖所声称的金额。

模型确实通过更大的工具读取看过这些内容，因此根因结论仍成立；但 anchor 本身不能独立支撑 claim。后续协议要求 source/payload anchor 携带短原文 `quote`，并验证 quote 位于声明范围内。旧产物保留为该校验缺口的回归样本，不人工改写为“通过”的新产物。

## M4 结论

- 开放式诊断可以识别“非 agent 失败”，避免错误生成 DeepRead patch；
- 支持证据、反证和反事实足以与人工判断比较；
- 仅验证“范围读过”不等于验证“范围支持 claim”，必须增加原文 quote 校验；
- 新 quote 协议的真实模型复跑可等待 API 资源空闲，不阻塞进入 M5 的归并设计。
