# 渐进实施路线

每个阶段先学习 HarnessFix 对应实现，再完成一个可以独立检查的 DeepRead 版本。后续阶段可以根据上一阶段经验调整，不把当前路线视为不可更改的最终设计。

## M0：冻结和理解 baseline（当前）

- [x] 固定 DeepRead `7fe3ba23...`；
- [x] 固定 ruc-ov-eval `fb8a301c...`；
- [x] 原样迁入 DeepRead 源码；
- [x] 区分历史基线与当前 HEAD 的后续优化；
- [x] 建立 ruc-ov `deepread_store.py` 的逐函数归属清单；
- [x] 保存历史 FinanceBench sample20 的脱敏参考配置；
- [ ] 登记一个历史 FinanceBench 小样本的预期运行产物。

完成条件：能明确说明历史运行依赖哪些 DeepRead/ruc-ov 文件，没有误带后续能力。

## M1：最小 baseline 执行闭环

对应 HarnessFix 的 `task_agent/open_deep_research + run_gaia_entry.py + eval_gaia.py`。

- [x] 定义不携带 gold label 的最小 `DocumentQATask` 和独立 `EvaluationReference`；
- [x] 从历史 `deepread_store.py` 提取可加载既有 corpus 的 global query runtime；
- [x] 建立 Markdown-only ingestion 和可注入 embedding provider 协议；
- [x] 从 active baseline 移除 OCR/PDF 运行路径；
- [x] 实现历史配置使用的 Volcengine multimodal embedding adapter；
- [x] 将 chat、embedding、rerank 的鉴权和传输实现移出 DeepRead target；
- [x] 移除 target 内部重复 CLI，建立框架级 `run_deepread.py`；
- [x] 一次运行输出 manifest、predictions、summary 和逐问题 trace；
- [x] 建立历史 store/run 的只读结构校验；
- [x] 建立不依赖 DeepRead 内部实现的 evaluator；
- [x] 支持逐题对齐历史 ruc-ov 评测结果；
- 用 5–10 个 FinanceBench 问题 smoke test。

完成条件：新项目不调用 ruc-ov pipeline 也能运行 v0，并能与历史产物逐问题比较。

## M2：轨迹协议与可观测性

对应 HarnessFix 的 trajectory、sanitizer 和 HTIR 构建，但重新定义为文档 QA 轨迹。

- [x] 检查 v0 现有 `llm_request/response`、`tool_call/result`、`final_answer` 事件；
- [x] 增加稳定 run/task/event ID；
- [x] 建立保留 raw JSONL 的 task-level trajectory compiler；
- [x] 将模型调用和 provider 生命周期 tracing 移出 DeepRead agent loop；
- [x] 将工具分发提取为 target-owned executor，并用框架 proxy 捕获 call/result；
- [x] 用 AgentObserver/AgentOutcome 收敛剩余 agent 语义事件和终止状态；
- [x] 建立诊断输入字段投影、源码 allowlist 和受限 artifact reader；
- [x] 在 evidence ladder 中增加 corpus source ref；
- [x] 表示检索候选、read 和最终证据覆盖；
- 处理并发乱序、重试、缺失事件和大结果裁剪。

完成条件：任一答案都可追溯到完整搜索/读取路径，且原始事件不因紧凑视图丢失。

## M3：评测结果和确定性失败信号

- [x] 对齐 generated answer、gold answer、gold evidence 与 trajectory；
- [x] 计算 evidence-to-corpus、candidate、read、answer 四级覆盖；
- [x] 区分执行异常、无答案、错误答案和评测可疑样本；
- [x] 生成可人工审阅的 bad-case 报告。

完成条件：在冻结样本上完成人工核查，并报告启发式信号的不确定性。

## M4：证据锚定诊断

对应 HarnessFix 的 `failure_analysis/run_analysis.py`，但不照搬 GAIA 固定分类。

- [x] 定义最早可干预节点、支持/反驳证据和反事实协议；
- [x] 用受限工具按需检查 DeepRead 源码；
- [x] 校验事件引用、代码引用、实际读取范围和诊断自洽性；
- [x] 与人工诊断比较节点、原因和修改范围（见 `docs/experiments/m4-diagnosis-00585.md`）。

## M5：归并、计划和隔离修改

对应 HarnessFix 的 `aggregate_results.py`、modify agent 和 diff audit。

- [x] 建立 repair-eligible diagnosis cohort，并隔离 evaluation/data review 与证据不足样本；
- [x] 将相似诊断归并为可证伪的 improvement hypothesis（协议、validator 与空 cohort 已验证；真实 recurring cohort 待合格诊断样本齐备）；
- [x] 声明目标 cohort、潜在副作用、允许修改范围和验证计划；
- [x] 实现受冻结 plan scope 约束、无通用 shell 权限的 candidate modification agent；
- [x] 实现 detached worktree candidate 隔离与 manifest 冻结（真实 candidate 等待 recurring `proceed` plan）；
- [x] 独立执行 diff/范围/静态审计；
- [x] 独立执行 candidate 固定测试审计，并冻结测试前后源码快照；
- [x] 实现与静态审计及冻结 plan 绑定的行为验证协议。

## M6：验证门禁与外层循环

对应 HarnessFix 的 train compare、validation gate 和 harness memory。

- [x] development bad case 与同数据集 holdout 的逐题配对门禁（协议与合成测试）；
- [x] promotion 级跨数据集回归与 token 成本门禁（协议与合成测试）；
- [x] 以不可变 outcome record 保存 accepted/rejected candidate revision；
- [x] 将 accepted revision 物化为确定性的 detached Git commit；
- [x] 建立线性不可变 baseline registry，并用 durable Git ref 保存 materialized commit；
- [x] 将 rejected outcome 转为可验证的紧凑 repair memory，并阻止完全相同的失败计划重试；
- [x] 隔离 sealed validation 反馈，并从已登记 accepted baseline 生成 preservation memory；
- [x] 将 development 回退显式导出为可重新诊断的 feedback artifact，同时只保留 sealed cohort 聚合；
- [x] 为 modification planner 提供 hypothesis-scope 源码读取，并在 candidate 模式复核完整 snapshot；
- [x] 生成不复制大产物、不泄漏 validation task ID 的终态 iteration report；
- [x] 建立 hash-chain 单轮账本，支持失败不推进、同阶段重试与 accepted/rejected 分支恢复；
- [x] 用冻结 runbook 自动执行/恢复各阶段，并将 stdout/stderr 摘要作为账本证据；
- [x] 支持诊断跳过、空 repair cohort 和无 candidate 的合法短路终态；
- [x] 用真实 FinanceBench 单题验证 baseline、judge、trajectory、bundle 和 no-candidate 终态恢复（见 `docs/experiments/m6-full-flow-smoke-00517.md`）；
- 在真实小样本上完成全流程；
- [x] 在验证协议中拒绝 final test role，使其不参与 candidate 选择。

完成条件：可以从 v0 自动完成一轮诊断、修改和候选晋升，并完整复现实验谱系。
