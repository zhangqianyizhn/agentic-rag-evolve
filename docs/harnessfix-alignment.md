# 与 HarnessFix 的结构对齐

| HarnessFix | AgenticRAGEvolve | 差异说明 |
|---|---|---|
| `task_agent/open_deep_research/` | `systems/deepread/` | DeepRead 除 agent 外还包含索引与检索，这些同样是被进化对象。 |
| `data/gaia_*` | `benchmarks/<dataset>/`（计划） | 同时保存统一 DocumentQA adapter 和冻结 split，不提交大型原始文档。 |
| `task_agent/run_gaia.sh`、`run_gaia_entry.py` | `runner/run_deepread.py` | Python 主入口已从 DeepRead target 外部注入 provider 和 candidate。 |
| `eval/eval_gaia.py` | `src/agentic_rag_evolve/evaluation/` | 保留独立 evaluator，但 FinanceBench 使用 token F1、evidence recall 和可注入的 0–4 LLM judge。 |
| 各 task agent 的 `.traj.json` | raw `deepread_trace.jsonl` + `trajectory/` compiler | 不强迫 DeepRead 直接输出通用轨迹；保留 raw 事实源，再生成带父子关系的 task-level 视图。 |
| `failure_analysis/` | `diagnostics/` + `diagnosis/` | 确定性 evidence/failure signals 负责分流，诊断 agent 使用开放式假设与强引用校验；不复用 GAIA 的固定失败类别、fix scope 和 HTIR。 |
| `enhancement_implementation/` | `evolution/repair`（计划） | 允许修改 agent、检索、索引与 prompt，范围由每次计划决定。 |
| `run_pipeline_gaia.py` | `run_pipeline_deepread.py`（后期） | 等各模块稳定后再编写总编排，避免先形成一个特化的大脚本。 |
| 复制 `enhanced_odr_vN` 目录 | detached Git worktree + 外部 manifest | 保留完整 runner/target 组合和精确 base commit；通过验证后再决定长期分支或 tag。 |
| validation gate 的 resolved 数量与成本报告 | development/promotion 两级逐题配对门禁 | DeepRead 显式限制平均指标、题级回退、运行异常和 token 成本；promotion 另需跨数据集 cohort。 |
| promotion 后推进 `current_base_version`、写 accepted/rejected harness memory | outcome registry + materialization/baseline registry + repair/preservation memory | outcome 本身不改变 Git 或 baseline；accepted 只有登记为 baseline 后才产生 preservation memory，rejected 只按源码路径交集反馈可复核事实；sealed validation 不泄露题级反馈，也不沿用固定 defect/operator 分类。 |
| validation regression feedback 重新进入 diagnosis | `deepread-regression-feedback-v1` 的 development-only 诊断权限 | 保留 HarnessFix 的再诊断闭环，但 holdout/cross-dataset 只给聚合统计；若没有 development 回退则不创建可诊断题目。 |
| modify/planning agent 按需读取 harness 源码 | hypothesis-scope `list_sources` / `read_source` | 只开放 diagnosis 已引用的 DeepRead 文件；proceed 前强制真实读取，candidate 重试还需持续匹配 static-audit snapshot。 |
| SWE iteration report 复制 plan markdown、diff、audit 和 compare | `deepread-iteration-report-v1` 终态投影 | 只引用并重验不可变 artifact，保存计数和决策；不复制大内容或 task ID，也不把派生报告当作恢复状态。 |
| 整数 `current_base_version` + 版本目录 | 线性 baseline entry + internal Git ref | 当前版本由不可变账本末端推导；完整仓库 commit 保留 framework/runner/target 的一致组合。 |

共同主线保持一致：冻结被测系统 → 运行 → 评测 → 诊断 → 聚合计划 → 隔离修改 → train/development 验证 → 晋升或拒绝。

HarnessFix 的诊断输出采用完整示例模板，但运行端只要求可解析 JSON，并在后处理中补字段、归一化类别；解析失败则生成低置信 fallback。`DefaultAgent` 会保留并重发完整 `messages`，本身没有自动压缩；实际长度控制来自 agent 主动调用带 `--char-limit` 的 trajectory/HTIR 摘要脚本。它的 GAIA 分析模型设置单次 300 秒超时和零 HTTP 重试。我们的诊断层参考这一“宽输入、稳输出”和按需读取方式，先规范化模型常见写法、最多纠错一次；长思考模型的实测默认超时放宽到 1,800 秒，共享账号出现并发限流后增加了 provider 内有限 429/5xx 重试（默认额外 3 次，遵循 `Retry-After` 或 15/30/60 秒退避），但不重启 agent 轮次。差异是继续校验 trajectory/source/payload 引用真实性，且不引入预枚举缺陷类别。

## 运行边界

与 HarnessFix 一致，CLI 属于稳定 benchmark harness，而不是被进化的 agent：

- `systems/deepread/DeepRead/` 不包含参数解析或可执行入口；
- `runner/` 负责路径、运行规模和输出协议；
- `src/agentic_rag_evolve/providers/` 负责配置、鉴权、HTTP、重试和厂商 SDK；
- DeepRead 只能通过 `ports.py` 中的 `ChatModel`、`EmbeddingModel` 和 `Reranker` 能力接口调用模型。

默认修复范围不包含 `runner/` 和 provider adapters。检索 query、候选融合、工具调用策略仍属于 DeepRead，可作为后续进化对象。
