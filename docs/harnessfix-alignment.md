# 与 HarnessFix 的结构对齐

| HarnessFix | AgenticRAGEvolve | 差异说明 |
|---|---|---|
| `task_agent/open_deep_research/` | `systems/deepread/` | DeepRead 除 agent 外还包含索引与检索，这些同样是被进化对象。 |
| `data/gaia_*` | `benchmarks/<dataset>/`（计划） | 同时保存统一 DocumentQA adapter 和冻结 split，不提交大型原始文档。 |
| `task_agent/run_gaia.sh`、`run_gaia_entry.py` | `runner/run_deepread.py` | Python 主入口已从 DeepRead target 外部注入 provider 和 candidate。 |
| `eval/eval_gaia.py` | `src/agentic_rag_evolve/evaluation/` | 保留独立 evaluator，但 FinanceBench 使用 token F1、evidence recall 和可注入的 0–4 LLM judge。 |
| 各 task agent 的 `.traj.json` | raw `deepread_trace.jsonl` + `trajectory/` compiler | 不强迫 DeepRead 直接输出通用轨迹；保留 raw 事实源，再生成带父子关系的 task-level 视图。 |
| `failure_analysis/` | `diagnostics/` + `evolution/diagnosis`（计划） | 已先实现确定性 evidence/failure signals 和受限输入包；不复用 GAIA 的固定失败类别和 HTIR，LLM 根因诊断留在后续模块。 |
| `enhancement_implementation/` | `evolution/repair`（计划） | 允许修改 agent、检索、索引与 prompt，范围由每次计划决定。 |
| `run_pipeline_gaia.py` | `run_pipeline_deepread.py`（后期） | 等各模块稳定后再编写总编排，避免先形成一个特化的大脚本。 |
| 复制 `enhanced_odr_vN` 目录 | candidate 隔离方式待定 | 在实际修改模块前比较 copy、branch 与 worktree。 |

共同主线保持一致：冻结被测系统 → 运行 → 评测 → 诊断 → 聚合计划 → 隔离修改 → train/development 验证 → 晋升或拒绝。

## 运行边界

与 HarnessFix 一致，CLI 属于稳定 benchmark harness，而不是被进化的 agent：

- `systems/deepread/DeepRead/` 不包含参数解析或可执行入口；
- `runner/` 负责路径、运行规模和输出协议；
- `src/agentic_rag_evolve/providers/` 负责配置、鉴权、HTTP、重试和厂商 SDK；
- DeepRead 只能通过 `ports.py` 中的 `ChatModel`、`EmbeddingModel` 和 `Reranker` 能力接口调用模型。

默认修复范围不包含 `runner/` 和 provider adapters。检索 query、候选融合、工具调用策略仍属于 DeepRead，可作为后续进化对象。
