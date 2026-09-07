# 与 HarnessFix 的结构对齐

| HarnessFix | AgenticRAGEvolve | 差异说明 |
|---|---|---|
| `task_agent/open_deep_research/` | `systems/deepread/` | DeepRead 除 agent 外还包含索引与检索，这些同样是被进化对象。 |
| `data/gaia_*` | `benchmarks/<dataset>/`（计划） | 同时保存统一 DocumentQA adapter 和冻结 split，不提交大型原始文档。 |
| `task_agent/run_gaia.sh`、`run_gaia_entry.py` | `runner/run_deepread.py`（计划） | 采用 Python 主入口，允许注入任意 DeepRead candidate。 |
| `eval/eval_gaia.py` | `evaluation/`（计划） | 除答案正确性外还要评价 gold evidence recall、成本和检索行为。 |
| `failure_analysis/` | `evolution/diagnosis`、`evolution/planning`（计划） | 不复用 GAIA 的固定失败类别和 HTIR。 |
| `enhancement_implementation/` | `evolution/repair`（计划） | 允许修改 agent、检索、索引与 prompt，范围由每次计划决定。 |
| `run_pipeline_gaia.py` | `run_pipeline_deepread.py`（后期） | 等各模块稳定后再编写总编排，避免先形成一个特化的大脚本。 |
| 复制 `enhanced_odr_vN` 目录 | candidate 隔离方式待定 | 在实际修改模块前比较 copy、branch 与 worktree。 |

共同主线保持一致：冻结被测系统 → 运行 → 评测 → 诊断 → 聚合计划 → 隔离修改 → train/development 验证 → 晋升或拒绝。
