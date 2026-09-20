# AgenticRAGEvolve

面向 DeepRead 文档问答系统的自动化诊断、修复与验证框架。项目参考 HarnessFix 的闭环结构，但会在开发过程中逐步验证架构决策，而不是预先固定一套缺陷分类或修复算子。

当前工作的第一目标是重建一个边界清晰、可独立运行、可复制修改的历史 DeepRead baseline。诊断和自动修复将在 baseline 的执行协议与轨迹协议稳定后实现。

## 冻结基线

初始迭代严格基于：

- DeepRead：`7fe3ba23f81d88ee83552ba7f38cd6cc25e6c1eb`
- ruc-ov-eval：`fb8a301cfd9cb92f19a5c95cd0066da1133b734b`

这一版本已有 global 模式所需的 `get_doc_structure`，但没有后续加入的 session pagination、跨轮检索去重、停滞提示和模型专项适配。版本锁定信息见 [baseline.lock.json](systems/deepread/baseline.lock.json)。

## 当前结构

```text
systems/deepread/
  DeepRead/                 # 以 7fe3ba2 为来源的工作源码
  ingestion.py              # 唯一 active 的 Markdown ingestion
  runtime.py                # 从 fb8a301 提取的 standalone global runtime
  baseline.lock.json        # DeepRead 与 ruc-ov-eval 来源 revision
  PROVENANCE.md             # 快照规则与下一步提取边界
src/agentic_rag_evolve/
  providers/                # target 不可见的模型配置、鉴权与传输实现
  evaluation/               # F1、证据召回、LLM judge 与历史结果对比
  diagnostics/              # evidence ladder、失败信号、受限输入包与报告
  diagnosis/                # 证据锚定的受限工具诊断 agent 与引用校验
  evolution/                # hypothesis/plan 后的 candidate 隔离、审计与终态记录
  validation/               # 行为门禁与 development-only 回退反馈
  reporting/                # 已完成演化轮次的紧凑可复核投影
  deepread_runner.py        # 稳定单数据集执行协议
  reference_validation.py  # golden store/run 只读校验
runner/                     # 框架 CLI；不属于被进化的 DeepRead
docs/
  architecture.md           # 当前阶段架构与代码归属
  roadmap.md                # 渐进实施顺序
  harnessfix-alignment.md   # 与 HarnessFix 的简要目录对齐
  baseline-extraction.md    # 历史 baseline 重建方案
```

## 实施主线

```text
历史源码冻结
    ↓
Markdown ingestion + 最小 DeepRead runtime + 隔离 gold label 的 DocumentQA 协议
    ↓
单数据集 runner / evaluator / trace
    ↓
与历史 ruc-ov baseline 对齐
    ↓
失败诊断 → 改进计划 → 隔离修改 → 验证门禁
```

详细说明见 [系统架构](docs/architecture.md)和[实施路线](docs/roadmap.md)。

## 当前入口

```bash
uv run python runner/validate_reference.py --store <store_index> --run <historical_run>
uv run python runner/run_deepread.py --dataset <dataset.jsonl> --store <store_index> --output <empty_run_dir>
uv run python runner/evaluate_deepread.py --dataset <dataset.jsonl> --predictions <predictions.jsonl> --output <empty_eval_dir> --judge
uv run python runner/compile_trajectory.py --trace <deepread_trace.jsonl> --predictions <predictions.jsonl> --output <empty_trajectory_dir>
uv run python runner/build_diagnostic_bundle.py --trajectory <trajectory.json> --evaluation <evaluation.json> --run-manifest <manifest.json> --store <store_index> --output <empty_bundle_dir>
uv run python runner/build_bad_case_report.py --bundle <bundle_dir> --output <empty_report_dir>
uv run python runner/run_diagnosis.py --bundle <bundle.json> --source-root . --output <empty_diagnosis_dir>
```

模型配置由仓库根目录 `.env` 提供，字段模板见 `.env.example`。评测设计及错误语义见 [DeepRead 评测模块](docs/evaluation.md)，事件与轨迹协议见 [DeepRead trajectory](docs/trajectory.md)，诊断 triage 见 [确定性失败信号](docs/failure-signals.md)，诊断输出与引用门禁见 [证据锚定诊断协议](docs/diagnosis.md)，行为门禁与回退再诊断边界见 [validation](docs/validation.md)，候选终态与固化见 [outcome registry](docs/outcome-registry.md)和[materialization](docs/materialization.md)，失败反馈见 [repair memory](docs/repair-memory.md)，演化基线见 [baseline registry](docs/baseline-registry.md)，轮次汇总见 [iteration report](docs/iteration-report.md)。
