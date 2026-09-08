# 诊断输入包与可见范围 v1

## 目标

诊断 agent 默认只看到一个 `deepread-diagnostic-input-v1` bundle。它不读取 raw JSONL、不遍历仓库，也不需要理解 provider、telemetry 或 trajectory compiler。bundle 只能用于 development/train bad case；包含 gold answer/evidence 的信息不得进入最终 test 自适应。

HarnessFix 的 GAIA failure analysis 同时提供 sanitized/raw trace 和 agent source directory，并主要通过 prompt 约束读取方式。本协议将这些约束变成 builder 投影、显式 manifest 和受限 reader 的代码级门禁。

## 默认内联内容

`bundle.json` 包含：

- `task`：task ID、sample ID、question；
- `run_context`：DeepRead 行为配置、模型名称、dataset/store fingerprint，不含 endpoint、API key 或 provider 实现；
- `evaluation`：`gold_answers`、`gold_evidence`、generated answer、终止信息、证据匹配、确定性指标和 judge 结论；
- `trajectory`：紧凑的模型决策和工具交互；
- `access`：源码/payload manifest、允许工具和明确排除层。

trajectory 顶层只允许：

```text
schema_version, task_id, question, status, answer,
token_usage, turns, summary
```

每个 turn 只允许 `round/model/tools/annotations/raw_event_range`。model 只允许模型名称、reasoning/content、content reference、token estimate、异常 retry/error 和 requested tools。tool 只允许 call ID、名称、参数、状态、结果或 result ref、错误以及 raw event anchors。新增 raw trace 字段不会自动进入诊断包，必须显式修改白名单。

## 源码可见范围

权威列表位于 `diagnostics/policy.py`，当前 18 个文件分为：

- agent loop/context：`agent/runner.py`、`agent/contracts.py`、`agent/llm.py`；
- prompt/tool protocol：`prompt/system.py`、`tool/schema.py`、`tool/fallback.py`、`tool/executor.py`；
- retrieval/read：`tool/retrieval.py`、BM25、regex、vector、hybrid、semantic、read_section 和 utils；
- indexing：`tool/corpus.py`、`index/markdown_parser.py`、`systems/deepread/ingestion.py`。

manifest 为每个文件保存 component、用途、SHA-256、字节数和行数，但不默认内联源码。明确排除：

- `src/agentic_rag_evolve/telemetry/`；
- `src/agentic_rag_evolve/providers/`；
- `src/agentic_rag_evolve/trajectory/`；
- `systems/deepread/runtime.py`；
- runner/evaluator 和其他进化框架代码。

诊断 agent 因此可以归因和建议 DeepRead accuracy 改动，但不会把采集、模型传输或评测实现误当作 agent 设计。

## 按需读取工具

v1 只允许三个工具：

1. `list_sources(component?)`：列出 manifest 中的源码；
2. `read_source(path, start_line, end_line)`：只接受 manifest 精确路径，每次最多 240 行和 30,000 字符，并校验源码哈希；
3. `read_payload(path, offset_chars, limit_chars)`：只接受 trajectory 引用的 bundle 内 payload，每次最多 12,000 字符，并校验字节数与 SHA-256。

reader 拒绝绝对路径、`..`、未登记源码、未引用 payload、篡改内容和超限读取。`runner/read_diagnostic_artifact.py` 是当前 CLI 适配；以后接入诊断 agent 时应直接包装同一个 `DiagnosticArtifactReader`，不能另写宽松的文件工具。

## 构建命令

```bash
python runner/build_diagnostic_bundle.py \
  --trajectory <task.trajectory.json> \
  --evaluation <evaluation.json> \
  --run-manifest <manifest.json> \
  --source-root <repository-root> \
  --output <empty-output-directory>
```

builder 校验 task/question/run 对齐、trajectory schema、source hash 和外置 payload hash，并将引用 payload 复制到 bundle，使诊断输入不依赖原 trajectory 目录。
