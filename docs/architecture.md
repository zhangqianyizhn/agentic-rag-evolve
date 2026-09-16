# 当前阶段的系统架构

这份架构是可演进的工作假设。项目刚开始时，优先建立可运行事实，再根据迁移、轨迹和诊断实验修订模块边界。

## 1. 仓库同时包含两类代码

AgenticRAGEvolve 不再被定义为只读取外部 ruc-ov 的“控制面”。它将同时维护：

1. **被进化系统**：一份边界清晰的 DeepRead baseline 及其后续候选版本；
2. **进化框架**：benchmark、runner、evaluator、诊断、修复和验证闭环。

这样修改 agent 不需要阅读整个 ruc-ov-eval，也允许索引、检索工具和 agent loop 一起成为潜在改进对象。

## 2. 初始版本是历史版本，不是当前 HEAD

基线固定为 DeepRead `7fe3ba23...` 和 ruc-ov-eval `fb8a301c...`。后续已有的 session pagination、跨轮去重、停滞提示等能力应被视为未来可能重新发现或引入的改进，不能出现在 v0 中。

`systems/deepread/DeepRead/` 以原样快照为起点；精确内容保存在 AgenticRAGEvolve commit `4c74375`。工作版本通过独立提交进行必要的边界整理，ruc-ov-eval 中属于 DeepRead runtime 的逻辑则逐项迁入相邻模块，并记录原文件、原 revision 和迁移理由。

## 3. 暂定职责边界

### 被进化系统 `systems/deepread/DeepRead`

包含任何会改变 DeepRead 行为或准确率的实现：

- Markdown/corpus 解析和目录树；
- embedding 输入选择、分片与索引结构；
- BM25、regex、vector、hybrid、semantic 检索；
- global 文档定位和 `get_doc_structure`；
- `read_section`；
- agent loop、prompt、工具 schema、上下文组织和停止条件；
- agent 行为相关配置。

索引建立不是评测辅助代码，因为分片、标题和层级结构直接决定检索效果。所有 benchmark 已提供 Markdown，因此 v0 的 active ingestion 只接受 `.md`/`.markdown`；OCR、PDF 解析及其服务配置不进入当前 baseline，也暂不作为进化目标。

具体模型 provider 不属于被进化系统。DeepRead 只声明 `ChatModel`、`EmbeddingModel` 和 `Reranker` 能力端口；API key、endpoint、HTTP、重试和厂商 SDK 位于 `src/agentic_rag_evolve/providers/`。这样修复 agent 可以改变检索 query 和候选使用方式，但默认看不到或修改模型传输实现。

### Benchmark

包含与某个数据集语义有关、但不应被 DeepRead 修改的代码：

- 原始数据到统一 DocumentQA task 的转换；
- train/development/test 切分；
- gold answer、gold evidence 和问题元数据；
- 数据集特定的答案标准化与评分规则。

### Runner / Evaluator

Runner 负责读取 `.env`/运行配置、构造 provider、用固定协议调用任意 DeepRead candidate，并产生 prediction、trajectory、成本和异常记录。Evaluator 只根据任务与输出评分，不导入 DeepRead 内部模块。

### Evolution

在基础执行闭环稳定后再实现：轨迹编译、失败诊断、跨样本归并、改进计划、隔离修改、diff 审计、验证门禁和进化记忆。

诊断 agent 不直接获得 raw trace 或仓库目录。框架先构建 task-level diagnostic bundle，并通过显式 source manifest 与受限 reader 提供按需读取；provider、telemetry、trajectory compiler 和 evaluator 默认不可见。具体协议见 `docs/diagnostic-input.md`。

## 4. 第一条垂直切片

第一条切片只覆盖一个小型 FinanceBench global 子集：

```text
DocumentQA tasks
    ↓
DeepRead v0 ingest/index
    ↓
DeepRead v0 agent retrieval + answer
    ↓
predictions + per-query JSONL trace
    ↓
deterministic metrics + optional LLM judge
```

选择 FinanceBench 是因为它同时包含 global 文档选择、年份/公司辨别、表格数值、多证据和计算问题。最初只需 5–10 个问题用于行为对齐，不把该小样本分数作为研究结论。

## 5. 当前不提前固定的决策

- 最终缺陷 taxonomy；
- 修复 operator registry；
- 诊断使用单 agent 还是多 agent；
- candidate 晋升后采用分支、tag 还是其他长期保存方式（生成阶段已采用 detached Git worktree，见 ADR 0002）；
- 多数据集调度和长期记忆存储实现；
- 哪些 telemetry 必须侵入 DeepRead runtime。

这些问题将在前置模块提供真实约束后分别形成 ADR。

## 6. 数据隔离原则

即使在原型阶段，也区分用于提出改进的 development 样本与最终 test。若 validation 回退被反馈给下一轮，它就已经成为 development 数据，不再承担无偏最终评测角色。标准答案和标准证据不能进入 test 阶段的修改提示。
