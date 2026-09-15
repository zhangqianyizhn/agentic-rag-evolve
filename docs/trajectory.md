# DeepRead trajectory v2

## 两层记录

`deepread_trace.jsonl` 是 append-only 的原始事实源，保存模型、provider、工具和最终答案事件。编译器不会修改它，而是在独立目录中为每个 task 生成 `<task_id>.trajectory.json`。

紧凑 trajectory 以一个 agent round 为一个 `turn`。每个 turn 合并一次模型决策以及该决策触发的全部工具交互；每个工具的 arguments/result 也在同一个对象中。正常的单次 provider attempt/success、request envelope、`context_delta_preview` 和 provider URL 不进入诊断视图。

模型 reasoning/content、工具参数和工具结果保持完整。`raw_event_range` 指向组成该 turn 的首尾 raw event；每个工具额外保留 call/result event ID。超过 4 KB 的工具结果写入 `payloads/`：主 trajectory 保留类型摘要、相对路径、字节数与 SHA-256，需要时可无损回读和校验。较低默认值使诊断 agent 像 HarnessFix 读取 HTIR 一样按需展开大结果，而不是在首轮接收完整检索正文。阈值可通过 `--inline-result-bytes` 调整，设为 `0` 可外置全部非空结果。

原始 JSONL 每行重复 `run_id`/`task_id` 是有意的：单行可以独立检索、恢复和并发归并。它不是诊断模型的输入，因此暂不采用依赖文件头状态的压缩格式。

## 标识与关联

- `run_id`：一次 runner 调用唯一；
- `task_id`：数据集稳定 ID，不使用问题文本代替；
- `event_id`：raw JSONL 中全局递增；
- `turn.round`：一次模型决策及其工具交互；
- `raw_event_range`：该 turn 对应的原始审计事件区间。

新 runner 从源头写入前三类 ID。编译器仍支持早期只有 question hash 的 trace，但仅在 hash 能唯一映射到一个 prediction 时才归属事件。

早期 raw log 的 `tool_result` 没有 `tool_call_id`。compiler 仅在结果紧邻一个可确定的最近 tool call 时建立 fallback parent，并记录 `legacy_inferred_tool_parent` warning；新版日志始终写入显式 ID。

## 当前语义结构

模型 request/response 和正常 provider 生命周期合并为 `turn.model`；tool call/result 合并为 `turn.tools[]`。只有重试、provider error、tool parse recovery 等异常信息才额外保留。未知 raw event 不会静默丢弃，而是形成 warning 并保留 raw event ID。

每个 trajectory 汇总轮数、工具调用次数、候选数量、涉及的文档、read_section 文档和终止状态。重复 event ID、非法/倒序时间、孤立事件、缺失终止事件和未能分配到 task 的事件会被显式报告。

## 与 HarnessFix 的差异

HarnessFix 的 `TracingLiteLLMModel` 与 task-local recorder 证明 wrapper/context 是可行机制；但其 agent 包仍显式导入 tracing，模型构造也显式选择 tracing model，因此不是完全无侵入。

本项目把 JSONL 写入、event ID、run/task scope 和 trajectory 编译放在进化框架的 telemetry/trajectory 层。模型 request/response 由框架的 `TracingChatModel` 自动记录，provider 重试通过 task-local trace context 归属，无需 DeepRead 计算 query hash 或传入 logger。工具分发与参数语义由 target 内的 `DeepReadToolExecutor` 负责，框架的 `TracingToolExecutor` 装饰它并自动记录完整 call/result。

DeepRead agent loop 不再依赖日志协议。它返回只包含 answer、termination reason 和 rounds completed 的 `AgentOutcome`，并通过小型 `AgentObserver` 报告文本工具调用恢复、无效参数和空响应。框架的 `TraceAgentObserver` 负责将这些类型化通知转成 raw event；runtime 根据 outcome 记录最终答案或最大轮次终止。

后续继续缩小埋点面：

1. [x] 用 chat-model wrapper 捕获模型 request/response、provider 重试和 token 信息；
2. [x] 用 tool executor proxy 捕获一次完整的工具交互。

这两类 wrapper 不能都放进模型 provider：工具执行不是模型传输职责。模型生成的 `tool_call_id` 仍由 agent 放入后续 tool message，这是模型协议本身所需的关联；它由 tracing proxy 消费，不会传给 target 的工具实现。agent 仅在框架无法可靠推断的语义分支（例如文本恢复出的工具调用）保留极少量 observer hook。

HarnessFix 需要统一多个异构 agent 的 trajectory；这里先忠实表示 DeepRead 的检索链路，不采用预设诊断 taxonomy，也不把 raw trace 提前压缩为只服务某类修复算子的格式。
