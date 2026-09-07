# DeepRead trajectory v1

## 两层记录

`deepread_trace.jsonl` 是 append-only 的原始事实源，保存模型、provider、工具和最终答案事件。编译器不会修改它，而是在独立目录中为每个 task 生成 `<task_id>.trajectory.json`。

紧凑 trajectory 删除 `context_delta_preview`、provider URL 等重复或非 agent 决策字段，并在对应 event 的 `omitted_fields` 中登记；`raw_event_id` 始终指向原始事件。工具结果、模型 reasoning/content 和工具参数当前保持完整，后续若引入裁剪，必须同时生成可回读的外部 payload 引用。

## 标识与关联

- `run_id`：一次 runner 调用唯一；
- `task_id`：数据集稳定 ID，不使用问题文本代替；
- `event_id`：raw JSONL 中全局递增；
- `step_NNNN`：task trajectory 内的顺序 ID；
- `parent_id`：连接 request/response、model/tool call 和 tool call/result。

新 runner 从源头写入前三类 ID。编译器仍支持早期只有 question hash 的 trace，但仅在 hash 能唯一映射到一个 prediction 时才归属事件。

早期 raw log 的 `tool_result` 没有 `tool_call_id`。compiler 仅在结果紧邻一个可确定的最近 tool call 时建立 fallback parent，并记录 `legacy_inferred_tool_parent` warning；新版日志始终写入显式 ID。

## 当前规范化事件

事件分为 `model.*`、`provider.*`、`tool.*`、`answer.final` 和 `run.*`。未知 raw event 不会丢弃，而是保存为 `raw.<event>` 并产生 warning。

每个 trajectory 汇总轮数、工具调用次数、候选数量、涉及的文档、read_section 文档和终止状态。重复 event ID、非法/倒序时间、孤立事件、缺失终止事件和未能分配到 task 的事件会被显式报告。

## 与 HarnessFix 的差异

HarnessFix 需要统一多个异构 agent 的 trajectory；这里先忠实表示 DeepRead 的检索链路，不采用预设诊断 taxonomy，也不把 raw trace 提前压缩为只服务某类修复算子的格式。
