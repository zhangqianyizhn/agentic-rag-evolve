# 证据锚定诊断协议 v1

## 与 HarnessFix 的关系

HarnessFix 的 GAIA diagnosis prompt 要求模型从预先枚举的 component、defect class 和 fix scope 中选择，并在同一次输出中提出修复。DeepRead v1 只借鉴“轨迹分析、源码定位、结构化输出”的主线，不复制这些任务特化枚举。

诊断层只回答：发生了什么、最早可以在哪里干预、哪些事实支持或反驳当前假设、什么反事实可以证伪它。修复算子和补丁留给后续 planning/repair 模块。

## 运行资格

`failure_signals.diagnosis_route` 在调用模型前进行分流：

- `incorrect_answer`、`partial_answer`、`no_answer` 和 `execution_failure` 可以进入 DeepRead 诊断；
- `evaluation_suspicious` 转交 evaluation review，不允许诊断 agent 在看不到 evaluator 源码时猜测 DeepRead 根因；
- `needs_judgment` 必须先取得可靠 judge；
- `pass` 不诊断。

## 输出

一个 `deepread-diagnosis-v1` 包含：

- `status`：`diagnosed`、`not_agent_failure` 或 `insufficient_evidence`；
- `failure_manifestation`：可直接观察的失败现象；
- `earliest_intervention`：trajectory turn 和可选 tool call，以及为何这是最早干预点；
- `root_cause_hypothesis`：开放式、可被反驳的根因假设；
- `supporting_evidence` 与 `contradicting_evidence`；
- `counterfactual`：行为变化、预期观测和 falsifier；
- `affected_sources`：实际读过的源码范围和 symbol；
- `uncertainties`。

协议会机械归一化模型偶尔输出的单个 `uncertainties` 字符串为单元素数组；这种处理不改变诊断语义。证据锚点、读取范围、轨迹节点和源码白名单仍执行严格校验。若一个可解析候选未通过校验，会另存为 `candidate.json`，便于修正协议后重新验证，而无需从审计摘要中猜测模型原文。

证据 anchor 只允许五种事实坐标：trajectory turn/tool、evidence coverage 层、evaluation 字段、allowlisted source 行区间、trajectory payload 字符区间。source 和 payload anchor 还必须提供一个不超过 500 字符的原文 `quote`；validator 不仅检查范围被实际读取，还检查 quote 确实位于所声明的行或字符范围内，避免“读过大段内容但引用坐标错位”。这里固定的是引用语法，不是缺陷类别。

## 工具与校验

诊断 agent 只能使用诊断 bundle 已声明的 `list_sources`、`read_source` 和 `read_payload`。Validator 校验：

- task、turn、tool call、evidence index/layer 和 evaluation 字段真实存在；
- source/payload 属于 bundle manifest；
- 模型引用的 source 行或 payload 字符必须被本轮工具调用完整覆盖；
- `diagnosed` 必须同时提供最早干预点、支持证据、反驳证据、反事实和至少一个源码范围；
- 未知字段、过长文本和过多 anchor 被拒绝。

模型常见的嵌套 anchor 写法和 `kind=judge` 会先被规范化为稳定的扁平协议，再进行事实校验。校验失败会连同精确的扁平 anchor 形状返回模型，最多重试一次。诊断 HTTP 调用默认单次等待 1,800 秒；每轮请求对 429/5xx 最多额外重试 3 次，优先遵循 `Retry-After`，否则按 15、30、60 秒退避。重试发生在 provider 内，不会重启 agent 或重复已经完成的轮次；attempt 数和等待时间写入对应 model audit event。相较 HarnessFix 的 300 秒默认值，这适配了长思考模型的实测时延波动和共享账号并发限流。单次 payload 最多返回 12,000 字符、源码最多返回 240 行，越界请求会安全截断并告知 `has_more`。默认不限制总工具调用数或模型输出 token，调查阶段由 12 轮 agent 上限终止；两者仍可通过 CLI 显式设置。完全被既有读取覆盖的 payload 范围会被拒绝，避免模型反复读取同一小片段。调查轮次结束后，框架额外提供一次不暴露工具的最终提交机会；证据不足时应返回 `insufficient_evidence`，不能继续检索。`audit.json` 将一次模型响应或工具调用各记录为一个事件；运行中通过原子替换逐轮 checkpoint，只保存请求字节数、耗时、finish reason、tool arguments、状态、token、验证错误和不含正文的候选结构摘要，不复制初始 bundle、源码内容或模型长文本。

## 命令

```bash
python runner/run_diagnosis.py \
  --bundle <diagnostic-bundle.json> \
  --source-root <repository-root> \
  --output <empty-output-directory> \
  --env-file .env
```

成功输出 `diagnosis.json` 和 `audit.json`；不符合资格的 bundle 只输出 status 为 `skipped` 的 `audit.json`，且不会调用模型。
