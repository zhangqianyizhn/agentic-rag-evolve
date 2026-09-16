# M5 归并与计划

HarnessFix 的 `aggregate_results.py` 在一次模型调用中同时完成固定类别归并、优先级排序、改进计划和 modify spec。DeepRead 不预设 defect class 或 repair operator，因此先建立一个确定性的 cohort 边界，再在后续模块中进行开放式语义归并。

## Hypothesis cohort

`runner/build_hypothesis_cohort.py` 接收一个或多个已经生成的 `deepread-diagnosis-v1`：

- 只有 `status=diagnosed` 可以进入 `eligible_diagnoses`；
- `not_agent_failure` 路由到 `evaluation_review`；
- `insufficient_evidence` 路由到 `evidence_review`；
- 重复 task、未知 schema/status 和覆盖已有输出都会被拒绝；
- 每条输入保留 SHA-256，整个 cohort 具有与输入顺序无关的稳定 ID。

这一层不自行聚类，也不从文本猜测 repair operator。它的职责是防止数据/评测问题或证据不足样本污染 DeepRead 修改计划。

```bash
python runner/build_hypothesis_cohort.py \
  --diagnosis <diagnosis-1.json> \
  --diagnosis <diagnosis-2.json> \
  --output <hypothesis-cohort.json>
```

下一模块只读取 `eligible_diagnoses`，生成包含目标 cohort、共同机制、反证和可证伪验证条件的 improvement hypothesis；源码修改范围和 candidate 创建仍留在更后面的 planning/modify 阶段。

## Improvement hypothesis aggregation

`runner/run_hypothesis_aggregation.py` 不使用固定 defect class、harness layer 或 repair operator。模型只能通过 task ID 和数组 index 引用 cohort 中已有的 supporting/contradicting evidence 与 affected source，validator 会拒绝未知任务、越界证据、虚构源码范围、重复分组和遗漏任务。

每个 hypothesis 明确记录：

- 共同机制与最早干预模式；
- inclusion/exclusion signals，用于界定目标 cohort；
- 概念级 behavior delta，而非补丁或实现步骤；
- 支持证据、反证和源码引用；
- expected observation、falsifier 与 regression guards；
- `singleton` 或 `recurring` 成熟度。

没有 eligible diagnosis 时直接生成 `no_eligible_diagnoses`，不调用模型。真实调用产生独立 `*.audit.json`，记录耗时、token、provider retry 和校验错误但不复制长 prompt；可解析但无效的候选保留为 `*.candidate.json`。

```bash
python runner/run_hypothesis_aggregation.py \
  --cohort <hypothesis-cohort.json> \
  --output <improvement-hypotheses.json> \
  --env-file .env
```

## Bounded modification plan

`runner/run_modification_planning.py` 将 hypothesis 转换为修改契约，但不创建 candidate、不读取未引用源码，也不执行编辑。与 HarnessFix 从 operator registry 推导 allowed paths 不同，DeepRead plan 只能选择 hypothesis 的 `affected_source_refs`；validator 再从原 diagnosis cohort 解析真实 path 与 symbol，模型不能直接提供文件路径。

主要门禁：

- `singleton` hypothesis 必须 `defer`，不能自动进入修改；
- `proceed` 必须覆盖 hypothesis 的全部 development task；
- allowed source 不得超出 hypothesis 已引用范围；
- 必须声明 preserved behavior、non-goals、holdout selection、expected observations、rollback conditions 与 regression scenarios；
- framework、runner、provider、benchmark、tests、docs 和 `.env` 固定为 forbidden roots；
- 输出自动计算 `max_files_to_modify`，并要求 modify agent 编辑前先检查全部 allowed sources。

无 hypothesis 时确定性输出 `no_plannable_hypotheses`，模型调用和 token 均为零。真实调用同样产生独立 audit 与失败 candidate。

```bash
python runner/run_modification_planning.py \
  --cohort <hypothesis-cohort.json> \
  --hypotheses <improvement-hypotheses.json> \
  --output <modification-plan.json> \
  --env-file .env
```
