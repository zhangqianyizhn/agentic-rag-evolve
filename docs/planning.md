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

`runner/run_modification_planning.py` 将 hypothesis 转换为修改契约，但不创建 candidate、不能读取未引用源码，也不执行编辑。与 HarnessFix 从 operator registry 推导 allowed paths 不同，DeepRead plan 只能选择 hypothesis 的 `affected_source_refs`；validator 再从原 diagnosis cohort 解析真实 path 与 symbol，模型不能直接提供文件路径。

planner 通过框架提供的 `list_sources` 和 `read_source` 按需理解实现。源码目录不会直接暴露为通用文件系统能力：source catalog 只包含当前 hypotheses 已引用、且位于 `systems/deepread/DeepRead/` 或 active `systems/deepread/ingestion.py` 下的文件，每次最多返回 240 行和 30,000 字符。`proceed` plan 选择的每个文件都必须在本轮真实读取过，仅读取目录或直接猜测实现会被 validator 拒绝；`defer` 不要求无意义地读取源码。

读取器在启动时冻结 allowlisted 文件 SHA256，并在每次工具调用和最终落盘前复核，因此 planning 过程中源码变化会产生 `source_changed`，不会输出 plan。若 `--candidate-audit` 存在，还会重新核验 candidate HEAD、完整 changed-path 集合和 candidate snapshot SHA256；这可以发现发生在 source catalog 之外的新增或修改。audit 只保存 revision、读取文件的 path/SHA256 和工具调用元数据，不复制源码内容。

主要门禁：

- `singleton` hypothesis 必须 `defer`，不能自动进入修改；
- `proceed` 必须覆盖 hypothesis 的全部 development task；
- allowed source 不得超出 hypothesis 已引用范围；
- 必须声明 preserved behavior、non-goals、holdout selection、expected observations、rollback conditions 与 regression scenarios；
- framework、runner、provider、benchmark、tests、docs 和 `.env` 固定为 forbidden roots；
- 输出自动计算 `max_files_to_modify`，并要求 modify agent 编辑前先检查全部 allowed sources。
- 若范围包含 active ingestion 或 Markdown parser，输出自动标记 `requires_store_rebuild=true`；该字段不是模型决定，不能通过提示词省略重建。

无 hypothesis 时确定性输出 `no_plannable_hypotheses`，模型调用和 token 均为零。真实调用同样产生独立 audit 与失败 candidate。

```bash
python runner/run_modification_planning.py \
  --cohort <hypothesis-cohort.json> \
  --hypotheses <improvement-hypotheses.json> \
  --source-root <baseline-or-candidate-worktree> \
  --candidate-audit <optional-candidate-audit.json> \
  --memory-context <optional-planning-memory.json> \
  --output <modification-plan.json> \
  --env-file .env
```

当已有 rejected candidate 时，先按 [repair memory](repair-memory.md) 生成与当前 hypotheses 绑定的紧凑上下文。planner 会复核其 artifact 谱系；完全重复的历史失败 attempt 即使模型再次提出，也会被 validator 拒绝。

初始 hypothesis planning 通常以当前 baseline checkout 作为 `--source-root`，此时 revision 表示 allowlisted source manifest。由 development regression feedback 触发的 candidate 再诊断/再计划则同时传入 candidate worktree 和其 static audit，使读取权限绑定到已审计 snapshot。二者使用同一工具协议，模型无需理解 Git、manifest 或 provider 实现。

## Candidate isolation and static audit

通过 plan gate 后，`runner/create_candidate.py` 从明确的 base commit 创建 detached Git worktree。manifest 位于 worktree 外部，冻结 plan hash、base commit、allowed paths 和文件预算；不自动创建分支或清理目录。选择依据见 `docs/decisions/0002-candidate-isolation.md`。

`runner/audit_candidate.py` 是只读审计，检查：

- candidate HEAD 仍等于 base commit；
- 当前 plan 文件与 manifest hash 一致；
- tracked/untracked changed paths 均在 allowed paths 内且不超过预算；
- forbidden roots 未被触碰；
- 不存在新增/修改 symlink；
- changed Python 文件可编译，且 `git diff --check` 通过。

本阶段只实现隔离和静态审计，不会在没有 recurring `proceed` plan 时创建真实 candidate。行为评测、development/holdout 运行和晋升/回滚属于 M6 validation gate。

## Candidate 固定测试审计

candidate 创建时同时冻结固定测试策略的 ID/SHA256。静态审计再冻结 candidate HEAD、变化路径和变化文件内容形成的 snapshot SHA256。`runner/audit_candidate_tests.py` 只执行这份仓库维护的固定测试策略，不接受 modification agent 提供的命令，并在测试前后检查 snapshot 未变化。行为验证必须同时匹配 plan SHA256、测试策略 SHA256、candidate snapshot 和通过的固定测试审计。
