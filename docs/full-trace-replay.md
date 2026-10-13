# 全量 agent 日志与 baseline 复用重跑

这一入口用于重新收集框架各阶段的真实输入输出，不改变 DeepRead baseline 行为，也不覆盖已有实验。只复用旧 baseline 的预测、原始 DeepRead trace 和已建索引；重新执行 judge、trajectory 编译、诊断输入包、诊断、hypothesis 聚合、规划，以及所有 `proceed` 计划的修改、审计、测试、必要的索引重建和候选运行/评估。

## 214 服务器完整命令

在 `/noraiddata/zhangqianyi/agentic-rag-evolve` 中运行，沿用当前 `.env`：

```bash
nohup .venv/bin/python -u -m runner.run_replay \
  --baseline-experiment /noraiddata/zhangqianyi/agentic-rag-evolve-data/financebench/runs/financebench-initial-v1 \
  --output /noraiddata/zhangqianyi/agentic-rag-evolve-data/financebench/replay-runs/full-trace-v1 \
  --repo-root /noraiddata/zhangqianyi/agentic-rag-evolve \
  --env-file /noraiddata/zhangqianyi/agentic-rag-evolve/.env \
  --diagnosis-max-rounds 20 \
  --modification-max-rounds 100 \
  --modification-max-tool-calls 200 \
  --workers 1 \
  --request-timeout 1800 \
  --request-max-retries 3 \
  > /noraiddata/zhangqianyi/agentic-rag-evolve-data/financebench/full-trace-v1.console.log 2>&1 < /dev/null &
```

这是一条连续命令，无需手动选择 plan。`workers=1` 让全部计划依次独立执行，减少账号并发限制；资源允许时可改为 `3`。默认是原 development 集上的完整 screening，不晋升、不访问虚构的 holdout。多轮晋升仍需配置真实的 `--validation-config` 和 `--max-iterations`。

需要先检查时，使用同样参数运行前台命令并加 `--dry-run`：没有模型调用、不写文件、不创建 worktree。中断后，确认旧进程已经退出，在原命令加 `--resume`；已经完成并通过 hash 校验的阶段不会重跑。若要重新采集全部非 baseline 中间阶段，请更换输出目录及 console 日志名，例如 `full-trace-v2`，不要使用旧目录加 `--resume`。

数据集、documents manifest、store、task 顺序以及 DeepRead 运行配置均从旧 experiment manifest 继承，无需重新整理数据。必须仍能访问其中的绝对路径和旧源码 revision 的 Git 对象。

## 新实验的阅读位置

相对于 `replay-runs/full-trace-v1/`：

| 阶段 | 主要输入/输出 | 全量模型/工具日志 |
| --- | --- | --- |
| baseline 复用 | `experiment/baseline/`；来源与 SHA256 见 `experiment/baseline_reuse.json` | 复用的 `experiment/baseline/deepread_trace.jsonl`，不伪造旧日志 |
| 重新 judge | `experiment/evaluation/evaluation.json` | `experiment/evaluation.judge.trace.json` |
| trajectory / 输入包 | `experiment/trajectories/`、`experiment/diagnostic_bundles/` | 无模型；由旧 DeepRead 原始 trace 重新生成 |
| diagnosis | `experiment/diagnoses/<task>/attempt-NNNN/diagnosis.json`、`audit.json` | 同目录 `agent_trace.json` |
| hypothesis 聚合 | `experiment/planning/cohort.json` → `hypotheses.json` | `experiment/planning/hypotheses.trace.json` |
| modification planning | cohort + hypotheses + 受限源码 → `plan.json` | `experiment/planning/plan.trace.json` |
| candidate modification | `repairs/round-0001/candidates/<plan>/attempt-NNNN/modify/attempt-MMMM/modification.json` | 同目录 `modification.trace.json` |
| 固定测试 | `repairs/round-0001/candidates/<plan>/attempt-NNNN/tests/attempt-MMMM/tests.json` | 同目录 `tests.logs/check-NNNN.stdout.log`、`stderr.log`，完整而非尾部截断 |
| candidate DeepRead | `repairs/round-0001/candidates/<plan>/attempt-NNNN/run-development/attempt-MMMM/run/` | `deepread_trace.jsonl` |
| candidate judge | `repairs/round-0001/candidates/<plan>/attempt-NNNN/eval-development/attempt-MMMM/evaluation/` | 同级 `evaluation.judge.trace.json` |
| 汇总 | `experiment/experiment_report.json`、`repairs/report.json` | 各 subprocess 阶段另有 `command.json`、`stdout.log`、`stderr.log` |

具体阶段名以 `state.json` 和报告中的路径引用为准。失败的诊断/修改 attempt 留在原目录；失败的聚合/规划完整 trace 随其他产物归档到 `experiment/planning/attempts/<stage>/attempt-NNNN/`；被中断的初始 judge batch 在 `experiment/evaluation_attempts/`。确定性 triage 跳过的任务也写 trace，但 `events=[]`、outcome 为 `skipped`，不会为补日志调用模型。

## 简洁且完整的 trace 协议

框架侧日志是 `evolution-agent-trace-v1` JSON。一次模型调用对应一个 `type=model` event，包含完整 request 参数、消息、完整原始 response（含 API 返回的 `reasoning_content`、tool calls、usage）、耗时，以及该调用内部的 HTTP 尝试/重试。一次工具调用对应一个 `type=tool` event，参数与结果放在一起；`replace_text` 的真实 `old_text/new_text`、`read_source` 内容和 `show_diff` 均完整保留。

首次请求记录全部 messages，后续用 `messages_prefix` + `messages_delta` 表达与前次请求共享的前缀和本轮新增/替换内容。它是无损编码，不是摘要、截断或压缩上下文，不影响 agent 收到的消息。需要逐轮还原时：

```python
import json
from pathlib import Path
from agentic_rag_evolve.telemetry.evolution import reconstruct_requests

trace = json.loads(Path("<trace-path>").read_text())
for payload in reconstruct_requests(trace):
    print(payload)
```

请求在 API 调用前以 `pending` 状态原子保存，完成后更新同一条记录；异常保留输入和错误。进程被强制杀死时，最后一条可能保持 `pending`，不能据此认定供应商没有收到请求。`response` 仅包括 API 实际返回的内容，不会声称获得供应商没有返回的内部思考。模型默认输出预算、timeout、retries 记录在 `provider_settings`；request 是传入 provider 的原始 payload，provider 注入的默认预算可由该设置还原。

trace 保留最终 outcome、入口代码 hash、输入文件路径/hash 和预算。现有 compact audit/diagnosis/plan/modification 协议保持不变；诊断 agent 不会自动读取自己的全量 trace，也不会因此增加模型上下文。接入代码在框架 `telemetry/evolution.py`，四个 agent 的接入只有 decorator 和工具结果记录点，DeepRead 目标源码没有变化。

日志不保存 `.env`、API key、Authorization headers 或 provider 对象 repr，但包含问题、标准答案、文档源码和工具读取内容，应按实验数据权限管理。

## 复用与失败保护

复用前检查原 baseline checkpoint 的四个必需产物 SHA256、dataset/documents manifest、实际 Markdown 内容、task 顺序、目标运行预算、目标/embedding/rerank 模型名、store manifest 和索引文件，以及原 revision 的完整 `systems/deepread` 文件集/内容。框架 revision 或 `EVOLUTION_LLM_*` 可变化，目标源码和 baseline 模型不可混用。历史产物没有记录 provider endpoint/维度等完整配置；校验不会声称验证了历史中未记录的字段，实际运行仍应沿用相同 target 配置。

baseline 文件被物理复制，既不是 symlink 也不是 hardlink，保持 byte-identical。复制先进入独立 attempt，校验后原子发布，部分复制保留供排查；所有旧实验文件、失败候选和 patch 均不修改。新 trace 加入阶段产物 hash 校验，planning retry 会一起归档 trace。

旧实验 `--resume` 会继续复用已完成旧阶段，不会补造缺失历史 full trace。要获得完整新数据，应使用本页的新目录重跑命令。
