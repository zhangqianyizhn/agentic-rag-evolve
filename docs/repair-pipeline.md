# 单命令修复与连续迭代

`runner/run_repairs.py` 接续已完成的 `run_experiment.py --mode diagnose` 实验。它自动读取全部 `decision=proceed` 的计划，无需人工指定 plan ID，不改变原诊断或原计划。

## 先比较当前三个计划

服务器更新到包含本入口的提交后，在仓库目录执行一条命令：

```bash
uv run --frozen python -m runner.run_repairs \
  --experiment /noraiddata/zhangqianyi/agentic-rag-evolve-data/financebench/runs/financebench-initial-v1 \
  --output /noraiddata/zhangqianyi/agentic-rag-evolve-data/financebench/repair-runs/initial-plans-v1 \
  --env-file /noraiddata/zhangqianyi/agentic-rag-evolve/.env \
  --workers 3
```

默认 `--workers 1` 串行执行所有 proceeding plan；`--workers 3` 可同时执行三个。每个候选具有独立 detached worktree、修改记录、静态审计、固定测试、运行产物和必要时重建的索引。候选均从同一个被冻结的 base commit 开始，不共享修改、不将三个补丁直接叠加。

命令连续执行创建、修改、两级审计、必要的索引重建、DeepRead inference、LLM judge 和逐题配对比较。默认使用原 baseline manifest 的全部任务 ID、max_rounds 和 retrieval_topk；原 baseline/diagnosis/planning 均复用。索引修改会对原 documents manifest 重建完整 global store，不只是目标 bad case 的文档。

没有验证配置时，模式是 **screening**：最终 `report.json` 包含各候选 baseline/candidate 平均分、满分数量、改善/回退题数及 token 比率，但不生成 accepted/rejected promotion outcome、不固化候选提交、不推进 baseline。`screening.json` 保存 development 逐题比较，不能当作独立泛化评估。已被诊断或人工检查的原 138 题不会自动伪装成 holdout。

## 自动选择与多轮演化

冻结额外验证数据后，加上：

```bash
--validation-config /absolute/path/repair-validation.json --max-iterations 3
```

配置模板为 [repair-validation.example.json](../config/repair-validation.example.json)。文件路径相对配置文件所在目录解析；数据采用现有 FinanceBench JSONL adapter 字段，即 `financebench_id`、`doc_name`、`question`、`answer`、`evidence`。其他数据集须转换到这一格式。

每轮运行全部 proceeding 计划，然后只在通过原 promotion gate 的候选中，按 holdout/cross-dataset 加权平均分变化排序，development 均值变化用于次级排序，plan ID 用于确定性打破平局。最多晋升一个候选；其他通过门禁的候选仍保留 accepted/eligible outcome，但不推进 baseline，也不强行归类为 rejected。所有候选都未通过时保持旧 baseline。

晋升者按原协议 materialize、登记 baseline、写 preservation memory。下一轮以新 commit 重新运行 development、评测、诊断和聚合，并带 memory 重新 planning；不会继续执行旧 baseline 的过时计划。拒绝者记录 repair memory；没有晋升的下一轮复用原 baseline 与诊断，带新的失败记忆重新 planning。验证集只提供聚合反馈，不进入 diagnosis。最终 test 不属于合法 cohort role。

development 默认是整个原实验任务集，必须与 holdout 互斥。附加配置必须同时包含同数据集 holdout 和其他数据集 cross_dataset；两者不会被自动抽样或伪造。baseline 验证数据的索引和运行结果会在同一 base commit 下复用。缺失的 baseline 验证索引由入口自动建立；所有需要重建的 candidate 索引分别从该候选源码生成。

## 恢复与检查

原命令追加 `--resume` 即可恢复；追加 `--dry-run` 可只核对输入、源码版本、数据角色和自动选中的计划，不创建文件、不调用模型。API 密钥不进入 manifest，可轮换；模型名称、输入文件、测试策略和原始实验结果被哈希冻结。未显式指定 `--base-revision` 时，首次使用 HEAD，恢复沿用首次冻结的 commit，不随当前 HEAD 漂移。

默认拒绝与原 baseline 不同的 LLM、embedding 或 reranker 模型；若只验证框架且接受这个对照混杂因素，可显式加 `--allow-model-change`，报告会记录这一变化，不能把结果作为严格修复收益。embedding 模型仍须与既有向量索引兼容。

每个阶段的失败日志和部分产物保存在独立 attempt 目录，不覆盖或删除。中断的修改可能已改动源码，因此新尝试会创建新的 worktree；成功候选和已完成评测继续复用。索引重建或 inference 中断后，重试使用新输出目录；**当前恢复粒度是阶段，不是逐 query/逐 embedding**，这类阶段可能需要重跑。下一轮 diagnosis 继续复用已有 experiment checkpoint 的题级诊断恢复能力。

共享账号限流时，可在 `--resume` 时把 `--workers 3` 降到 `--workers 1`；这只改变并发调度，不改变冻结的实验数据、模型或目标代码。

结果与日志：

```text
<output>/
  manifest.json                # 冻结配置、非秘密模型名称和输入哈希
  report.json                  # 整体结果；失败时记录 resume_required
  round-0001/report.json       # 本轮候选比较及唯一晋升者
  round-0001/candidates/<plan>/attempt-0001/
    state.json
    worktree/
    <stage>/attempt-0001/{command.json,stdout.log,stderr.log,...}
    screening.json            # screening 模式
  outcomes/                   # promotion 模式
  baselines/
  memory/
```

本入口复用已有候选、门禁、outcome 和 baseline 协议，但有独立的 batch checkpoint；旧 `run_iteration.py` 的 ledger/runbook 入口仍保留，不要求手工生成 runbook。它不是实验调度服务，也不会无限迭代：达到 `max_iterations` 或无 proceeding plan 后结束。

## 与 HarnessFix 的对应

`run_pipeline_gaia.py:953` 的外层循环自动聚合一份包含多个修复点的计划，随后串行 modify、train compare、validation gate；没有人工选 plan 的阶段，也没有并行多个代码候选的分支。通过 gate 才推进 `current_base_version`，失败则保留旧版本并记录反馈。

这里保留自动计划、验证后晋升和下一轮反馈，但将已生成的独立计划分别作为候选实验。`workers` 只控制同轮候选并发；演化轮次始终串行，且只允许一个通过门禁的候选成为下一轮起点。这有助于区分三个机制的效果，尤其当前 BM25 与 regex 计划的源码范围有重叠，直接合并会影响归因。
