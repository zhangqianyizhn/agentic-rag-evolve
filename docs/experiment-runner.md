# 统一实验入口

`runner/run_experiment.py` 将初始 DeepRead 实验串行为一个可恢复命令。底层模块仍保持独立，统一入口负责固定输入、检查阶段产物、记录哈希并在 `--resume` 时跳过已完成阶段。

## 模式

- `baseline`：preflight、store、baseline run、evaluation、trajectory；
- `diagnose`：在 baseline 模式之后继续执行 diagnostic bundle、逐题 diagnosis、hypothesis cohort、hypothesis aggregation、modification planning 和终态报告。

`diagnose` 必须启用 judge。无需诊断的正确题仍生成确定性 bundle，但 diagnosis 会按 route 自动跳过，不产生模型调用。

如果所有 modification plan 均 defer，流程生成 `no_candidate` 终态报告。如果存在 proceeding plan，流程生成 `candidate_planned` 报告并停止。候选修改与晋升必须等 development、holdout、cross-dataset cohort 冻结后再进入现有 candidate/validation 编排，不能用单一 development 数据集自动晋升。

## 服务器示例

以下示例把原始 Markdown 保留在 home，把 store 和实验产物写到空间更充足的磁盘：

```bash
cd /home/zhangqianyi/agentic-rag-evolve

uv run --frozen python runner/run_experiment.py \
  --experiment-id financebench-initial-v1 \
  --mode diagnose \
  --documents /home/zhangqianyi/agentic-rag-evolve-data/financebench/documents.jsonl \
  --dataset /home/zhangqianyi/agentic-rag-evolve-data/financebench/splits/compatible_all.jsonl \
  --store /noraiddata/zhangqianyi/agentic-rag-evolve-data/financebench/stores/baseline-v0 \
  --output /noraiddata/zhangqianyi/agentic-rag-evolve-data/financebench/runs/financebench-initial-v1 \
  --source-root /home/zhangqianyi/agentic-rag-evolve \
  --env-file /home/zhangqianyi/agentic-rag-evolve/.env \
  --max-rounds 50 \
  --retrieval-topk 5 \
  --request-timeout 1800 \
  --request-max-retries 3
```

先用一题验证真实 provider：

```bash
uv run --frozen python runner/run_experiment.py \
  --experiment-id financebench-smoke-v1 \
  --mode baseline \
  --documents /home/zhangqianyi/agentic-rag-evolve-data/financebench/documents.jsonl \
  --dataset /home/zhangqianyi/agentic-rag-evolve-data/financebench/splits/compatible_all.jsonl \
  --store /noraiddata/zhangqianyi/agentic-rag-evolve-data/financebench/stores/baseline-v0 \
  --output /noraiddata/zhangqianyi/agentic-rag-evolve-data/financebench/runs/financebench-smoke-v1 \
  --source-root /home/zhangqianyi/agentic-rag-evolve \
  --env-file /home/zhangqianyi/agentic-rag-evolve/.env \
  --limit 1
```

中断或可重试的 API 失败后，使用完全相同的参数并增加：

```bash
--resume
```

恢复时会重新核对 Git revision、dataset/documents 哈希及所有已完成 artifact。输入、源码或完成产物发生变化时拒绝混跑。diagnosis 失败会保留原 attempt；恢复时创建下一 attempt，不覆盖诊断审计记录。

## 产物

```text
<output>/
  experiment_manifest.json
  experiment_state.json
  preflight.json
  baseline/
  evaluation/
  trajectories/
  diagnostic_bundles/        # diagnose 模式
  diagnoses/                 # diagnose 模式；每题按 attempt 保存
  planning/                  # diagnose 模式
  experiment_report.json
```

如果 store 尚未建立，入口自动建库；存在合法 `STORE_MANIFEST.json` 时复用并校验。非空但没有合法 manifest 的 store 不会被覆盖。
