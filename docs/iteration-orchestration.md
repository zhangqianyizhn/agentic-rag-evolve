# 单轮编排与恢复

单轮运行由两份不同职责的文件控制：

- ledger 是框架事实源。`manifest.json` 固定 iteration 与起始 baseline；`events/` 中每个完成或失败事件都绑定前一事件和实际产物哈希。
- runbook 是冻结的执行说明。每一步只提供无 shell 的 argv 数组和预期产物路径；API 配置仍由各 CLI 从 `.env` 读取，不写入 runbook 或 ledger。

最小用法：

```bash
uv run python runner/manage_iteration.py init \
  --root artifacts/iterations/i1 \
  --iteration-id i1 \
  --baseline-id baseline-0000 \
  --baseline-commit <40-char-commit>

uv run python runner/run_iteration.py \
  --ledger-root artifacts/iterations/i1 \
  --runbook configs/iterations/i1.runbook.json
```

runbook 的结构如下。路径相对 `workspace`；glob 只允许相对路径且必须恰好匹配一个文件。

```json
{
  "schema_version": "deepread-iteration-runbook-v1",
  "iteration_id": "i1",
  "workspace": "/absolute/project/path",
  "steps": {
    "baseline_run": {
      "command": ["uv", "run", "python", "runner/run_deepread.py", "..."],
      "artifacts": {"primary": "runs/i1/manifest.json"}
    }
  }
}
```

执行器只运行 ledger 报告的 `next_step`。命令失败或产物缺失/协议不符时写入 failed event，next step 保持不变；修复外部条件后再次运行即可重试。已完成步骤的输入产物或 runbook 一旦变化，恢复校验会失败，避免在不知情的情况下混合两次实验。

当前固定阶段为 baseline run、evaluation、trajectory、diagnostic bundle、diagnosis、hypothesis/plan、candidate 创建与修改、两级审计、validation gate、outcome，以及按 outcome 选择的终态分支。accepted 必须继续 materialization 和 baseline advance；rejected 直接进入 repair memory。两者最终都生成 iteration report。
