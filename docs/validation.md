# Candidate 行为验证协议

`runner/check_validation_gate.py` 对同一批任务的 baseline 与 candidate 评测结果做逐题配对比较。它只接受已经通过静态审计和固定测试审计、且与冻结 modification plan 及 candidate snapshot 哈希一致的 candidate，避免拿错 plan、审计后替换代码或让未测试代码进入行为验证。

## 固定测试审计

`runner/create_candidate.py` 在创建 worktree 时就把仓库维护的 `config/candidate-test-policy.json` ID 和 SHA256 冻结进 manifest。`runner/audit_candidate_tests.py` 只接受这份被冻结的策略；策略不接受 shell 或模型提供的命令，目前只允许框架构造 `python -m unittest discover` 参数。执行环境不继承模型 API 密钥，并把 `PYTHONPATH` 指向 candidate worktree。

静态审计对 candidate HEAD、变化路径以及变化文件的内容/模式生成 snapshot SHA256。固定测试运行前后均重新计算；审计后代码变化、测试过程改写代码、非零退出、超时、无法解析测试数量或测试数量低于策略基线都会失败。完整 stdout/stderr 只保存哈希和有界尾部，避免把大日志复制到后续门禁输入。

## 两级门禁

- `development`：必须包含用于快速反馈的 `development` cohort，以及同数据集但未参与诊断的 `holdout` cohort。
- `promotion`：除上述两类外，还必须包含至少一个 `cross_dataset` cohort，才具备晋升资格。

`test` 不是合法 cohort role。最终 test 不进入诊断、调参或 candidate 选择循环。

每个 cohort 显式指定唯一任务 ID、主指标和阈值。当前支持 `f1`、`recall`、`accuracy_normalized`；主指标缺失、非有限或超出 `[0, 1]` 时直接报错，不用其他指标替代。门禁同时检查：

- 主指标平均变化；
- 改善题数（仅 development 可要求）；
- 回退题数；
- candidate 预测异常数不超过 baseline；
- answer 运行记录中的 input/output token 总量比例。

development 与其余验证 cohort 按 `(dataset, task_id)` 必须互斥。问题文本和 sample ID 也必须在 baseline/candidate 两侧一致。每个 cohort 还会冻结 baseline/candidate evaluation 文件的 SHA256，供下游反馈和编排重新核验。

## 回退反馈边界

`runner/build_regression_feedback.py` 将一个已经落盘的 validation gate 转成 `deepread-regression-feedback-v1`。它重新核对 gate、suite 和 static audit 的 SHA256、candidate/plan/snapshot ID，并把 candidate snapshot 作为下一轮诊断的源码版本。

```bash
uv run python runner/build_regression_feedback.py \
  --gate <development-gate.json> \
  --suite <development-suite.json> \
  --candidate-audit <candidate-audit.json> \
  --output <regression-feedback.json>
```

该产物不是普通的门禁报告，而是下一轮诊断的权限边界：

- 只有 `development` cohort 的回退 task ID 和 evaluation artifact 引用可以进入诊断；
- `holdout` 与 `cross_dataset` 只保留 cohort 名称、数据集、回退数量、平均变化、token 比例和失败原因，不暴露 task ID 或 evaluation 路径；
- 若只有 sealed cohort 回退，状态为 `no_development_regressions`，不会把这些题转成可诊断样本；
- `test` 仍不属于 validation role，产物固定记录 `final_test_accessed: false`；
- feedback ID 由 gate 哈希、candidate snapshot 与 development 回退集合确定，输出文件不可覆盖。

这样借鉴 HarnessFix 的 regression feedback / rediagnosis 闭环时，不会把 validation 样本在失败后悄悄转成训练数据。后续诊断 bundle 构建器只应接受这里的 `diagnosis_scope.task_ids`，并在读取源码时重新核验 candidate snapshot。

## 与 HarnessFix 的差异

HarnessFix 的 validation gate 主要以 resolved 数量决定是否通过，并将成本变化作为报告项。DeepRead 的单题分数和证据质量可能连续变化，因此这里采用逐题配对指标、显式 regression budget 和 token cost gate；同时把“快速开发检查”和“可晋升检查”分成两级，防止每次小迭代都运行完整跨数据集验证。

门禁只作确定性判定，不运行 DeepRead，也不隐式选择数据。suite 中的 cohort 和阈值必须由已审阅的 modification plan/实验配置明确给出。

## 当前边界

目前已用合成评测产物验证协议、错误拒绝路径、promotion 成功路径、accepted/rejected outcome registry 和 development-only regression feedback。仓库中尚无满足 recurring hypothesis 与 `proceed` plan 的真实修复 candidate，因此没有伪造一次真实晋升结果。单轮外层编排仍属于后续实现。
