# Candidate 行为验证协议

`runner/check_validation_gate.py` 对同一批任务的 baseline 与 candidate 评测结果做逐题配对比较。它只接受已经通过静态审计、且与冻结 modification plan 哈希一致的 candidate，避免拿错 plan 或未审计代码进入行为验证。

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

development 与其余验证 cohort 按 `(dataset, task_id)` 必须互斥。问题文本和 sample ID 也必须在 baseline/candidate 两侧一致。

## 与 HarnessFix 的差异

HarnessFix 的 validation gate 主要以 resolved 数量决定是否通过，并将成本变化作为报告项。DeepRead 的单题分数和证据质量可能连续变化，因此这里采用逐题配对指标、显式 regression budget 和 token cost gate；同时把“快速开发检查”和“可晋升检查”分成两级，防止每次小迭代都运行完整跨数据集验证。

门禁只作确定性判定，不运行 DeepRead，也不隐式选择数据。suite 中的 cohort 和阈值必须由已审阅的 modification plan/实验配置明确给出。

## 当前边界

目前已用合成评测产物验证协议、错误拒绝路径和 promotion 成功路径。仓库中尚无满足 recurring hypothesis 与 `proceed` plan 的真实修复 candidate，因此没有伪造一次真实晋升结果。candidate 的固定测试执行审计、accepted/rejected 持久化和外层循环仍属于后续实现。
