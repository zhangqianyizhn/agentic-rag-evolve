# 当前模型下的 FinanceBench 历史 bad-case 复筛

## 目的

历史 sample20 运行中有 4 个 judge 分数低于 4 的问题。由于当前 `.env` 已更换模型，本实验不直接复用旧失败结论，而是重新运行 baseline 与 judge，判断是否存在足够的 recurring failure 来验证 proceeding candidate 路径。

## 复筛结果

| task | 历史分数 | 当前分数 | 结论 |
|---|---:|---:|---|
| `financebench_id_00494` | 1 | 4 | 当前已正确回答 |
| `financebench_id_01290` | 3 | 4 | 当前已正确回答 |
| `financebench_id_05718` | 2 | 4 | 当前已正确回答 |
| `financebench_id_00585` | 1 | 1 | 进入真实诊断 |

`00585` 的初次诊断运行出现了工具循环：12 个调查轮、18 次工具请求，累计约 327k 输入 token 和 99k 输出 token，最终 `max_rounds`，没有 diagnosis。审计表明模型多次读取已经完整覆盖的 payload 小范围，且在最后一轮工具预算耗尽后没有提交 JSON 的机会。

框架随后拒绝完全被既有读取覆盖的 payload 范围，并在调查预算结束后保留一次不暴露工具的最终提交轮。相同冻结 bundle 的重跑在 5 轮、9 次工具调用后成功，累计约 103k 输入 token 和 66k 输出 token，未发生 validation failure。

## 诊断结论

成功诊断的状态是 `not_agent_failure`。Boeing 10-K 的税率调节表明确披露 FY2022 `(0.6)%`、FY2021 `14.7%`；DeepRead 的答案与披露一致。gold 使用从税前亏损和税收费用/优惠推导的另一种符号约定，得到 `+0.62%` 和 `-14.76%`。两者绝对值一致，分歧来自亏损年份的符号约定，而非检索、读取或源码故障。

因此 4 个历史低分样本在当前模型下产生 0 个 repair-eligible diagnosis。hypothesis aggregation 与 planning 均以 0 次模型调用产生空结果，终态为 `no_candidate`，baseline 未改变。

## 后续

不能通过放宽 singleton/recurrence 门禁来强制验证 modifier。真实 proceeding 路径需要扩大 development split，先由当前 baseline + judge 自动筛选当前模型仍然失败的样本，再从至少两个独立任务中确认共同机制。
