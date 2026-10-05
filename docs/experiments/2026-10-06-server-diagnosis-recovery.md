# 服务器诊断失败恢复验证

## 版本与失败位置

排查前，本机、GitHub `main` 和服务器均为 `9a544839b4c600dd043beb963c79af1de721bd28`。实验 `financebench-initial-v1` 最初基于 `2a68092` 启动，冻结了 83 篇 Markdown、106,028 个段落和 138 个任务。

store、baseline、evaluation、trajectory 和 diagnostic bundles 均完成。baseline 成功 138/138，judge 成功 138/138，平均 accuracy_normalized 为 0.875。失败点是 `financebench_id_00540` 的 diagnosis，而非回答、embedding 或 API 故障。

原 attempt 先因 root-cause 文本超过 1,500 字符被拒绝，随后三次格式修复分别把 earliest_intervention 写为字符串、含 action/expected_effect 的对象、含 change/expected_observation/falsifier 的对象。最终错误为 `earliest_intervention has unknown fields`。Prompt 没有提供该字段的精确结构，也没有完整列出 validator 的长度/数量约束。

## 修复和真实验证

修复提交 `836d39f` 补全输出协议，并允许新的 attempt 恢复已有 candidate 与旧 audit 中成功读取的证据，直接执行无工具格式修复。恢复重新核验 bundle/source/payload 哈希，保留旧 attempt；本次恢复复用了 6 次 source 读取和 11 次 payload 读取。

服务器全套 195 项测试通过。以原 manifest 的全部配置加 `--resume` 启动后，前六个阶段全部跳过，已完成诊断继续复用。

`financebench_id_00540/attempt-0002` 在一次模型调用后通过 validator：调用模式 repair、耗时 140.746 秒、没有新工具调用、没有 validation retry。新调用记录 input_tokens=55,867、output_tokens=23,551，其中 reasoning_tokens=21,277；reasoning 是 output 的明细，不应再相加。旧 attempt 保持 validation_error。

已完成阶段的文件 SHA256 与 checkpoint 一致；恢复日志为实验输出目录中的 `resume-836d39f.log`。验证结束时实验已越过原失败点，继续处理其他诊断，尚未生成完整实验终态报告。

## 结论边界

本次证明诊断协议修复和跨提交恢复有效，不代表已验证自动修复与晋升。恢复出的诊断区分了 gold 使用期末库存与原回答使用平均库存这一计算口径差异；validator 只验证引用和结构，不能据此宣称任何金融计算口径必然错误。是否属于应修改的 agent 行为仍需后续证据和验证。
