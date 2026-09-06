# 开发与历史管理

## Git 边界

本仓库保存框架代码、schema、测试、配置模板、ADR 和小型脱敏 fixture。`ruc-ov-eval`、DeepRead 与 HarnessFix 都是外部参考仓库，不复制到本仓库，也不把本地绝对路径写进已提交配置。

大体积日志、corpus、模型输出、worktree 和实验运行目录分别放在 `artifacts/`、`runs/`、`worktrees/`，默认忽略。可复现信息由小型 manifest 进入版本控制。

## 提交粒度

- 一个提交只完成一个可验证变化；架构决策、schema、实现和测试可按依赖顺序拆分；
- 使用前缀：`docs:`、`feat:`、`fix:`、`test:`、`refactor:`、`chore:`、`exp:`；
- `exp:` 只提交实验 manifest、汇总与结论，不提交原始大文件；
- 自动生成的 DeepRead 补丁在独立分支/worktree 验证，通过门禁后再人工决定是否迁移到目标仓库；
- 不改写已用于实验的提交；新修正通过追加提交表达，保证 revision 可追踪。

推荐分支形式：

```text
main
feature/m1-artifact-loader
experiment/<hypothesis-id>/<candidate-id>
```

## 每个模块的完成定义

1. 数据契约和错误行为已文档化；
2. 单元测试覆盖正常、缺失、损坏和旧版本输入；
3. 至少一个来自真实 ruc-ov 产物的小型 smoke test；
4. `git diff --check`、测试和类型/静态检查通过；
5. 变更日志能关联 issue/hypothesis/experiment ID。

## 实验谱系

每个实验 manifest 至少记录：

- 唯一 ID、创建时间、父实验与 hypothesis ID；
- AgenticRAGEvolve revision、ruc-ov revision、DeepRead revision；
- 数据集、split 文件哈希、配置文件哈希；
- agent/judge/embedding 模型标识与关键参数；
- 补丁哈希、运行命令、产物位置和状态；
- 汇总指标、逐数据集指标、成本指标和门禁决策。

API key、完整 prompt 中的秘密、用户目录和服务端内部地址不得进入 manifest。

## 架构决策

影响多个模块或实验可比性的决定写入 `docs/decisions/`。决策一旦被后续实验使用，不直接覆盖原文；新增 ADR 标记替代关系。
