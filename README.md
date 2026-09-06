# AgenticRAGEvolve

面向 DeepRead / ruc-ov 的自动化 RAG 改进系统。项目把人工执行的“bad case 对比、原因定位、修改系统、重新评测”变成可复现、可审计的闭环，同时把跨数据集泛化作为一等目标。

当前阶段只建立系统边界、模块契约与开发路线，不实现具体诊断或修复逻辑。

## 问题边界

输入主要来自 `ruc-ov-eval`：

- `generated_answers.json`：问题、标准答案、标准证据、生成答案、召回文本及 token/延迟；
- `qa_eval_detailed_results.json`：逐问题正确性评价；
- `deepread_run.log`：按 `query_id` 记录的模型调用、工具调用、工具结果和最终答案；
- DeepRead corpus、目录树/id map、实验配置和对应 Git revision。

输出不是一组写死的经验规则，而是：可定位到轨迹证据和代码位置的诊断、可验证的改进假设、隔离生成的补丁，以及跨数据集验证报告。

## 闭环

```text
实验登记 -> 产物归一化 -> 失败诊断 -> 跨样本归并 -> 改进提案
   ^                                                |
   |                                                v
接受/拒绝 <- 多级验证门禁 <- 隔离补丁应用 <- 风险与范围检查
```

与 HarnessFix 的关键区别是：

- 使用面向文档 QA/RAG 的轨迹表示，而不是复用通用 agent 的固定 HTIR 分类；
- 缺陷标签和修复方式允许由证据归纳产生，不要求落入预枚举类别/算子；
- 标准答案和标准证据只用于诊断与训练集分析，测试集仅用于最终评价；
- 改进是否成立由 bad-case、同数据集留出集、跨数据集回归和成本约束共同决定。

详细设计见 [系统架构](docs/architecture.md)，实施顺序见 [路线图](docs/roadmap.md)，版本管理约定见 [开发流程](docs/development.md)。

## 计划目录

```text
src/agentic_rag_evolve/
  artifacts/      # ruc-ov / DeepRead 产物适配与校验
  trace_ir/       # QueryTrace 与证据引用的统一表示
  diagnosis/      # 单样本因果诊断与诊断质量检查
  synthesis/      # 跨样本/跨数据集假设归并
  repair/         # 修改范围、补丁生成、静态检查
  evaluation/     # 实验运行、配对比较、泛化门禁
  memory/         # 假设、补丁、结果与否定经验
  orchestration/  # 可恢复的闭环状态机
```

## 当前状态

- [x] 架构与模块边界
- [x] 分阶段实施路线
- [x] Git/实验历史管理约定
- [ ] M1：评测产物登记与归一化
- [ ] M2：DeepRead 轨迹 IR
- [ ] M3 及以后：诊断、归并、修复和验证闭环
