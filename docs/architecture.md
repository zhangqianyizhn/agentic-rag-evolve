# 系统架构

## 1. 设计目标

系统需要同时解决两类目标：一是把针对单个数据集的人工 bad-case 优化自动化；二是继续探索 global DeepRead 的准确率上限。二者共享同一闭环，但验收标准不同：局部修复要解决明确失败，通用改进还必须通过跨数据集门禁。

首版只面向 DeepRead，但适配层不能把核心逻辑绑定到某个数据集。新系统作为独立控制面，通过路径和 Git revision 引用 `ruc-ov-eval` 与 DeepRead，不复制其代码和数据。

## 2. 核心数据对象

后续实现先稳定以下对象，再引入 LLM 诊断或自动改码。

### ExperimentManifest

记录一次可复现实验：数据集/切分、评测配置、目标仓库与 revision、模型和检索参数、输入输出路径、随机性设置、父实验及变更假设。密钥只记录环境变量名，不保存值。

### QueryTrace

以一个问题为单位对齐：

- 问题、样本/文档标识、标准答案和标准证据；
- 生成答案、指标、延迟与 token；
- 有序的模型/工具事件；
- 每次搜索的 query、scope、候选、rank、score、文档/节点/段落坐标；
- read 操作、实际进入上下文的证据和最终引用；
- corpus/id map/config/revision 等来源引用。

原始事件必须保留不可变引用。IR 中的每个诊断证据都应能回到原文件和事件，而不是只保存摘要。

### Diagnosis

诊断不是强制单选的固定 taxonomy，而是结构化因果论证：失败表现、最早可干预节点、支持/反驳证据、候选根因、置信度、建议修改范围，以及仍需做的判别实验。可以附加开放标签用于聚类，但标签不决定修复算子。

### ImprovementHypothesis

描述一组失败为何可能由同一机制导致、预期改善哪些 cohort、可能伤害哪些 cohort、最小修改范围、所需消融实验和接受条件。

### PatchCandidate / ValidationReport

补丁记录目标 revision、diff、生成依据和风险；验证报告保存配对实验结果、逐 cohort 变化、成本变化、统计不确定性及接受/拒绝原因。失败补丁也进入记忆库。

## 3. 主要模块

### A. 实验登记与产物适配 `artifacts`

发现并校验 ruc-ov 的 `generated_answers.json`、`qa_eval_detailed_results.json`、`deepread_run.log`、配置与 corpus 元数据；建立稳定 ID，把分片运行和合并运行统一起来。此层只做无损读取和 schema 校验。

### B. DeepRead 轨迹编译 `trace_ir`

按 `query_id` 合并 JSONL 事件，并与评测记录对齐。显式表示 global 模式的两阶段行为：全库分片检索、文档定位/目录加载、文档内检索、section read、答案合成。编译器要容忍乱序并发日志、重试、缺失事件和旧版本 schema。

### C. 结果分解与单样本诊断 `diagnosis`

先计算确定性信号，再交给诊断 agent：标准证据是否可映射到 corpus、是否进入候选、是否被 agent 读取、答案是否受已读证据支持、是否存在错误文档/时间/单位混合，以及工具异常、轮次耗尽和停滞等执行信号。

这些是观测维度，不是封闭缺陷枚举。诊断 agent 必须给出事件锚点和反事实：如果在某个最早节点采取何种不同动作，失败为何可能被避免。

### D. Cohort 构建与假设归并 `synthesis`

按可解释特征聚合诊断，例如问题组成性、证据跨度、文档数量、答案类型、所需检索阶段、工具轨迹模式。系统从诊断中归纳改进假设，合并重复原因，并把“通用机制”和“数据集特有策略”分开报告。

### E. 修复规划与补丁执行 `repair`

修改对象包括 prompt、工具 schema/反馈、检索与排序、目录加载策略、状态管理、停止条件、上下文预算和评测集成。这里不使用固定 operator registry；模型可以提出新修改，但必须声明允许文件、修改预算、预期机制和验证计划。补丁只在临时 worktree/分支应用，并经过 diff、语法、单测和敏感信息检查。

### F. 实验与验证门禁 `evaluation`

统一调用现有 ruc-ov runner，支持断点恢复、缓存、并发和失败重试。验证从低成本到高成本逐级进行：

1. 静态检查和单元测试；
2. 目标 bad case 重放；
3. 同数据集未参与诊断的 validation；
4. 多数据集回归集；
5. 完整评测或候选间竞赛。

门禁同时比较 Accuracy、evidence Recall、拒答行为、token、延迟与异常率。接受规则不能只看被分析的 bad case。

### G. 进化记忆 `memory`

保存实验谱系、诊断、假设、补丁和验证结果。区分事实证据、模型推断和最终决策；记录被拒补丁及失败条件，避免循环尝试。记忆检索以机制和轨迹特征为主，数据集名仅作为一个特征。

### H. 闭环编排 `orchestration`

用可恢复状态机连接各模块，每一步写入 manifest 和完成标记。支持从任意阶段继续、预算上限、人工审批点和候选并行比较。编排层不包含诊断规则或数据集业务逻辑。

## 4. 与现有代码的接口

首版直接利用下列现有接口，不先改 ruc-ov：

- `ov_test/src/core/deepread_store.py`：DeepRead 配置入口、`run_agent` 调用、`collected_texts` 与 token 汇总；
- `ov_test/src/pipeline.py`：生成记录、Recall 计算、LLM judge 和 DeepRead 日志统计；
- `DeepRead/agent/runner.py` 与 `agent/logger.py`：模型/工具/final_answer 事件；
- `DeepRead/tool/*`：BM25、regex、vector、hybrid、semantic、read_section 及 global 文档目录加载行为；
- 各 adapter：数据集标准答案、标准证据和答案后处理语义。

当 M1/M2 暴露出缺失的可观测字段时，再以最小补丁增强 DeepRead 日志。不要先为了“完整 telemetry”大范围修改被测系统。

## 5. 防止数据集过拟合

- 数据按 diagnosis/train、validation、test 三种用途隔离；test gold 不进入诊断和补丁提示。
- 每个改进至少报告 micro、macro-dataset 与最差数据集变化，不能用总体均值掩盖回退。
- 优先做 leave-one-dataset-out：在若干数据集诊断/修复，在未见数据集验证迁移。
- 对模型、温度、并发、语料版本和 judge 版本做配对控制；必要时重复运行估计方差。
- 将 prompt 中显式数据集知识标记为 dataset-specific，不与通用候选混为一谈。
- 诊断质量单独评估；不能因补丁偶然涨分就倒推诊断正确。

## 6. 明确不在首版做的事

- 不重新实现 ruc-ov 的数据 adapter、指标或 DeepRead 检索器；
- 不直接在主工作目录上让修改 agent 自由改码；
- 不从全部测试集 bad case 反复调参后仍把测试分数当泛化结果；
- 不以一份固定缺陷分类表或修复算子表限制系统探索空间；
- 不把大体积 corpus、日志、模型输出或密钥提交到 Git。
