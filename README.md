# AgenticRAGEvolve

面向 DeepRead 文档问答系统的自动化诊断、修复与验证框架。项目参考 HarnessFix 的闭环结构，但会在开发过程中逐步验证架构决策，而不是预先固定一套缺陷分类或修复算子。

当前工作的第一目标是重建一个边界清晰、可独立运行、可复制修改的历史 DeepRead baseline。诊断和自动修复将在 baseline 的执行协议与轨迹协议稳定后实现。

## 冻结基线

初始迭代严格基于：

- DeepRead：`7fe3ba23f81d88ee83552ba7f38cd6cc25e6c1eb`
- ruc-ov-eval：`fb8a301cfd9cb92f19a5c95cd0066da1133b734b`

这一版本已有 global 模式所需的 `get_doc_structure`，但没有后续加入的 session pagination、跨轮检索去重、停滞提示和模型专项适配。版本锁定信息见 [baseline.lock.json](systems/deepread/baseline.lock.json)。

## 当前结构

```text
systems/deepread/
  DeepRead/                 # 7fe3ba2 的原样源码快照
  baseline.lock.json        # DeepRead 与 ruc-ov-eval 来源 revision
  PROVENANCE.md             # 快照规则与下一步提取边界
src/agentic_rag_evolve/     # 新框架代码（后续逐模块增加）
docs/
  architecture.md           # 当前阶段架构与代码归属
  roadmap.md                # 渐进实施顺序
  harnessfix-alignment.md   # 与 HarnessFix 的简要目录对齐
  baseline-extraction.md    # 历史 baseline 重建方案
```

## 实施主线

```text
历史源码冻结
    ↓
最小 DeepRead runtime + DocumentQA 协议
    ↓
单数据集 runner / evaluator / trace
    ↓
与历史 ruc-ov baseline 对齐
    ↓
失败诊断 → 改进计划 → 隔离修改 → 验证门禁
```

详细说明见 [系统架构](docs/architecture.md)和[实施路线](docs/roadmap.md)。
