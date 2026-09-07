# ADR-0001：在本仓库维护冻结的 DeepRead baseline

- 状态：Accepted
- 日期：2026-09-07

## 背景

当前可运行的 global DeepRead 分散在 DeepRead 与 ruc-ov-eval 两个仓库。直接让诊断/修改 agent 面对 ruc-ov-eval，会暴露其他 RAG 系统、大量数据预处理和评测编排代码；只引用 DeepRead 仓库又会遗漏 global 运行适配和入库逻辑。

## 决策

AgenticRAGEvolve 自身维护一份可运行的 DeepRead baseline。初始源码固定为 DeepRead `7fe3ba23f81d88ee83552ba7f38cd6cc25e6c1eb`，运行适配行为以 ruc-ov-eval `fb8a301cfd9cb92f19a5c95cd0066da1133b734b` 为准。

先原样保存 DeepRead 源码，再逐项提取 ruc-ov 中属于 DeepRead runtime 的部分。每次提取都记录来源和行为对齐测试，不把当前 HEAD 的后续优化带入 v0。

## 影响

仓库不仅是进化控制面，也包含被进化系统。前期会有一定源码重复，但换来明确的修改边界、可复现历史起点和独立 runner。何时以及如何生成 candidate 暂不在本 ADR 中决定。
