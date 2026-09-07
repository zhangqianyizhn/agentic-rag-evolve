# Baseline provenance

`DeepRead/` 是 DeepRead revision `7fe3ba23f81d88ee83552ba7f38cd6cc25e6c1eb` 的原样 Git archive，不是当前 DeepRead checkout 的副本。

在 baseline 行为对齐完成前：

- 不做格式化、重命名或无关清理；
- 不引入该 revision 之后的功能；
- 若必须修正迁移阻塞问题，使用单独提交并记录与上游快照的 diff；
- 上游许可证保留在 `DeepRead/LICENSE`。

ruc-ov-eval revision `fb8a301cfd9cb92f19a5c95cd0066da1133b734b` 暂未整体复制。下一阶段只提取其 `ov_test/src/core/deepread_store.py` 中构成 DeepRead global runtime 的逻辑，并通过行为测试确认等价。
