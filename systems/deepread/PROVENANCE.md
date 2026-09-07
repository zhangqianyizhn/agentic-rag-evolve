# Baseline provenance

`DeepRead/` 在 AgenticRAGEvolve commit `4c74375` 中是 DeepRead revision `7fe3ba23f81d88ee83552ba7f38cd6cc25e6c1eb` 的原样 Git archive，不是当前 DeepRead checkout 的副本。后续行为保持型适配以独立提交叠加，因此可以始终与该快照比较。

在 baseline 行为对齐完成前：

- 不做格式化、重命名或无关清理；
- 不引入该 revision 之后的功能；
- 若必须修正迁移阻塞问题，使用单独提交并记录与上游快照的 diff；
- 上游许可证保留在 `DeepRead/LICENSE`。

ruc-ov-eval revision `fb8a301cfd9cb92f19a5c95cd0066da1133b734b` 不整体复制。`runtime.py` 提取其共享 `DocIndex`、固定参数调用和 token 汇总行为；`ingestion.py` 提取其 Markdown → corpus → embedding/id-map 主路径。原 DeepRead 对 `src.core.token_tracer_util` 的反向依赖由等价的本地 thread-local tracker 替代。

所有目标 benchmark 都已有 Markdown，因此工作版本删除了上游快照中的 `index/pdf_parser.py`、`index/paddleocr.sh` 和旧 parse CLI。原始文件仍保存在 commit `4c74375`，这一裁剪不改变约定输入上的 DeepRead 问答行为。
