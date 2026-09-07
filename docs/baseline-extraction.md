# DeepRead v0 提取说明

## 来源

- DeepRead 源码：`7fe3ba23f81d88ee83552ba7f38cd6cc25e6c1eb`；
- global 运行适配：ruc-ov-eval `fb8a301cfd9cb92f19a5c95cd0066da1133b734b`；
- DeepRead 原始路径：仓库根目录；
- ruc-ov 适配原始路径：`ov_test/src/core/deepread_store.py`。

## 已完成

`systems/deepread/DeepRead/` 最初由 `git archive` 从指定 DeepRead revision 原样生成，完整快照保存在 AgenticRAGEvolve commit `4c74375`。工作版本随后删除未处于历史 global 评测主路径的 PDF/OCR 和旧 parse CLI，保留 agent、prompt、工具、Markdown parser 和上游 LICENSE。

该版本的 `get_doc_structure` 已支持 global 检索先发现文档、再加载指定文档目录；搜索工具仍使用 wrapper 提供的固定 top-k。不存在 `agent/search_session.py`，`run_agent` 也没有 pagination、检索结果去重或停滞策略参数。

## `deepread_store.py` 逐项归属

| 历史对象/函数 | 处理方式 | 理由 |
|---|---|---|
| `DeepReadResource`、`DeepReadResult` | 不迁移 | 这是为了模拟 ruc-ov vector store 接口的包装类型，由新的 run-result schema 替代。 |
| `DeepReadWrapper.__init__` | 拆分 | DeepRead 参数进入 runtime config；输出目录和 logger 进入 runner context。 |
| `from_config` | 重写 | 历史函数直接读取 ruc-ov YAML 结构，新 runner 应依赖自己的版本化配置。 |
| `_pdf_to_markdown_pymupdf` | 不迁移 | 所有 benchmark 已提供 Markdown，OCR/PDF 暂不进入 baseline。 |
| `_sample_dir`、`_corpus_path` | 不原样迁移 | 在 global 模式中没有形成有效核心抽象，路径由 workspace layout 统一管理。 |
| `count_tokens` | 迁入 runner telemetry | 不改变 DeepRead 策略，只负责运行成本统计。 |
| `build_uri_map` | 不迁移 | ruc-ov store/pipeline 兼容接口。 |
| `ingest` | 拆分 | 调度、monitor 属于 runner；Markdown corpus、embedder 和索引建立属于 DeepRead ingestion。 |
| `_ingest_one` | 部分迁入 ingestion | 只保留 Markdown 复制、corpus、embedding 和 id map；PDF/OCR 分支不迁移。 |
| `_get_doc_index` | 迁入 global runtime | 扫描全部 corpus 并建立共享 `DocIndex` 是 global DeepRead 的关键行为。 |
| `invalidate_doc_index_cache` | 迁入 global runtime | 与共享索引生命周期绑定。 |
| `retrieve` | 拆分 | `run_agent` 参数绑定属于 runtime；ruc-ov result 包装和全局 token tracker 不迁移。 |
| `process_retrieval_results` | 不迁移 | ruc-ov vector store 兼容接口。 |
| `read_resource` | 暂不迁移 | 未处于 DeepRead 回答主路径；如后续 runner 需要文档审计，再定义明确接口。 |
| `clear` | 迁入 workspace 管理 | 属于运行产物生命周期，不属于 agent 策略。 |
| `main` | 不迁移 | 包含本机 OCR 地址和一次性手工测试。 |

关联依赖的初步处理：`StandardDoc` 由统一 DocumentQA/DocumentInput 协议替代；`BenchmarkMonitor` 与通用 logger 属于 runner；火山 embedding client 放入 provider adapter，不能嵌入核心领域对象。

## 历史 FinanceBench 配置

`systems/deepread/reference/financebench_sample20.yaml` 保存 `fb8a301c...` 的关键行为参数，但不保存原配置中的明文 API key。初始对齐参数为 vector 开启、hybrid/semantic 关闭、neighbor window `1,-1`、最多 50 轮、retrieval top-k 5、单线程和复用已建索引。历史 `use_pymupdf` 字段不进入新 runtime，因为输入统一使用已有 Markdown。

首次提取必须以行为保持为目标。命名整理、依赖倒置、日志增强等变化应在 baseline 对齐后单独提交，避免无法判断结果差异来自迁移还是优化。
