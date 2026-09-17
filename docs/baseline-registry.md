# Baseline registry

baseline registry 是外层循环选择下一轮代码起点的唯一协议：

```text
<registry-root>/
  .baseline-registry.lock
  entries/
    0000-<digest>.json
    0001-<digest>.json
    ...
```

generation 0 必须显式指定完整 AgenticRAGEvolve 仓库 commit。DeepRead/ruc-ov 的冻结 revision 继续作为源码 provenance；实际 candidate 需要 runner、framework 和 target 的一致组合，因此演化 baseline 使用完整仓库 commit。

```bash
uv run python runner/init_baseline_registry.py \
  --registry-root artifacts/baselines \
  --repo-root . \
  --revision <full-repository-commit>
```

后续只能用 `deepread-candidate-materialization-v1` artifact 推进：

```bash
uv run python runner/advance_baseline_registry.py \
  --registry-root artifacts/baselines \
  --materialization artifacts/materializations/<id>.json
```

推进会验证 materialization ID、accepted outcome 哈希、commit/tree、父 commit 以及当前 registry tip。entry 按 generation 连续组成单链；完整性校验失败、从旧 baseline 分叉或重复推进都会被拒绝。写操作由文件锁串行化。

每代 entry 对应一个 `refs/agentic-rag-evolve/baselines/<baseline-id>` durable ref。它仅保证 commit 不被 Git GC，不创建分支、不改变 HEAD。

读取当前 baseline：

```bash
uv run python runner/show_current_baseline.py \
  --registry-root artifacts/baselines
```

当前尚无真实 recurring `proceed` repair candidate，因此仓库没有伪造或初始化实验 registry；第一次真实外层循环启动时再选择明确的 genesis commit。
