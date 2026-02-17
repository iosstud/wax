# Wax Hot-Path Specialization Investigation

## Scope and Method
- Measure first, then specialize only measured hotspots.
- No blanket existential/protocol witness-table rewrites.
- Focused files:
  - `Sources/Wax/UnifiedSearch/UnifiedSearch.swift`
  - `Sources/Wax/UnifiedSearch/UnifiedSearchEngineCache.swift`
  - `Sources/Wax/RAG/FastRAGContextBuilder.swift`
  - `Sources/Wax/VectorSearchSession.swift`
  - `Sources/Wax/WaxSession.swift`

## Baseline Measurements (pre-specialization)

### Ingest throughput (`testIngestHybridBatchedPerformance`)
- smoke (200 docs): `0.103s` (`~1941.7 docs/s`)
- standard (1000 docs): `0.309s` (`~3236.2 docs/s`)
- stress (5000 docs): `2.864s` (`~1745.8 docs/s`)
- 10k: `7.756s` (`~1289.3 docs/s`)

### Search latency
- warm CPU smoke: `0.0015s` (`~666.7 ops/s`)
- warm CPU standard: `0.0033s` (`~303.0 ops/s`)
- warm CPU stress: `0.0072s` (`~138.9 ops/s`)
- 10k CPU hybrid benchmark: `0.103s` (`~9.7 ops/s`) per measured iteration

### Recall (`testMemoryOrchestratorRecallPerformance`)
- smoke: `0.103s`
- standard: `0.101s`
- stress: blocked in current harness (`signal 11`)

### FastRAG builder
- fast mode: `0.102s`
- dense cached: `0.102s`

## Hotspots (ranked by trace evidence)

### Top exclusive (10k trace)
1. `outlined copy of WALEntry`: `3379ms`
2. `initializeWithCopy for PendingMutation`: `2741ms`
3. Allocation/retain/release frames (`_xzm_free`, `_xzm_xzone_malloc_tiny`, `swift_release`, `swift_retain`) dominate remaining top-exclusive

### Top inclusive (10k trace)
1. `WaxVectorSearchSession.putWithEmbedding(...)`: `29148ms`
2. `Wax.withWriteLock`: `26459ms`
3. `Wax.put.../Wax.putLocked...`: `~26069ms`
4. `USearchIndex.add(key:vector:)`: `2703ms`
5. `WaxVectorSearchSession.stageForCommit()`: `2682ms`

### Warm-search trace (search subset)
- `Wax.search(_:engineOverrides:)`: `29ms` subset-inclusive
- `rrfFusionResults(...)` (`UnifiedSearch.swift`): `17ms` subset-inclusive
- sorting internals (`MutableCollection.sort`): `~11ms` subset-inclusive
- cache cold-load path present but smaller in warm runs (`UnifiedSearchEngineCache.textEngine`, `.vectorEngine`)

### FastRAG trace
- `FastRAGContextBuilder.build(...)`: `1114ms` inclusive
- dominant children: tokenization/encoding init + sorting

## Cost Separation Evidence
- Actor-hop cost:
  - `OptimizationComparisonBenchmark.testActorVsTaskHopTokenCounter`
  - direct actor `7.675ms` vs per-call Task hop `8.120ms` (`~5.5%` better direct)
- Allocation/copy cost:
  - dominant in 10k exclusive profile (copy/destroy + malloc/free/retain/release)
- Serialization cost:
  - `BufferSerializationBenchmark`:
    - save buffer `0.1382ms` vs file `2.1032ms` (`15.2x`)
    - load buffer `0.2580ms` vs file `0.5851ms` (`2.3x`)
    - total `~6.8x` buffer advantage
- Dynamic dispatch cost:
  - witness frames visible in vector paths but not dominant versus allocation/serialization
  - broad existential rewrites not justified by profile share

## Implemented Specializations (targeted)

### 1) `UnifiedSearchEngineCache.textEngine(for:)`
- File: `Sources/Wax/UnifiedSearch/UnifiedSearchEngineCache.swift`
- Change: removed duplicate `readStagedLexIndexBytes()` fetch on staged path (single read reused).
- Why: measured cold-path serialization/actor overhead; no blanket protocol optimization.
- Risk: low (fallback behavior retained when staged stamp exists but bytes are absent).

### 2) `rrfFusionResults` + rerank allocation tightening
- File: `Sources/Wax/UnifiedSearch/UnifiedSearch.swift`
- Change:
  - pre-reserved dictionary/array capacities in RRF fusion
  - replaced closure-heavy map chains with reserved arrays and loops
  - replaced `Candidate` array copies with `(index, score)` scoring to reduce temporary result copies
  - reduced per-candidate phrase-match closure allocations
- Why: measured sort/fusion costs in warm-search hot path.
- Risk: low-medium (ordering semantics preserved: score, source rank, frameId tie-breaks).

### 3) Concrete-engine fast path in `WaxVectorSearchSession`
- File: `Sources/Wax/VectorSearchSession.swift`
- Change:
  - introduced internal concrete engine dispatch (`USearchVectorEngine` / `MetalVectorEngine`) for hot calls
  - hot methods now avoid repeated existential dispatch on `add`, `addBatch`, `remove`, `search`, `stageForCommit`
  - staged embedding extraction moved to single-pass array fill to reduce temporary allocation overhead
- Why: profile shows this path dominates ingest-inclusive time.
- Risk: medium (dual concrete paths must stay behaviorally identical).

### 4) Concrete-engine staging fast path in `WaxSession`
- File: `Sources/Wax/WaxSession.swift`
- Change:
  - internal concrete engine enum for staging path
  - `stageVectorForCommit` uses concrete dispatch + single-pass frame/vector extraction
- Why: high-frequency staging boundary in ingest path.
- Risk: medium (internal path duplication, no public API changes).

### 5) Quoted-preview normalization guard in rerank
- File: `Sources/Wax/UnifiedSearch/UnifiedSearch.swift`
- Change: rerank comparisons use de-highlighted preview text for scoring checks, avoiding snippet marker artifacts.
- Why: preserves deterministic phrase matching intent when snippet markup inserts `[`/`]`.
- Risk: low (normalization-only, no API change).

## Candidate Optimizations (remaining prioritized)

1. `FastRAGContextBuilder.build(...)` token counter warm path
- Location: `Sources/Wax/RAG/FastRAGContextBuilder.swift`
- Expected impact: `~5-15%` builder-path CPU/cold reduction
- Risk: medium
- Why targeted: measured builder/tokenization hotspot; not global protocol rewrite

2. Session-local vector search override gating (only when safe)
- Location: `Sources/Wax/WaxSession.swift` search path
- Expected impact: small-moderate warm search latency win when pending-embedding visibility semantics are preserved
- Risk: medium-high
- Why targeted: measured cache/engine acquisition path, avoid broad rewrite

3. Additional RRF/rerank microbench harness for p95-sort pressure
- Location: `Tests/WaxIntegrationTests` benchmark additions
- Expected impact: tighter evidence for next specialization step
- Risk: low
- Why targeted: prevents unmeasured over-optimization

## Throughput/Latency Impact Status

### Current measured status
- Baselines above are confirmed measurement anchors.
- Targeted functional tests for new specialization paths passed before unrelated build breakage:
  - `unifiedSession_vectorSearchWorksBeforeAndAfterCommit`
  - `unifiedSession_putEmbeddingBatchPersistsSearchOrder`
  - `hybridSearchRankingDiagnosticsTopKIsScopedAndStable`
  - `fastRAGUsingSessionMatchesWaxSearchDeterministically`

### Post-change benchmark reprofile status
- Full post-change ingest/search/recall reprofiling is **temporarily blocked** by unrelated pre-existing orchestrator compile errors in workspace (`MemoryOrchestrator.swift` path), outside the scoped specialization files.
- No rollback of unrelated user edits was performed.

## Reproducible Benchmark/Profile Commands
Scratch path: `/tmp/wax-hotpath-build`

1. `WAX_RUN_XCTEST_BENCHMARKS=1 WAX_BENCHMARK_SCALE=smoke swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testIngestHybridBatchedPerformance`
2. `WAX_RUN_XCTEST_BENCHMARKS=1 swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testIngestHybridBatchedPerformance`
3. `WAX_RUN_XCTEST_BENCHMARKS=1 WAX_BENCHMARK_SCALE=stress swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testIngestHybridBatchedPerformance`
4. `WAX_RUN_XCTEST_BENCHMARKS=1 WAX_BENCHMARK_10K=1 swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testIngestHybridBatchedPerformance10KDocs`
5. `WAX_RUN_XCTEST_BENCHMARKS=1 WAX_BENCHMARK_SAMPLES=1 swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testUnifiedSearchHybridWarmLatencySamplesCPUOnly`
6. `WAX_RUN_XCTEST_BENCHMARKS=1 WAX_BENCHMARK_SCALE=smoke WAX_BENCHMARK_SAMPLES=1 swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testUnifiedSearchHybridWarmLatencySamplesCPUOnly`
7. `WAX_RUN_XCTEST_BENCHMARKS=1 WAX_BENCHMARK_SCALE=stress WAX_BENCHMARK_SAMPLES=1 swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testUnifiedSearchHybridWarmLatencySamplesCPUOnly`
8. `WAX_RUN_XCTEST_BENCHMARKS=1 WAX_BENCHMARK_10K=1 swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testUnifiedSearchHybridPerformance10KDocsCPU`
9. `WAX_RUN_XCTEST_BENCHMARKS=1 WAX_BENCHMARK_SCALE=smoke swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testMemoryOrchestratorRecallPerformance`
10. `WAX_RUN_XCTEST_BENCHMARKS=1 swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter RAGPerformanceBenchmarks/testMemoryOrchestratorRecallPerformance`
11. `WAX_BENCHMARK_OPTIMIZATION=1 swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter OptimizationComparisonBenchmark`
12. `swift test --scratch-path /tmp/wax-hotpath-build --disable-swift-testing --skip-build --filter BufferSerializationBenchmark/testBufferSerializationVsFileBased`

### Time Profiler capture pattern
- `xctrace record --template 'Time Profiler' --all-processes --time-limit <N>s --output /tmp/<trace>.trace`
- `xctrace export --input /tmp/<trace>.trace --output /tmp/<trace>_timeprofile.xml --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]'`

## Proposed Specialization Rollout Plan

### Quick wins (low risk, measurable)
1. Keep `UnifiedSearchEngineCache` single-read staged path (implemented)
2. Keep RRF/rerank allocation tightening (implemented)
3. Keep concrete dispatch for session vector staging/write hot calls (implemented)

### Deeper changes (higher risk)
1. FastRAG token counter lifecycle/warmup specialization
2. Safe session-local vector override to avoid cache path only when pending-state correctness is guaranteed

### Rollback criteria
1. Functional: deterministic ordering/content regressions in `UnifiedSearchTests` or `FastRAG*` tests
2. Performance: `<3%` improvement on target benchmark or `>5%` regression on unaffected benchmark
3. Stability: new crashes/flakes in benchmark/test runs
4. Complexity: duplication/branching without measurable ROI after two benchmark passes

## Explicit Rejection of Blanket Optimization
- No global existential/protocol witness-table rewrite was applied.
- Specialization is constrained to measured hot boundaries where ROI was evidenced (RRF/sort allocations, staged engine reads, vector session hot operations).
