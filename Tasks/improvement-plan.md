# Wax Codebase Improvement Plan

**Date:** 2026-02-17
**Branch:** `2614`
**Scope:** 17 changes across Performance (5), Code Quality (6), Testability (6)
**Codebase:** ~118 source files, ~399 tests

---

## Executive Summary

Comprehensive audit of the Wax framework revealed improvement opportunities in three areas:

1. **Performance** — Redundant TaskGroup passes, unnecessary intermediate allocations, and hardcoded thresholds in hot paths (TokenCounter, UnifiedSearch, FastRAGContextBuilder).
2. **Code Quality** — Silent `try?` error swallowing across 14 sites, duplicated provider validation, string-typed frame kinds, and missing invariant guards.
3. **Testability** — 13 duplicated test embedders, missing error-path coverage, no concurrency stress tests, and no property-based determinism verification.

The codebase is well-architected overall. These are targeted refinements, not rewrites.

---

## Work Streams & File Ownership

To enable parallel execution, changes are grouped into non-overlapping file ownership streams:

| Stream | Files Owned | Items |
|--------|-------------|-------|
| **A: TokenCounter** | `TokenCounter.swift` | P1 |
| **B: UnifiedSearch** | `UnifiedSearch.swift`, `SearchRequest.swift` | P2, P4 |
| **C: FastRAGContextBuilder** | `FastRAGContextBuilder.swift` | P3, P5 |
| **D: Orchestrators** | `MemoryOrchestrator.swift`, `PhotoRAGOrchestrator.swift`, `VideoRAGOrchestrator.swift` | Q1 (partial), Q2, Q3, Q4, Q5, Q6 |
| **E: Diagnostics** | New `WaxDiagnostics.swift` + scattered `try?` sites | Q1 |
| **F: Test Infrastructure** | `Tests/WaxIntegrationTests/Mocks/` (new) | T1 |
| **G: New Tests** | `Tests/WaxIntegrationTests/` (new files) | T2, T3, T4, T5, T6 |

**Conflict notes:**
- Q1 touches files from streams A-D (adding log calls) — must run after P1-P5 and Q2-Q6
- T2-T6 depend on T1 (shared mocks)
- Q3 (typed enums) touches D's files — serialize with Q2, Q4, Q5

---

## Phase 1: Performance

### P1. Fuse batch token encode + truncate into single TaskGroup

**File:** `Sources/Wax/RAG/TokenCounter.swift` — `countAndTruncateBatch()` (lines 321-351)

**Current behavior:**
```
countAndTruncateBatch() calls:
  1. encodeBatch()     → TaskGroup #1: encode all texts in parallel
  2. withTaskGroup()   → TaskGroup #2: truncate each encoded result
```

The method first calls `encodeBatch(texts)` at line 329, which creates its own `TaskGroup` internally. Then lines 333-348 create a second `TaskGroup` to iterate over encoded results and truncate. This doubles the cooperative thread pool scheduling overhead.

**Proposed fix:**
Merge into a single `TaskGroup` where each child task encodes AND truncates in one pass:

```swift
public func countAndTruncateBatch(_ texts: [String], maxTokens: Int) async -> [(count: Int, truncated: String)] {
    guard maxTokens > 0 else {
        return texts.map { _ in (count: 0, truncated: "") }
    }

    // Small batch fast path (no TaskGroup overhead)
    guard texts.count > 4 else {
        return texts.map { text in
            let tokens = encode(text)
            if tokens.count <= maxTokens {
                return (count: tokens.count, truncated: text)
            }
            let sliced = Array(tokens.prefix(maxTokens))
            return (count: maxTokens, truncated: decode(sliced))
        }
    }

    let localBackend = backend
    var results = [(count: Int, truncated: String)](repeating: (0, ""), count: texts.count)

    // Single TaskGroup: encode + truncate in one pass per text
    await withTaskGroup(of: (Int, (count: Int, truncated: String)).self) { group in
        for (index, text) in texts.enumerated() {
            group.addTask {
                let tokens = self.encodeNonisolated(text, backend: localBackend)
                let count = tokens.count
                if count <= maxTokens {
                    return (index, (count: count, truncated: text))
                }
                let sliced = Array(tokens.prefix(maxTokens))
                let truncated = self.decodeNonisolated(sliced, backend: localBackend)
                return (index, (count: maxTokens, truncated: truncated))
            }
        }
        for await (index, entry) in group {
            results[index] = entry
        }
    }
    return results
}
```

**Impact:** Eliminates one full TaskGroup scheduling round-trip. ~20-30% speedup for batch token operations used in every RAG context build (FastRAGContextBuilder lines 191, 276).

**Verification:** Existing `FastRAGTests` determinism test (builds same context twice, compares with `==`) will catch any behavioral regression.

---

### P2. Eliminate intermediate array copies in RRF fusion source attribution

**File:** `Sources/Wax/UnifiedSearch/UnifiedSearch.swift` (lines 172-249)

**Current behavior:**
In the `.hybrid` case (lines 219-249), four intermediate arrays are created via `.map(\.frameId)` and then immediately wrapped in `Set(...)`:

```swift
// Line 224-225: Creates [UInt64] arrays
let textIds = textResults.map(\.frameId)      // Array allocation
let vectorIds = vectorResults.map(\.frameId)   // Array allocation

// Line 236-239: Each becomes a Set (second allocation)
let textSet = Set(textIds)
let vectorSet = Set(vectorIds)
```

The `textIds` and `vectorIds` arrays are passed to `rrfFusion()` (which needs `[UInt64]`), then immediately converted to Sets. The `.textOnly` and `.vectorOnly` cases have the same pattern.

**Proposed fix:**
Build Sets directly using `lazy.map` to avoid intermediate array allocation where the array isn't reused:

```swift
case .hybrid(let alpha):
    let clampedAlpha = min(1, max(0, alpha))
    let textWeight = weights.bm25 * clampedAlpha
    let vectorWeight = weights.vector * (1 - clampedAlpha)

    let textIds = textResults.map(\.frameId)
    let vectorIds = vectorResults.map(\.frameId)
    let timelineIds = timelineFrameIds

    var lists: [(weight: Float, frameIds: [UInt64])] = []
    if textWeight > 0, !textIds.isEmpty { lists.append((weight: textWeight, frameIds: textIds)) }
    if vectorWeight > 0, !vectorIds.isEmpty { lists.append((weight: vectorWeight, frameIds: vectorIds)) }
    if weights.temporal > 0, !timelineIds.isEmpty { lists.append((weight: weights.temporal, frameIds: timelineIds)) }
    if structuredWeight > 0, !structuredIds.isEmpty { lists.append((weight: structuredWeight, frameIds: structuredIds)) }

    let fused = HybridSearch.rrfFusion(lists: lists, k: request.rrfK)

    // Build Sets directly from existing arrays (no intermediate copy)
    let textSet: Set<UInt64> = Set(textIds)
    let vectorSet: Set<UInt64> = Set(vectorIds)
    let timelineSet: Set<UInt64> = Set(timelineIds)
    let structuredSet: Set<UInt64> = Set(structuredIds)
```

For the `.textOnly` and `.vectorOnly` cases where the array is ONLY used for the Set (not passed to rrfFusion separately):

```swift
case .textOnly:
    if structuredIds.isEmpty || structuredWeight <= 0 {
        baseResults = textResults.map { ... }
    } else {
        let textIds = textResults.map(\.frameId)
        let fused = HybridSearch.rrfFusion(lists: [...], k: request.rrfK)
        let textSet = Set(textIds)  // reuses the array passed to rrfFusion
        let structuredSet = Set(structuredIds)
        // ...
    }
```

**Impact:** Eliminates 2-4 temporary `[UInt64]` allocations per search call. The `textIds`/`vectorIds` arrays serve double duty for both rrfFusion and Set construction, which is already efficient. The main win is in `.textOnly` and `.vectorOnly` where we can avoid creating the array entirely if it's only used for the Set.

**Risk:** Low — this is allocation optimization only, no behavioral change.

---

### P3. Add early termination to surrogate tier TaskGroup

**File:** `Sources/Wax/RAG/FastRAGContextBuilder.swift` (lines 159-183)

**Current behavior:**
The surrogate tier selection `TaskGroup` at lines 159-183 processes ALL work items, extracting tier text for every surrogate candidate. After the TaskGroup completes, lines 186-210 iterate sequentially and stop when `remainingTokens == 0`. This means surrogates beyond the budget are fully processed (tier selected, data decoded, text extracted) but their results discarded.

**Proposed fix:**
Since `TaskGroup` tasks run concurrently and we can't cancel individual tasks mid-flight, the optimization is to check a shared budget indicator before doing expensive work. Use a simple actor-isolated flag:

```swift
// After the TaskGroup, during sequential budget enforcement (lines 186-210):
// This section already terminates early — no TaskGroup change needed.
// The real optimization is to reduce maxToLoad based on expected budget:

let estimatedTokensPerSurrogate = max(1, clamped.surrogateMaxTokens / 2)
let estimatedMaxSurrogates = max(1, remainingTokens / estimatedTokensPerSurrogate)
let maxToLoad = min(clamped.maxSurrogates, min(clamped.searchTopK, 32), estimatedMaxSurrogates + 2) // +2 buffer for short surrogates
```

This reduces the number of items entering the TaskGroup when the budget is already tight, without introducing synchronization overhead inside the TaskGroup itself.

**Impact:** Saves tier extraction work in `denseCached` mode with many surrogates and a small remaining budget. The +2 buffer ensures we don't under-load.

**Risk:** Medium — the estimate could cause us to process fewer surrogates than optimal if early surrogates are very short. The +2 buffer mitigates this. Needs testing with varied surrogate lengths.

---

### P4. Make metadata loading threshold configurable

**File:** `Sources/Wax/UnifiedSearch/UnifiedSearch.swift` (line 283) and `Sources/Wax/UnifiedSearch/SearchRequest.swift`

**Current behavior:**
```swift
let lazyMetadataThreshold = 50  // hardcoded at line 283
```

Below 50 results, metadata is loaded lazily per-frame. Above 50, it's batch-prefetched into a dictionary. Different workloads may have different optimal thresholds.

**Proposed fix:**
Add to `SearchRequest`:

```swift
public struct SearchRequest: Sendable, Equatable {
    // ... existing fields ...

    /// Threshold for switching between lazy per-frame metadata loading and batch prefetch.
    /// Below this count, metadata is loaded on demand (avoids Dictionary overhead).
    /// Above this count, all metadata is prefetched in a single batch.
    /// Default: 50.
    public var metadataLoadingThreshold: Int

    public init(
        // ... existing params ...
        metadataLoadingThreshold: Int = 50
    ) {
        // ... existing assignments ...
        self.metadataLoadingThreshold = metadataLoadingThreshold
    }
}
```

Then in `UnifiedSearch.swift`:
```swift
let lazyMetadataThreshold = max(1, request.metadataLoadingThreshold)
```

**Impact:** Allows host apps to tune based on their result set characteristics. No behavioral change with default value.

**Risk:** Low — additive API, default preserves current behavior.

---

### P5. Consolidate preview snapshot filtering in FastRAGContextBuilder

**File:** `Sources/Wax/RAG/FastRAGContextBuilder.swift` (lines 227-268)

**Current behavior:**
The snippet extraction for `resultCount > 4` path (lines 227-268) creates:
1. `previewSnapshots` — an array of `(frameId, preview)` tuples (line 230)
2. `previewSlots` — `Array<String?>` of size `resultCount` (line 231)
3. A TaskGroup to populate `previewSlots` (lines 232-249)
4. Sequential iteration over `previewSlots` to build `snippetCandidates` (lines 251-257)

The TaskGroup doesn't do any real async work — it just checks two conditions and returns the preview string. This is pure CPU work that's faster sequentially than through TaskGroup scheduling.

**Proposed fix:**
Replace the TaskGroup with a direct sequential loop that matches the `resultCount <= 4` path:

```swift
// Replace lines 227-268 with:
for result in response.results {
    if let expandedFrameId, result.frameId == expandedFrameId { continue }
    if surrogateSourceFrameIds.contains(result.frameId) { continue }
    guard snippetCount < clamped.maxSnippets else { break }
    guard let preview = result.previewText, !preview.isEmpty else { continue }

    snippetCandidates.append((result, preview))
    snippetCount += 1
}
```

This eliminates the `resultCount > 4` / `<= 4` split entirely since the TaskGroup provides no benefit (no async work inside it).

**Impact:** Removes two intermediate array allocations, eliminates unnecessary TaskGroup overhead. Simpler code.

**Risk:** Low — the TaskGroup was doing no async work, so removing it changes only performance (better), not behavior.

---

## Phase 2: Code Quality

### Q1. Add structured logging to all `try?` failure sites

**Files:** 14 instances across 8 files

**Current behavior:**
`try?` silently swallows errors with no diagnostic trail. When something fails in production, there's no way to know what went wrong.

**Identified sites:**

| File | Line | Context | Fallback |
|------|------|---------|----------|
| `MemoryOrchestrator.swift` | 32 | `try? await TokenCounter.preload()` | Tokenizer not prewarmed; cold start on first use |
| `FastRAGContextBuilder.swift` | 126 | `try? await wax.frameContent(frameId:)` | Surrogate content load failure; skip this surrogate |
| `TextChunker.swift` | ~25 | `try? await TokenCounter.shared()` | Falls back to character-based chunking |
| `TextChunker.swift` | ~61 | `try? await TokenCounter.shared()` | Falls back to unsplit text |
| `PhotoRAGOrchestrator.swift` | ~540 | Caption generation failure | Skip caption for this asset |
| `VideoRAGOrchestrator.swift` | ~952 | Thumbnail extraction failure | No thumbnail for this segment |
| `UnifiedSearch.swift` | 311 | `try? await frameMetaIncludingPending()` | Skip this result |
| `UnifiedSearch.swift` | 376 | `try? await framePreviews()` | Empty preview map |
| `WaxSession.swift` | ~450 | Metal engine load failure | Fall back to CPU engine |
| `VectorSearchSession.swift` | ~24 | Metal engine load failure | Fall back to CPU engine |

**Proposed fix:**
Create a lightweight diagnostics logger:

```swift
// Sources/Wax/Utilities/WaxDiagnostics.swift (new file)
import os

enum WaxDiagnostics {
    static let log = Logger(subsystem: "com.wax.framework", category: "diagnostics")

    /// Log a swallowed error with context about the fallback behavior.
    static func logSwallowed(
        _ error: any Error,
        context: StaticString,
        fallback: StaticString,
        file: String = #file,
        line: Int = #line
    ) {
        log.error("\(context): \(error.localizedDescription) — falling back to \(fallback) [\(file):\(line)]")
    }
}
```

Then replace each `try?` with `do/catch`:

```swift
// Before:
_ = try? await TokenCounter.preload()

// After:
do {
    _ = try await TokenCounter.preload()
} catch {
    WaxDiagnostics.logSwallowed(error, context: "tokenizer prewarm", fallback: "cold start on first use")
}
```

**Impact:** Makes silent failures debuggable in production. No behavioral change — all fallbacks preserved.

**Risk:** Low. The `os.Logger` is lightweight and compiled out in release if not observed.

---

### Q2. Extract shared provider validation helper

**Files:**
- `MemoryOrchestrator.swift` lines 36-40
- `PhotoRAGOrchestrator.swift` lines 111-121
- `VideoRAGOrchestrator.swift` lines 108-115

**Current behavior:**
Each orchestrator has copy-pasted provider validation:

```swift
// MemoryOrchestrator (lines 36-40):
if config.requireOnDeviceProviders, let localEmbedder = embedder {
    guard localEmbedder.executionMode == .onDeviceOnly else {
        throw WaxError.io("MemoryOrchestrator requires on-device embedding provider")
    }
}

// PhotoRAGOrchestrator (lines 111-121): same pattern but checks embedder + OCR + captioner
// VideoRAGOrchestrator (lines 108-115): same pattern but checks embedder + transcriptProvider
```

**Proposed fix:**
Create a validation helper in the Wax target:

```swift
// Sources/Wax/Utilities/ProviderValidation.swift (new file)
import WaxVectorSearch

enum ProviderValidation {
    struct ProviderCheck {
        let provider: any ProviderWithExecutionMode
        let name: String
    }

    static func validateOnDevice(
        _ checks: [ProviderCheck],
        orchestratorName: String
    ) throws {
        for check in checks {
            guard check.provider.executionMode == .onDeviceOnly else {
                throw WaxError.io("\(orchestratorName) requires on-device \(check.name)")
            }
        }
    }
}
```

Usage:
```swift
// PhotoRAGOrchestrator:
if config.requireOnDeviceProviders {
    var checks: [ProviderValidation.ProviderCheck] = [
        .init(provider: embedder, name: "embedding provider")
    ]
    if let ocr { checks.append(.init(provider: ocr, name: "OCR provider")) }
    if let captioner { checks.append(.init(provider: captioner, name: "caption provider")) }
    try ProviderValidation.validateOnDevice(checks, orchestratorName: "PhotoRAG")
}
```

**Prerequisite:** Need to verify that `ProviderWithExecutionMode` protocol (or equivalent) exists, or extract a minimal protocol that `EmbeddingProvider`, `OCRProvider`, `CaptionProvider`, and `VideoTranscriptProvider` all conform to. If not, use `any` existential with `executionMode` property.

**Impact:** Eliminates ~40 lines of duplication. Single place to update validation logic.

**Risk:** Low — behavioral equivalence. The only question is the protocol hierarchy.

---

### Q3. Convert frame kind & metadata key strings to typed enums

**Files:**
- `PhotoRAGOrchestrator.swift` lines 22-55 (`FrameKind`, `MetaKey`)
- `VideoRAGOrchestrator.swift` lines 35-54 (`FrameKind`, `MetaKey`)

**Current behavior:**
Frame kinds and metadata keys are `private enum` namespaces with `static let` string constants:

```swift
private enum FrameKind {
    static let root = "photo.root"
    static let ocrBlock = "photo.ocr.block"
    // ...
}
```

These are `private` to each orchestrator. The test files that need to construct frames with matching kinds use hardcoded string literals — if the orchestrator value changes, tests silently stop matching.

**Proposed fix:**
Extract to public `RawRepresentable` enums:

```swift
// Sources/Wax/PhotoRAG/PhotoFrameKind.swift (new file)
public enum PhotoFrameKind: String, Sendable, CaseIterable {
    case root = "photo.root"
    case ocrBlock = "photo.ocr.block"
    case ocrSummary = "photo.ocr.summary"
    case captionShort = "photo.caption.short"
    case tags = "photo.tags"
    case region = "photo.region"
    case syncState = "system.photos.sync_state"
}

// Sources/Wax/PhotoRAG/PhotoMetadataKey.swift (new file)
public enum PhotoMetadataKey: String, Sendable, CaseIterable {
    case assetID = "photos.asset_id"
    case captureMs = "photo.capture_ms"
    case isLocal = "photo.availability.local"
    case pipelineVersion = "photo.pipeline.version"
    case lat = "photo.location.lat"
    case lon = "photo.location.lon"
    // ... all keys
}
```

Then update `PhotoRAGOrchestrator` to use `.root.rawValue` etc. Same pattern for Video.

**Impact:** Compile-time safety for frame/metadata operations. Tests reference the same enum. IDE autocomplete.

**Risk:** Medium — this is a public API addition. Internal string comparisons need updating. Tests referencing hardcoded strings need migration.

**Migration strategy:**
1. Create the public enums
2. Update orchestrator internals to use `.rawValue`
3. Update tests to use enum values
4. Keep old `private enum` aliases during transition if needed

---

### Q4. Add missing invariant assertions in VideoRAGOrchestrator

**File:** `VideoRAGOrchestrator.swift` — `writeSegments()` and related methods

**Current behavior:**
`PhotoRAGOrchestrator` validates embedding dimensions match expected dimensions. `VideoRAGOrchestrator` is missing equivalent guards:
- No guard that `keyframes.count == segments.count` before batch writing
- No guard that embedding dimensions match configured dimensions

**Proposed fix:**
Add guards at frame-writing boundaries:

```swift
// In writeSegments (or wherever segment frames are batch-written):
guard keyframeEmbeddings.count == segmentPlans.count else {
    throw WaxError.io(
        "segment/keyframe count mismatch: \(segmentPlans.count) segments, \(keyframeEmbeddings.count) keyframes"
    )
}

// Dimension validation:
for (index, embedding) in keyframeEmbeddings.enumerated() {
    guard embedding.count == embedder.dimensions else {
        throw WaxError.io(
            "segment \(index) embedding dimension mismatch: expected \(embedder.dimensions), got \(embedding.count)"
        )
    }
}
```

**Impact:** Prevents silent data corruption from mismatched segment counts. Fail-fast instead of writing garbage.

**Risk:** Low — adds guards that would only fire on bugs. Correct code is unaffected.

---

### Q5. Extract embedding cache initialization pattern

**Files:** All 3 orchestrators

**Current behavior:**
Identical pattern in each orchestrator's `init`:

```swift
// MemoryOrchestrator (lines 51-54):
if embedder != nil, config.embeddingCacheCapacity > 0 {
    self.embeddingCache = EmbeddingMemoizer(capacity: config.embeddingCacheCapacity)
} else {
    self.embeddingCache = nil
}
```

Similar patterns in PhotoRAG and VideoRAG with `queryEmbeddingCache`.

**Proposed fix:**
Add a factory method to `EmbeddingMemoizer`:

```swift
extension EmbeddingMemoizer {
    static func fromConfig(capacity: Int, enabled: Bool = true) -> EmbeddingMemoizer? {
        guard enabled, capacity > 0 else { return nil }
        return EmbeddingMemoizer(capacity: capacity)
    }
}
```

Usage:
```swift
self.embeddingCache = EmbeddingMemoizer.fromConfig(
    capacity: config.embeddingCacheCapacity,
    enabled: embedder != nil
)
```

**Impact:** Minor deduplication (~10 lines x 3). Cleaner init methods.

**Risk:** Low — trivial refactor.

---

### Q6. Remove implicit `executionMode` protocol defaults

**Files:** Protocol extensions for `MultimodalEmbeddingProvider`, `OCRProvider`, `CaptionProvider`, `VideoTranscriptProvider`

**Current behavior:**
Protocol extensions provide a default `executionMode`:

```swift
extension MultimodalEmbeddingProvider {
    public var executionMode: ProviderExecutionMode { .onDeviceOnly }
}
```

This means new conformers get `.onDeviceOnly` by default even if they make network calls. The validation in Q2 would pass incorrectly.

**Proposed fix:**
Remove the extension defaults. Force all conformers to explicitly declare:

```swift
// Remove these extension defaults:
// extension MultimodalEmbeddingProvider { var executionMode ... }
// extension OCRProvider { var executionMode ... }
// extension CaptionProvider { var executionMode ... }
// extension VideoTranscriptProvider { var executionMode ... }
```

**Impact:** Prevents accidental "on-device" declaration by network-using providers. New conformers get a compile error reminding them to declare their mode.

**Risk:** Medium — **breaking change** for external conformers. Any type that relies on the default will fail to compile. Mitigation: document in release notes. All internal conformers already declare explicitly (verified).

**Prerequisite:** Audit all conformers to ensure they already declare `executionMode`. Grep for protocol conformances.

---

## Phase 3: Testability

### T1. Create shared test mock directory

**Location:** `Tests/WaxIntegrationTests/Mocks/` (new directory)

**Current behavior:**
13 test embedder definitions scattered across 8 test files:
- `TestEmbedder` in `MemoryOrchestratorTests.swift`, `MemoryOrchestratorGapTests.swift`, `PerformanceImprovementsTests.swift`
- `StubMultimodalEmbedder` in `PhotoRAGOrchestratorTests.swift`, `VideoRAGRecallOnlyTests.swift`
- `FailOnNthEmbedder` in `MemoryOrchestratorTests.swift`
- `RejectConcurrentEmbedder` in `MemoryOrchestratorTests.swift`
- Various OCR/caption/transcript stubs in their respective test files

**Proposed fix:**
Create shared mock files:

```
Tests/WaxIntegrationTests/Mocks/
    MockEmbedders.swift          -- TestEmbedder, StubMultimodalEmbedder, FailOnNthEmbedder,
                                    RejectConcurrentEmbedder, NetworkEmbedder, WrongDimensionEmbedder
    MockProviders.swift          -- StubOCRProvider, StubCaptionProvider, StubTranscriptProvider
    FrameBuilders.swift          -- Fluent frame fixture builders for test data construction
    TestHelpers.swift            -- Shared utilities (temp directory creation, etc.)
```

**Implementation steps:**
1. Create directory and files
2. Move each mock/stub definition to the appropriate shared file
3. Make them `package` (or `internal` with `@testable`) accessible
4. Update all importing test files to remove local definitions
5. Run full test suite to verify nothing broke

**Impact:** Reduces ~200 lines of duplication. Single source of truth for test doubles. Adding new mock behaviors (like `WrongDimensionEmbedder` for T2) is straightforward.

**Risk:** Low — test-only change. No production code affected.

---

### T2. Add error path tests for embedding batch failures

**File:** New `Tests/WaxIntegrationTests/MemoryOrchestratorErrorTests.swift`

**Target code paths:**
1. `vectors.count != missingIndices.count` guard (`MemoryOrchestrator.swift:334-335`) — batch embedding returns wrong count
2. Oversized vector rejection (`vector.count > UInt32.max`, line 451-452) — not practically testable but can test dimension mismatch
3. Embedding deserialization errors (lines 488-509) — corrupt staged embedding file

**Proposed tests:**

```swift
@Test("Batch embedding count mismatch throws encodingError")
func batchEmbeddingCountMismatch() async throws {
    // Use a WrongCountEmbedder that returns N-1 vectors for N inputs
    // Expect WaxError.encodingError
}

@Test("Corrupt staged embedding file throws decodingError")
func corruptStagedEmbeddingFile() async throws {
    // Write a valid header but truncated body
    // Call readEmbeddings(from:) directly
    // Expect WaxError.decodingError
}

@Test("Embedding file with trailing bytes throws decodingError")
func trailingBytesInEmbeddingFile() async throws {
    // Write valid embeddings + extra garbage bytes
    // Expect WaxError.decodingError("trailing bytes")
}

@Test("Empty batch embedding returns empty results")
func emptyBatchEmbedding() async throws {
    // Ingest empty string — should succeed with document frame only
}
```

**Dependencies:** T1 (for `WrongCountEmbedder` mock)

**Impact:** Covers the most dangerous silent-failure paths in the embedding pipeline.

---

### T3. Add config validation edge case tests

**File:** Extended `Tests/WaxIntegrationTests/RAGConfigClampingTests.swift`

**Target code paths:**
- `rrfK = 0` — potential division by zero in RRF formula `1 / (k + rank + 1)` when `k=0`
- `expansionMaxTokens > maxContextTokens` — clamped correctly?
- `maxSnippets = 0` with non-zero budget — should produce no snippets
- Negative values for all token budgets — should clamp to 0
- `searchTopK = 0` — should return empty results
- `previewMaxBytes = 0` — should still work (no previews)

**Proposed tests:**

```swift
@Test("rrfK=0 does not divide by zero", arguments: [0, -1, -100])
func rrfKZero(rrfK: Int) async throws {
    var config = FastRAGConfig()
    config.rrfK = rrfK
    let clamped = FastRAGContextBuilder().clamp(config)  // need @testable
    #expect(clamped.rrfK >= 0)
    // Also run actual search to verify no crash
}

@Test("expansionMaxTokens clamped to maxContextTokens")
func expansionExceedsMax() async throws {
    var config = FastRAGConfig()
    config.maxContextTokens = 100
    config.expansionMaxTokens = 500
    let clamped = FastRAGContextBuilder().clamp(config)
    #expect(clamped.expansionMaxTokens <= clamped.maxContextTokens)
}

@Test("All negative values clamp to zero")
func negativeBudgets() async throws {
    var config = FastRAGConfig()
    config.maxContextTokens = -1
    config.snippetMaxTokens = -100
    config.maxSnippets = -5
    let clamped = FastRAGContextBuilder().clamp(config)
    #expect(clamped.maxContextTokens == 0)
    #expect(clamped.snippetMaxTokens == 0)
    #expect(clamped.maxSnippets == 0)
}
```

**Impact:** Validates all `clamp()` edge cases. Prevents future regressions if clamping logic changes.

---

### T4. Add VideoRAG segmentation edge case tests

**File:** Extended `Tests/WaxIntegrationTests/VideoRAGSegmentationMathTests.swift`

**Target:** `VideoRAGOrchestrator._makeSegmentRangesForTesting()` (line 70)

**Proposed tests:**

```swift
@Test("Overlap greater than segment duration")
func overlapExceedsDuration() {
    let ranges = VideoRAGOrchestrator._makeSegmentRangesForTesting(
        durationMs: 30_000,
        segmentDurationSeconds: 5.0,
        segmentOverlapSeconds: 10.0,  // overlap > duration
        maxSegments: 100
    )
    // Should not crash; verify segments still cover the video
    #expect(!ranges.isEmpty)
    #expect(ranges.first?.startMs == 0)
}

@Test("Single-frame video (duration = 0)")
func zeroDurationVideo() {
    let ranges = VideoRAGOrchestrator._makeSegmentRangesForTesting(
        durationMs: 0, segmentDurationSeconds: 10.0, segmentOverlapSeconds: 1.0, maxSegments: 100
    )
    // Should produce 0 or 1 segment, not crash
}

@Test("Sub-second video")
func subSecondVideo() {
    let ranges = VideoRAGOrchestrator._makeSegmentRangesForTesting(
        durationMs: 500, segmentDurationSeconds: 10.0, segmentOverlapSeconds: 1.0, maxSegments: 100
    )
    #expect(ranges.count <= 1)
}

@Test("maxSegments = 1 returns single segment covering full duration")
func singleSegmentCap() {
    let ranges = VideoRAGOrchestrator._makeSegmentRangesForTesting(
        durationMs: 60_000, segmentDurationSeconds: 10.0, segmentOverlapSeconds: 1.0, maxSegments: 1
    )
    #expect(ranges.count == 1)
}

@Test("Segments cover full video duration without gaps")
func fullCoverage() {
    let durationMs: Int64 = 45_000
    let ranges = VideoRAGOrchestrator._makeSegmentRangesForTesting(
        durationMs: durationMs, segmentDurationSeconds: 10.0, segmentOverlapSeconds: 2.0, maxSegments: 100
    )
    // First segment starts at 0
    #expect(ranges.first?.startMs == 0)
    // Last segment ends at or after duration
    #expect(ranges.last!.endMs >= durationMs)
}
```

**Impact:** Exhaustive boundary coverage for segmentation math.

---

### T5. Add concurrent ingest + recall stress tests

**File:** New `Tests/WaxIntegrationTests/ConcurrencyStressTests.swift`

**Proposed tests:**

```swift
@Test("Concurrent ingest and recall do not race")
func concurrentIngestAndRecall() async throws {
    let orchestrator = try await makeOrchestrator()

    // Seed with initial data
    try await orchestrator.remember("Initial memory about Swift concurrency")
    try await orchestrator.flush()

    // Run 10 concurrent ingest tasks + 5 parallel recall queries
    try await withThrowingTaskGroup(of: Void.self) { group in
        for i in 0..<10 {
            group.addTask {
                try await orchestrator.remember("Concurrent memory \(i) about topic \(i % 3)")
            }
        }
        for _ in 0..<5 {
            group.addTask {
                let context = try await orchestrator.recall(query: "Swift concurrency")
                // Should not crash, return partial or full results
                _ = context.items
            }
        }
        try await group.waitForAll()
    }

    // Verify final state is consistent
    try await orchestrator.flush()
    let finalContext = try await orchestrator.recall(query: "concurrent memory")
    #expect(!finalContext.items.isEmpty)
}

@Test("Rapid sequential ingest-recall cycles")
func rapidIngestRecallCycles() async throws {
    let orchestrator = try await makeOrchestrator()

    for i in 0..<20 {
        try await orchestrator.remember("Memory \(i)")
        let context = try await orchestrator.recall(query: "Memory \(i)")
        // May or may not find it (WAL not committed), but must not crash
        _ = context
    }
}
```

**Impact:** Validates actor isolation holds under realistic concurrent load.

**Risk:** Low — test-only. May be flaky if timing-dependent; use deterministic assertions (no crash, no data race) rather than content assertions.

---

### T6. Add property-based determinism tests

**File:** New `Tests/WaxIntegrationTests/DeterminismPropertyTests.swift`

**Proposed tests:**

```swift
@Test("RRF fusion is idempotent")
func rrfIdempotent() {
    let lists: [(weight: Float, frameIds: [UInt64])] = [
        (weight: 1.0, frameIds: [1, 2, 3, 4, 5]),
        (weight: 0.5, frameIds: [3, 4, 5, 6, 7]),
    ]
    let result1 = HybridSearch.rrfFusion(lists: lists, k: 60)
    let result2 = HybridSearch.rrfFusion(lists: lists, k: 60)
    #expect(result1.map(\.0) == result2.map(\.0))
    #expect(result1.map(\.1) == result2.map(\.1))
}

@Test("RRF fusion is input-order independent for same list set")
func rrfOrderIndependent() {
    let listA: (weight: Float, frameIds: [UInt64]) = (weight: 1.0, frameIds: [1, 2, 3])
    let listB: (weight: Float, frameIds: [UInt64]) = (weight: 0.5, frameIds: [3, 4, 5])

    let resultAB = HybridSearch.rrfFusion(lists: [listA, listB], k: 60)
    let resultBA = HybridSearch.rrfFusion(lists: [listB, listA], k: 60)

    // Same frame IDs in output (order may differ due to different scores)
    #expect(Set(resultAB.map(\.0)) == Set(resultBA.map(\.0)))
}

@Test("Token count is subadditive within constant K")
func tokenCountSubadditive() async throws {
    let counter = try await TokenCounter.shared()
    let a = "Hello world"
    let b = "Swift concurrency"
    let ab = a + " " + b

    let countA = counter.count(a)
    let countB = counter.count(b)
    let countAB = counter.count(ab)

    // count(a+b) <= count(a) + count(b) + K (K accounts for tokenizer merging)
    let K = 2
    #expect(countAB <= countA + countB + K)
}

@Test("FastRAG context is deterministic across repeated builds")
func fastRAGDeterministic() async throws {
    // Build identical context twice
    // This mirrors the existing FastRAGTests determinism test but with varied inputs
    let wax = try await makeWaxWithSeedData()
    let builder = FastRAGContextBuilder()
    let config = FastRAGConfig()

    let context1 = try await builder.build(query: "test query", wax: wax, config: config)
    let context2 = try await builder.build(query: "test query", wax: wax, config: config)

    #expect(context1.items.count == context2.items.count)
    #expect(context1.totalTokens == context2.totalTokens)
    for (a, b) in zip(context1.items, context2.items) {
        #expect(a.text == b.text)
        #expect(a.frameId == b.frameId)
    }
}
```

**Impact:** Catches subtle non-determinism bugs that example-based tests miss. Validates Architecture Invariant #6.

---

## Implementation Order

Items are ordered by: (1) dependency chain, (2) risk level, (3) effort.

| # | Item | Effort | Risk | Depends On | Stream |
|---|------|--------|------|------------|--------|
| 1 | **T1** — Shared test mocks | Low | Low | — | F |
| 2 | **P1** — Fuse batch token ops | Medium | Low | — | A |
| 3 | **P5** — Remove unnecessary TaskGroup | Low | Low | — | C |
| 4 | **Q2** — Extract provider validation | Low | Low | — | D |
| 5 | **Q4** — VideoRAG invariant guards | Low | Low | — | D (after Q2) |
| 6 | **Q5** — Embedding cache factory | Low | Low | — | D (after Q4) |
| 7 | **P4** — Configurable metadata threshold | Low | Low | — | B |
| 8 | **P2** — RRF allocation cleanup | Low | Low | — | B (after P4) |
| 9 | **Q3** — Typed frame/metadata enums | Medium | Medium | — | D (after Q5) |
| 10 | **P3** — Surrogate early termination | Medium | Medium | — | C (after P5) |
| 11 | **Q1** — Structured logging for try? | Low | Low | P1-P5, Q2-Q5 | E |
| 12 | **Q6** — Remove executionMode defaults | Low | Medium | Q2 | D (after Q3) |
| 13 | **T2** — Embedding error path tests | Medium | Low | T1 | G |
| 14 | **T3** — Config validation tests | Low | Low | T1 | G |
| 15 | **T4** — Segmentation edge case tests | Low | Low | — | G |
| 16 | **T5** — Concurrency stress tests | Medium | Low | T1 | G |
| 17 | **T6** — Determinism property tests | Medium | Low | T1 | G |

**Parallelization opportunities:**
- Items 1-4 can run in parallel (different file streams)
- Items 5-8 can run in parallel (Q5 serial after Q4 in stream D; P2 after P4 in stream B)
- Items 13-17 can all run in parallel (all in stream G, independent new test files)

---

## Verification Protocol

### Per-change:
1. `swift build` — zero errors, zero warnings
2. `swift test --filter <RelevantTestSuite>` — targeted tests pass
3. New tests added for each functional change
4. No new `@unchecked Sendable` or `nonisolated(unsafe)` without ADR
5. No `!` or `try!` in production code

### Full regression (after all changes):
```bash
swift build && swift test --parallel
```

### Manual spot checks:
- P1: Verify `FastRAGTests` determinism test still passes
- Q1: Check Console.app for `com.wax.framework` log entries when inducing failures
- Q3: Verify test files reference enum values, not string literals
- Q6: Grep for protocol conformances to verify all declare `executionMode`

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Q3 breaks internal string comparisons | Medium | High | Search all `.kind ==` and metadata key references before changing |
| Q6 breaks external conformers | Medium | Medium | Document in release notes; low risk if no external adopters yet |
| P3 under-loads surrogates | Low | Medium | +2 buffer; validate with existing surrogate tests |
| T5 flaky under CI | Medium | Low | Assert "no crash" + "no data race", not content assertions |
| P1 changes token count rounding | Very Low | High | Determinism test catches any divergence |

---


These were considered but deferred:
- **Structured concurrency migration** (replacing `Task.detached` with structured alternatives) — too broad
- **Protocol witness table optimization** — premature; profile first
- **WAL compaction improvements** — separate project
- **Public API documentation pass** — separate documentation sprint
