# Implemented application architecture

The live application is `enterprise/`. The original `src/` research code is retained separately.

1. Authenticated uploads are parsed into heading/clause-aware dynamic chunks with source locations and explicit metadata. One SQLite transaction stores the document, its chunks, audit event and knowledge-revision increment.
2. Queries are restricted by tenant, department read access and repository snapshot. Baseline RAG retrieves relevant evidence without specialized applicability filtering.
3. CAKE builds a generic query plan: governing, explanation, comparison, announcement or history. It resolves scope, approval, explicit lineages and effective dates. Supporting evidence is retained separately for explanation/comparison questions. Shared clause numbers in unrelated documents do not by themselves establish a shared rule.
4. AMO checks a key containing user permissions, knowledge revision, snapshot, full question/context and model settings before corpus loading. Valid hits bypass corpus loading, embedding and retrieval/resolution. Misses execute the same resolution path as CAKE and save the resolved context. Final answers are not cached.
5. The shared generator either quotes source excerpts explicitly or runs local Qwen. Qwen draft tokens stream to the browser. Citation identifiers are checked before the final response is persisted. Source-reference validity does not establish semantic entailment; factual review remains necessary.
6. Conversation ownership is checked server-side. Short referential follow-ups reuse the previous question for retrieval while current scope is applied to the follow-up.
7. Evaluation labels are isolated from the engine. Users can import datasets. Raw recall, selected evidence metrics, literal generated-claim checks and measured timing distributions are reported separately.

## Retrieval and memory

Offline retrieval uses generic text normalization and BM25. The neural path uses BGE-M3 and permission-filtered local Qdrant indexes, fused with lexical ranks. Up to four dense indexes are retained in-process, with clients closed on eviction. A changed visible corpus builds a new index lazily; this is not a production incremental vector-index service.

AMO context and query caches are bounded in-process LRU caches. Corpus revisions protect against stale reuse after uploads, metadata changes or deletions. Exact normalized question identity intentionally avoids unsafe paraphrase reuse. Buffered usage observations are flushed to SQLite on memory inspection and clean shutdown. Logical archive/consolidation tiers retain source history; utility is advisory, not a learned optimizer.

## Explainability

Sources retain original files, page/section references and version metadata. The interface exposes governing/supporting evidence, exclusions with reasons, announcement dates, conflicts and per-rule lineage diffs. An administrator can edit approval, source type, authority and clause-to-rule mappings, invalidating affected cached revisions. No graph database is claimed.
