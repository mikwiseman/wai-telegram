# Incremental archive memory

The original Telegram records remain authoritative. Search is a derived view. The Batch index groups complete exchanges by chat, UTC day and thread, with separately chunked document text. Every passage keeps its source message IDs, speaker and date. Inputs are bounded to 1,400 tokens and use OpenAI text-embedding-3-large at 3,072 dimensions. PostgreSQL stores half precision vectors and a separate HNSW index alongside the existing exact search.

## Lifecycle

Database triggers mark changed source days in the same transaction as a new message, edit, transcription or deletion. A minute worker prepares changed days, reconciles submitted Batch jobs and submits bounded new work. Unchanged passage hashes retain vectors. The mutable end of today's conversation waits five minutes before submission to avoid paying repeatedly for a growing passage.

Changed source revisions immediately leave semantic retrieval. Rebuilding a day removes obsolete passages; access is checked again at claim, upload, import and retrieval. Results arriving after correction or revocation cannot restore stale knowledge. Exact source search remains available while Batch work is pending (OpenAI's completion window is up to 24 hours).

Batch receipts precede network writes. Automatic SDK write retries are disabled; ambiguous submissions are reconciled through provider metadata. Partial results retry only missing embeddings. Invalid model/dimension/source IDs fail visibly. Billing quota failures hold the queue; no model substitution or repeated quota retries. Capacity checks preserve 8 GB of free root-disk space.

## Operation

Apply migration 033 before enabling. Set `MEMORY_BATCH_ENABLED=true`, `MEMORY_BATCH_ORGANIZATION` to the verified organization and `MEMORY_BATCH_QUEUE_TOKENS` below that project's provider limit. Keep `MEMORY_SEARCH_ENABLED=false` until coverage, permissions and real retrieval are verified.

```
python -m app.cli.memory_batches --seed --prepare-all
python -m app.cli.memory_batches --advance
python -m app.cli.memory_batches
```

The CLI and Celery share a Redis submission lock. Bulk preparation uses locked source-day rows and can resume after interruption. Account data-status responses report only that account's current passages, pending source days and semantic coverage. Operators additionally see aggregated Batch states and reported token usage. Text coverage and media extraction coverage are separate.

Changing `MEMORY_SEARCH_ENABLED` rolls search between the old and new indexes without deleting sources. Keep the old index through the measured cutover. Do not delete private exports or databases to fit the new index. A blocked provider receipt requires an operator to resolve the provider condition before releasing its claims.

## Forgetting

A message deletion invalidates its day immediately and removes derived passages on preparation. Disabling a chat or user denies retrieval immediately. A semantic index cannot delete upstream Telegram messages, original files, backups or a copied canonical note: an explicit erasure operation must enumerate those systems and verify each receipt. Archiving and superseding knowledge are different from erasure; neither should silently delete historical evidence.

## Verification

Tests cover Unicode text preservation, corrections and attribution, source isolation, malformed results, partial imports, unknown submissions, quota stops, disk stops, and a PostgreSQL lifecycle with edit-during-Batch, repeat import, access revocation and deletion. The PostgreSQL contract runs in CI against migrated pgvector and exercises the actual hybrid search query.

References: [OpenAI Batch](https://developers.openai.com/api/docs/guides/batch), [pgvector](https://github.com/pgvector/pgvector). Provider limits, prices and deployment coverage are recorded separately at rollout time.
