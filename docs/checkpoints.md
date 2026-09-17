# Durable execution checkpoints

`kenkui.checkpoints.checkpointing(store)` opts a pipeline execution into a
caller-provided, job-scoped durable store:

```python
from kenkui.checkpoints import checkpointing

with checkpointing(job_store):
    pipeline.write("book.m4b", workers=8)
```

The `CheckpointStore` protocol has two operations:

- `save(key, source_path, metadata)` uploads a file and atomically makes its
  JSON-compatible metadata and immutable bytes available to future attempts.
- `restore(key, destination_path)` downloads and verifies a committed file,
  returning its metadata, or `None` when that key has never been committed.

Implementations must be thread-safe, isolate jobs, verify payload size and digest,
and fence stale attempts. Do not share a store namespace across unrelated jobs.
Kenkui Server supplies the R2/PostgreSQL implementation. Core has no cloud SDK
dependency and unchanged behavior outside this scope.

Completed chapter files include their segment timing metadata and applied
pauses. Identity includes the source/plan fingerprint, selected voice assets,
engine configuration and checkpoint schema. A restart restores compatible
chapters before starting workers. Incomplete chapters are rendered again.
Completed synthesis can proceed directly to assembly; partial M4B encoding is
not retained.

Casting uses a private SQLite store for the scope. Consistent snapshots are
saved after committed changes, including individual model responses, attribution
and resolved casting. Concurrent attribution calls carry the checkpoint context
into their threads. This preserves partial attribution work across interruptions.

Uploads are synchronous. A chapter is reported complete only after durable
publication. Storage failures stop execution; callers own retry and retention
policies. Use the same job namespace on recovery. Do not reuse checkpoints after
changing rendering semantics without updating the render/checkpoint schema.
