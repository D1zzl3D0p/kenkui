# NLP Pipeline Redesign: Adapter-Protocol Architecture

**Date**: 2026-05-13  
**Status**: Draft  
**Context**: The ebook→quotes pipeline (character extraction → coreference resolution → quote attribution) has been fragile — many changes break the pipeline. The goal is to reduce the tool surface to three well-defined backends (BookNLP, Ollama, LiteLLM), make each step independently configurable, and ground the architecture in 12-factor principles so the code is stable, maintainable, and easy to reason about.

---

## Problem Statement

The current `nlp/__init__.py` (1182 lines) mixes orchestration, caching, segment assembly, spaCy loading, and quote extraction. The provider factory (`providers/__init__.py`) hard-codes `OllamaProvider` regardless of config. BookNLP is partially integrated but not wired to the provider protocol. LiteLLM doesn't exist in the codebase. Modal execution is scaffolded but never consulted. `AppConfig` mixes TTS, NLP, audio, and credential concerns in one struct.

The result: changing any part of the pipeline risks breaking an unrelated part. There is no stable interface between "Step 1: character extraction" and "Step 2: quote attribution."

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | Adapter-Protocol split | Builds on existing NLPProvider shape; factory dispatch is the single wiring point |
| Config split | NLPConfig + TTSConfig | Each config class maps to a clear env prefix; services can take only what they need |
| Execution context | Local/Modal as separate wrapper, not per-tool subclass | Symmetry: all tools configured the same way regardless of where they run |
| Cache key | `(step, tool, model)` | Shared across LOCAL/MODAL (same output); different runs coexist by tool+model |
| Failure mode | Retry with backoff, save partial cache, then fail | Fail fast is too abrupt; automatic fallback to another tool silently hides config issues |
| Prompt structure | Preserved | Output quality is good; instability is architectural, not prompt-driven |
| Streaming | Callbacks + Job polling | CLI doesn't need streaming; polling is sufficient for kentui's progress display |

---

## Architecture

### Layers

```
┌─────────────────────────────────────────────────────────┐
│  services/nlp_service.py  (public kenkui API surface)    │
│  fast_scan() / full_analysis() / attribute_only()        │
└───────────────────┬─────────────────────────────────────┘
                    │ creates
┌───────────────────▼─────────────────────────────────────┐
│  nlp/pipeline.py  (NLPPipeline)                          │
│  extract() / attribute() / run()                         │
│  extract_job() / attribute_job()  → Job                  │
│  validate_tools()  → list[ValidationResult]              │
│                                                          │
│  wraps: _retry.py, _cache.py, signal handler             │
└──────────┬────────────────────────┬─────────────────────┘
           │                        │
┌──────────▼──────────┐  ┌─────────▼──────────────────────┐
│ ExtractionProvider  │  │ AttributionProvider             │
│                     │  │                                 │
│ LocalExtraction     │  │ LocalAttribution                │
│   Provider(adapter) │  │   Provider(adapter)             │
│ ModalExtraction     │  │ ModalAttribution                │
│   Provider(adapter) │  │   Provider(adapter)             │
└──────────┬──────────┘  └─────────┬──────────────────────┘
           │ delegates              │ delegates
┌──────────▼──────────┐  ┌─────────▼──────────────────────┐
│ ExtractionAdapter   │  │ AttributionAdapter              │
│                     │  │                                 │
│ BookNLP             │  │ BookNLP                         │
│ Ollama              │  │ Ollama                          │
│ LiteLLM             │  │ LiteLLM                         │
└─────────────────────┘  └────────────────────────────────┘
```

### Config Layer

```python
# config/nlp.py
class NLPConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KENKUI_NLP_")

    extraction_tool:   ExtractionTool  = ExtractionTool.OLLAMA   # BOOKNLP | OLLAMA | LITELLM
    extraction_mode:   ExecutionMode   = ExecutionMode.LOCAL      # LOCAL | MODAL
    extraction_model:  str             = "llama3.2"

    attribution_tool:  AttributionTool = AttributionTool.OLLAMA
    attribution_mode:  ExecutionMode   = ExecutionMode.LOCAL
    attribution_model: str             = "llama3.2"

    ollama_url:            str        = "http://localhost:11434"
    litellm_api_base:      str | None = None
    litellm_api_key:       str | None = Field(None, alias="KENKUI_NLP_LITELLM_API_KEY")

    retry_max_attempts:    int        = 3
    retry_backoff_base:    float      = 2.0    # seconds; doubles per attempt

# config/tts.py  — extracted from AppConfig
class TTSConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KENKUI_TTS_")
    # (workers, m4b_bitrate, temp, lsd_decode_steps, post_processing, apostrophe_mode, etc.)
```

**12-factor note**: Both classes source values from the environment first (`KENKUI_NLP_*`, `KENKUI_TTS_*`), then TOML file, then field defaults. Credentials (API keys) come only from env vars, never from files.

### Protocol Layer

Two protocol pairs with identical method signatures but different responsibilities:

- **Adapter** = tool implementation (BookNLP, Ollama, LiteLLM). Knows how to call the tool.
- **Provider** = execution context (Local, Modal). Knows where to run the adapter.

`LocalExtractionProvider` simply delegates to its adapter. `ModalExtractionProvider` serializes the call and dispatches it to Modal. Both satisfy the same `ExtractionProvider` protocol so the pipeline never knows which execution context is in use.

```python
# nlp/providers/_base.py

class ExtractionAdapter(Protocol):
    """Tool implementation: extracts characters + coreference → CharacterRoster.
    Knows how to call a specific tool (BookNLP / Ollama / LiteLLM).
    Does not know where it runs (local vs Modal)."""
    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster: ...

class AttributionAdapter(Protocol):
    """Tool implementation: assigns speaker per quote in a chapter.
    Knows how to call a specific tool. Does not know where it runs."""
    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult: ...

# Execution-context wrappers: same method signatures as the adapters they wrap.
# The pipeline only ever touches ExtractionProvider / AttributionProvider.
class ExtractionProvider(Protocol):
    def build_roster(self, chapters, series_roster, progress_callback, book_path) -> CharacterRoster: ...

class AttributionProvider(Protocol):
    def attribute_chapter(self, chapter, roster, progress_callback) -> AttributionResult: ...
```

### Provider Implementations

| File | Adapter(s) | Notes |
|------|-----------|-------|
| `providers/booknlp.py` | `BookNLPExtractionAdapter` | Migrated from `booknlp_roster.py`; uses `booknlp.BookNLP` with `entity,quote,coref` pipeline |
| | `BookNLPAttributionAdapter` | New; reads the `.book` JSON quote events already generated by the same pipeline run (currently discarded by `booknlp_roster.py`) |
| `providers/ollama.py` | `OllamaExtractionAdapter` | Extracted from `OllamaProvider.build_roster()` |
| | `OllamaAttributionAdapter` | Extracted from `OllamaProvider.attribute_chapter()` |
| `providers/litellm.py` | `LiteLLMExtractionAdapter` | New; LiteLLM chat completions + instructor schema enforcement |
| | `LiteLLMAttributionAdapter` | New; same prompt structure as Ollama, different client |
| `providers/local.py` | `LocalExtractionProvider(adapter)` | Runs adapter in current process |
| | `LocalAttributionProvider(adapter)` | Runs adapter in current process |
| `providers/modal.py` | `ModalExtractionProvider(adapter, config)` | Dispatches to Modal function; polls for result |
| | `ModalAttributionProvider(adapter, config)` | Same |

### Factory

```python
# providers/_factory.py

def get_extraction_provider(config: NLPConfig) -> ExtractionProvider:
    adapter = _make_extraction_adapter(config)
    match config.extraction_mode:
        case ExecutionMode.LOCAL:  return LocalExtractionProvider(adapter)
        case ExecutionMode.MODAL:  return ModalExtractionProvider(adapter, config)
        case _: raise ValueError(f"Unknown extraction mode: {config.extraction_mode}")

def get_attribution_provider(config: NLPConfig) -> AttributionProvider:
    adapter = _make_attribution_adapter(config)
    match config.attribution_mode:
        case ExecutionMode.LOCAL:  return LocalAttributionProvider(adapter)
        case ExecutionMode.MODAL:  return ModalAttributionProvider(adapter, config)
        case _: raise ValueError(f"Unknown attribution mode: {config.attribution_mode}")

def _make_extraction_adapter(config: NLPConfig) -> ExtractionAdapter:
    match config.extraction_tool:
        case ExtractionTool.BOOKNLP: return BookNLPExtractionAdapter(config)
        case ExtractionTool.OLLAMA:  return OllamaExtractionAdapter(config)
        case ExtractionTool.LITELLM: return LiteLLMExtractionAdapter(config)
        case _: raise ValueError(f"Unknown extraction tool: {config.extraction_tool}")

def _make_attribution_adapter(config: NLPConfig) -> AttributionAdapter:
    # same pattern
```

### Pipeline Orchestrator

```python
# nlp/pipeline.py

class NLPPipeline:
    def __init__(self, config: NLPConfig) -> None:
        self._config = config
        self._extraction = get_extraction_provider(config)
        self._attribution = get_attribution_provider(config)

    def validate_tools(self) -> list[ValidationResult]:
        """Pre-flight check. Never raises; surfaces errors in results."""
        ...

    def extract(
        self,
        book_path: Path,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
        cache_key: CacheMeta | None = None,
        use_cache: bool = True,
    ) -> CharacterRoster:
        """Run Step 1. Checks cache first; retries on transient errors; saves cache on success."""
        ...

    def attribute(
        self,
        book_path: Path,
        chapters: list[Chapter],
        roster: CharacterRoster,
        progress_callback: Callable[[int, str], None] | None = None,
        cache_key: CacheMeta | None = None,
        use_cache: bool = True,
    ) -> NLPResult:
        """Run Step 2 per chapter. Checks cache; retries; saves on success."""
        ...

    def run(
        self,
        book_path: Path,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
        extraction_cache_key: CacheMeta | None = None,
        attribution_cache_key: CacheMeta | None = None,
        use_cache: bool = True,
    ) -> NLPResult:
        """Full pipeline: extract then attribute.

        BookNLP optimization (implementation detail): when both extraction and
        attribution tools are BookNLP, the pipeline MAY share the raw .book
        JSON output between steps to avoid running BookNLP twice on the same
        text. This is an internal optimization and does not affect the public API.
        """
        roster = self.extract(book_path, chapters, series_roster, progress_callback, extraction_cache_key, use_cache)
        return self.attribute(book_path, chapters, roster, progress_callback, attribution_cache_key, use_cache)

    def extract_job(self, book_path, chapters, ...) -> Job:
        """Non-blocking. Runs extract() in a thread. Poll Job.poll() for status."""
        ...

    def attribute_job(self, book_path, chapters, roster, ...) -> Job:
        ...
```

### Cache Layer

```python
# nlp/_cache.py

@dataclass
class CacheMeta:
    path: Path
    step: Literal["extraction", "attribution"]
    tool: str
    model: str
    created_at: datetime
    description: str
    book_hash: str

# Cache key: {hash}-{step}-{tool}-{slugified-model}.json
# LOCAL and MODAL share the same cache (same output for same tool+model).
# Sorted by created_at descending (most recent first) from list_caches().

def list_caches(book_path: Path) -> list[CacheMeta]: ...
def get_cache(book_path: Path, step: str, tool: str, model: str) -> T | None: ...
def put_cache(result: T, book_path: Path, step: str, tool: str, model: str, description: str) -> Path: ...
def delete_cache(meta: CacheMeta) -> None: ...

# All writes are atomic: write to .tmp, then os.replace().
```

### Retry + Graceful Shutdown

```python
# nlp/_retry.py

RETRYABLE_ERRORS = (OSError, TimeoutError, json.JSONDecodeError, httpx.HTTPError)
NOT_RETRYABLE = (KeyboardInterrupt, SystemExit, ValueError, TypeError)

def with_retry(fn: Callable, max_attempts: int = 3, backoff_base: float = 2.0) -> Callable:
    """Exponential backoff retry. Re-raises on final attempt or non-retryable errors."""
    ...
```

SIGTERM handling in `NLPPipeline.extract()` and `.attribute()`:
- Intercept SIGTERM during a step
- Flush any partial cache to disk (via atomic write)
- Re-raise as `KeyboardInterrupt` so the caller can handle cleanup
- Restore original SIGTERM handler after the step completes

### Job Polling

```python
@dataclass
class Job:
    job_id: str
    status: JobStatus        # PENDING | RUNNING | DONE | FAILED
    progress: int            # 0-100
    message: str
    result: CharacterRoster | NLPResult | None   # populated on DONE
    error: Exception | None                       # populated on FAILED

    def poll(self) -> Job: ...        # non-blocking snapshot of current state
    def wait(self, timeout: float | None = None) -> Job: ...
    def cancel(self) -> None: ...

class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
```

---

## File Change Summary

| File | Action |
|------|--------|
| `config/nlp.py` | **Create** — `NLPConfig` (Pydantic BaseSettings) |
| `config/tts.py` | **Create** — `TTSConfig` extracted from `AppConfig` |
| `nlp/providers/_base.py` | **Replace** — `NLPProvider` → `ExtractionAdapter + AttributionAdapter + ExtractionProvider + AttributionProvider` |
| `nlp/providers/_factory.py` | **Create** — `get_extraction_provider()`, `get_attribution_provider()` |
| `nlp/providers/local.py` | **Create** — `LocalExtractionProvider`, `LocalAttributionProvider` |
| `nlp/providers/modal.py` | **Create** — `ModalExtractionProvider`, `ModalAttributionProvider` |
| `nlp/providers/booknlp.py` | **Create** — migrated from `booknlp_roster.py` + new `BookNLPAttributionAdapter` |
| `nlp/providers/ollama.py` | **Refactor** — split `OllamaProvider` into two adapters |
| `nlp/providers/litellm.py` | **Create** — `LiteLLMExtractionAdapter`, `LiteLLMAttributionAdapter` |
| `nlp/providers/__init__.py` | **Replace** — delegate to `_factory.py` |
| `nlp/pipeline.py` | **Create** — `NLPPipeline`, `Job`, `JobStatus`, `ValidationResult` |
| `nlp/_cache.py` | **Create** — `CacheMeta`, cache CRUD functions extracted from `__init__.py` |
| `nlp/_retry.py` | **Create** — `with_retry()` decorator |
| `nlp/__init__.py` | **Shrink** — ~100 lines: exports, `book_hash`, `CONFIG_DIR` seam |
| `nlp/booknlp_roster.py` | **Delete** — content moved to `providers/booknlp.py` |
| `services/nlp_service.py` | **Update** — replace `get_provider(config)` with `NLPPipeline(nlp_config)` |
| `nlp/quotes.py` | Unchanged |
| `nlp/chunker.py` | Unchanged |
| `nlp/models.py` | Unchanged |
| `nlp/annotator.py` | Unchanged (prompts preserved) |
| `nlp/attribution.py` | Minor refactor for testability; used by `OllamaAttributionAdapter` |

---

## What Moves to kentui

- Interactive cache picker (calls `list_caches()`, presents choices, passes `CacheMeta` back)
- `validate_tools()` result display (format + print `ValidationResult` list)
- Tool selection UI in the add wizard (maps user choices to `NLPConfig` fields)

---

## Verification

### Unit tests (per-adapter)
- `test_booknlp_extraction_adapter.py` — mock `booknlp.BookNLP`, assert `CharacterRoster` output
- `test_ollama_extraction_adapter.py` — mock `LLMClient`, assert roster shape
- `test_litellm_extraction_adapter.py` — mock `litellm.completion`, assert roster shape
- Same pattern for attribution adapters

### Integration test
- `test_nlp_pipeline_integration.py` — run `NLPPipeline(NLPConfig(extraction_tool=OLLAMA, attribution_tool=OLLAMA))` against a small fixture ebook; assert `NLPResult` has attributed chapters

### Cache tests
- Verify `list_caches()` sorts by timestamp descending
- Verify `put_cache()` is atomic (no partial writes on SIGTERM simulation)
- Verify separate tool runs produce separate cache files that coexist

### Validation test
- `validate_tools()` with Ollama unreachable → `ValidationResult(ok=False)` for affected step
- `validate_tools()` with BookNLP not installed → `ValidationResult(ok=False, message="booknlp not installed")`

### Config test
- `NLPConfig()` with `KENKUI_NLP_EXTRACTION_TOOL=booknlp` env var → `extraction_tool == BOOKNLP`
- `TTSConfig()` separate from `NLPConfig()` — no field overlap
