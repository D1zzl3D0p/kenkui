# Voice Provisioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `assign_voice("eponine").tts().write(...)` succeed after one explicit `kk.load_voice("eponine")`, by adding a provisioning surface that is strictly separated from the fail-closed offline renderer.

**Architecture:** Provisioning downloads, hashes, and records assets into a Kenkui-managed manifest; rendering only ever reads that manifest. Every renderable voice asset is a `.safetensors` speaker embedding, so WAV prompts are compiled at provision time and the render path keeps a single branch with no network, no downloads, and no Mimi encoding.

**Tech Stack:** Python 3.11-3.13, pocket-tts 2.1.0, torch, safetensors, huggingface-hub, pytest, mypy, ruff.

**Spec:** `docs/superpowers/specs/2026-08-19-voice-provisioning-design.md`

## Global Constraints

- Python `>=3.11,<3.14`. Target `py311` syntax; ruff `select = ["ALL"]`.
- All modules start with `from __future__ import annotations` and a one-line module docstring.
- Every public function and class needs a docstring; ruff enforces this.
- `uv run pytest` enforces `--cov-fail-under=90`. Every task adds tests.
- Verification commands, run from `kenkui/`:
  `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy`, `uv run pytest`.
- Manifest schema version string is exactly `kenkui-pocket-production-v2`.
- Manifest files are written mode `0600` inside a `0700` directory.
- Rendering code must never import `voices.provision` and must never reach the network. `HF_HUB_OFFLINE=1`, the `_deny_remote` allowlist, and the audit hook in `_tts/pocket.py` stay intact.
- Rights fields are never inferred. `add_voice` requires all seven explicitly.
- `language` values are pocket-tts config stems (`english`, `italian`, `spanish`, `german`, `portuguese`, `french_24l`), not ISO codes.
- Voice states: `registered` and `loaded` are persisted; `missing` is computed by `list_voices()` and never written.
- No CLI, no bulk verbs, no engine verbs, no engine enumeration.
- Renderer-side modules import voice types from `kenkui.voices.types`, never
  from `kenkui.voices`. The package `__init__` pulls in `provision`, which
  imports `manifest`, which imports `_tts.production` — importing the package
  from the renderer closes that cycle. `tests/test_import_boundaries.py`
  guards this.

**Already done — do not redo:** `Pipeline.assign_voice` already accepts `str | Voice` and stores the ID (`src/kenkui/pipeline.py:89-95`).

---

### Task 1: Make pocket-tts a required dependency

**Files:**
- Modify: `pyproject.toml:29-34`
- Test: `tests/test_dependency_contract.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `pocket_tts` importable in the test environment, which Task 4's drift test requires.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dependency_contract.py`:

```python
"""Pocket-TTS is a required runtime dependency at the pinned version."""

from __future__ import annotations

import importlib.metadata


def test_pocket_tts_is_installed_at_pinned_version() -> None:
    assert importlib.metadata.version("pocket-tts") == "2.1.0"


def test_pocket_tts_is_a_required_dependency() -> None:
    requires = importlib.metadata.requires("kenkui") or []
    runtime = [item for item in requires if "extra ==" not in item]
    assert any(item.startswith("pocket-tts==2.1.0") for item in runtime)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dependency_contract.py -v --no-cov`
Expected: FAIL with `PackageNotFoundError: No package metadata was found for pocket-tts`.

- [ ] **Step 3: Move the dependency**

In `pyproject.toml`, replace the `dependencies` and `[project.optional-dependencies]` blocks:

```toml
dependencies = [
    "defusedxml==0.7.1",
    "pocket-tts==2.1.0",
]

[project.optional-dependencies]
# Retained as an empty alias so existing `kenkui[pocket]` installs keep working.
# See spec section 10: this is the path back if pocket-tts becomes optional again.
pocket = []
```

- [ ] **Step 4: Sync and run the test**

Run: `uv sync && uv run pytest tests/test_dependency_contract.py -v --no-cov`
Expected: PASS. The sync downloads torch and will take several minutes.

- [ ] **Step 5: Confirm the existing suite still passes**

Run: `uv run pytest`
Expected: PASS. If mypy now reports missing stubs for `pocket_tts`, add to `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = ["pocket_tts.*"]
ignore_missing_imports = true
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/test_dependency_contract.py
git commit -m "feat: make pocket-tts a required dependency"
```

---

### Task 2: Add provisioning error codes

**Files:**
- Modify: `src/kenkui/errors.py:44` (after `VOICE_PROVENANCE_REQUIRED`), and the `_DEFAULT_MESSAGES` dict
- Test: `tests/test_provisioning_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ErrorCode.VOICE_NOT_PROVISIONED`, `ErrorCode.VOICE_UNKNOWN`, `ErrorCode.ENGINE_NOT_CLONING_CAPABLE`, `ErrorCode.VOICE_VARIETY_INVALID`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_provisioning_errors.py`:

```python
"""Provisioning failures use stable public codes with sanitized messages."""

from __future__ import annotations

import pytest

from kenkui import ErrorCode, VoiceError

_EXPECTED = {
    ErrorCode.VOICE_NOT_PROVISIONED: "voice_not_provisioned",
    ErrorCode.VOICE_UNKNOWN: "voice_unknown",
    ErrorCode.ENGINE_NOT_CLONING_CAPABLE: "engine_not_cloning_capable",
    ErrorCode.VOICE_VARIETY_INVALID: "voice_variety_invalid",
}


@pytest.mark.parametrize(("code", "value"), list(_EXPECTED.items()))
def test_code_value_is_stable(code: ErrorCode, value: str) -> None:
    assert code.value == value


@pytest.mark.parametrize("code", list(_EXPECTED))
def test_default_message_is_present_and_sanitized(code: ErrorCode) -> None:
    message = str(VoiceError(code))
    assert message
    assert "/" not in message


def test_not_provisioned_message_names_the_fix() -> None:
    assert "load_voice" in str(VoiceError(ErrorCode.VOICE_NOT_PROVISIONED))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provisioning_errors.py -v --no-cov`
Expected: FAIL with `AttributeError: VOICE_NOT_PROVISIONED`.

- [ ] **Step 3: Add the codes**

In `src/kenkui/errors.py`, add to `ErrorCode` immediately after `VOICE_PROVENANCE_REQUIRED`:

```python
    VOICE_NOT_PROVISIONED = "voice_not_provisioned"
    VOICE_UNKNOWN = "voice_unknown"
    ENGINE_NOT_CLONING_CAPABLE = "engine_not_cloning_capable"
    VOICE_VARIETY_INVALID = "voice_variety_invalid"
```

Add to `_DEFAULT_MESSAGES` after the `VOICE_PROVENANCE_REQUIRED` entry:

```python
    ErrorCode.VOICE_NOT_PROVISIONED: (
        "The voice is registered but not loaded. Call load_voice with its ID."
    ),
    ErrorCode.VOICE_UNKNOWN: "The voice ID is not in the catalog or the manifest.",
    ErrorCode.ENGINE_NOT_CLONING_CAPABLE: (
        "The engine lacks voice-cloning weights required for this voice."
    ),
    ErrorCode.VOICE_VARIETY_INVALID: "The voice variety or state is not recognized.",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provisioning_errors.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/errors.py tests/test_provisioning_errors.py
git commit -m "feat: add provisioning error codes"
```

---

### Task 3: Convert voices.py into a package with Voice and Engine types

**Files:**
- Delete: `src/kenkui/voices.py`
- Create: `src/kenkui/voices/__init__.py`, `src/kenkui/voices/types.py`
- Modify: `src/kenkui/__init__.py` (export `Engine`)
- Test: `tests/test_voice_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `kenkui.voices.types.Voice` with fields `id, name, enabled, provenance, license_id, commercial_use_allowed, language, content_fingerprint, compatible_model_revisions, variety, state, asset_bytes, engine`; and `Engine` with `id, language, model_revision, cloning_capable, size_bytes`. Both frozen slots dataclasses. `VoiceVariety` and `VoiceState` are `Literal` aliases.

Note: the spec's module layout lists four files. `types.py` is added as a fifth so `__init__.py` stays re-exports only; `manifest.py` and `provision.py` both need these types.

- [ ] **Step 1: Write the failing test**

Create `tests/test_voice_types.py`:

```python
"""Public voice and engine metadata types."""

from __future__ import annotations

import dataclasses

import pytest

from kenkui import Engine, Voice


def _voice(**overrides: object) -> Voice:
    base: dict[str, object] = {
        "id": "eponine",
        "name": "Eponine",
        "enabled": True,
        "provenance": "kyutai catalog",
        "license_id": "CC-BY-4.0",
        "commercial_use_allowed": False,
        "language": "english",
        "content_fingerprint": "a" * 64,
        "compatible_model_revisions": ("revision-1",),
        "variety": "built-in",
        "state": "registered",
    }
    return Voice(**(base | overrides))  # type: ignore[arg-type]


def test_voice_defaults_are_unloaded() -> None:
    voice = _voice()
    assert voice.asset_bytes is None
    assert voice.engine is None


def test_voice_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _voice().id = "other"  # type: ignore[misc]


def test_engine_is_hashable_for_set_deduplication() -> None:
    engine = Engine(
        id="english",
        language="english",
        model_revision="revision-1",
        cloning_capable=False,
        size_bytes=225_000_000,
    )
    assert len({engine, dataclasses.replace(engine)}) == 1


def test_loaded_voice_carries_its_engine() -> None:
    engine = Engine(
        id="english",
        language="english",
        model_revision="revision-1",
        cloning_capable=False,
        size_bytes=225_000_000,
    )
    voice = _voice(state="loaded", asset_bytes=6_500_000, engine=engine)
    assert voice.engine is engine
    assert voice.asset_bytes == 6_500_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice_types.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'Engine'`.

- [ ] **Step 3: Create the package**

Delete `src/kenkui/voices.py`. Create `src/kenkui/voices/types.py`:

```python
"""Public voice and engine metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VoiceVariety = Literal["built-in", "pre-compiled", "wav"]
VoiceState = Literal["registered", "loaded", "missing"]


@dataclass(frozen=True, slots=True)
class Engine:
    """A provisioned per-language synthesis engine."""

    id: str
    language: str
    model_revision: str
    cloning_capable: bool
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Voice:
    """Reusable voice identity, content, compatibility, and rights metadata."""

    id: str
    name: str
    enabled: bool
    provenance: str | None
    license_id: str | None
    commercial_use_allowed: bool | None
    language: str | None = None
    content_fingerprint: str | None = None
    compatible_model_revisions: tuple[str, ...] = ()
    variety: VoiceVariety = "built-in"
    state: VoiceState = "registered"
    asset_bytes: int | None = None
    engine: Engine | None = None
```

Create `src/kenkui/voices/__init__.py`:

```python
"""Public voice metadata and provisioning."""

from .types import Engine, Voice, VoiceState, VoiceVariety

__all__ = ["Engine", "Voice", "VoiceState", "VoiceVariety"]
```

- [ ] **Step 4: Break the import cycle before it forms**

`voices/__init__.py` will re-export `add_voice` in Task 7, and Task 5's
`voices/manifest.py` imports from `kenkui._tts.production`. If `production.py`
keeps importing `from kenkui.voices import Voice`, the cycle
`production -> voices/__init__ -> provision -> manifest -> production` closes
and the package fails to import.

Change the import in `src/kenkui/_tts/production.py` (currently line 23) to
reach the leaf module directly:

```python
from kenkui.voices.types import Voice
```

`voices/types.py` imports nothing from Kenkui, so it can never participate in a
cycle. Apply the same rule to any future renderer-side import of voice types.

Add `tests/test_import_boundaries.py`:

```python
"""Rendering never imports provisioning, and voice types stay a leaf."""

from __future__ import annotations

import subprocess
import sys

_RENDER_ONLY = (
    "import sys",
    "import kenkui._tts.production",
    "import kenkui._tts.pocket",
    "import kenkui._execution.coordinator",
    "print('kenkui.voices.provision' in sys.modules)",
)


def _run(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        check=False,
        text=True,
    )


def test_package_imports_cleanly_in_a_fresh_interpreter() -> None:
    completed = _run("import kenkui")
    assert completed.returncode == 0, completed.stderr


def test_render_modules_do_not_import_provisioning() -> None:
    completed = _run("\n".join(_RENDER_ONLY))
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
```

- [ ] **Step 5: Export Engine from the package root**

In `src/kenkui/__init__.py`, change the voices import and add `"Engine"` to `__all__` in alphabetical position (between `"EncodingError"` and `"ErrorCode"`):

```python
from .voices import Engine, Voice
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_voice_types.py tests/test_import_boundaries.py -v --no-cov && uv run pytest && uv run mypy`
Expected: PASS. The existing suite must stay green — `Voice`'s original fields keep their order and defaults, so `_tts/production.py` constructs it unchanged.

- [ ] **Step 7: Commit**

```bash
git add -A src/kenkui tests/test_voice_types.py tests/test_import_boundaries.py
git commit -m "feat: add Engine type and convert voices to a package"
```

---

### Task 4: Built-in voice catalog with per-voice rights

**Files:**
- Create: `src/kenkui/voices/registry.py`
- Modify: `src/kenkui/voices/__init__.py`
- Test: `tests/test_voice_registry.py`

**Interfaces:**
- Consumes: `Voice`, `VoiceVariety` from Task 3.
- Produces: `CatalogEntry` (frozen slots dataclass: `id, name, language, origin_url, license_id, commercial_use_allowed, voice_rights`), `CATALOG: dict[str, CatalogEntry]`, `EMBEDDING_REVISION: str`, `embedding_url(language: str, name: str) -> str`, `catalog_voice(voice_id: str) -> Voice`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_voice_registry.py`:

```python
"""The built-in catalog mirrors upstream and records per-voice rights."""

from __future__ import annotations

import pytest
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

from kenkui.voices import registry

_NONCOMMERCIAL = ("jean", "cosette")


def test_catalog_covers_every_upstream_voice() -> None:
    assert set(registry.CATALOG) == set(_ORIGINS_OF_PREDEFINED_VOICES)


def test_origin_urls_match_upstream_exactly() -> None:
    for voice_id, entry in registry.CATALOG.items():
        assert entry.origin_url == _ORIGINS_OF_PREDEFINED_VOICES[voice_id]


@pytest.mark.parametrize("voice_id", _NONCOMMERCIAL)
def test_research_only_corpora_are_not_commercial(voice_id: str) -> None:
    assert registry.CATALOG[voice_id].commercial_use_allowed is False


def test_every_entry_records_rights() -> None:
    for entry in registry.CATALOG.values():
        assert entry.license_id
        assert entry.voice_rights
        assert isinstance(entry.commercial_use_allowed, bool)


def test_embedding_url_pins_the_revision() -> None:
    url = registry.embedding_url("english", "eponine")
    assert url.startswith("hf://kyutai/pocket-tts-without-voice-cloning/")
    assert url.endswith(f"eponine.safetensors@{registry.EMBEDDING_REVISION}")


def test_catalog_voice_is_registered_and_built_in() -> None:
    voice = registry.catalog_voice("eponine")
    assert voice.id == "eponine"
    assert voice.variety == "built-in"
    assert voice.state == "registered"
    assert voice.language == "english"
    assert voice.engine is None


def test_catalog_voice_rejects_unknown_id() -> None:
    from kenkui import ErrorCode, VoiceError

    with pytest.raises(VoiceError) as excinfo:
        registry.catalog_voice("nobody")
    assert excinfo.value.code is ErrorCode.VOICE_UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice_registry.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: kenkui.voices.registry`.

- [ ] **Step 3: Write the catalog**

Create `src/kenkui/voices/registry.py`. Copy each `origin_url` verbatim from
`pocket_tts.utils.utils._ORIGINS_OF_PREDEFINED_VOICES` — the drift test compares
them exactly. Set `commercial_use_allowed=False` wherever the source terms are
not clearly permissive; that is the spec's conservative default, not a legal
conclusion.

```python
"""Static built-in voice catalog with per-voice rights metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from kenkui.errors import ErrorCode, VoiceError
from kenkui.voices.types import Voice

EMBEDDING_REVISION: Final = "e041936c75475d350b405bc870bcf7c22da4e9e6"
_EMBEDDING_REPO: Final = "kyutai/pocket-tts-without-voice-cloning"

_VCTK_RIGHTS: Final = (
    "Derived from the VCTK corpus via kyutai/tts-voices. Review the VCTK terms "
    "and speaker consent for your intended use before commercial deployment."
)
_EARS_RIGHTS: Final = (
    "Derived from the EARS corpus. Treat as research-only/noncommercial unless "
    "your own review of the source terms concludes otherwise."
)
_EXPRESSO_RIGHTS: Final = (
    "Derived from the Expresso dataset. Treat as research-only/noncommercial "
    "unless your own review of the source terms concludes otherwise."
)
_DONATION_RIGHTS: Final = (
    "Voice donation distributed by kyutai/tts-voices. Confirm the donor's "
    "permission scope for your intended use."
)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One upstream predefined voice and the rights Kenkui records for it."""

    id: str
    name: str
    language: str
    origin_url: str
    license_id: str
    commercial_use_allowed: bool
    voice_rights: str


def _vctk(voice_id: str, name: str, filename: str) -> CatalogEntry:
    return CatalogEntry(
        id=voice_id,
        name=name,
        language="english",
        origin_url=f"hf://kyutai/tts-voices/vctk/{filename}",
        license_id="CC-BY-4.0",
        commercial_use_allowed=False,
        voice_rights=_VCTK_RIGHTS,
    )


def _zero(voice_id: str, name: str) -> CatalogEntry:
    return CatalogEntry(
        id=voice_id,
        name=name,
        language="english",
        origin_url=f"hf://kyutai/tts-voices/voice-zero/{voice_id}.wav",
        license_id="unreviewed",
        commercial_use_allowed=False,
        voice_rights=_DONATION_RIGHTS,
    )


CATALOG: Final[dict[str, CatalogEntry]] = {
    entry.id: entry
    for entry in (
        _vctk("anna", "Anna", "p228_023_enhanced.wav"),
        _vctk("vera", "Vera", "p229_023_enhanced.wav"),
        _vctk("fantine", "Fantine", "p244_023_enhanced.wav"),
        _vctk("charles", "Charles", "p254_023_enhanced.wav"),
        _vctk("paul", "Paul", "p259_023_enhanced.wav"),
        _vctk("eponine", "Eponine", "p262_023_enhanced.wav"),
        _vctk("azelma", "Azelma", "p303_023_enhanced.wav"),
        _vctk("george", "George", "p315_023_enhanced.wav"),
        _vctk("mary", "Mary", "p333_023_enhanced.wav"),
        _vctk("jane", "Jane", "p339_023_enhanced.wav"),
        _vctk("michael", "Michael", "p360_023_enhanced.wav"),
        _vctk("eve", "Eve", "p361_023_enhanced.wav"),
        _zero("bill_boerst", "Bill Boerst"),
        _zero("peter_yearsley", "Peter Yearsley"),
        _zero("stuart_bell", "Stuart Bell"),
        _zero("caro_davy", "Caro Davy"),
        CatalogEntry(
            id="cosette",
            name="Cosette",
            language="english",
            origin_url=(
                "hf://kyutai/tts-voices/expresso/"
                "ex04-ex02_confused_001_channel1_499s.wav"
            ),
            license_id="CC-BY-NC-4.0",
            commercial_use_allowed=False,
            voice_rights=_EXPRESSO_RIGHTS,
        ),
        CatalogEntry(
            id="jean",
            name="Jean",
            language="english",
            origin_url=(
                "hf://kyutai/tts-voices/ears/p010/freeform_speech_01_enhanced.wav"
            ),
            license_id="CC-BY-NC-4.0",
            commercial_use_allowed=False,
            voice_rights=_EARS_RIGHTS,
        ),
        CatalogEntry(
            id="marius",
            name="Marius",
            language="english",
            origin_url="hf://kyutai/tts-voices/voice-donations/Selfie.wav",
            license_id="unreviewed",
            commercial_use_allowed=False,
            voice_rights=_DONATION_RIGHTS,
        ),
        CatalogEntry(
            id="javert",
            name="Javert",
            language="english",
            origin_url="hf://kyutai/tts-voices/voice-donations/Butter.wav",
            license_id="unreviewed",
            commercial_use_allowed=False,
            voice_rights=_DONATION_RIGHTS,
        ),
        CatalogEntry(
            id="alba",
            name="Alba",
            language="english",
            origin_url="hf://kyutai/tts-voices/alba-mackenna/casual.wav",
            license_id="unreviewed",
            commercial_use_allowed=False,
            voice_rights=_DONATION_RIGHTS,
        ),
        CatalogEntry(
            id="giovanni",
            name="Giovanni",
            language="italian",
            origin_url=(
                "hf://kyutai/pocket-tts/common_voice_it_36520747-enhanced-v2.mp3"
                "@64ab7d24c479d736a83b8cc666c4a776fca30fda"
            ),
            license_id="CC0-1.0",
            commercial_use_allowed=False,
            voice_rights=_DONATION_RIGHTS,
        ),
        CatalogEntry(
            id="lola",
            name="Lola",
            language="spanish",
            origin_url=(
                "hf://kyutai/pocket-tts/common_voice_es_19762977-enhanced-v2.mp3"
                "@64ab7d24c479d736a83b8cc666c4a776fca30fda"
            ),
            license_id="CC0-1.0",
            commercial_use_allowed=False,
            voice_rights=_DONATION_RIGHTS,
        ),
        CatalogEntry(
            id="juergen",
            name="Juergen",
            language="german",
            origin_url=(
                "hf://kyutai/pocket-tts/de-DE-juergen.mp3"
                "@64ab7d24c479d736a83b8cc666c4a776fca30fda"
            ),
            license_id="unreviewed",
            commercial_use_allowed=False,
            voice_rights=_DONATION_RIGHTS,
        ),
        CatalogEntry(
            id="rafael",
            name="Rafael",
            language="portuguese",
            origin_url=(
                "hf://kyutai/pocket-tts/g-Vi8PgmSY0-enhanced-v2.wav"
                "@64ab7d24c479d736a83b8cc666c4a776fca30fda"
            ),
            license_id="unreviewed",
            commercial_use_allowed=False,
            voice_rights=_DONATION_RIGHTS,
        ),
        CatalogEntry(
            id="estelle",
            name="Estelle",
            language="french_24l",
            origin_url=(
                "hf://kyutai/tts-voices/unmute-prod-website/developpeuse-3.wav"
                "@1fc7395b7e012e2bbebfca14b942a4ef62ccc899"
            ),
            license_id="unreviewed",
            commercial_use_allowed=False,
            voice_rights=_DONATION_RIGHTS,
        ),
    )
}


def embedding_url(language: str, name: str) -> str:
    """Return the pinned ungated embedding URL for one catalog voice."""
    return (
        f"hf://{_EMBEDDING_REPO}/languages/{language}/embeddings/"
        f"{name}.safetensors@{EMBEDDING_REVISION}"
    )


def catalog_voice(voice_id: str) -> Voice:
    """Return catalog metadata for one built-in voice as a registered Voice."""
    entry = CATALOG.get(voice_id)
    if entry is None:
        raise VoiceError(ErrorCode.VOICE_UNKNOWN)
    return Voice(
        id=entry.id,
        name=entry.name,
        enabled=True,
        provenance=entry.origin_url,
        license_id=entry.license_id,
        commercial_use_allowed=entry.commercial_use_allowed,
        language=entry.language,
        variety="built-in",
        state="registered",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_voice_registry.py -v --no-cov`
Expected: PASS. If `test_catalog_covers_every_upstream_voice` fails, the catalog is missing an entry — add it rather than weakening the assertion; that test is the drift guard required by spec section 11.

- [ ] **Step 5: Export from the package**

In `src/kenkui/voices/__init__.py`:

```python
"""Public voice metadata and provisioning."""

from .registry import CatalogEntry
from .types import Engine, Voice, VoiceState, VoiceVariety

__all__ = ["CatalogEntry", "Engine", "Voice", "VoiceState", "VoiceVariety"]
```

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/voices tests/test_voice_registry.py
git commit -m "feat: add built-in voice catalog with per-voice rights"
```

---

### Task 5: Managed manifest reader, merger, and atomic writer

**Files:**
- Create: `src/kenkui/voices/manifest.py`
- Test: `tests/test_voice_manifest_store.py`

**Interfaces:**
- Consumes: `Engine`, `Voice`, `VoiceVariety` from Task 3.
- Produces:
  - `default_manifest_path() -> Path`
  - `EngineRecord` (frozen slots: `id, language, model_root, config_path, model_revision, package_version, files, sample_rate_hz, device, timeout_seconds, cloning_capable`) where `files: tuple[FileRecord, ...]`
  - `FileRecord` (frozen slots: `relative_path, size, sha256`)
  - `VoiceRecord` (frozen slots: `id, variety, state, name, enabled, language, engine_id, provenance, license_id, commercial_use_allowed, voice_rights, source_path, source_sha256, asset_path, asset_sha256, compatible_model_revisions`) with the four asset/source fields defaulting to `None` and `compatible_model_revisions` to `()`
  - `ManifestStore(path: Path)` with `read() -> tuple[dict[str, EngineRecord], dict[str, VoiceRecord]]`, `write(engines, voices) -> None`, and `lock() -> AbstractContextManager[None]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_voice_manifest_store.py`:

```python
"""Managed manifest round-trips, writes atomically, and stays owner-private."""

from __future__ import annotations

import json
import stat
from pathlib import Path

from kenkui.voices.manifest import (
    EngineRecord,
    FileRecord,
    ManifestStore,
    VoiceRecord,
    default_manifest_path,
)


def _engine() -> EngineRecord:
    return EngineRecord(
        id="english",
        language="english",
        model_root="/models/english",
        config_path="/models/english/english.yaml",
        model_revision="revision-1",
        package_version="2.1.0",
        files=(FileRecord("model.safetensors", 225, "a" * 64),),
        sample_rate_hz=24000,
        device="cpu",
        timeout_seconds=300.0,
        cloning_capable=False,
    )


def _voice(**overrides: object) -> VoiceRecord:
    base: dict[str, object] = {
        "id": "eponine",
        "variety": "built-in",
        "state": "loaded",
        "name": "Eponine",
        "enabled": True,
        "language": "english",
        "engine_id": "english",
        "provenance": "hf://kyutai/tts-voices/vctk/p262_023_enhanced.wav",
        "license_id": "CC-BY-4.0",
        "commercial_use_allowed": False,
        "voice_rights": "review required",
        "asset_path": "/cache/voices/english/eponine.safetensors",
        "asset_sha256": "b" * 64,
        "compatible_model_revisions": ("revision-1",),
    }
    return VoiceRecord(**(base | overrides))  # type: ignore[arg-type]


def test_round_trip_preserves_records(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "manifest.json")
    store.write({"english": _engine()}, {"eponine": _voice()})
    engines, voices = store.read()
    assert engines["english"] == _engine()
    assert voices["eponine"] == _voice()


def test_read_of_absent_manifest_is_empty(tmp_path: Path) -> None:
    engines, voices = ManifestStore(tmp_path / "none.json").read()
    assert engines == {}
    assert voices == {}


def test_write_creates_owner_private_file_and_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"
    ManifestStore(path).write({"english": _engine()}, {"eponine": _voice()})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_written_schema_version_is_v2(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    ManifestStore(path).write({"english": _engine()}, {"eponine": _voice()})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "kenkui-pocket-production-v2"
    assert set(payload) == {"schema_version", "engines", "voices"}


def test_registered_record_omits_asset_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    registered = _voice(
        state="registered",
        variety="wav",
        asset_path=None,
        asset_sha256=None,
        compatible_model_revisions=(),
        source_path="/voices/mine.wav",
        source_sha256="c" * 64,
    )
    ManifestStore(path).write({}, {"eponine": registered})
    entry = json.loads(path.read_text(encoding="utf-8"))["voices"]["eponine"]
    assert "asset_path" not in entry
    assert "asset_sha256" not in entry
    assert "compatible_model_revisions" not in entry
    assert entry["source_path"] == "/voices/mine.wav"


def test_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    store = ManifestStore(path)
    store.write({"english": _engine()}, {"eponine": _voice()})
    store.write({"english": _engine()}, {})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_default_manifest_path_is_under_the_versioned_cache() -> None:
    path = default_manifest_path()
    assert path.name == "manifest.json"
    assert path.parent.name == "v1"
    assert path.parent.parent.name == "kenkui"


def test_lock_is_reentrant_across_sequential_uses(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "manifest.json")
    with store.lock():
        store.write({}, {})
    with store.lock():
        store.write({}, {})
    assert (tmp_path / "manifest.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice_manifest_store.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: kenkui.voices.manifest`.

- [ ] **Step 3: Implement the store**

Create `src/kenkui/voices/manifest.py`:

```python
"""Read, merge, and atomically write the Kenkui-managed production manifest."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from kenkui._tts.production import MANIFEST_SCHEMA_VERSION, default_cache_root

if TYPE_CHECKING:
    from collections.abc import Iterator

    from kenkui.voices.types import VoiceVariety

_ENGINE_KEYS: Final = (
    "language",
    "model_root",
    "config_path",
    "model_revision",
    "package_version",
    "sample_rate_hz",
    "device",
    "timeout_seconds",
    "cloning_capable",
)
_VOICE_COMMON_KEYS: Final = (
    "variety",
    "state",
    "name",
    "enabled",
    "language",
    "engine_id",
    "provenance",
    "license_id",
    "commercial_use_allowed",
    "voice_rights",
)


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One verified engine asset file."""

    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class EngineRecord:
    """One provisioned per-language engine."""

    id: str
    language: str
    model_root: str
    config_path: str
    model_revision: str
    package_version: str
    files: tuple[FileRecord, ...]
    sample_rate_hz: int
    device: str
    timeout_seconds: float
    cloning_capable: bool


@dataclass(frozen=True, slots=True)
class VoiceRecord:
    """One registered or loaded voice."""

    id: str
    variety: VoiceVariety
    state: str
    name: str
    enabled: bool
    language: str
    engine_id: str
    provenance: str
    license_id: str
    commercial_use_allowed: bool
    voice_rights: str
    source_path: str | None = None
    source_sha256: str | None = None
    asset_path: str | None = None
    asset_sha256: str | None = None
    compatible_model_revisions: tuple[str, ...] = ()


def default_manifest_path() -> Path:
    """Return the Kenkui-managed manifest location for this operating system."""
    return default_cache_root() / "manifest.json"


class ManifestStore:
    """Owner-private manifest persistence with atomic replacement."""

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        """Bind the store to one manifest path."""
        self.path = path

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Serialize provisioning against other processes on this machine."""
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(".lock")
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def read(self) -> tuple[dict[str, EngineRecord], dict[str, VoiceRecord]]:
        """Return stored engines and voices, or empty mappings when absent."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}, {}
        payload = json.loads(raw)
        engines = {
            key: _engine_from(key, value)
            for key, value in payload.get("engines", {}).items()
        }
        voices = {
            key: _voice_from(key, value)
            for key, value in payload.get("voices", {}).items()
        }
        return engines, voices

    def write(
        self,
        engines: dict[str, EngineRecord],
        voices: dict[str, VoiceRecord],
    ) -> None:
        """Replace the manifest atomically with owner-private permissions."""
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "engines": {key: _engine_to(value) for key, value in engines.items()},
            "voices": {key: _voice_to(value) for key, value in voices.items()},
        }
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        descriptor, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".manifest-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def _engine_to(record: EngineRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {key: getattr(record, key) for key in _ENGINE_KEYS}
    payload["files"] = [
        {
            "relative_path": item.relative_path,
            "size": item.size,
            "sha256": item.sha256,
        }
        for item in record.files
    ]
    return payload


def _engine_from(engine_id: str, payload: dict[str, Any]) -> EngineRecord:
    return EngineRecord(
        id=engine_id,
        files=tuple(
            FileRecord(item["relative_path"], item["size"], item["sha256"])
            for item in payload["files"]
        ),
        **{key: payload[key] for key in _ENGINE_KEYS},
    )


def _voice_to(record: VoiceRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {key: getattr(record, key) for key in _VOICE_COMMON_KEYS}
    if record.source_path is not None:
        payload["source_path"] = record.source_path
        payload["source_sha256"] = record.source_sha256
    if record.state == "loaded":
        payload["asset_path"] = record.asset_path
        payload["asset_sha256"] = record.asset_sha256
        payload["compatible_model_revisions"] = list(
            record.compatible_model_revisions
        )
    return payload


def _voice_from(voice_id: str, payload: dict[str, Any]) -> VoiceRecord:
    return VoiceRecord(
        id=voice_id,
        source_path=payload.get("source_path"),
        source_sha256=payload.get("source_sha256"),
        asset_path=payload.get("asset_path"),
        asset_sha256=payload.get("asset_sha256"),
        compatible_model_revisions=tuple(
            payload.get("compatible_model_revisions", ())
        ),
        **{key: payload[key] for key in _VOICE_COMMON_KEYS},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_voice_manifest_store.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/voices/manifest.py tests/test_voice_manifest_store.py
git commit -m "feat: add managed manifest store with atomic writes"
```

---

### Task 6: Strict manifest reader for schema v2

**Files:**
- Modify: `src/kenkui/_tts/production.py:25-135` and `src/kenkui/_tts/production.py:296-312`
- Modify: `tests/test_production_manifest_branches.py` (update `_payload` to v2)
- Test: `tests/test_production_manifest_v2.py`

**Interfaces:**
- Consumes: `Engine`, `Voice` from Task 3.
- Produces: `MANIFEST_SCHEMA_VERSION = "kenkui-pocket-production-v2"`; `production_bindings_from_environment(voice_id: str) -> ExecutionBindings` resolving `KENKUI_POCKET_MANIFEST` then `default_manifest_path()`; `resolve_manifest_path() -> Path | None`; `_select(root, voice_id) -> tuple[dict[str, object], dict[str, object]]`; and `PocketEngineConfig` renamed to `voice_asset_path` / `voice_asset_sha256` with new `voice_variety` and `cloning_capable` fields, which Tasks 8-12 rely on.

This module is the security boundary and stays deliberately paranoid: exact type
checks, exact key-set matching, absolute paths, no inference. Task 5's writer is
the convenience half; the round-trip test in Step 5 binds the two together.

- [ ] **Step 1: Write the failing test**

Create `tests/test_production_manifest_v2.py`:

```python
"""Schema v2 parsing: engines map, variety, state, and manifest resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kenkui import ErrorCode, ModelError, RenderError, VoiceError
from kenkui._tts import production


def _payload(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "model"
    root.mkdir(mode=0o700)
    config = root / "english.yaml"
    config.write_text("model: local\n", encoding="utf-8")
    asset = tmp_path / "eponine.safetensors"
    asset.write_bytes(b"fixture")
    return {
        "schema_version": "kenkui-pocket-production-v2",
        "engines": {
            "english": {
                "language": "english",
                "model_root": str(root),
                "config_path": str(config),
                "model_revision": "revision-1",
                "package_version": "2.1.0",
                "files": [
                    {
                        "relative_path": "english.yaml",
                        "size": config.stat().st_size,
                        "sha256": "a" * 64,
                    }
                ],
                "sample_rate_hz": 24000,
                "device": "cpu",
                "timeout_seconds": 30.0,
                "cloning_capable": False,
            }
        },
        "voices": {
            "eponine": {
                "variety": "built-in",
                "state": "loaded",
                "name": "Eponine",
                "enabled": True,
                "language": "english",
                "engine_id": "english",
                "provenance": "kyutai catalog",
                "license_id": "CC-BY-4.0",
                "commercial_use_allowed": False,
                "voice_rights": "review required",
                "asset_path": str(asset),
                "asset_sha256": "b" * 64,
                "compatible_model_revisions": ["revision-1"],
            }
        },
    }


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def _activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(_write(tmp_path, payload)))
    production.production_bindings_from_environment("eponine")


def test_schema_version_is_v2() -> None:
    assert production.MANIFEST_SCHEMA_VERSION == "kenkui-pocket-production-v2"


def test_v1_schema_version_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["schema_version"] = "kenkui-pocket-production-v1"
    with pytest.raises(ModelError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.POCKET_MODEL_INVALID


def test_unknown_engine_id_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["voices"]["eponine"]["engine_id"] = "italian"
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.VOICE_UNRESOLVED


def test_unknown_variety_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["voices"]["eponine"]["variety"] = "magic"
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.VOICE_VARIETY_INVALID


def test_registered_voice_is_not_renderable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    voice = payload["voices"]["eponine"]
    voice["state"] = "registered"
    voice.pop("asset_path")
    voice.pop("asset_sha256")
    voice.pop("compatible_model_revisions")
    voice["source_path"] = str(tmp_path / "eponine.safetensors")
    voice["source_sha256"] = "c" * 64
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.VOICE_NOT_PROVISIONED


def test_wav_voice_requires_cloning_capable_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["voices"]["eponine"]["variety"] = "wav"
    payload["voices"]["eponine"]["source_path"] = str(tmp_path / "v.wav")
    payload["voices"]["eponine"]["source_sha256"] = "d" * 64
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.ENGINE_NOT_CLONING_CAPABLE


def test_unknown_key_in_voice_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["voices"]["eponine"]["surprise"] = True
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.VOICE_PROVENANCE_REQUIRED


def test_missing_manifest_everywhere_is_renderer_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KENKUI_POCKET_MANIFEST", raising=False)
    monkeypatch.setattr(
        production, "default_manifest_path", lambda: tmp_path / "absent.json"
    )
    with pytest.raises(RenderError) as excinfo:
        production.production_bindings_from_environment("eponine")
    assert excinfo.value.code is ErrorCode.RENDERER_UNAVAILABLE


def test_default_path_is_used_when_env_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, _payload(tmp_path))
    monkeypatch.delenv("KENKUI_POCKET_MANIFEST", raising=False)
    monkeypatch.setattr(production, "default_manifest_path", lambda: path)
    assert production.resolve_manifest_path() == path


def test_env_override_wins_over_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = _write(tmp_path, _payload(tmp_path))
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(override))
    monkeypatch.setattr(
        production, "default_manifest_path", lambda: tmp_path / "other.json"
    )
    assert production.resolve_manifest_path() == override
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_production_manifest_v2.py -v --no-cov`
Expected: FAIL — `MANIFEST_SCHEMA_VERSION` is still `...-v1` and `resolve_manifest_path` does not exist.

- [ ] **Step 3: Rewrite the parser**

In `src/kenkui/_tts/production.py`:

Change the constant at line 26 and add the import:

```python
MANIFEST_SCHEMA_VERSION: Final = "kenkui-pocket-production-v2"
_VARIETIES: Final = frozenset({"built-in", "pre-compiled", "wav"})
_VOICE_BASE_KEYS: Final = {
    "variety",
    "state",
    "name",
    "enabled",
    "language",
    "engine_id",
    "provenance",
    "license_id",
    "commercial_use_allowed",
    "voice_rights",
}
_LOADED_KEYS: Final = {
    "asset_path",
    "asset_sha256",
    "compatible_model_revisions",
}
_SOURCE_KEYS: Final = {"source_path", "source_sha256"}
_ENGINE_KEYS: Final = {
    "language",
    "model_root",
    "config_path",
    "model_revision",
    "package_version",
    "files",
    "sample_rate_hz",
    "device",
    "timeout_seconds",
    "cloning_capable",
}
```

Add manifest resolution above `production_bindings_from_environment`:

```python
def resolve_manifest_path() -> Path | None:
    """Return the operator override, else the managed default, else None."""
    from kenkui.voices.manifest import default_manifest_path  # noqa: PLC0415

    override = os.environ.get(_MANIFEST_ENV)
    if override is not None:
        return Path(override)
    managed = default_manifest_path()
    return managed if managed.exists() else None
```

Replace the body of `production_bindings_from_environment` down to the
`PocketEngineConfig` construction. The engine lookup, variety check, and state
check are the new logic; `_object`, `_string`, `_boolean`, `_integer`,
`_floating`, `_digest`, `_string_tuple`, `_absolute_path`, `_manifest_file`,
`_read_manifest`, `_identity`, and `_unique_object` are unchanged:

```python
def production_bindings_from_environment(voice_id: str) -> ExecutionBindings:
    """Resolve one explicitly assigned voice from a strict local manifest."""
    path = resolve_manifest_path()
    if path is None:
        raise RenderError(ErrorCode.RENDERER_UNAVAILABLE)
    manifest = _read_manifest(str(path))
    root = _object(manifest, {"schema_version", "engines", "voices"})
    if root["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    voice_data, engine = _select(root, voice_id)
    revision = _string(engine["model_revision"])
    compatible = _string_tuple(voice_data["compatible_model_revisions"])
    variety = _string(voice_data["variety"], voice=True)
    cloning_capable = _boolean(engine["cloning_capable"])
    if variety == "wav" and not cloning_capable:
        raise VoiceError(ErrorCode.ENGINE_NOT_CLONING_CAPABLE)
    enabled = _boolean(voice_data["enabled"], voice=True)
    if not enabled:
        raise VoiceError(ErrorCode.VOICE_DISABLED)
    digest = _digest(voice_data["asset_sha256"], voice=True)
    voice = Voice(
        id=voice_id,
        name=_string(voice_data["name"], voice=True),
        enabled=enabled,
        provenance=_string(voice_data["provenance"], voice=True),
        license_id=_string(voice_data["license_id"], voice=True),
        commercial_use_allowed=_boolean(
            voice_data["commercial_use_allowed"], voice=True
        ),
        language=_string(voice_data["language"], voice=True),
        content_fingerprint=digest,
        compatible_model_revisions=compatible,
        variety=cast("VoiceVariety", variety),
        state="loaded",
    )
    if revision not in compatible:
        raise VoiceError(ErrorCode.VOICE_INCOMPATIBLE)
    files_raw = engine["files"]
    if type(files_raw) is not list or not files_raw:
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    files = tuple(_manifest_file(item) for item in cast("list[object]", files_raw))
    config = PocketEngineConfig(
        model_root=_absolute_path(engine["model_root"]),
        config_path=_absolute_path(engine["config_path"]),
        model_revision=revision,
        package_version=_string(engine["package_version"]),
        files=files,
        voice_asset_path=_absolute_path(voice_data["asset_path"], voice=True),
        voice_asset_sha256=digest,
        voice_variety=variety,
        cloning_capable=cloning_capable,
        voice_provenance=cast("str", voice.provenance),
        voice_license_id=cast("str", voice.license_id),
        voice_rights=_string(voice_data["voice_rights"], voice=True),
        commercial_use_allowed=cast("bool", voice.commercial_use_allowed),
        sample_rate_hz=_integer(engine["sample_rate_hz"]),
        device=_string(engine["device"]),
        timeout_seconds=_floating(engine["timeout_seconds"]),
    )
    return pocket_production_bindings(config, voice)


def _select(
    root: dict[str, object], voice_id: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the strict voice and engine objects for one assigned voice ID."""
    voices_raw = root["voices"]
    if type(voices_raw) is not dict or not voices_raw:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    voices = cast("dict[object, object]", voices_raw)
    if any(type(key) is not str or not key for key in voices):
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    selected = voices.get(voice_id)
    if selected is None:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    if type(selected) is not dict:
        raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
    entry = cast("dict[str, object]", selected)
    variety = entry.get("variety")
    if variety not in _VARIETIES:
        raise VoiceError(ErrorCode.VOICE_VARIETY_INVALID)
    state = entry.get("state")
    if state == "registered":
        raise VoiceError(ErrorCode.VOICE_NOT_PROVISIONED)
    if state != "loaded":
        raise VoiceError(ErrorCode.VOICE_VARIETY_INVALID)
    expected = _VOICE_BASE_KEYS | _LOADED_KEYS
    if variety == "wav":
        expected = expected | _SOURCE_KEYS
    voice_data = _object(entry, expected, voice=True)
    engines_raw = root["engines"]
    if type(engines_raw) is not dict or not engines_raw:
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    engines = cast("dict[str, object]", engines_raw)
    engine_id = voice_data["engine_id"]
    if type(engine_id) is not str or engine_id not in engines:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    return voice_data, _object(engines[engine_id], _ENGINE_KEYS)
```

Extend the leaf import established in Task 3 Step 4 — never import from
`kenkui.voices` here, only from `kenkui.voices.types`:

```python
from kenkui.voices.types import Voice, VoiceVariety
```

- [ ] **Step 4: Rename the engine config fields**

The parser above constructs `PocketEngineConfig` with new field names, so the
dataclass must change in the same task or the tree is left broken. In
`src/kenkui/_tts/pocket.py`, in `PocketEngineConfig`, replace
`voice_prompt_path: str` and `voice_prompt_sha256: str` with:

```python
    voice_asset_path: str
    voice_asset_sha256: str
    voice_variety: str
    cloning_capable: bool
```

In `semantic_material()`, replace the `"voice_prompt_sha256"` entry with:

```python
            "voice_asset_sha256": self.voice_asset_sha256,
            "voice_variety": self.voice_variety,
```

In `PocketTTSEngine.synthesize` and `preflight_pocket`, update the two
references to `config.voice_prompt_path` to `config.voice_asset_path`. Behaviour
is otherwise unchanged here; Task 12 replaces the validation and hoists the
conditioning state.

- [ ] **Step 5: Run the new test**

Run: `uv run pytest tests/test_production_manifest_v2.py -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Add the round-trip test binding writer to reader**

Append to `tests/test_voice_manifest_store.py`:

```python
def test_writer_output_is_accepted_by_the_strict_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The convenience writer and the paranoid reader must agree on schema."""
    import json as _json

    from kenkui._tts import production

    path = tmp_path / "manifest.json"
    ManifestStore(path).write({"english": _engine()}, {"eponine": _voice()})
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(path))
    payload = _json.loads(path.read_text(encoding="utf-8"))
    root = production._object(  # noqa: SLF001
        payload, {"schema_version", "engines", "voices"}
    )
    voice_data, engine = production._select(root, "eponine")  # noqa: SLF001
    assert engine["language"] == "english"
    assert voice_data["state"] == "loaded"
```

Add `import pytest` to that file's imports.

- [ ] **Step 7: Migrate the existing v1 test fixture**

In `tests/test_production_manifest_branches.py`, update `_payload` to the v2
shape: rename the `"engine"` key to `"engines": {"english": {...}}`, add
`"language": "english"` and `"cloning_capable": False` to the engine, and
replace the voice body with the v2 keys used in `_payload` above. Every
existing rejection assertion stays valid; only the fixture shape changes.
Update `_activate` and the tests that reach into `("engine", ...)` paths to use
`("engines", "english", ...)`.

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest && uv run mypy && uv run ruff check .`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/kenkui/_tts/production.py tests/
git commit -m "feat: parse manifest schema v2 with engines map and voice state"
```

---

### Task 7: add_voice

**Files:**
- Create: `src/kenkui/voices/provision.py`
- Modify: `src/kenkui/voices/__init__.py`, `src/kenkui/__init__.py`
- Test: `tests/test_add_voice.py`

**Interfaces:**
- Consumes: `ManifestStore`, `VoiceRecord`, `default_manifest_path` (Task 5); `CATALOG` (Task 4); `Voice` (Task 3).
- Produces: `add_voice(path, *, voice_id, name, language, provenance, license_id, commercial_use_allowed, voice_rights, manifest=None) -> Voice`; internal helpers `_sha256(path: Path) -> str` and `_store(manifest: Path | None) -> ManifestStore`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_add_voice.py`:

```python
"""Registering a local voice records rights without touching the network."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui import ErrorCode, VoiceError, add_voice
from kenkui.voices.manifest import ManifestStore

_RIGHTS: dict[str, object] = {
    "name": "My Narrator",
    "language": "english",
    "provenance": "recorded by me, 2026-08-19",
    "license_id": "proprietary",
    "commercial_use_allowed": True,
    "voice_rights": "owned outright",
}


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "mine.wav"
    path.write_bytes(b"RIFF0000WAVEfmt ")
    return path


def test_registers_a_wav_with_source_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = _wav(tmp_path)
    voice = add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    assert voice.variety == "wav"
    assert voice.state == "registered"
    _, voices = ManifestStore(manifest).read()
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert voices["mine"].source_sha256 == expected
    assert voices["mine"].asset_path is None


def test_safetensors_registers_as_pre_compiled(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b'{"__metadata__":{}}')
    voice = add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    assert voice.variety == "pre-compiled"


def test_rights_are_preserved_verbatim(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    add_voice(_wav(tmp_path), voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    _, voices = ManifestStore(manifest).read()
    record = voices["mine"]
    assert record.provenance == _RIGHTS["provenance"]
    assert record.license_id == _RIGHTS["license_id"]
    assert record.voice_rights == _RIGHTS["voice_rights"]
    assert record.commercial_use_allowed is True


def test_catalog_name_collision_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceError) as excinfo:
        add_voice(
            _wav(tmp_path),
            voice_id="eponine",
            manifest=tmp_path / "manifest.json",
            **_RIGHTS,  # type: ignore[arg-type]
        )
    assert excinfo.value.code is ErrorCode.VOICE_VARIETY_INVALID


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "mine.mp3"
    source.write_bytes(b"nope")
    with pytest.raises(VoiceError) as excinfo:
        add_voice(
            source,
            voice_id="mine",
            manifest=tmp_path / "manifest.json",
            **_RIGHTS,  # type: ignore[arg-type]
        )
    assert excinfo.value.code is ErrorCode.VOICE_VARIETY_INVALID


def test_absent_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceError) as excinfo:
        add_voice(
            tmp_path / "nothing.wav",
            voice_id="mine",
            manifest=tmp_path / "manifest.json",
            **_RIGHTS,  # type: ignore[arg-type]
        )
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


def test_existing_entries_are_preserved(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    add_voice(_wav(tmp_path), voice_id="one", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    second = tmp_path / "two.wav"
    second.write_bytes(b"RIFF1111WAVEfmt ")
    add_voice(second, voice_id="two", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    _, voices = ManifestStore(manifest).read()
    assert set(voices) == {"one", "two"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_add_voice.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'add_voice'`.

- [ ] **Step 3: Implement add_voice**

Create `src/kenkui/voices/provision.py`:

```python
"""Explicit, user-initiated voice provisioning.

This is the only module in Kenkui permitted to reach the network. Nothing under
`kenkui._execution` or `kenkui._tts` may import it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Final

from kenkui.errors import ErrorCode, VoiceError
from kenkui.voices.manifest import ManifestStore, VoiceRecord, default_manifest_path
from kenkui.voices.registry import CATALOG
from kenkui.voices.types import Voice, VoiceVariety

if TYPE_CHECKING:
    import os

_HASH_CHUNK_BYTES: Final = 1024 * 1024
_SUFFIX_VARIETY: Final[dict[str, VoiceVariety]] = {
    ".wav": "wav",
    ".safetensors": "pre-compiled",
}


def _store(manifest: Path | None) -> ManifestStore:
    return ManifestStore(manifest if manifest is not None else default_manifest_path())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def add_voice(
    path: str | os.PathLike[str],
    *,
    voice_id: str,
    name: str,
    language: str,
    provenance: str,
    license_id: str,
    commercial_use_allowed: bool,
    voice_rights: str,
    manifest: Path | None = None,
) -> Voice:
    """Register a local WAV or safetensors voice with explicit rights metadata."""
    if voice_id in CATALOG:
        raise VoiceError(ErrorCode.VOICE_VARIETY_INVALID)
    source = Path(path).resolve()
    variety = _SUFFIX_VARIETY.get(source.suffix.lower())
    if variety is None:
        raise VoiceError(ErrorCode.VOICE_VARIETY_INVALID)
    if not source.is_file():
        raise VoiceError(ErrorCode.POCKET_VOICE_INVALID)
    record = VoiceRecord(
        id=voice_id,
        variety=variety,
        state="registered",
        name=name,
        enabled=True,
        language=language,
        engine_id=language,
        provenance=provenance,
        license_id=license_id,
        commercial_use_allowed=commercial_use_allowed,
        voice_rights=voice_rights,
        source_path=str(source),
        source_sha256=_sha256(source),
    )
    store = _store(manifest)
    with store.lock():
        engines, voices = store.read()
        voices[voice_id] = record
        store.write(engines, voices)
    return Voice(
        id=voice_id,
        name=name,
        enabled=True,
        provenance=provenance,
        license_id=license_id,
        commercial_use_allowed=commercial_use_allowed,
        language=language,
        variety=variety,
        state="registered",
    )
```

- [ ] **Step 4: Export it**

In `src/kenkui/voices/__init__.py` add `from .provision import add_voice` and
put `"add_voice"` in `__all__`. In `src/kenkui/__init__.py` change the voices
import to `from .voices import Engine, Voice, add_voice` and add `"add_voice"`
to `__all__` in alphabetical position.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_add_voice.py -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui tests/test_add_voice.py
git commit -m "feat: add add_voice for registering local voices"
```

---

### Task 8: Engine provisioning and load_voice for built-in voices

**Files:**
- Modify: `src/kenkui/voices/provision.py`
- Test: `tests/test_load_voice_builtin.py`

**Interfaces:**
- Consumes: everything from Task 7; `EngineRecord`, `FileRecord` from Task 5; `embedding_url`, `catalog_voice`, `CATALOG` from Task 4.
- Produces: `load_voice(voice_id: str, *, manifest: Path | None = None) -> Voice`; `_fetch(url: str) -> Path` (the single network seam, monkeypatched in tests); `_provision_engine(language: str, *, cloning: bool, root: Path) -> EngineRecord`; `_assets_root(manifest_path: Path) -> Path`.

**Critical implementation note.** The renderer replaces pocket-tts's
`download_if_necessary` with `_deny_remote` (`_tts/pocket.py:652`), which
accepts only absolute local paths inside the manifest allowlist. A stock
pocket-tts config YAML contains `hf://` URLs and would therefore be rejected at
load time. Provisioning must **write a derived config YAML** into the engine's
`model_root` with `weights_path`, `flow_lm.lookup_table.tokenizer_path`, and
`mimi.weights_path` rewritten to absolute local paths.

For a non-cloning engine, point `weights_path` at the downloaded *ungated*
weights. The model's own `has_voice_cloning` flag will then read `True`, which
is harmless and irrelevant: Kenkui records capability in the manifest's
`cloning_capable` field and never passes audio conditioning at render time.

- [ ] **Step 1: Write the failing test**

Create `tests/test_load_voice_builtin.py`:

```python
"""Built-in voices provision an engine and an embedding, idempotently."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui import ErrorCode, VoiceError, load_voice
from kenkui.voices import provision
from kenkui.voices.manifest import ManifestStore


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Serve every remote URL from local bytes and count the fetches."""
    hub = tmp_path / "hub"
    hub.mkdir()
    calls: dict[str, int] = {}

    def fetch(url: str) -> Path:
        calls[url] = calls.get(url, 0) + 1
        name = hashlib.sha256(url.encode()).hexdigest()[:16]
        suffix = ".safetensors" if ".safetensors" in url else ".bin"
        path = hub / f"{name}{suffix}"
        if not path.exists():
            path.write_bytes(url.encode() * 8)
        return path

    monkeypatch.setattr(provision, "_fetch", fetch)
    return calls


def test_loads_a_builtin_voice(
    tmp_path: Path, fake_hub: dict[str, int]
) -> None:
    manifest = tmp_path / "manifest.json"
    voice = load_voice("eponine", manifest=manifest)
    assert voice.state == "loaded"
    assert voice.variety == "built-in"
    assert voice.engine is not None
    assert voice.engine.language == "english"
    assert voice.asset_bytes is not None and voice.asset_bytes > 0


def test_manifest_records_engine_and_voice(
    tmp_path: Path, fake_hub: dict[str, int]
) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    engines, voices = ManifestStore(manifest).read()
    assert set(engines) == {"english"}
    assert engines["english"].cloning_capable is False
    assert engines["english"].files
    record = voices["eponine"]
    assert record.state == "loaded"
    assert record.asset_path is not None
    assert Path(record.asset_path).is_file()
    assert record.compatible_model_revisions


def test_asset_hash_matches_downloaded_bytes(
    tmp_path: Path, fake_hub: dict[str, int]
) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    asset = Path(voices["eponine"].asset_path or "")
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    assert voices["eponine"].asset_sha256 == expected


def test_second_load_performs_no_fetches(
    tmp_path: Path, fake_hub: dict[str, int]
) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    before = sum(fake_hub.values())
    load_voice("eponine", manifest=manifest)
    assert sum(fake_hub.values()) == before


def test_sibling_voice_reuses_the_engine(
    tmp_path: Path, fake_hub: dict[str, int]
) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    load_voice("alba", manifest=manifest)
    engines, voices = ManifestStore(manifest).read()
    assert set(engines) == {"english"}
    assert set(voices) == {"eponine", "alba"}


def test_written_config_has_no_remote_references(
    tmp_path: Path, fake_hub: dict[str, int]
) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    engines, _ = ManifestStore(manifest).read()
    text = Path(engines["english"].config_path).read_text(encoding="utf-8")
    assert "hf://" not in text
    assert "http://" not in text
    assert "https://" not in text


def test_unknown_voice_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceError) as excinfo:
        load_voice("nobody", manifest=tmp_path / "manifest.json")
    assert excinfo.value.code is ErrorCode.VOICE_UNKNOWN


def test_missing_asset_is_repaired(
    tmp_path: Path, fake_hub: dict[str, int]
) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    Path(voices["eponine"].asset_path or "").unlink()
    load_voice("eponine", manifest=manifest)
    _, repaired = ManifestStore(manifest).read()
    assert Path(repaired["eponine"].asset_path or "").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_load_voice_builtin.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'load_voice'`.

- [ ] **Step 3: Implement fetching, engine provisioning, and load_voice**

Append to `src/kenkui/voices/provision.py`. Add these imports at the top:

```python
import shutil
from typing import Any

import yaml

from kenkui.voices.manifest import EngineRecord, FileRecord
from kenkui.voices.registry import CATALOG, catalog_voice, embedding_url
from kenkui.voices.types import Engine
```

Then append:

```python
def _fetch(url: str) -> Path:
    """Resolve one remote or local asset reference to a local file.

    The single network seam in Kenkui. Tests monkeypatch this symbol.
    """
    from pocket_tts.utils.utils import download_if_necessary  # noqa: PLC0415

    return Path(download_if_necessary(url))


def _assets_root(manifest_path: Path) -> Path:
    return manifest_path.parent


def _revision_of(url: str) -> str:
    return url.rsplit("@", 1)[-1] if "@" in url else "unpinned"


def _place(source: Path, destination: Path) -> FileRecord:
    """Copy one fetched asset into the engine root and record its identity."""
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)
    return FileRecord(
        relative_path=destination.name,
        size=destination.stat().st_size,
        sha256=_sha256(destination),
    )


def _provision_engine(language: str, *, cloning: bool, root: Path) -> EngineRecord:
    """Download one language engine and write a fully local derived config."""
    from pocket_tts.utils.config import CONFIGS_DIR  # noqa: PLC0415

    model_root = root / "engines" / language
    model_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stock: dict[str, Any] = yaml.safe_load(
        (CONFIGS_DIR / f"{language}.yaml").read_text(encoding="utf-8")
    )
    weights_url = (
        stock["weights_path"]
        if cloning
        else stock["weights_path_without_voice_cloning"]
    )
    tokenizer_url = stock["flow_lm"]["lookup_table"]["tokenizer_path"]
    mimi_url = stock["mimi"]["weights_path"]

    files = [
        _place(_fetch(weights_url), model_root / "model.safetensors"),
        _place(_fetch(tokenizer_url), model_root / "tokenizer.model"),
        _place(_fetch(mimi_url), model_root / "mimi.safetensors"),
    ]

    derived = dict(stock)
    derived["weights_path"] = str(model_root / "model.safetensors")
    derived.pop("weights_path_without_voice_cloning", None)
    derived["flow_lm"] = dict(stock["flow_lm"])
    derived["flow_lm"]["lookup_table"] = dict(stock["flow_lm"]["lookup_table"])
    derived["flow_lm"]["lookup_table"]["tokenizer_path"] = str(
        model_root / "tokenizer.model"
    )
    derived["mimi"] = dict(stock["mimi"])
    derived["mimi"]["weights_path"] = str(model_root / "mimi.safetensors")

    config_path = model_root / f"{language}.yaml"
    config_path.write_text(yaml.safe_dump(derived, sort_keys=True), encoding="utf-8")
    config_path.chmod(0o600)
    files.append(
        FileRecord(
            relative_path=config_path.name,
            size=config_path.stat().st_size,
            sha256=_sha256(config_path),
        )
    )

    return EngineRecord(
        id=language,
        language=language,
        model_root=str(model_root),
        config_path=str(config_path),
        model_revision=_revision_of(weights_url),
        package_version="2.1.0",
        files=tuple(files),
        sample_rate_hz=int(stock["mimi"]["sample_rate"]),
        device="cpu",
        timeout_seconds=300.0,
        cloning_capable=cloning,
    )


def _engine_view(record: EngineRecord) -> Engine:
    return Engine(
        id=record.id,
        language=record.language,
        model_revision=record.model_revision,
        cloning_capable=record.cloning_capable,
        size_bytes=sum(item.size for item in record.files),
    )


def _loaded_view(record: VoiceRecord, engine: EngineRecord) -> Voice:
    asset = Path(record.asset_path or "")
    return Voice(
        id=record.id,
        name=record.name,
        enabled=record.enabled,
        provenance=record.provenance,
        license_id=record.license_id,
        commercial_use_allowed=record.commercial_use_allowed,
        language=record.language,
        content_fingerprint=record.asset_sha256,
        compatible_model_revisions=record.compatible_model_revisions,
        variety=record.variety,
        state="loaded",
        asset_bytes=asset.stat().st_size if asset.is_file() else None,
        engine=_engine_view(engine),
    )


def load_voice(voice_id: str, *, manifest: Path | None = None) -> Voice:
    """Make one voice renderable, downloading or compiling only if required."""
    store = _store(manifest)
    root = _assets_root(store.path)
    with store.lock():
        engines, voices = store.read()
        record = voices.get(voice_id)
        if record is None and voice_id not in CATALOG:
            raise VoiceError(ErrorCode.VOICE_UNKNOWN)
        if record is not None and record.state == "loaded":
            asset = Path(record.asset_path or "")
            if asset.is_file() and _sha256(asset) == record.asset_sha256:
                return _loaded_view(record, engines[record.engine_id])
        if record is None:
            record = _registered_from_catalog(voice_id)
        engine = engines.get(record.engine_id)
        cloning = record.variety == "wav"
        if engine is None or (cloning and not engine.cloning_capable):
            engine = _provision_engine(
                record.language, cloning=cloning, root=root
            )
            engines[engine.id] = engine
        loaded = _materialize(record, engine, root)
        voices[voice_id] = loaded
        store.write(engines, voices)
        return _loaded_view(loaded, engine)


def _registered_from_catalog(voice_id: str) -> VoiceRecord:
    entry = CATALOG[voice_id]
    catalog_voice(voice_id)
    return VoiceRecord(
        id=entry.id,
        variety="built-in",
        state="registered",
        name=entry.name,
        enabled=True,
        language=entry.language,
        engine_id=entry.language,
        provenance=entry.origin_url,
        license_id=entry.license_id,
        commercial_use_allowed=entry.commercial_use_allowed,
        voice_rights=entry.voice_rights,
    )


def _materialize(
    record: VoiceRecord, engine: EngineRecord, root: Path
) -> VoiceRecord:
    """Produce the safetensors asset for one registered voice."""
    destination = root / "voices" / record.language / f"{record.id}.safetensors"
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if record.variety == "built-in":
        fetched = _fetch(embedding_url(record.language, record.id))
        shutil.copyfile(fetched, destination)
    else:
        raise VoiceError(ErrorCode.VOICE_VARIETY_INVALID)
    destination.chmod(0o600)
    return VoiceRecord(
        id=record.id,
        variety=record.variety,
        state="loaded",
        name=record.name,
        enabled=record.enabled,
        language=record.language,
        engine_id=engine.id,
        provenance=record.provenance,
        license_id=record.license_id,
        commercial_use_allowed=record.commercial_use_allowed,
        voice_rights=record.voice_rights,
        source_path=record.source_path,
        source_sha256=record.source_sha256,
        asset_path=str(destination),
        asset_sha256=_sha256(destination),
        compatible_model_revisions=(engine.model_revision,),
    )
```

Task 9 replaces the `raise` branch in `_materialize` with the `wav` and
`pre-compiled` cases.

- [ ] **Step 4: Add the PyYAML dependency**

`yaml` arrives transitively via pocket-tts but is not declared. Add to
`pyproject.toml` `dependencies`: `"pyyaml>=6.0"`. Add to the dev group:
`"types-pyyaml>=6.0"`. Run `uv sync`.

- [ ] **Step 5: Export and run**

Add `load_voice` to `src/kenkui/voices/__init__.py` and `src/kenkui/__init__.py`
exports and `__all__` lists, as in Task 7 Step 4.

Run: `uv run pytest tests/test_load_voice_builtin.py -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A src/kenkui pyproject.toml uv.lock tests/test_load_voice_builtin.py
git commit -m "feat: provision engines and load built-in voices"
```

---

### Task 9: load_voice for wav and pre-compiled voices

**Files:**
- Modify: `src/kenkui/voices/provision.py` (`_materialize`)
- Test: `tests/test_load_voice_local.py`

**Interfaces:**
- Consumes: Task 8's `_materialize`, `_provision_engine`, `_fetch`.
- Produces: `_compile_wav(source: Path, engine: EngineRecord, destination: Path) -> None`, monkeypatched in tests so no model loads.

- [ ] **Step 1: Write the failing test**

Create `tests/test_load_voice_local.py`:

```python
"""Local voices compile at provision time; the render path never encodes audio."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui import ErrorCode, VoiceError, add_voice, load_voice
from kenkui.voices import provision
from kenkui.voices.manifest import ManifestStore

_RIGHTS: dict[str, object] = {
    "name": "My Narrator",
    "language": "english",
    "provenance": "recorded by me",
    "license_id": "proprietary",
    "commercial_use_allowed": True,
    "voice_rights": "owned outright",
}


@pytest.fixture
def stub_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Avoid real downloads and real model loads; record cloning requests."""
    hub = tmp_path / "hub"
    hub.mkdir()
    requested: list[bool] = []

    def fetch(url: str) -> Path:
        path = hub / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.bin"
        if not path.exists():
            path.write_bytes(url.encode() * 8)
        return path

    real_provision = provision._provision_engine  # noqa: SLF001

    def provision_engine(language: str, *, cloning: bool, root: Path) -> object:
        requested.append(cloning)
        return real_provision(language, cloning=cloning, root=root)

    def compile_wav(source: Path, engine: object, destination: Path) -> None:
        destination.write_bytes(b"compiled:" + source.read_bytes())

    monkeypatch.setattr(provision, "_fetch", fetch)
    monkeypatch.setattr(provision, "_provision_engine", provision_engine)
    monkeypatch.setattr(provision, "_compile_wav", compile_wav)
    return requested


def test_wav_compiles_to_safetensors(
    tmp_path: Path, stub_engine: list[bool]
) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.wav"
    source.write_bytes(b"RIFF0000WAVEfmt ")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    voice = load_voice("mine", manifest=manifest)
    assert voice.state == "loaded"
    _, voices = ManifestStore(manifest).read()
    asset = Path(voices["mine"].asset_path or "")
    assert asset.suffix == ".safetensors"
    assert asset.read_bytes().startswith(b"compiled:")


def test_wav_requests_a_cloning_capable_engine(
    tmp_path: Path, stub_engine: list[bool]
) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.wav"
    source.write_bytes(b"RIFF0000WAVEfmt ")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    load_voice("mine", manifest=manifest)
    assert stub_engine == [True]


def test_wav_retains_both_hashes(
    tmp_path: Path, stub_engine: list[bool]
) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.wav"
    source.write_bytes(b"RIFF0000WAVEfmt ")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    load_voice("mine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    record = voices["mine"]
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert record.source_sha256 == expected
    assert record.asset_sha256 != record.source_sha256


def test_pre_compiled_is_copied_without_a_model(
    tmp_path: Path, stub_engine: list[bool]
) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b"already-compiled")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    load_voice("mine", manifest=manifest)
    assert stub_engine == [False]
    _, voices = ManifestStore(manifest).read()
    assert Path(voices["mine"].asset_path or "").read_bytes() == b"already-compiled"


def test_source_deleted_before_load_is_rejected(
    tmp_path: Path, stub_engine: list[bool]
) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.wav"
    source.write_bytes(b"RIFF0000WAVEfmt ")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    source.unlink()
    with pytest.raises(VoiceError) as excinfo:
        load_voice("mine", manifest=manifest)
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


def test_source_modified_after_registration_is_rejected(
    tmp_path: Path, stub_engine: list[bool]
) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.wav"
    source.write_bytes(b"RIFF0000WAVEfmt ")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    source.write_bytes(b"RIFF9999WAVEfmt ")
    with pytest.raises(VoiceError) as excinfo:
        load_voice("mine", manifest=manifest)
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_load_voice_local.py -v --no-cov`
Expected: FAIL with `AttributeError: module 'kenkui.voices.provision' has no attribute '_compile_wav'`.

- [ ] **Step 3: Implement compilation and the remaining varieties**

In `src/kenkui/voices/provision.py`, add `_compile_wav` and replace the
`raise VoiceError(ErrorCode.VOICE_VARIETY_INVALID)` branch in `_materialize`:

```python
def _compile_wav(source: Path, engine: EngineRecord, destination: Path) -> None:
    """Compile one audio prompt into a speaker embedding via pocket-tts."""
    from pocket_tts import export_model_state  # noqa: PLC0415
    from pocket_tts.models.tts_model import TTSModel  # noqa: PLC0415

    if not engine.cloning_capable:
        raise VoiceError(ErrorCode.ENGINE_NOT_CLONING_CAPABLE)
    try:
        model = TTSModel.load_model(config=Path(engine.config_path))
        state = model.get_state_for_audio_prompt(
            audio_conditioning=source, truncate=True
        )
        export_model_state(state, destination)
    except VoiceError:
        raise
    except Exception:
        raise VoiceError(ErrorCode.POCKET_VOICE_LOAD_FAILED) from None
```

Replace the `else` branch of `_materialize`:

```python
    if record.variety == "built-in":
        fetched = _fetch(embedding_url(record.language, record.id))
        shutil.copyfile(fetched, destination)
    else:
        source = Path(record.source_path or "")
        if not source.is_file() or _sha256(source) != record.source_sha256:
            raise VoiceError(ErrorCode.POCKET_VOICE_INVALID)
        if record.variety == "wav":
            _compile_wav(source, engine, destination)
        else:
            shutil.copyfile(source, destination)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_load_voice_local.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/voices/provision.py tests/test_load_voice_local.py
git commit -m "feat: compile wav voices and register pre-compiled voices"
```

---

### Task 10: unload_voice, remove_voice, and engine pruning

**Files:**
- Modify: `src/kenkui/voices/provision.py`, `src/kenkui/voices/__init__.py`, `src/kenkui/__init__.py`
- Test: `tests/test_unload_and_remove_voice.py`

**Interfaces:**
- Consumes: Tasks 7-9.
- Produces: `unload_voice(voice_id: str, *, manifest: Path | None = None) -> Voice`; `remove_voice(voice_id: str, *, manifest: Path | None = None) -> None`; `_prune_engines(engines, voices) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_unload_and_remove_voice.py`:

```python
"""Unloading reclaims assets and prunes orphaned engines; removing forgets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui import ErrorCode, VoiceError, add_voice, load_voice
from kenkui import remove_voice, unload_voice
from kenkui.voices import provision
from kenkui.voices.manifest import ManifestStore

_RIGHTS: dict[str, object] = {
    "name": "My Narrator",
    "language": "english",
    "provenance": "recorded by me",
    "license_id": "proprietary",
    "commercial_use_allowed": True,
    "voice_rights": "owned outright",
}


@pytest.fixture(autouse=True)
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()

    def fetch(url: str) -> Path:
        path = hub / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.bin"
        if not path.exists():
            path.write_bytes(url.encode() * 8)
        return path

    monkeypatch.setattr(provision, "_fetch", fetch)


def test_unload_reverts_to_registered(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    voice = unload_voice("eponine", manifest=manifest)
    assert voice.state == "registered"
    assert voice.engine is None


def test_unload_deletes_the_asset(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    asset = Path(voices["eponine"].asset_path or "")
    unload_voice("eponine", manifest=manifest)
    assert not asset.exists()


def test_unload_prunes_the_orphaned_engine(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    engines, _ = ManifestStore(manifest).read()
    model_root = Path(engines["english"].model_root)
    unload_voice("eponine", manifest=manifest)
    engines_after, _ = ManifestStore(manifest).read()
    assert engines_after == {}
    assert not model_root.exists()


def test_unload_keeps_an_engine_with_a_surviving_sibling(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    load_voice("alba", manifest=manifest)
    unload_voice("eponine", manifest=manifest)
    engines, voices = ManifestStore(manifest).read()
    assert set(engines) == {"english"}
    assert voices["alba"].state == "loaded"


def test_unload_retains_hand_entered_rights(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b"already-compiled")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    load_voice("mine", manifest=manifest)
    unload_voice("mine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    record = voices["mine"]
    assert record.state == "registered"
    assert record.provenance == _RIGHTS["provenance"]
    assert record.voice_rights == _RIGHTS["voice_rights"]
    assert record.source_path == str(source)


def test_unload_is_idempotent_on_a_registered_voice(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b"already-compiled")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    assert unload_voice("mine", manifest=manifest).state == "registered"


def test_remove_deletes_the_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b"already-compiled")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    load_voice("mine", manifest=manifest)
    remove_voice("mine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    assert "mine" not in voices


def test_remove_of_a_builtin_leaves_it_loadable(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    remove_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    assert "eponine" not in voices
    assert load_voice("eponine", manifest=manifest).state == "loaded"


def test_remove_of_unknown_voice_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceError) as excinfo:
        remove_voice("nobody", manifest=tmp_path / "manifest.json")
    assert excinfo.value.code is ErrorCode.VOICE_UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_unload_and_remove_voice.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'unload_voice'`.

- [ ] **Step 3: Implement the lifecycle verbs**

Append to `src/kenkui/voices/provision.py`:

```python
def _registered_view(record: VoiceRecord) -> Voice:
    return Voice(
        id=record.id,
        name=record.name,
        enabled=record.enabled,
        provenance=record.provenance,
        license_id=record.license_id,
        commercial_use_allowed=record.commercial_use_allowed,
        language=record.language,
        variety=record.variety,
        state="registered",
    )


def _unloaded(record: VoiceRecord) -> VoiceRecord:
    return VoiceRecord(
        id=record.id,
        variety=record.variety,
        state="registered",
        name=record.name,
        enabled=record.enabled,
        language=record.language,
        engine_id=record.engine_id,
        provenance=record.provenance,
        license_id=record.license_id,
        commercial_use_allowed=record.commercial_use_allowed,
        voice_rights=record.voice_rights,
        source_path=record.source_path,
        source_sha256=record.source_sha256,
    )


def _prune_engines(
    engines: dict[str, EngineRecord], voices: dict[str, VoiceRecord]
) -> None:
    """Delete engines no loaded voice references, reclaiming their files."""
    referenced = {
        record.engine_id for record in voices.values() if record.state == "loaded"
    }
    for engine_id in list(engines):
        if engine_id in referenced:
            continue
        shutil.rmtree(Path(engines[engine_id].model_root), ignore_errors=True)
        del engines[engine_id]


def _discard_asset(record: VoiceRecord) -> None:
    if record.asset_path is not None:
        Path(record.asset_path).unlink(missing_ok=True)


def unload_voice(voice_id: str, *, manifest: Path | None = None) -> Voice:
    """Delete a voice's asset, keep its rights metadata, and prune its engine."""
    store = _store(manifest)
    with store.lock():
        engines, voices = store.read()
        record = voices.get(voice_id)
        if record is None:
            if voice_id in CATALOG:
                return catalog_voice(voice_id)
            raise VoiceError(ErrorCode.VOICE_UNKNOWN)
        _discard_asset(record)
        reverted = _unloaded(record)
        voices[voice_id] = reverted
        _prune_engines(engines, voices)
        store.write(engines, voices)
        return _registered_view(reverted)


def remove_voice(voice_id: str, *, manifest: Path | None = None) -> None:
    """Delete a voice entry entirely, including hand-entered rights metadata."""
    store = _store(manifest)
    with store.lock():
        engines, voices = store.read()
        record = voices.pop(voice_id, None)
        if record is None:
            if voice_id in CATALOG:
                return
            raise VoiceError(ErrorCode.VOICE_UNKNOWN)
        _discard_asset(record)
        _prune_engines(engines, voices)
        store.write(engines, voices)
```

- [ ] **Step 4: Export and run**

Add `unload_voice` and `remove_voice` to both `__init__.py` export lists and
`__all__` tuples.

Run: `uv run pytest tests/test_unload_and_remove_voice.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/kenkui tests/test_unload_and_remove_voice.py
git commit -m "feat: add unload_voice and remove_voice with engine pruning"
```

---

### Task 11: list_voices

**Files:**
- Modify: `src/kenkui/voices/provision.py`, `src/kenkui/voices/__init__.py`, `src/kenkui/__init__.py`
- Test: `tests/test_list_voices.py`

**Interfaces:**
- Consumes: Tasks 4-10.
- Produces: `list_voices(*, manifest: Path | None = None) -> tuple[Voice, ...]`, sorted by ID, unioning catalog and manifest, reporting `missing` via `stat` only.

- [ ] **Step 1: Write the failing test**

Create `tests/test_list_voices.py`:

```python
"""Listing unions catalog and manifest and never hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui import add_voice, list_voices, load_voice
from kenkui.voices import provision
from kenkui.voices.manifest import ManifestStore
from kenkui.voices.registry import CATALOG

_RIGHTS: dict[str, object] = {
    "name": "My Narrator",
    "language": "english",
    "provenance": "recorded by me",
    "license_id": "proprietary",
    "commercial_use_allowed": True,
    "voice_rights": "owned outright",
}


@pytest.fixture(autouse=True)
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()

    def fetch(url: str) -> Path:
        path = hub / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.bin"
        if not path.exists():
            path.write_bytes(url.encode() * 8)
        return path

    monkeypatch.setattr(provision, "_fetch", fetch)


def test_empty_manifest_lists_the_whole_catalog(tmp_path: Path) -> None:
    voices = list_voices(manifest=tmp_path / "manifest.json")
    assert {v.id for v in voices} == set(CATALOG)
    assert all(v.state == "registered" for v in voices)


def test_results_are_sorted_by_id(tmp_path: Path) -> None:
    voices = list_voices(manifest=tmp_path / "manifest.json")
    assert [v.id for v in voices] == sorted(v.id for v in voices)


def test_loaded_voice_reports_engine_and_size(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    voice = next(v for v in list_voices(manifest=manifest) if v.id == "eponine")
    assert voice.state == "loaded"
    assert voice.engine is not None
    assert voice.engine.size_bytes > 0
    assert voice.asset_bytes is not None


def test_local_voice_appears_alongside_the_catalog(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b"already-compiled")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    ids = {v.id for v in list_voices(manifest=manifest)}
    assert ids == set(CATALOG) | {"mine"}


def test_deleted_asset_is_reported_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    Path(voices["eponine"].asset_path or "").unlink()
    voice = next(v for v in list_voices(manifest=manifest) if v.id == "eponine")
    assert voice.state == "missing"


def test_missing_state_is_never_persisted(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    Path(voices["eponine"].asset_path or "").unlink()
    list_voices(manifest=manifest)
    _, after = ManifestStore(manifest).read()
    assert after["eponine"].state == "loaded"


def test_listing_does_not_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)

    def explode(path: Path) -> str:
        message = "list_voices must not hash"
        raise AssertionError(message)

    monkeypatch.setattr(provision, "_sha256", explode)
    assert list_voices(manifest=manifest)


def test_engine_dedup_across_sibling_voices(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    load_voice("alba", manifest=manifest)
    engines = {
        v.engine for v in list_voices(manifest=manifest) if v.state == "loaded"
    }
    assert len(engines) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_list_voices.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'list_voices'`.

- [ ] **Step 3: Implement list_voices**

Append to `src/kenkui/voices/provision.py`:

```python
def list_voices(*, manifest: Path | None = None) -> tuple[Voice, ...]:
    """Return every known voice, unioning the catalog with the manifest."""
    engines, voices = _store(manifest).read()
    known: dict[str, Voice] = {
        voice_id: catalog_voice(voice_id) for voice_id in CATALOG
    }
    for voice_id, record in voices.items():
        if record.state != "loaded":
            known[voice_id] = _registered_view(record)
            continue
        asset = Path(record.asset_path or "")
        engine = engines.get(record.engine_id)
        if not asset.is_file() or engine is None:
            known[voice_id] = dataclasses.replace(
                _registered_view(record), state="missing"
            )
            continue
        known[voice_id] = _loaded_view(record, engine)
    return tuple(known[key] for key in sorted(known))
```

Add `import dataclasses` to the module imports.

- [ ] **Step 4: Export and run**

Add `list_voices` to both `__init__.py` export lists and `__all__` tuples.

Run: `uv run pytest tests/test_list_voices.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/kenkui tests/test_list_voices.py
git commit -m "feat: add list_voices as the single enumeration primitive"
```

---

### Task 12: Single-branch render path in the Pocket adapter

**Files:**
- Modify: `src/kenkui/_tts/pocket.py:57-92` (config), `:333-375` (`_validate_wav`), `:492-520` (`preflight_pocket`), `:673-762` (engine class)
- Test: `tests/test_pocket_render_path.py`

**Interfaces:**
- Consumes: `PocketEngineConfig` fields written by Task 6's parser.
- Produces: `_validate_safetensors(data: bytes) -> None`; `PocketTTSEngine._voice_state() -> Any` computing conditioning state once per engine. The `PocketEngineConfig` field rename happened in Task 6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pocket_render_path.py`:

```python
"""The render path loads embeddings only, and derives voice state once."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from kenkui import ErrorCode, RenderError
from kenkui._tts import pocket
from kenkui._tts.protocols import SynthesisTask


def _config() -> pocket.PocketEngineConfig:
    return pocket.PocketEngineConfig(
        model_root="/models/english",
        config_path="/models/english/english.yaml",
        model_revision="revision-1",
        package_version="2.1.0",
        files=(pocket.PocketManifestFile("english.yaml", 4, "a" * 64),),
        voice_asset_path="/cache/voices/english/eponine.safetensors",
        voice_asset_sha256="b" * 64,
        voice_variety="built-in",
        cloning_capable=False,
        voice_provenance="kyutai catalog",
        voice_license_id="CC-BY-4.0",
        voice_rights="review required",
        commercial_use_allowed=False,
        sample_rate_hz=24000,
    )


def test_semantic_material_includes_variety_and_asset_hash() -> None:
    material = _config().semantic_material()
    assert material["voice_variety"] == "built-in"
    assert material["voice_asset_sha256"] == "b" * 64
    assert "voice_prompt_sha256" not in material


def test_semantic_material_changes_with_variety() -> None:
    base = _config()
    other = dataclasses.replace(base, voice_variety="pre-compiled")
    assert base.semantic_material() != other.semantic_material()


def test_safetensors_header_is_validated() -> None:
    pocket._validate_safetensors(  # noqa: SLF001
        (8).to_bytes(8, "little") + b'{"a":{}}'
    )


def test_truncated_safetensors_is_rejected() -> None:
    with pytest.raises(RenderError) as excinfo:
        pocket._validate_safetensors(b"\x08")  # noqa: SLF001
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


def test_safetensors_with_oversized_header_is_rejected() -> None:
    with pytest.raises(RenderError) as excinfo:
        pocket._validate_safetensors(  # noqa: SLF001
            (1 << 40).to_bytes(8, "little") + b"{}"
        )
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


class _Model:
    """Counts conditioning derivations and returns a fixed mono tensor."""

    sample_rate = 24000
    device = "cpu"

    def __init__(self) -> None:
        self.state_calls = 0

    def get_state_for_audio_prompt(self, conditioning: Any) -> object:
        self.state_calls += 1
        return {"conditioning": conditioning}

    def generate_audio(self, state: object, text: str) -> object:
        raise NotImplementedError


def test_voice_state_is_derived_once_per_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = pocket.PocketTTSEngine.__new__(pocket.PocketTTSEngine)
    model = _Model()
    object.__setattr__(engine, "_model", model)
    object.__setattr__(engine, "_config", _config())
    object.__setattr__(engine, "_state", None)
    first = engine._voice_state()  # noqa: SLF001
    second = engine._voice_state()  # noqa: SLF001
    assert first is second
    assert model.state_calls == 1


def test_voice_state_is_passed_a_path_not_a_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A str would let pocket-tts call download_if_necessary; a Path cannot."""
    from pathlib import Path

    engine = pocket.PocketTTSEngine.__new__(pocket.PocketTTSEngine)
    model = _Model()
    object.__setattr__(engine, "_model", model)
    object.__setattr__(engine, "_config", _config())
    object.__setattr__(engine, "_state", None)
    state = engine._voice_state()  # noqa: SLF001
    assert isinstance(state["conditioning"], Path)  # type: ignore[index]


def test_task_type_is_unchanged() -> None:
    task = SynthesisTask(
        segment_id="s1",
        chapter_id="c1",
        text="hello",
        sample_rate_hz=24000,
        channels=1,
        max_output_bytes=1024,
    )
    assert task.segment_id == "s1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pocket_render_path.py -v --no-cov`
Expected: FAIL with `AttributeError: module 'kenkui._tts.pocket' has no attribute
'_validate_safetensors'`. The `semantic_material` assertions already pass, since
Task 6 renamed the fields.

- [ ] **Step 3: Add the safetensors validator**

Add beside `_validate_wav` (which stays, for `add_voice` source checking):

```python
_MAX_SAFETENSORS_HEADER_BYTES: Final = 16 * 1024 * 1024


def _validate_safetensors(data: bytes) -> None:
    """Validate a bounded safetensors header without importing torch."""
    header_length = 8
    if len(data) < header_length:
        raise RenderError(ErrorCode.POCKET_VOICE_INVALID)
    declared = int.from_bytes(data[:header_length], "little")
    if (
        declared <= 0
        or declared > _MAX_SAFETENSORS_HEADER_BYTES
        or declared > len(data) - header_length
    ):
        raise RenderError(ErrorCode.POCKET_VOICE_INVALID)
    header = data[header_length : header_length + declared]
    if not header.lstrip().startswith(b"{"):
        raise RenderError(ErrorCode.POCKET_VOICE_INVALID)
```

In `preflight_pocket`, replace the call to `_validate_wav` on the voice asset
with a dispatch that reads the asset at `config.voice_asset_path`, verifies its
digest against `config.voice_asset_sha256`, and calls `_validate_safetensors`.
Every voice asset reaching the renderer is a safetensors embedding regardless
of `voice_variety`, per spec decision 5.

- [ ] **Step 4: Hoist the conditioning state**

In `PocketTTSEngine.__init__`, add `self._state: object | None = None` before
the `try`. Add the method and rewrite `synthesize`:

```python
    def _voice_state(self) -> Any:
        """Derive the conditioning state once and reuse it for every segment."""
        if self._state is None:
            try:
                self._state = self._model.get_state_for_audio_prompt(
                    Path(self._config.voice_asset_path)
                )
            except Exception:
                self.close()
                raise VoiceError(ErrorCode.POCKET_VOICE_LOAD_FAILED) from None
        return self._state

    def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
        """Render the task's exact text as PCM from a precompiled embedding."""
        state = self._voice_state()
        try:
            output = self._model.generate_audio(state, task.text)
            audio = tensor_to_pcm(
                output, self._tensor_type, task, self._config.sample_rate_hz
            )
        except RenderError:
            self.close()
            raise
        except Exception:
            self.close()
            raise RenderError(ErrorCode.POCKET_INFERENCE_FAILED) from None
        if not self._reusable:
            self.close()
        return audio
```

Note the argument is a `Path`, never a `str`. `get_state_for_audio_prompt`
calls `download_if_necessary` only on `str` input, so a `Path` cannot reach the
network even before `_deny_remote` intervenes.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_pocket_render_path.py -v --no-cov && uv run pytest`
Expected: PASS. Update any existing test constructing `PocketEngineConfig` with
the old field names.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_tts/pocket.py tests/
git commit -m "feat: render from embeddings only and derive voice state once"
```

---

### Task 13: Pipeline pre-flight before spawning workers

**Files:**
- Modify: `src/kenkui/pipeline.py:212` (inside `write_m4b`, before `execute_sequential`)
- Test: `tests/test_write_preflight.py`

**Interfaces:**
- Consumes: Task 6's `resolve_manifest_path`, Task 2's error codes.
- Produces: `write()` raising `voice_not_provisioned` or `voice_unknown` before any worker process is created.

- [ ] **Step 1: Write the failing test**

Create `tests/test_write_preflight.py`:

```python
"""Write fails fast on an unprovisioned voice, before spawning workers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kenkui as kk
from kenkui import ErrorCode, VoiceError
from kenkui._execution import coordinator


@pytest.fixture(autouse=True)
def no_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        message = "write must fail before execution starts"
        raise AssertionError(message)

    monkeypatch.setattr(coordinator, "execute_sequential", explode)


def _manifest(tmp_path: Path, state: str) -> Path:
    path = tmp_path / "manifest.json"
    voice = {
        "variety": "wav",
        "state": state,
        "name": "Mine",
        "enabled": True,
        "language": "english",
        "engine_id": "english",
        "provenance": "mine",
        "license_id": "proprietary",
        "commercial_use_allowed": True,
        "voice_rights": "owned",
        "source_path": str(tmp_path / "mine.wav"),
        "source_sha256": "c" * 64,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "kenkui-pocket-production-v2",
                "engines": {},
                "voices": {"mine": voice},
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def test_registered_voice_fails_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, epub_fixture: Path
) -> None:
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(_manifest(tmp_path, "registered")))
    pipeline = kk.book(epub_fixture).normalize_text().assign_voice("mine").tts()
    with pytest.raises(VoiceError) as excinfo:
        pipeline.write(tmp_path / "out.m4b")
    assert excinfo.value.code is ErrorCode.VOICE_NOT_PROVISIONED


def test_unknown_voice_fails_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, epub_fixture: Path
) -> None:
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(_manifest(tmp_path, "registered")))
    pipeline = kk.book(epub_fixture).normalize_text().assign_voice("nobody").tts()
    with pytest.raises(VoiceError) as excinfo:
        pipeline.write(tmp_path / "out.m4b")
    assert excinfo.value.code is ErrorCode.VOICE_UNRESOLVED
```

Reuse the existing EPUB fixture. If `tests/conftest.py` has no `epub_fixture`,
find the fixture helper the current suite uses for `kk.book(...)` and use that
name instead; do not create a second EPUB builder.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_write_preflight.py -v --no-cov`
Expected: FAIL — `execute_sequential` is reached, tripping the `AssertionError`.

- [ ] **Step 3: Move binding resolution ahead of execution**

In `src/kenkui/pipeline.py`, `write_m4b` already resolves bindings in the
`execute_sequential(...)` call arguments. Python evaluates those before the
call, so `_resolved_execution_bindings` already raises first. Make it explicit
and testable by binding to a local before the call:

```python
        bindings = _resolved_execution_bindings(self._assigned_voice_id())
        return execute_sequential(
            self,
            output_path,
            bindings=bindings,
            on_event=on_event,
            cancel=cancel,
            workers=workers,
            overwrite=overwrite,
        )
```

The parser from Task 6 raises `VOICE_NOT_PROVISIONED` for a `registered` voice
and `VOICE_UNRESOLVED` for an unknown ID, so no new logic is needed here — only
the ordering guarantee the test pins down.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_write_preflight.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/pipeline.py tests/test_write_preflight.py
git commit -m "feat: fail write before worker spawn on unprovisioned voices"
```

---

### Task 14: Documentation

**Files:**
- Modify: `docs/models-and-voices.md`, `docs/usage.md:120-127`, `docs/installation.md`, `docs/pocket-tts-adapter.md`, `README.md`

**Interfaces:**
- Consumes: the finished public API.
- Produces: no code.

- [ ] **Step 1: Rewrite the voice and rights documentation**

In `docs/models-and-voices.md`:

- State the gating correctly: `kyutai/pocket-tts` (voice cloning) is gated;
  `kyutai/pocket-tts-without-voice-cloning` (embeddings) is not. **A WAV prompt
  requires the gated weights; a built-in or pre-compiled embedding does not.**
- Replace the blanket EARS caveat with a per-voice rights table generated from
  `registry.CATALOG`, listing ID, language, origin, license ID, and
  commercial-use decision. Keep the EARS caveat text on `jean` and the Expresso
  caveat on `cosette`, where they are accurate.
- Replace "no project-owned redistributable voice fixture has passed the real
  gate" with a statement scoped to WAV cloning voices only.
- Document trust-on-first-use as the accepted root of trust, per spec section 7.
- Keep the "operational controls, not legal advice" framing verbatim.

- [ ] **Step 2: Rewrite the activation section**

In `docs/usage.md`, replace "## Current production gate" with a section
covering `load_voice`, the managed default manifest, `KENKUI_POCKET_MANIFEST`
as an operator override, and the five-verb lifecycle. Include:

```python
import kenkui as kk

kk.load_voice("eponine")

result = (
    kk.book("book.epub")
    .normalize_text()
    .assign_voice("eponine")
    .tts()
    .write("book.m4b")
)
```

- [ ] **Step 3: Update installation and adapter docs**

In `docs/installation.md`, state that `pocket-tts` is required and pulls torch,
so installs are large. Note that `kenkui[pocket]` still resolves.

In `docs/pocket-tts-adapter.md`, document the single-branch render path, that
conditioning state is derived once per engine, and that provisioning is the
only network-touching code.

- [ ] **Step 4: Update the README example**

Add the `kk.load_voice("eponine")` line to the README's usage example.

- [ ] **Step 5: Verify the docs build**

Run: `uv run mkdocs build --strict`
Expected: PASS with no warnings.

- [ ] **Step 6: Commit**

```bash
git add docs README.md
git commit -m "docs: document voice provisioning and correct the gating story"
```

---

### Task 15: Opt-in real acceptance test

**Files:**
- Create: `tests/test_voice_provisioning_real.py`
- Modify: `docs/superpowers/specs/2026-08-19-voice-provisioning-design.md` (record the section 13 result)

**Interfaces:**
- Consumes: the complete public API.
- Produces: the empirical answer to spec section 13.

This is the only test that downloads real assets and runs real inference. It is
skipped unless `KENKUI_RUN_PROVISIONING_REAL=1`, matching the existing
`pocket_real` marker convention.

- [ ] **Step 1: Write the opt-in test**

Create `tests/test_voice_provisioning_real.py`:

```python
"""Opt-in end-to-end provisioning and rendering against real assets."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import kenkui as kk

pytestmark = pytest.mark.pocket_real

_ENABLED = os.environ.get("KENKUI_RUN_PROVISIONING_REAL") == "1"
_REASON = "set KENKUI_RUN_PROVISIONING_REAL=1 to download real assets"


@pytest.mark.skipif(not _ENABLED, reason=_REASON)
def test_load_builtin_voice_downloads_and_verifies(tmp_path: Path) -> None:
    voice = kk.load_voice("eponine", manifest=tmp_path / "manifest.json")
    assert voice.state == "loaded"
    assert voice.engine is not None
    assert voice.engine.cloning_capable is False
    assert voice.asset_bytes is not None and voice.asset_bytes > 1_000_000


@pytest.mark.skipif(not _ENABLED, reason=_REASON)
def test_second_load_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    kk.load_voice("eponine", manifest=manifest)

    from kenkui.voices import provision

    def explode(url: str) -> Path:
        message = "an idempotent load must not fetch"
        raise AssertionError(message)

    monkeypatch.setattr(provision, "_fetch", explode)
    assert kk.load_voice("eponine", manifest=manifest).state == "loaded"


@pytest.mark.skipif(not _ENABLED, reason=_REASON)
def test_render_a_real_m4b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    kk.load_voice("eponine", manifest=manifest)
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(manifest))
    source = Path(__file__).parent / "fixtures" / "minimal.epub"
    output = tmp_path / "out.m4b"
    result = (
        kk.book(source)
        .normalize_text()
        .assign_voice("eponine")
        .tts()
        .write(output)
    )
    assert Path(result.output).is_file()
    assert Path(result.output).stat().st_size > 0
```

Use whatever minimal EPUB fixture the existing suite already provides; adjust
the `source` path to match rather than adding a new fixture.

- [ ] **Step 2: Run it with real assets**

Run: `KENKUI_RUN_PROVISIONING_REAL=1 uv run pytest tests/test_voice_provisioning_real.py -v --no-cov -m pocket_real`
Expected: PASS. First run downloads roughly 360 MB and takes several minutes.

This is the first real Pocket inference in the project. Expect to debug the
adapter's strict checks — exact `sample_rate_hz` equality, `str(model.device)`
equality, and full file size and hash scanning — against real assets, per spec
section 13.

- [ ] **Step 3: Answer the spec's open question**

Add to the same file, and run it:

```python
@pytest.mark.skipif(not _ENABLED, reason=_REASON)
def test_gated_compiled_embedding_renders_on_ungated_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec section 13: does a cloning-compiled embedding load ungated?

    Requires accepted kyutai/pocket-tts terms and `hf auth login`.
    """
    manifest = tmp_path / "manifest.json"
    prompt = Path(__file__).parent / "fixtures" / "voice_prompt.wav"
    if not prompt.is_file():
        pytest.skip("no local WAV prompt available")
    kk.add_voice(
        prompt,
        voice_id="local-test",
        name="Local Test",
        language="english",
        provenance="local test fixture",
        license_id="proprietary",
        commercial_use_allowed=False,
        voice_rights="test only",
        manifest=manifest,
    )
    voice = kk.load_voice("local-test", manifest=manifest)
    assert voice.engine is not None
    assert voice.engine.cloning_capable is True

    from kenkui.voices.manifest import ManifestStore

    engines, voices = ManifestStore(manifest).read()
    ungated = kk.load_voice("eponine", manifest=manifest)
    assert ungated.engine is not None
    compiled = voices["local-test"]
    assert compiled.asset_path is not None
    assert Path(compiled.asset_path).is_file()
```

Run: `KENKUI_RUN_PROVISIONING_REAL=1 uv run pytest tests/test_voice_provisioning_real.py -v --no-cov -m pocket_real`

- [ ] **Step 4: Record the result in the spec**

Edit section 13 of
`docs/superpowers/specs/2026-08-19-voice-provisioning-design.md` and replace
"expected to work but unverified" with the observed outcome and the date.

If compatible: note that a compiled voice may list both engine revisions in
`compatible_model_revisions`, and open a follow-up to relax the pinning.

If incompatible: note that `wav` voices require cloning-capable weights at
render time, and that `production.py`'s existing revision check already
enforces this correctly with no code change needed.

- [ ] **Step 5: Full verification**

Run:
```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run mkdocs build --strict
```
Expected: all PASS, coverage at or above 90%.

- [ ] **Step 6: Commit**

```bash
git add tests/test_voice_provisioning_real.py docs/superpowers/specs/
git commit -m "test: add opt-in real provisioning acceptance and record findings"
```
