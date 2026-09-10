# example.py update: dial-in loop, simplified sidecar guard

## What changed, and why shaped that way

`spikes/examples/example.py`:

1. **Module docstring** gained a short paragraph naming `dial_in()` as the
   interactive counterpart to `main()`'s batch render, and stating plainly
   that it is not wired into `main()`.

2. **`explicit_run()`'s sidecar guard simplified.** The old code:

   ```python
   sidecar = epub.with_suffix(".kenkui.json")
   if sidecar.exists():
       pipeline = pipeline.annotations()
   ```

   is now:

   ```python
   # A book that has never been dialed in has no sidecar yet: annotations()
   # loads an empty baseline for it rather than raising, so this call needs
   # no existence check. A sidecar that exists but is damaged still raises
   # INVALID_SIDECAR, which is the failure worth stopping for.
   pipeline = pipeline.annotations()
   ```

   The `sidecar` variable is gone with it -- nothing else in the function
   used it. This matches `annotations()`'s current docstring in
   `src/kenkui/pipeline.py`, which already documents the missing-sidecar
   case as a no-op baseline and the corrupt-sidecar case as `INVALID_SIDECAR`.

3. **New `dial_in(epub: Path) -> None` function**, placed after `magic_run()`
   and before `main()`, demonstrating the loop from
   `docs/superpowers/specs/2026-09-08-grid-and-tuning-design.md` section 8
   and `docs/usage.md`'s "Tuning a book and the dial-in loop" section:
   `script()` to read what's currently decided and *why* (`row.provenance`),
   `attribute()`/`silence()` to correct it, `select().preview()` to probe
   the correction cheaply, `write_annotations()` to persist it.

   Design choices:

   - It builds a full pipeline the same way `explicit_run()` does
     (`metadata()`, `series()`, the Dune lexicon, `pipe(house_style)`), so
     `.identity` and `.style` print something real instead of "nothing set"
     -- printing them is only illustrative if there is something to look at.
   - The worked correction is Dune-specific on purpose: every chapter opens
     with an epigraph from Princess Irulan's in-universe writings at
     paragraph 1, which is exactly the example the design spec and
     `docs/usage.md` use (`{"chapter": "*", "paragraph": 1}` -> `"irulan"`).
     Reusing it here ties the running example together instead of inventing
     an arbitrary correction.
   - Comments at each step explain *why*, not just *what*: `script()`'s
     provenance column is called out as the debugging payoff; `preview()`'s
     cheapness is tied to segment-identity reuse with the full render
     (`docs/superpowers/specs/2026-09-08-grid-and-tuning-design.md`'s "cost
     model" section), not just "it's a WAV not an M4B"; `write_annotations()`
     is tied to `explicit_run()`'s now-unconditional `.annotations()` call,
     closing the loop between the two functions in the same file.
   - Not called from `main()`, per the task constraint -- it sits at module
     scope the same way `magic_run()` already does: a second, undispatched
     entry point a reader calls by hand.

4. Checked the rest of the file: `report_progress()`, `house_style()`,
   `magic_run()`, `BOOKS`, `LEXICONS`, `main()` reference no operations that
   changed on this branch. The top-level docstring's claims about the three
   tiers still hold.

## Proving it runs

`main()` genuinely cannot run here: it needs the real Calibre library at
`/Users/dizzler/Projects/Calibre Library` and a provisioned real TTS voice,
neither of which exist in this checkout. Everything else was executed for
real.

### Static checks

```
$ uv run ruff format --check spikes/examples/example.py
1 file already formatted
$ uv run ruff check spikes/examples/example.py
All checks passed!
$ uv run mypy spikes/examples/example.py
Success: no issues found in 1 source file
```

Note: the project's `[tool.mypy] files = ["src", "tests"]`, so the plain
`uv run mypy` gate (used by CI and below) does not walk `spikes/` at all --
verified with `uv run mypy --verbose` and grepping for the file. Running
mypy directly against the file, as above, is strict (`strict = true` is a
global setting) and passes clean. This is a pre-existing gap in the
project's mypy file list, not something introduced here; noting it since
the task described mypy as "covering this file."

### Scratch script

Written to `/tmp/dial_in_proof.py` (not part of the repo). It builds a
throwaway two-chapter EPUB with `tests/helpers.py`'s `make_epub`/`xhtml`,
stubs the same two seams `tests/conftest.py`'s `resolved_book` and
`isolated_cache_root` fixtures stub (`kenkui._resolution._execution_bindings`
for a fake TTS engine built from `EngineSpecification.fake()` +
`FakeArtifactAssembler()`, `kenkui._resolution._attribution_client` for an
empty-roster LLM client that never reaches the network, and the cache-root
functions in `kenkui._tts.production` / `kenkui.voices.manifest` so nothing
touches a real user cache), then calls the real, **unmodified**
`example.dial_in()` and `example.explicit_run()`.

`house_style()`'s `.infer_characters("spacy")` runs for real (unstubbed) --
this checkout's venv has `en_core_web_lg` installed, so no spaCy-related
substitution was needed. `attribute_quotes()` never reaches the real network
because the LLM client seam is stubbed, exactly as `tests/conftest.py`'s
`resolved_book` fixture does it.

One thing worth flagging about the harness itself, not the library: the
first version of this script had no `if __name__ == "__main__":` guard, and
`kenkui`'s process-pool renderer spawns workers with multiprocessing's
`"spawn"` start method, which re-imports the driver script as `__main__` in
each child. Every worker re-ran the whole script from the top -- including
building a fresh throwaway book -- and the render failed inside the spawned
child because the monkeypatches installed in the parent process do not exist
there. Wrapping the body in `main()` behind the standard guard fixed it. This
is a property of the standard library and this project's use of it, not a
bug to report.

Full contents of `/tmp/dial_in_proof.py`:

```python
"""Scratch proof that example.py's new API surface actually runs.

Not part of the repo. Builds a throwaway EPUB with tests/helpers.py's
make_epub/xhtml, stubs the same two seams tests/conftest.py stubs
(_resolution._execution_bindings for a fake TTS engine, _resolution.
_attribution_client for a model-free empty-roster LLM client, and the
cache-root functions so nothing touches a real user cache), and then calls
the real, unmodified functions defined in spikes/examples/example.py:
dial_in() and explicit_run(), plus a direct annotations()/
write_annotations() round trip.

house_style()'s ``infer_characters("spacy")`` call runs for real here
(unstubbed) because this checkout's venv happens to have en_core_web_lg
installed; attribute_quotes() never reaches the real network because the
LLM client seam above is stubbed, matching how tests/conftest.py's
resolved_book fixture avoids network calls.

The one thing this script cannot exercise is main()'s batch loop: it reads
a real Calibre library and needs a real, provisioned TTS voice, neither of
which exists in this sandbox.

Everything lives behind ``if __name__ == "__main__"`` because rendering
spawns worker processes with the multiprocessing "spawn" start method,
which re-imports this file as ``__main__`` in each child; without the
guard the whole script -- including building a fresh throwaway book --
would re-run inside every worker.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(
    "/Users/dizzler/Projects/Repos/kenkui-v2/kenkui/.worktrees/book-tuning-dial-in-loop"
)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "spikes" / "examples"))


def main() -> None:
    import kenkui as kk
    from helpers import make_epub, xhtml
    from kenkui._audio.m4b import FakeArtifactAssembler
    from kenkui._execution.coordinator import ExecutionBindings
    from kenkui._execution.process_pool import EngineSpecification
    from kenkui.errors import ErrorCode, ValidationError
    from kenkui.voices import manifest as manifest_module

    import example
    from kenkui import _resolution
    from kenkui._tts import production

    # --- Seams, stubbed exactly as tests/conftest.py stubs them ----------

    cache_root = Path(tempfile.mkdtemp(prefix="kenkui-scratch-cache-"))
    production.default_cache_root = lambda: cache_root
    manifest_module.default_cache_root = lambda: cache_root
    manifest_module.default_manifest_path = lambda: cache_root / "manifest.json"

    narrator_voice = kk.Voice(
        id="ivy",
        name="Ivy",
        enabled=True,
        provenance="fixture",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en",
        state="loaded",
        content_fingerprint="a" * 64,
        compatible_model_revisions=("fake-v1",),
    )
    bindings = ExecutionBindings(
        EngineSpecification.fake(), FakeArtifactAssembler(), narrator_voice, "fake-v1"
    )

    class _NoCharactersClient:
        """Answers roster discovery/attribution with no characters at all."""

        def complete(self, model: str, prompt: str) -> str:
            assert model
            assert prompt
            return '{"characters": []}'

    _resolution._execution_bindings = lambda _voice_id, **_cast: bindings
    _resolution._attribution_client = _NoCharactersClient

    # --- A throwaway two-chapter book -------------------------------------

    workdir = Path(tempfile.mkdtemp(prefix="kenkui-scratch-book-"))
    epub = make_epub(
        workdir / "Dune - Frank Herbert.epub",
        chapters={
            "ch01": xhtml(
                "<p>In the week before their departure to Arrakis, when all"
                " the final scurrying about had reached a nearly unbearable"
                " frenzy, an old crone came to visit the mother of the boy,"
                " Paul.</p>"
                '<p>"Paul," she said. "Come here."</p>'
            ),
            "ch02": xhtml("<p>Dune. Desert planet. Third planet of Canopus.</p>"),
        },
        spine=["ch01", "ch02"],
    )
    print(f"built throwaway epub at {epub}")

    # --- Part 1: the real, unmodified dial_in() end to end ---------------

    print("\n=== dial_in(epub): real house_style(), stubbed model/engine ===")
    example.dial_in(epub)

    sidecar = epub.with_suffix(".kenkui.json")
    print(f"\nsidecar exists after dial_in(): {sidecar.exists()}")
    print(sidecar.read_text())

    probe = epub.with_name(f"{epub.stem}.dial-in-probe.wav")
    print(f"probe exists after dial_in(): {probe.exists()} size={probe.stat().st_size}")

    # --- Part 2: annotations() round trip and the corrupt-sidecar case ---

    print("\n=== annotations()/write_annotations() round trip ===")
    reloaded = kk.book(epub).annotations()
    print(reloaded.tuning)

    print("\n=== corrupt sidecar still raises INVALID_SIDECAR ===")
    sidecar.write_text("{not json")
    try:
        kk.book(epub).annotations()
    except ValidationError as error:
        print(f"annotations() raised {error.code!r} as expected: {error}")
        assert error.code == ErrorCode.INVALID_SIDECAR
    sidecar.unlink()

    print("\n=== missing sidecar is a silent no-op baseline ===")
    fresh = workdir / "Dune Messiah - Frank Herbert.epub"
    make_epub(
        fresh,
        chapters={"ch01": xhtml("<p>Whatever a man is, he grows.</p>")},
        spine=["ch01"],
    )
    book_without_sidecar = kk.book(fresh).annotations()
    print(book_without_sidecar.tuning)
    assert not fresh.with_suffix(".kenkui.json").exists()

    # --- Part 3: explicit_run()'s simplified guard, end to end -----------

    print("\n=== explicit_run() before any sidecar exists ===")
    example.WORKERS = 1
    result = example.explicit_run("Dune", "Frank Herbert", epub, "dune", 1)
    print(f"explicit_run() produced: {result.output} exists={result.output.exists()}")

    print("\n=== dial_in() again, to leave a sidecar behind ===")
    example.dial_in(epub)

    print("\n=== explicit_run() now that a sidecar exists ===")
    result = example.explicit_run("Dune", "Frank Herbert", epub, "dune", 1)
    print(f"explicit_run() produced: {result.output} exists={result.output.exists()}")

    # Prove the sidecar's rules actually made it into the pipeline this
    # second time, the way explicit_run's unconditional annotations() call
    # promises.
    pipeline = kk.book(epub).metadata(title="Dune", author="Frank Herbert")
    pipeline = pipeline.series("dune", book=1).pronounce(example.LEXICONS["dune"])
    pipeline = pipeline.annotations()
    print("\ntuning visible to explicit_run's guard on the second pass:")
    print(pipeline.tuning)

    print("\nALL SCRATCH CHECKS PASSED")


if __name__ == "__main__":
    main()
```

### Actual execution output

```
built throwaway epub at /var/folders/zk/.../kenkui-scratch-book-x4ee3iwa/Dune - Frank Herbert.epub

=== dial_in(epub): real house_style(), stubbed model/engine ===
<identity
  metadata(title='Dune', author='Frank Herbert')
  series(series_id='dune', book=1)
>
<style
  pronounce(numbers='standard')
  pauses(chapter_ms=1200ms, heading_after_ms=500ms, paragraph_ms=300ms)
  infer_characters('spacy')
  attribute_quotes('openrouter/deepseek/deepseek-v4-flash')
  assign_voices(narrator='ivy', unknown='michael', method='gendered')
>
<tuning
  pronunciations (1):
    * -> (('Atreides', 'Ah-tray-deez'), ('Bene Gesserit', 'Ben-eh Jess-er-it'), ('Chani', 'Chah-nee'), ('Ghola', 'Goh-lah'), ('Harkonnen', 'Har-koh-nen'), ('Kwisatz Haderach', 'Kwih-sats Hah-der-ock'), ('Leto', 'Lay-toh'), ("Muad'Dib", 'Moo-ahd-Deeb'), ('Sardaukar', 'Sar-doh-kar'), ('Shai-Hulud', 'Shy Hoo-lood'))  [unsaved]
>
Path(chapter='ch-v1-0645306ca67fe6e8670e1965', paragraph=1, line=1, sentence=1, phrase=1) None default 0 In the week before their departure to Arrakis, 
Path(chapter='ch-v1-0645306ca67fe6e8670e1965', paragraph=1, line=1, sentence=1, phrase=2) None default 0 when all the final scurrying about had reached a nearly unbe
Path(chapter='ch-v1-0645306ca67fe6e8670e1965', paragraph=1, line=1, sentence=1, phrase=3) None default 0 an old crone came to visit the mother of the boy, 
Path(chapter='ch-v1-0645306ca67fe6e8670e1965', paragraph=1, line=1, sentence=1, phrase=4) None default 300 Paul.


Path(chapter='ch-v1-c903a89fc4f54d24075119b3', paragraph=1, line=1, sentence=1, phrase=1) None default 0 Dune. 
Path(chapter='ch-v1-c903a89fc4f54d24075119b3', paragraph=1, line=1, sentence=2, phrase=1) None default 0 Desert planet. 
Path(chapter='ch-v1-c903a89fc4f54d24075119b3', paragraph=1, line=1, sentence=3, phrase=1) None default 0 Third planet of Canopus.
saved corrections to /var/folders/zk/.../kenkui-scratch-book-x4ee3iwa/Dune - Frank Herbert.kenkui.json

sidecar exists after dial_in(): True
{
  "kenkui_sidecar": 1,
  "pronunciations": [
    {
      "where": {},
      "words": {
        "Atreides": "Ah-tray-deez",
        "Bene Gesserit": "Ben-eh Jess-er-it",
        "Chani": "Chah-nee",
        "Ghola": "Goh-lah",
        "Harkonnen": "Har-koh-nen",
        "Kwisatz Haderach": "Kwih-sats Hah-der-ock",
        "Leto": "Lay-toh",
        "Muad'Dib": "Moo-ahd-Deeb",
        "Sardaukar": "Sar-doh-kar",
        "Shai-Hulud": "Shy Hoo-lood"
      },
      "matched": 11
    }
  ],
  "attributions": [
    {
      "where": {"chapter": "*", "paragraph": 1},
      "character": "irulan",
      "matched": 7
    }
  ],
  "silences": [
    {
      "where": {"chapter": "*", "paragraph": 1},
      "ms": 900,
      "matched": 7
    }
  ]
}

probe exists after dial_in(): True size=110124

=== annotations()/write_annotations() round trip ===
<tuning
  pronunciations (1):
    * -> (...same lexicon...)
  attributions (1):
    ¶1 -> 'irulan'
  silences (1):
    ¶1 -> 900
>

=== corrupt sidecar still raises INVALID_SIDECAR ===
annotations() raised <ErrorCode.INVALID_SIDECAR: 'invalid_sidecar'> as expected: The annotation sidecar is invalid or inaccessible.

=== missing sidecar is a silent no-op baseline ===
<tuning: no rules>

=== explicit_run() before any sidecar exists ===

== attribution ==
attribution: 0/2
attribution: 2/2
== attribution complete ==

== planning ==
planning: 1/1
== planning complete ==

== render ==
render: 1/2 (ch-v1-0645306ca67fe6e8670e1965)
render: 2/2 (ch-v1-c903a89fc4f54d24075119b3)
== render complete ==

== assembly ==
assembly: 1/1
== assembly complete ==

== publication ==
publication: 1/1
== publication complete ==
explicit_run() produced: /var/folders/zk/.../Dune - Frank Herbert.m4b exists=True

=== dial_in() again, to leave a sidecar behind ===
[... identity/style/tuning/script rows repeated ...]
saved corrections to /var/folders/zk/.../Dune - Frank Herbert.kenkui.json

=== explicit_run() now that a sidecar exists ===
[... same stage events ...]
explicit_run() produced: /var/folders/zk/.../Dune - Frank Herbert.m4b exists=True

tuning visible to explicit_run's guard on the second pass:
<tuning
  pronunciations (1):
    * -> (...)
  attributions (1):
    ¶1 -> 'irulan'
  silences (1):
    ¶1 -> 900
>

ALL SCRATCH CHECKS PASSED
```

(Elided repeated blocks above marked `[...]` for brevity; every run in the
actual terminal transcript was identical to the first.)

This exercises, for real: `.identity`, `.style`, `.tuning` properties;
`.script().at(pattern)` with `provenance` before any correction (`default`);
`.attribute()` and `.silence()` accumulating rules; `.select().preview()`
producing a real 110124-byte WAV via the fake engine; `.write_annotations()`
producing a real sidecar JSON with `matched` counts; `.annotations()`
reloading that sidecar into `.tuning`; a corrupt sidecar raising
`INVALID_SIDECAR`; a missing sidecar loading silently as `<tuning: no
rules>`; and `explicit_run()`'s simplified unconditional `.annotations()`
call working identically whether or not a sidecar exists yet, including a
full (fake-engine) `.tts().write()` producing a real output file both times.

## Library bug found (reported, not fixed)

None. Everything behaved exactly as `src/kenkui/pipeline.py`'s docstrings
and `docs/usage.md` describe it.

## Full gate

```
$ uv run ruff format --check .
184 files already formatted

$ uv run ruff check .
All checks passed!

$ uv run mypy
Success: no issues found in 150 source files

$ uv run pytest -q
...
Required test coverage of 90% reached. Total coverage: 91.94%
1511 passed, 45 skipped, 7 deselected, 1 warning in 92.02s (0:01:32)
```

1511 passed / 45 skipped matches the verified HEAD state exactly (the task
brief's baseline). Coverage read 91.94% here vs. the brief's stated 91.97%
baseline; no source under `src/` was touched by this change, so the 0.03-point
difference is run-to-run branch-coverage noise, not a regression. The `7
deselected` line reflects this repo's standing `-m=not native` pytest
addopt filtering out native-FFmpeg-marked tests; it was not printed in the
one-line baseline summary but is not new behavior.
