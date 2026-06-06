"""Tests for _split_paragraph_by_quotes and _merge_consecutive_segments.

These functions live in kenkui.nlp and are the core of per-quotation voice
assignment.  Tests are written before the implementation (TDD).
"""

from __future__ import annotations

from kenkui.models import Segment

# ---------------------------------------------------------------------------
# Import helpers under test (will fail until implemented)
# ---------------------------------------------------------------------------


def _import_helpers():
    from kenkui.nlp import _merge_consecutive_segments, _split_paragraph_by_quotes
    return _split_paragraph_by_quotes, _merge_consecutive_segments


# ---------------------------------------------------------------------------
# _split_paragraph_by_quotes
# ---------------------------------------------------------------------------


class TestSplitParagraphByQuotes:
    """Tests for _split_paragraph_by_quotes(para, para_quotes) -> list[tuple[str,str]]"""

    def _make_quote(self, qid, text, para_index=0):
        """Return a (Quote, AttributionItem)-like pair as simple objects."""
        from kenkui.nlp.models import AttributionItem, Quote
        q = Quote(id=qid, text=text, para_index=para_index, char_offset=0)
        a = AttributionItem(quote_id=qid, speaker="Alice", emotion="neutral")
        return q, a

    def _split(self, para, para_quotes):
        fn, _ = _import_helpers()
        return fn(para, para_quotes)

    def test_narrator_only_paragraph_no_quotes(self):
        """A paragraph with no quotes should return a single NARRATOR span."""
        result = self._split("She walked down the hall.", [])
        assert result == [("She walked down the hall.", "NARRATOR")]

    def test_single_attributed_quote(self):
        """A quoted span with attribution → character span; surrounding text → NARRATOR."""
        para = '"Hello," she said.'
        q, a = self._make_quote(0, '"Hello,"')
        result = self._split(para, [(q, a)])
        # Should have Alice for the quote, NARRATOR for " she said."
        speakers = [s for _, s in result]
        assert "Alice" in speakers
        assert "NARRATOR" in speakers
        # Alice span must contain the quote text
        alice_texts = [t for t, s in result if s == "Alice"]
        assert any("Hello" in t for t in alice_texts)

    def test_narrator_before_and_after_quote(self):
        """Narrator text before the quote and after should both be NARRATOR spans."""
        para = 'She asked, "How are you?" and waited.'
        q, a = self._make_quote(0, '"How are you?"')
        result = self._split(para, [(q, a)])
        speakers = [s for _, s in result]
        assert speakers.count("NARRATOR") >= 1
        assert "Alice" in speakers

    def test_two_quotes_different_speakers(self):
        """Two attributed quotes with different speakers produce separate character spans."""
        from kenkui.nlp.models import AttributionItem, Quote
        para = '"Hi," said Alice. "Hello," said Bob.'
        q1 = Quote(id=0, text='"Hi,"', para_index=0, char_offset=0)
        a1 = AttributionItem(quote_id=0, speaker="Alice", emotion="neutral")
        q2 = Quote(id=1, text='"Hello,"', para_index=0, char_offset=0)
        a2 = AttributionItem(quote_id=1, speaker="Bob", emotion="neutral")
        fn, _ = _import_helpers()
        result = fn(para, [(q1, a1), (q2, a2)])
        speakers = [s for _, s in result]
        assert "Alice" in speakers
        assert "Bob" in speakers

    def test_unmatched_quote_falls_back_to_narrator(self):
        """A quote in the paragraph that has no attribution should be NARRATOR."""
        para = '"Mystery line." She stared.'
        # No quotes in para_quotes → everything NARRATOR
        fn, _ = _import_helpers()
        result = fn(para, [])
        # With no attributions, should return the whole paragraph as NARRATOR
        speakers = [s for _, s in result]
        assert all(s == "NARRATOR" for s in speakers)

    def test_empty_paragraph_returns_narrator(self):
        fn, _ = _import_helpers()
        result = fn("", [])
        assert result == [("", "NARRATOR")]

    def test_full_coverage_no_whitespace_lost(self):
        """Concatenating all spans should reconstruct the original paragraph."""
        from kenkui.nlp.models import AttributionItem, Quote
        para = 'He said, "Let\'s go," and stood up.'
        q = Quote(id=0, text='"Let\'s go,"', para_index=0, char_offset=0)
        a = AttributionItem(quote_id=0, speaker="Bob", emotion="neutral")
        fn, _ = _import_helpers()
        result = fn(para, [(q, a)])
        reconstructed = "".join(t for t, _ in result)
        assert reconstructed == para


# ---------------------------------------------------------------------------
# _merge_consecutive_segments
# ---------------------------------------------------------------------------


class TestMergeConsecutiveSegments:
    """Tests for _merge_consecutive_segments(segments) -> list[Segment]"""

    def _merge(self, segments):
        _, fn = _import_helpers()
        return fn(segments)

    def test_empty_list(self):
        assert self._merge([]) == []

    def test_single_segment_unchanged(self):
        segs = [Segment(text="Hello", speaker="Alice", index=0)]
        result = self._merge(segs)
        assert len(result) == 1
        assert result[0].text == "Hello"
        assert result[0].speaker == "Alice"

    def test_no_consecutive_same_speaker(self):
        """Alternating speakers → no merging."""
        segs = [
            Segment(text="Hello", speaker="Alice", index=0),
            Segment(text="Hi", speaker="NARRATOR", index=1),
            Segment(text="Goodbye", speaker="Bob", index=2),
        ]
        result = self._merge(segs)
        assert len(result) == 3

    def test_merge_two_narrator_segments(self):
        """Two consecutive NARRATOR segments should be merged."""
        segs = [
            Segment(text="Para one.", speaker="NARRATOR", index=0),
            Segment(text="Para two.", speaker="NARRATOR", index=1),
        ]
        result = self._merge(segs)
        assert len(result) == 1
        assert result[0].speaker == "NARRATOR"
        assert "Para one." in result[0].text
        assert "Para two." in result[0].text

    def test_merge_two_character_segments(self):
        """Two consecutive same-character segments should be merged."""
        segs = [
            Segment(text='"Hello,"', speaker="Alice", index=0),
            Segment(text='"How are you?"', speaker="Alice", index=1),
        ]
        result = self._merge(segs)
        assert len(result) == 1
        assert result[0].speaker == "Alice"
        assert "Hello" in result[0].text
        assert "How are you?" in result[0].text

    def test_interleaved_no_merge(self):
        """A-N-A pattern should NOT merge across the NARRATOR span."""
        segs = [
            Segment(text='"Hello,"', speaker="Alice", index=0),
            Segment(text="she said,", speaker="NARRATOR", index=1),
            Segment(text='"goodbye."', speaker="Alice", index=2),
        ]
        result = self._merge(segs)
        assert len(result) == 3

    def test_indices_rewritten_after_merge(self):
        """Merged segment list must have contiguous 0-based indices."""
        segs = [
            Segment(text="A", speaker="NARRATOR", index=0),
            Segment(text="B", speaker="NARRATOR", index=1),
            Segment(text="C", speaker="Alice", index=2),
        ]
        result = self._merge(segs)
        assert [s.index for s in result] == list(range(len(result)))

    def test_narrator_join_uses_double_newline(self):
        """Merged NARRATOR spans should be separated by '\\n\\n'."""
        segs = [
            Segment(text="First paragraph.", speaker="NARRATOR", index=0),
            Segment(text="Second paragraph.", speaker="NARRATOR", index=1),
        ]
        result = self._merge(segs)
        assert "\n\n" in result[0].text

    def test_character_join_uses_space(self):
        """Merged character spans should be separated by a space."""
        segs = [
            Segment(text='"Line one."', speaker="Alice", index=0),
            Segment(text='"Line two."', speaker="Alice", index=1),
        ]
        result = self._merge(segs)
        assert " " in result[0].text


# ---------------------------------------------------------------------------
# _attribution_to_segments — slug normalisation and pronoun remap (Fix 1 + 4b)
# ---------------------------------------------------------------------------


def _make_chapter(paragraphs: list[str]):
    """Return a minimal Chapter-like object."""
    from kenkui.models import Chapter
    return Chapter(index=0, title="Ch 1", paragraphs=paragraphs)


def _make_attr_result(items: list[dict]):
    """Build an AttributionResult from a list of dicts."""
    from kenkui.nlp.models import AttributionItem, AttributionResult
    return AttributionResult(
        attributions=[AttributionItem(**item) for item in items]
    )


def _make_roster():
    """Return a permissive CharacterRoster for tests that don't exercise roster validation.

    Contains the slugs used by TestAttributionToSegmentsSlugs and
    TestAttributionToSegmentsPronounRemap so those tests survive the roster
    validation pass added in _attribution_to_segments.
    """
    from kenkui.nlp.models import CharacterRecord, CharacterRoster
    return CharacterRoster(characters=[
        CharacterRecord(slug="darrow", canonical_name="Darrow"),
        CharacterRecord(slug="elizabeth_bennet", canonical_name="Elizabeth Bennet"),
    ])


class TestAttributionToSegmentsSlugs:
    """Fix 1: all attribution speakers are slugified before segments are built."""

    def _run(self, paragraphs, items):
        from kenkui.nlp import _attribution_to_segments
        chapter = _make_chapter(paragraphs)
        attr_result = _make_attr_result(items)
        roster = _make_roster()
        return _attribution_to_segments(chapter, attr_result, roster)

    def test_canonical_speaker_becomes_slug(self):
        """Speaker 'Darrow' (canonical case) → segment speaker 'darrow' (slug)."""
        para = '"I will win," Darrow said.'
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "Darrow", "emotion": "neutral", "confidence": 1}],
        )
        speakers = {s.speaker for s in segments}
        assert "darrow" in speakers, f"Expected 'darrow' in {speakers}"
        assert "Darrow" not in speakers, "Canonical 'Darrow' should be slugified"

    def test_multiword_canonical_becomes_slug(self):
        """Speaker 'Elizabeth Bennet' → slug 'elizabeth_bennet'."""
        para = '"Indeed," said Elizabeth Bennet.'
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "Elizabeth Bennet", "emotion": "neutral", "confidence": 1}],
        )
        speakers = {s.speaker for s in segments}
        assert "elizabeth_bennet" in speakers, f"Expected 'elizabeth_bennet' in {speakers}"

    def test_narrator_sentinel_unchanged(self):
        """'NARRATOR' is a sentinel and must not be slugified."""
        para = '"Hello," she said.'
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "NARRATOR", "emotion": "neutral", "confidence": 1}],
        )
        speakers = {s.speaker for s in segments}
        assert "NARRATOR" in speakers, f"NARRATOR sentinel should be preserved: {speakers}"

    def test_unknown_sentinel_unchanged(self):
        """'Unknown' is a sentinel and must not be slugified."""
        para = '"Hello," she said.'
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "Unknown", "emotion": "neutral", "confidence": 1}],
        )
        speakers = {s.speaker for s in segments}
        assert "Unknown" in speakers, f"Unknown sentinel should be preserved: {speakers}"

    def test_missing_quote_fallback_uses_slug(self):
        """First quote attributed to 'Darrow'; second quote missing from attribution.
        The fallback should carry 'darrow' (the already-slugified form), not 'Darrow'.
        """
        para1 = '"I will win," Darrow said.'
        para2 = '"Forward," she called.'
        segments = self._run(
            [para1, para2],
            # Only quote 0 attributed — quote 1 is missing, must fall back to 'darrow'
            [{"quote_id": 0, "speaker": "Darrow", "emotion": "neutral", "confidence": 1}],
        )
        # Find the segment covering the second paragraph's quote
        char_speakers = {s.speaker for s in segments if s.speaker not in ("NARRATOR",)}
        assert "darrow" in char_speakers, (
            f"Fallback speaker should be slug 'darrow', got: {char_speakers}"
        )
        assert "Darrow" not in char_speakers, (
            f"Canonical 'Darrow' should not appear after slugification: {char_speakers}"
        )


class TestAttributionToSegmentsPronounRemap:
    """Fix 4b: pronoun speaker slugs are remapped to NARRATOR."""

    def _run(self, paragraphs, items):
        from kenkui.nlp import _attribution_to_segments
        chapter = _make_chapter(paragraphs)
        attr_result = _make_attr_result(items)
        roster = _make_roster()
        return _attribution_to_segments(chapter, attr_result, roster)

    def test_pronoun_she_becomes_narrator(self):
        """Speaker 'she' is a pronoun → remapped to NARRATOR."""
        para = '"I am here," she said.'
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "she", "emotion": "neutral", "confidence": 1}],
        )
        speakers = {s.speaker for s in segments}
        assert "NARRATOR" in speakers, f"Pronoun 'she' should remap to NARRATOR: {speakers}"
        assert "she" not in speakers, f"Raw pronoun 'she' should be gone: {speakers}"

    def test_pronoun_he_becomes_narrator(self):
        """Speaker 'he' is a pronoun → remapped to NARRATOR."""
        para = '"Come here," he said.'
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "he", "emotion": "neutral", "confidence": 1}],
        )
        speakers = {s.speaker for s in segments}
        assert "NARRATOR" in speakers, f"Pronoun 'he' should remap to NARRATOR: {speakers}"
        assert "he" not in speakers

    def test_pronoun_they_becomes_narrator(self):
        """Speaker 'they' is a pronoun → remapped to NARRATOR."""
        para = '"We are ready," they said.'
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "they", "emotion": "neutral", "confidence": 1}],
        )
        speakers = {s.speaker for s in segments}
        assert "NARRATOR" in speakers
        assert "they" not in speakers

    def test_pronoun_capitalized_becomes_narrator(self):
        """Capitalized pronoun from LLM (e.g. 'She') is slugified to 'she' then remapped.

        This locks in the slug-before-remap ordering dependency: if slugification
        ran *after* remap, 'She' would not be caught by the pronoun check.
        """
        para = '"I am here," She said.'
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "She", "emotion": "neutral", "confidence": 1}],
        )
        speakers = {s.speaker for s in segments}
        assert "NARRATOR" in speakers, (
            f"Capitalized pronoun 'She' should be slugified to 'she' then remapped to NARRATOR: {speakers}"
        )
        assert "She" not in speakers, f"Raw capitalized pronoun 'She' should be gone: {speakers}"
        assert "she" not in speakers, f"Slug 'she' should also be gone (remapped): {speakers}"

    def test_non_pronoun_character_not_remapped(self):
        """A real character name is not affected by pronoun remap."""
        para = '"Fight," Darrow said.'
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "Darrow", "emotion": "neutral", "confidence": 1}],
        )
        speakers = {s.speaker for s in segments}
        # After slugification 'Darrow' → 'darrow'; it is not a pronoun, must not be NARRATOR
        assert "darrow" in speakers
        non_narrator = speakers - {"NARRATOR"}
        assert non_narrator, "Character 'darrow' should not be remapped to NARRATOR"


# ---------------------------------------------------------------------------
# _attribution_to_segments — ghost speaker roster validation
#
# Regression tests for:
#   WARNING: speaker 'screwface' has no voice mapping — will use narrator fallback
#   WARNING: speaker 'computer' has no voice mapping — will use narrator fallback
#   WARNING: speaker 'screw' has no voice mapping — will use narrator fallback
#
# Root cause: _attribution_to_segments slugified speakers but never validated
# them against the roster.  LLM hallucinations ('computer') and short-form
# aliases not in the alias list ('screw' for 'Screwface') propagated to the
# audio worker with no voice mapping.
# ---------------------------------------------------------------------------


def _make_real_roster(*slugs: str):
    """Return a CharacterRoster with one CharacterRecord per slug."""
    from kenkui.nlp.models import CharacterRecord, CharacterRoster
    return CharacterRoster(characters=[
        CharacterRecord(slug=s, canonical_name=s.replace("_", " ").title())
        for s in slugs
    ])


class TestAttributionToSegmentsRosterValidation:
    """Ghost speakers (slugs not in the roster) must be remapped to 'Unknown'.

    After the fix, the 'no voice mapping' WARNING in parsing.py:_warn_unresolvable_speakers
    must never fire for hallucinated or alias-truncated speakers because they will
    have been rewritten to 'Unknown' (a sentinel that the audio worker handles via
    narrator-voice fallback) before the segments are returned.
    """

    def _run(self, paragraphs, items, roster):
        from kenkui.nlp import _attribution_to_segments
        chapter = _make_chapter(paragraphs)
        attr_result = _make_attr_result(items)
        return _attribution_to_segments(chapter, attr_result, roster)

    # --- Valid slugs must pass through ---

    def test_valid_roster_slug_passes_through(self):
        """A slug present in the roster is NOT remapped — voice lookup will succeed."""
        para = '"Forward," Darrow said.'
        roster = _make_real_roster("darrow", "sevro")
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "darrow", "emotion": "neutral", "confidence": 5}],
            roster,
        )
        speakers = {s.speaker for s in segments}
        assert "darrow" in speakers, (
            f"Valid roster slug 'darrow' must not be remapped: {speakers}"
        )

    # --- Regression: LLM hallucination ('computer') ---

    def test_hallucinated_slug_computer_remapped_to_unknown(self):
        """'computer' is not a character; LLM hallucination must become 'Unknown'.

        Regression: speaker 'computer' has no voice mapping — will use narrator fallback
        """
        para = '"Access granted," the computer said.'
        roster = _make_real_roster("darrow", "sevro", "screwface")
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "computer", "emotion": "neutral", "confidence": 3}],
            roster,
        )
        speakers = {s.speaker for s in segments}
        assert "computer" not in speakers, (
            f"Hallucinated slug 'computer' must be remapped, not kept: {speakers}"
        )
        assert "Unknown" in speakers, (
            f"Hallucinated slug must become 'Unknown': {speakers}"
        )

    # --- Regression: short-form alias ('screw' for 'screwface') ---

    def test_unregistered_alias_screw_remapped_to_unknown(self):
        """'screw' is a short form of 'screwface' not in the alias list → 'Unknown'.

        Regression: speaker 'screw' has no voice mapping — will use narrator fallback
        """
        para = '"Move," Screw said.'
        roster = _make_real_roster("screwface")  # 'screw' is not a registered slug
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "screw", "emotion": "neutral", "confidence": 2}],
            roster,
        )
        speakers = {s.speaker for s in segments}
        assert "screw" not in speakers, (
            f"Unregistered alias 'screw' must be remapped, not kept: {speakers}"
        )
        assert "Unknown" in speakers, (
            f"Unregistered alias must become 'Unknown': {speakers}"
        )

    # --- Regression: roster truncation ('screwface' dropped by LLM truncation) ---

    def test_slug_from_truncated_roster_remapped_to_unknown(self):
        """If 'screwface' was dropped from the roster by LLM truncation, attribution
        may return it anyway (hallucination).  It must be caught and remapped.

        Regression: speaker 'screwface' has no voice mapping — will use narrator fallback
        """
        para = '"Die," Screwface said.'
        # Roster does NOT contain 'screwface' — simulates truncated roster
        roster = _make_real_roster("darrow", "sevro")
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "screwface", "emotion": "neutral", "confidence": 4}],
            roster,
        )
        speakers = {s.speaker for s in segments}
        assert "screwface" not in speakers, (
            f"Slug from truncated roster must be caught: {speakers}"
        )
        assert "Unknown" in speakers, (
            f"Slug from truncated roster must become 'Unknown': {speakers}"
        )

    # --- Sentinels must not be touched ---

    def test_narrator_sentinel_not_remapped(self):
        """NARRATOR is a sentinel value; roster validation must skip it."""
        para = '"Hello," she said.'
        roster = _make_real_roster("darrow")
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "NARRATOR", "emotion": "neutral", "confidence": 1}],
            roster,
        )
        speakers = {s.speaker for s in segments}
        assert "NARRATOR" in speakers, f"NARRATOR sentinel must survive: {speakers}"

    def test_unknown_sentinel_not_remapped(self):
        """'Unknown' is already the fallback sentinel; roster validation must skip it."""
        para = '"Something," someone said.'
        roster = _make_real_roster("darrow")
        segments = self._run(
            [para],
            [{"quote_id": 0, "speaker": "Unknown", "emotion": "neutral", "confidence": 1}],
            roster,
        )
        speakers = {s.speaker for s in segments}
        assert "Unknown" in speakers, f"Unknown sentinel must survive: {speakers}"

    # --- Warning emission ---

    def test_ghost_speaker_emits_warning_with_slug_name(self, caplog):
        """When a ghost slug is found, a WARNING must name it so it's diagnosable."""
        import logging
        para = '"Beep," the computer said.'
        roster = _make_real_roster("darrow")
        chapter = _make_chapter([para])
        attr_result = _make_attr_result(
            [{"quote_id": 0, "speaker": "computer", "emotion": "neutral", "confidence": 3}]
        )
        from kenkui.nlp import _attribution_to_segments
        with caplog.at_level(logging.WARNING, logger="kenkui.nlp"):
            _attribution_to_segments(chapter, attr_result, roster)
        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("computer" in m for m in warning_messages), (
            f"WARNING must name the ghost slug 'computer': {warning_messages}"
        )


# ---------------------------------------------------------------------------
# _attribution_to_segments — attribution gap (missing quote IDs) logging
#
# Regression test for:
#   WARNING: Chapter N: quote id=X missing from LLM attribution
#
# Root cause: truncated attribution responses leave trailing quote IDs
# unattributed.  The fallback (last-seen-speaker) fills them in but this
# should be loudly logged so operators know attribution quality degraded.
# ---------------------------------------------------------------------------


class TestAttributionGapWarning:
    """When the LLM skips quote IDs, a WARNING must fire for each missing one."""

    def test_missing_quote_emits_warning(self, caplog):
        """Two quotes extracted, only one attributed → WARNING for the missing one."""
        import logging
        para1 = '"First," Darrow said.'
        para2 = '"Second," Darrow said.'
        roster = _make_real_roster("darrow")
        chapter = _make_chapter([para1, para2])
        attr_result = _make_attr_result(
            # Only quote 0 attributed; quote 1 is absent → must trigger WARNING
            [{"quote_id": 0, "speaker": "darrow", "emotion": "neutral", "confidence": 5}]
        )
        from kenkui.nlp import _attribution_to_segments
        with caplog.at_level(logging.WARNING, logger="kenkui.nlp"):
            _attribution_to_segments(chapter, attr_result, roster)
        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("missing" in m.lower() for m in warning_messages), (
            f"WARNING about missing quote must fire: {warning_messages}"
        )

    def test_no_gap_warning_when_all_attributed(self, caplog):
        """All quotes attributed → no 'missing' WARNING fires."""
        import logging
        para = '"Hello," Darrow said.'
        roster = _make_real_roster("darrow")
        chapter = _make_chapter([para])
        attr_result = _make_attr_result(
            [{"quote_id": 0, "speaker": "darrow", "emotion": "neutral", "confidence": 5}]
        )
        from kenkui.nlp import _attribution_to_segments
        with caplog.at_level(logging.WARNING, logger="kenkui.nlp"):
            _attribution_to_segments(chapter, attr_result, roster)
        missing_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "missing" in r.message.lower()
        ]
        assert not missing_warnings, (
            f"No missing-quote WARNING expected when fully attributed: "
            f"{[r.message for r in missing_warnings]}"
        )
