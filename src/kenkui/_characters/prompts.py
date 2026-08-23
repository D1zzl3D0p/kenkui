"""Versioned prompt text.

PROMPT_VERSION is an attribution-store key input. Bump it whenever any prompt
below changes, or a stored result produced by the previous wording will be
reused silently under the new one.
"""

from __future__ import annotations

PROMPT_VERSION = "characters-v1"

# Speakers carried into the next window so the model can hold A-B-A-B
# conversational momentum across a boundary it cannot see past.
CONTINUITY_SPEAKERS = 4

ROSTER_PROMPT = """\
List the speaking characters in this passage from a novel.

Return ONLY JSON:
{{"characters": [{{"id": "...", "name": "...", "gender": "..."}}]}}

- "id": lowercase, words joined by hyphens, derived from the name. Stable
  across the whole book, so use the fullest form you see: "elizabeth-bennet",
  not "lizzy".
- "name": the character's name as it appears in the text.
- "gender": "feminine", "masculine", or null when the text does not say.
  Do not guess from a name alone.
- Include only characters who speak or are addressed. Omit places, objects,
  and groups.
- Never return a pronoun as a name.

Passage:
---
{passage}
---
"""

ATTRIBUTION_PROMPT = """\
Identify who speaks each numbered quote in this passage from a novel.

Return ONLY JSON:
{{"attributions": [{{"quote_id": 0, "speaker": "..."}}]}}

CHARACTERS
Return the id exactly as written here, never the display name.
{roster}

Recently speaking, for continuity: {recent}

Passage:
---
{passage}
---

Quotes, in order. Attribute every one; do not add, skip, or reorder:
{quotes}

RULES
- "speaker": an id from the list above, or "unknown".
- Never answer with a pronoun. If the speaker is identifiable only by pronoun
  and the passage does not disambiguate, answer "unknown".
- A quoted run that is a title, label, acronym, or a word used as a term is not
  speech. Answer "unknown" for it.
- Prefer "unknown" over a guess. An unattributed line is narrated, which is
  correct-sounding; a wrongly attributed line is audibly wrong.
"""
