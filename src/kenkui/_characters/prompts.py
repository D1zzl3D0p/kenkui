"""Versioned prompt text.

PROMPT_VERSION is an attribution-store key input. Bump it whenever any prompt
below changes, or a stored result produced by the previous wording will be
reused silently under the new one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

PROMPT_VERSION = "characters-v4"

# Speakers carried into the next window so the model can hold A-B-A-B
# conversational momentum across a boundary it cannot see past.
CONTINUITY_SPEAKERS = 4

# The role words that state a gender outright. An innkeeper or a guard may be
# anyone, and casting them from a gendered pool would be a guess; a woman is a
# woman, and sending her to the whole pool is a coin flip on a speaker the text
# has already identified.
ROLE_GENDERS: Mapping[str, str] = {
    "woman": "feminine",
    "girl": "feminine",
    "old-woman": "feminine",
    "young-woman": "feminine",
    "first-woman": "feminine",
    "second-woman": "feminine",
    "third-woman": "feminine",
    "man": "masculine",
    "boy": "masculine",
    "old-man": "masculine",
    "young-man": "masculine",
    "first-man": "masculine",
    "second-man": "masculine",
    "third-man": "masculine",
}

ROSTER_PROMPT = """\
List the speaking characters in this passage from a novel.

Return ONLY JSON:
{{"characters": [{{"id": "...", "name": "...", "gender": "..."}}], "narrator": "..."}}

- "id": lowercase, words joined by hyphens, derived from the name. Stable
  across the whole book, so use the fullest form you see: "elizabeth-bennet",
  not "lizzy".
- "name": the character's name as it appears in the text.
- "gender": "feminine", "masculine", or null when the text does not say.
  Do not guess from a name alone.
- Include only characters who speak or are addressed. Omit places, objects,
  and groups.
- Never return a pronoun as a name.
- "narrator": when the passage is written in the first person, the "id" of the
  character narrating it, taken from the list you are returning. Omit this key
  or return null when the passage is written in the third person, or when the
  narrator is never named.

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
- "speaker": an id from the list above when one fits, otherwise the name or
  role of whoever speaks, or "unknown" when the passage does not say.
- Never answer with a pronoun. If the speaker is identifiable only by pronoun
  and the passage does not disambiguate, answer "unknown".
- A quoted run that is a title, label, acronym, or a word used as a term is not
  speech. Answer "unknown" for it.
- Prefer "unknown" over a guess. An unattributed line is narrated, which is
  correct-sounding; a wrongly attributed line is audibly wrong.
- A character marked "narrates this book" tells it in the first person. A quote
  tagged "I said", "I asked" or "said I" is spoken by them: answer with their
  id. Do not answer "unknown" for those, and never answer with the pronoun.
- When the text identifies a speaker without naming them -- "the lookout in the
  bow", "the first man", "the innkeeper" -- answer with a short lowercase noun
  phrase for them, words joined by hyphens: "lookout", "first-man",
  "innkeeper", "old-woman". Prefer this over "unknown" whenever the text says
  who is speaking at all.
- When the text names a speaker who is not in the list above, answer with their
  name. A speaker the list missed is still a speaker.
"""
