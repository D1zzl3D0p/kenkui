"""Versioned prompt text.

PROMPT_VERSION is an attribution-store key input. Bump it whenever any prompt
below changes, or a stored result produced by the previous wording will be
reused silently under the new one.
"""

from __future__ import annotations

PROMPT_VERSION = "characters-v3"

# Speakers carried into the next window so the model can hold A-B-A-B
# conversational momentum across a boundary it cannot see past.
CONTINUITY_SPEAKERS = 4

# The roles a speaker may be identified by when the text names no one. Closed
# on purpose: a pattern that accepts any short lowercase word cannot tell a
# role from a hallucinated name, and would turn every invented speaker into a
# cast voice. A word outside this list resolves to unknown, exactly as before
# roles existed.
ROLE_WORDS: frozenset[str] = frozenset(
    {
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "stranger",
        "voice",
        "first-man",
        "second-man",
        "third-man",
        "first-woman",
        "second-woman",
        "third-woman",
        "old-man",
        "old-woman",
        "young-man",
        "young-woman",
        "guard",
        "soldier",
        "servant",
        "innkeeper",
        "cook",
        "farmer",
        "merchant",
        "sailor",
        "lookout",
        "watchman",
        "driver",
        "porter",
        "messenger",
        "priest",
        "doctor",
        "nurse",
        "clerk",
        "captain",
    }
)

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
- "speaker": an id from the list above, or "unknown".
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
  bow", "the first man", "the innkeeper" -- answer with the matching word from
  this list, if one fits: man, woman, boy, girl, child, stranger, voice,
  first-man, second-man, third-man, first-woman, second-woman, third-woman,
  old-man, old-woman, young-man, young-woman, guard, soldier, servant,
  innkeeper, cook, farmer, merchant, sailor, lookout, watchman, driver,
  porter, messenger, priest, doctor, nurse, clerk, captain. Prefer this over
  "unknown" whenever the text says who is speaking at all and one of these
  words fits.
"""
