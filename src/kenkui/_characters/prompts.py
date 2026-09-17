"""Versioned prompt text.

PROMPT_VERSION is an attribution-store key input. Bump it whenever any prompt
below changes, or a stored result produced by the previous wording will be
reused silently under the new one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

PROMPT_VERSION = "characters-v8"

IDENTITY_PROMPT = """\
Below is the cast list extracted from a novel. Each numbered entry is one
character as found so far: the names it appears under, how often it is
mentioned, and a few short excerpts from the book.

Some entries may be the same individual as another entry under a different
name: a nickname, a title or epithet used for that one person, a name given to
them later in the story, or a formal and an informal name.

Return ONLY JSON: {"same_person": [[1, 7], [4, 12, 30]], "not_individuals": [5, 9]}

- Each inner list holds the numbers of entries that are all one individual.
- List only entries that belong to a group of two or more. Leave everything
  else out.
- People who share a surname or a title are different individuals unless the
  excerpts show otherwise: husband and wife, parent and child, two sisters,
  two people with the same first name.
- If you are not sure, leave them out. Merging two different people is much
  worse than missing a match.

- not_individuals: numbers of entries that are not one individual character at
  all: a place, a group or a people, an organisation, an object or concept, the
  title of a book, or a word used to address many different people ("Sire",
  "my Lord"). A title that names different people at different points in the
  story ("the Duke") also goes here. Someone known mainly by an epithet or a
  title held by one person ("the Dragon", "the Emperor", "the Mayor") IS an
  individual: do not list them. If unsure, do not list it.

ENTRIES
"""

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

_QUALIFIED_ROLE = re.compile(
    r"(?:(?:first|second|third|old|young)-)?(male|female)-([a-z]+)"
)


def role_gender(role: str) -> str | None:
    """Read explicit role gender, without guessing from occupations or names.

    The compact qualified form is requested by attribution. Restricting its
    grammar avoids treating a female proctor's assistant as necessarily female.
    """
    if role in ROLE_GENDERS:
        return ROLE_GENDERS[role]
    match = _QUALIFIED_ROLE.fullmatch(role)
    if match is None or match[2] in {"male", "female"}:
        return None
    return "masculine" if match[1] == "male" else "feminine"


ROSTER_PROMPT = """\
List the speaking characters in this passage from a novel.

Return ONLY JSON:
{{"characters": [{{"id": "...", "name": "...", "gender": "..."}}], "narrator": "..."}}

- "id": lowercase, words joined by hyphens, derived from the name. Stable
  across the whole book, so use the fullest form you see: "elizabeth-bennet",
  not "lizzy".
- "name": the character's name as it appears in the text.
- "gender": "feminine", "masculine", or null when the text does not say.
  Use explicit descriptions and pronouns referring to this person, including
  action beats and possessives ("he gestured", "his booklet"). Do not borrow
  the gender of someone they address or mention. Do not guess from a name alone.
- Keep different people separate even when they share an occupation or title.
  For unnamed speakers use stable distinguishing roles, such as "male-proctor"
  and "female-proctor", instead of merging both into "proctor".
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
{{"attributions": [{{"quote_id": 0, "speaker": "..."}}],
  "speaker_genders": {{"speaker-id": null}}}}

CHARACTERS
Return the id exactly as written here, never the display name.
{roster}

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
- Distinguish different people sharing a role within this passage. A male
  proctor and a female proctor are separate speakers: use "male-proctor" and
  "female-proctor" consistently, including where the text later shortens either
  to "the proctor". For multiple people of the same gender and role, use
  "first-male-guard", "second-male-guard", and so on. Do not merge them into a
  generic roster entry that conflates different people.
- For an unnamed speaker whose gender is explicit, preserve it in the role:
  "male-<occupation>" or "female-<occupation>" (a short single-word occupation).
  Read actions and possessives as well as dialogue tags: "he gestured" and
  "his booklet" can identify the speaker even without "he said". Check who
  each pronoun refers to; another person mentioned nearby is not the speaker.
  Keep the same qualified role for all that person's lines in the passage.
  Never infer gender from an occupation, name, or stereotype; omit the qualifier
  when the text does not establish it.
- When the text names a speaker who is not in the list above, answer with their
  name. A speaker the list missed is still a speaker.

- Also return speaker_genders: each speaking character's gender once, keyed by
  the exact speaker id used above. Use "masculine", "feminine", or null when the
  passage does not establish it.
"""
