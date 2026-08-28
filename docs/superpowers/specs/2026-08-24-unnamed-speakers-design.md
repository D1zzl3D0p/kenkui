# Speakers without names of their own

Status: proposal. Measured in `evals/attribution/` over five books and three
series; that harness holds the evidence and the routes that failed, so this
describes only what to build.

## What is missing

Attribution answers with a roster character or with `unknown`, and `unknown`
renders as narration. Two kinds of speaker fall through that.

A **first-person narrator** says `"..." I said`. The roster is built from names
governing speech verbs, and "I" is not a name, so they never enter it; the
prompt forbids answering with a pronoun, so a model cannot name them either.
This is 9.1% of dialogue in one measured book and 15.9% in another, where the
narrator is the largest speaker present and scores zero speech acts.

A **role speaker** is identified by the text but not named: "the lookout in the
bow", "the first man", "a scornful voice". These are 2.9% of one book's
dialogue, and the largest carries a scene — ten quotes over 5,400 characters
for the unnamed woman who attacks Mat in the Baerlon stable.

Alongside them, the roster's own **alias handling** is the largest correctness
risk in the existing pipeline, and is fixed here because both new speaker kinds
depend on it.

## Identity, first

A wrong merge misroutes dialogue that every later stage then treats as
settled, so this precedes the rest. `evals/attribution/identity.py` holds the
decisions as pure functions over names and counts; the roster builder composes
them.

Over-merging puts two people in one voice, which is the failure the attribution
design exists to prevent. Under-merging gives one person two voices, audible
but locally consistent. **Under-merging is the default.**

`same_person` separates names by honorific position. Two differing prefix
honorifics mean two people — "Mr Elliot" and "Miss Elliot", "Mr Geary" and
"Mrs Geary", who both speak. One honorific against none does not: "Brightlord
Dalinar" and "Dalinar Kholin" are one Dalinar. With titles set aside, names
denote one person when what remains is equal or nested, so "Charles Hayter"
stays apart from "Charles Musgrove". `detect_titles` learns a book's own
honorifics from its names, which is how "brightlord" and "brightness" are
found without being listed.

`resolve_short_forms` attaches a bare name to its entity, or refuses when two
could claim it. A refused short form is dropped, not assigned: keeping it would
split one person, assigning it would merge two, and dropping it leaves the
model to name a full form from the passage.

Two names with no shared tokens — "the Dragon" and "Sarkan" — are left split.
No signal separates them from two characters who simply never meet, and the
routes that promised to are recorded against the code rather than here. One
person with two consistent voices is the cheaper error, and no stage is added
to chase it.

## The narrator is a flag, not an id

A first-person narrator is a character already on the roster who happens to
deliver the narration. The roster says that about them; it does not mint a
second entry beside their own.

Detection needs both halves. A book qualifies only if it carries first-person
dialogue tags, which third-person books do not — three measured books have
exactly zero, and skipping that check lets a place be nominated. Within
qualifying chapters the narrator is the name *addressed* without *speaking*:
everyone talks to them by name while the narration says "I said" rather than
"Borlú said". The ratio is decisive — 37 addressed against 1 speaking, against
27 and 91 for a third-person protagonist. Chapters are identified individually
so alternating points of view remain expressible, but the narrator is chosen
once from their combined evidence, because choosing per chapter picks whoever
that chapter happened to address.

`ATTRIBUTION_PROMPT` marks the narrating character in the roster block and says
that a quote tagged "I said" belongs to them. The answer is that character's
ordinary id, and `_resolve` additionally accepts the bare word `narrator` for
it. Models shorten a long id and lose the answer — a namespaced narrator id
cost 5.6 points of coverage that way — so the alias is what keeps the coverage
while leaving one id per person, and therefore one voice across their narration
and their dialogue.

Their spoken lines and their narration end up in different voices without any
new rule. `_castable` drops `narrator_voice_id` from the pool for every
character, so a narrating character cannot be assigned the narration voice.

## Roles

A role speaker becomes an ordinary `CharacterProfile` scoped to one chapter.
Every role measured is confined to a single chapter, which is the shape
`chapter_ids` already describes, so the solver's behaviour follows unchanged:
the role cannot take a voice already used in its scene, two roles in one scene
are forced apart, and the voice is reused freely elsewhere, because chapter
40's guard is not chapter 12's.

The model answers a bare slug — `guard`, `lookout`, `first-man` — and the
resolver scopes it to the chapter being attributed, which attribution already
iterates. Nothing longer: an id the model must reproduce gets shortened and
rejected, which cost 5.6 points of coverage when a narrator id was namespaced.

Gender comes from the model, in the same answer. Asked directly it decided
eight of nine roles and returned null for the ninth, correctly, because "the
lookout in the bow sang out" does not say. Gender narrows the pool only under
the `gendered` method, so a null is a narrowing and not a failure.

## What does not change

**Extraction.** Quoted runs that nobody utters — a song title, a nickname —
are extracted as dialogue and should stay that way. `CastPlan.voice_for`
returns the narrator for a span with no speaker, so they already render as
narration, and `_fragments` adds no silence at a speaker boundary. Changing
extraction would move the span partition, and with it chunk boundaries, segment
identities and every cached segment in the chapter, for nothing audible. The
rules that would do it also cost more than they save: suppressing a quoted
pronoun would silence `"You!"`, `"Anyone."` and `"Who . . . ?"` to catch two
non-utterances.

**Casting.** No change. Both new speaker kinds are `CharacterProfile` values
the existing solver colours.

**The store.** No schema change. `characters` carries id, display name, gender,
volume and ordinal, and both kinds fit unaltered.

**Identity and cache.** A book with neither a narrator nor a role must produce
byte-identical segment identities, an unchanged plan fingerprint, and no
invalidated cache entry. Whatever these stages contribute enters a segment's
identity only when the feature is in play, and its key is absent rather than
null when it is not, as the spoken-form schema and the break tiers already are.
Model traffic stays in the parent process, where attribution already runs.

## Risks

Pool exhaustion, because roles raise the per-chapter speaker count. One
measured book peaks at twelve distinct speakers in a chapter, against a pool
that must cover the maximum rather than the total. Report the count before
rendering rather than leaving it to be heard.

Over-attribution, because `role` is an escape hatch a model may reach for where
`unknown` was right. One tested model already names characters for anonymous
crowd speech, which cost it precision against a better-calibrated one. Watch
the false-positive rate on quotes known to be anonymous.

Slug drift, because the model chooses the role word and does not choose stably:
one model at temperature zero returned "watchman" and "watchmen" for the same
speaker across two runs. Ids reach the cast and the cast reaches the plan
fingerprint. A closed, versioned role vocabulary answers this, at the cost of
the model's freedom to describe a role the list does not hold.

## Staging

Identity first, because everything after it trusts the roster. The narrator
next: it is measured end to end, and moved one book from 69.2% to 93.4%
attributed while leaving an already-good one at 98.6%. Roles last, being the
least validated.

Each step bumps `PROMPT_VERSION`, which keys the attribution store, and each is
measurable alone. Tests precede implementation, per CONTRIBUTING: deterministic,
offline, branch-covered, with the provider supplied through the existing
`Client` protocol.
