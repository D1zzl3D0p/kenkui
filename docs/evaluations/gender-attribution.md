# Speaker and gender attribution evaluation

On 2026-09-17, the default `openrouter/deepseek/deepseek-v4-flash` model was
compared on five passages from four locally available books. Adding a separate
per-speaker gender table to the role-aware prompt produced no measured loss of
speaker accuracy in this sample and recovered gender evidence missed by role
names. This is a small, exploratory passage evaluation, not a whole-book
accuracy estimate or a guarantee about future model/provider versions.

## Controlled results

All requests used temperature 0, reasoning effort `none`, and a 6,000-token
output limit. The controlled comparison pinned OpenRouter to DeepInfra with
fallback disabled. Each passage was run twice per variant. The first four
variants were specified before outputs were inspected; the combined fifth
variant was an exploratory follow-up on the same passages, not a held-out test.

| Prompt | Speaker identity | Final character gender | Both correct | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Original v6 | 241/248 (97.2%) | 209/232 (90.1%) | 209/232 | 4,309 |
| Role-aware v7 | 248/248 (100%) | 224/232 (96.6%) | 224/232 | 4,218 |
| Compact, gender on each quote | 240/248 (96.8%) | 216/232 (93.1%) | 216/232 | 6,189 |
| Compact, gender once per speaker | 242/248 (97.6%) | 232/232 (100%) | 226/232 | 5,084 |
| Role-aware plus per-speaker gender | 248/248 (100%) | 232/232 (100%) | 232/232 | 5,003 |

There were no missing quote IDs, duplicate IDs, malformed completions, or output
truncations among the 50 successful controlled responses. Three initial HTTP
429 failures were retained separately and retried at lower concurrency. These
transport failures are not counted as successful model answers.

The combined variant used 18.6% more output tokens than v7: about 79 extra
output tokens per passage call. Input grew by 57 tokens per call (1.9%). Its
output was smaller than the per-quote encoding. Reported dollar costs vary with
provider caching, and concurrency differed on the follow-up, so this experiment
does not establish a latency or price improvement.

## What changed in the books

| Passage | Quotes | Main finding |
| --- | ---: | --- |
| The Poppy War, opening of chapter 1 | 38 | Original attribution merged or split references to the two proctors. Role-aware instructions kept them distinct in both runs. |
| Pride and Prejudice, opening exchange | 32 | All controlled variants kept the Bennets' alternating and split dialogue correct. |
| Dune, opening passage | 30 | All controlled variants distinguished the speakers and the two non-dialogue quoted runs. |
| Red Rising, prologue | 4 | The role-aware prompt still returned a role without a recognized gender qualifier. The separate gender table recovered masculine gender for all four lines. |
| Red Rising, opening of chapter 1 | 21 | First-person dialogue and the other named speakers stayed correctly attributed. |

The per-quote variant returned the right gender for every scored quote, but
sometimes assigned different people the same generic role ID. Aggregation then
correctly abstained on that conflicting character. The compact per-speaker
variant sometimes gave the same male proctor two IDs: the gender stayed right,
but a cast could give him two voices. Gender correctness alone is therefore
not enough to evaluate audiobook casting.

These results do not support the claim that more instructions necessarily
reduce accuracy. They also do not prove there is no multitask interference.
Here, explicit identity instructions mattered more than minimizing the number
of output fields. Keep identity resolution and gender separate in the data,
while retaining the instructions needed to identify each person consistently.

## Labels and scoring

The 125 quote ranges came from Kenkui's canonical grid. Passages were selected
before calling the model and truncated at a paragraph boundary near 9,000
characters; two passages were shorter. Labels were assigned from those passages
before inspecting outputs and sealed with a SHA-256 digest. They were reviewed
by the coding assistant, not independently adjudicated by multiple humans.

One quote in The Poppy War has an ambiguous examiner identity and is excluded
from speaker scoring, leaving 124 scored identities per repetition. Gender is
scored on 116 source-supported quotes per repetition: the examiner, two
non-speech quoted runs, Hawat's one line, and Barlow's five lines are excluded.
Hawat and Barlow were excluded because the selected context did not offer
sufficiently direct evidence for an unambiguous passage-only gender label.
The experiment does not measure abstention accuracy on a broad set of speakers
whose gender is unknown.

A fixed, manually supplied roster of named characters was held constant across
variants. It contained no unnamed proctors or orator. Prompts received IDs and
names, not the roster's gender values. Scoring started with supported genders
for the named profiles and applied the same role parsing and dialogue-tag
checks to all variants. Direct fields were accepted only for attributed
speakers and only when their evidence agreed. Thus this isolates attribution
and gender recovery; it is not an evaluation of automatic roster discovery.

Named IDs were resolved through Kenkui's normal ID/alias resolver. Unnamed role
IDs were aligned one-to-one with labeled people by maximum agreement. This
allows equivalent role spellings but penalizes merges and splits. Non-speech
quotes must remain unknown. Repeated quotes are not independent samples;
percentages should not be read as statistical confidence bounds.

Ten real combined-variant responses were also replayed through the edited
library's attribution parser, evidence aggregation, SQLite storage, and cached
reuse. Every replay retained the expected genders and reused its stored result
without another model call. The all-unnamed prologue replay supplied a
non-speaking named roster entry to exercise the parser path, since an empty
roster intentionally skips attribution. No audiobook synthesis or listening
assessment was performed.

## Default routing exposed a separate problem

An initial 40-request run used OpenRouter's default provider routing. Some
responses labeled for Dune and Pride and Prejudice instead contained the four
orator assignments from Red Rising, accompanied by that shorter prompt's token
count. Another response was unrelated JSON. These anomalous responses reported
OpenInference as their provider; an additional response reported the token
count of a different prompt variant.

This is observed request/response inconsistency, not evidence that the model
was overloaded by gender instructions. The exact upstream cause is unverified.
The data were preserved, not silently removed from a default-routing accuracy
claim. The controlled rerun separated this routing issue from prompt design.
Production routing has not been changed by this work. Provider policy and
stronger request/response validation warrant separate investigation.

## Library behavior

The v8 response adds `speaker_genders` alongside `attributions`. Gender is
encoded once per speaker as `masculine`, `feminine`, or `null`, using exactly the
same speaker ID as the quote assignments. The role-aware instructions remain;
legacy gender-qualified role parsing is also retained as a fallback.

Invalid values and entries for speakers who were not attributed any dialogue
are ignored. Conflicting evidence across aliases or chapters is logged as
`attribution_gender_ambiguous` and not used as a majority vote. Accepted evidence
is applied before dialogue-tag checks and reviewed overrides, and persisted
separately so a spaCy refresh cannot erase it. The prompt version changes the
attribution cache identity. The final prompt differs from the evaluated
combined prompt only in line wrapping.

Local reproducibility artifacts are under the git-ignored
`evals/gender_attribution/` directory: sealed labels and protocol, prompts, raw
responses, scoring code, aggregate results, transport failures, and replay
results. Copyrighted passages and provider responses are not shipped in the
repository. No provider keys are stored in these artifacts.
