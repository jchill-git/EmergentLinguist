# Morphology (Evan)

The morphology rules live in `conlang_transformation_functions.py`, in the section marked `MORPHOLOGY (Evan)`. There is one section per WALS feature (features that describe the same rule share one), and `sentence_transform()` runs them all through `apply_morphology` as its last step.

| WALS | Value | English → conlang (morphology only) |
|---|---|---|
| 30A, 31A | Two genders, non-sex-based | the old man → oa-old man-mo |
| 33A, 34A | Plural prefix, on all nouns | the stars → xea-star-mo |
| 35A, 44A | Person stem + plural affix; no gender in pronouns | he, she, it → it; they → xea-it |
| 37A, 38A | Definite affix; no indefinite article | the sky → sky-mi; a lamp → lamp |
| 46A | Interrogative-based indefinites | someone, no one → who-do; something → what-do |
| 49A, 51A, 52A, 98A | 6-7 cases, suffixes, "with" is one case, nominative-accusative | in the sky → sky-mi-zh; has a lamp → has lamp-qo |
| 53A | First, two-th, three-th | the second hour → oa-taka-two hour-mo |
| 57A | Possessive prefixes | her sons → a-xea-son |
| 41A, 27A | Two-way demonstratives; no reduplication | no change: this, that |

Case endings: `-qo` object, `-zh` in/at/on, `-ph` to/into, `-ko` from, `-co` owner (of, 's), `-go` with. Other prepositions (near, over, for ...) stay as words, after their noun (85A).

## Markers from other steps

Other steps mark the words they find. `apply_morphology` trusts those marks and only spells them:

| Marker | Added by | Becomes |
|---|---|---|
| `[questionToken]` | `yes_no_to_particle` | **soos**, first in the sentence |
| `appMarker` | `mark_applicative` | **bhu-** prefix on the verb (109A) |
| `causMarker` | `mark_causative_and_reorder` | **-erg** suffix on the verb (111A) |
| `accMarker` | `mark_accusatives` | dropped: the `-qo` ending replaces it |

The marks also decide two case endings. The receiver under a verb with `appMarker` takes `-qo` ("gave the child a book" → child-mo-qo). The person made to do something under a causative takes `-qo` if the second verb has no object ("made him run" → it-qo) and `-ph` if it does ("made him eat the soup" → it-ph).

The purpose marker **-arg** (110A) is not spelled yet, because no step adds a purpose marker. Whichever step adds one, add its name next to `CAUSATIVE_MARKER` and spell it in `_spell_markers`.

## Inputs and outputs

```
apply_morphology(records: list[TokenRecord]) -> list[TokenRecord]
```

- **Input:** TokenRecords in their current order (by `current_index`), after every other step. Fields read: `surface`, `lemma`, `pos`, `tag`, `dep`, `original_index`, `head_index`, `current_index`, `markers`. The input list is not modified.
- **Output:** a new list of copied TokenRecords in the same order, with `current_index` renumbered 0..n-1.
  - `surface` is the conlang spelling. `pos`, `tag` and `dep` are unchanged.
  - English words that became endings or prefixes are left out: "the", "a/an", the prepositions above, "'s", and his/her/my/... before a noun. Words that depended on one are re-attached to its head, so every `head_index` still points to a word in the list.
  - The markers above are removed from `markers`.

Every rule reads a word's English spelling from a copy taken at the start, so it does not matter that earlier rules have already changed `surface`. The order of the rules only decides where affixes go: whole words change first, then endings (`sky-mi-zh`), then prefixes (`a-xea-son`, `oa-taka-two`).

## Place in `sentence_transform`

```
build_token_records(nlp(sentence))
→ yes_no_to_particle → passive_to_active        # sentence rewrites
→ apply_word_order                              # word order (81A, 85A, 86A, 87A, 90A)
→ mark_causative_and_reorder                    # clause shuffle keeps each clause's order
→ mark_accusatives → mark_applicative           # markers
→ apply_morphology                              # morphology; spells the markers
→ translate_words                               # English base words -> conlang words
→ render_sentence → remove dashes               # written form: "sky-mi-zh" -> "skymizh"
```

The dashes between word and affixes are only removed at the very end, so every step before rendering (and `render_sentence(records)` itself) still shows the parts.

`translate_words` (section `WORD TRANSLATION (Evan)`) swaps the base word inside each surface for a conlang word and keeps the affixes: `xea-star-mo-qo` → `xea-cdr-mo-qo`. Names, punctuation and the conlang's own words (`it`, `soos`, `who-do`) stay. It's a trigram cipher: each word is split into 3-letter chunks, each chunk is swapped for a 1-, 2- or 4-letter chunk fixed at setup (`WORD_SEED`), and the chunks are shuffled with a seed made from the word, so a word always gets the same translation. Words of 7 letters or more lose their last two letters first. `evan_cipher_playground.py prompt` writes an LLM test prompt from the passages.

Morphology runs last because the other steps read English words. Word order needs "the", "of" and "in" to find phrases, and `passive_to_active` swaps he/him by spelling. 86A turns "X of Y" into a possessor with dep `poss`, which gets `-co`. Any new step that reads English words must go before `apply_morphology`.

## Checking the output

Run `sample_sentence_transformation.ipynb` top to bottom. It has a **morphology only** cell that runs `apply_morphology` on one sentence without the other steps. The side-by-side cell shows the full pipeline.

`evan_playground.py` prints the first 15 sentences both ways, with morphology on English word order and through the full pipeline:

```
.venv/bin/python evan_playground.py
```
